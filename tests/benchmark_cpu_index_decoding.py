"""Sequential, alternating baseline/candidate CPU benchmarks; no binary swapping.

Example:
  OMP_PROC_BIND=true OMP_PLACES=cores python tests/benchmark_cpu_index_decoding.py \
      --baseline validation/n6/baseline/src/DeepGPR/lib \
      --candidate validation/n6/candidate/src/DeepGPR/lib --output validation/n6/bench

Each process warms up once, then records five samples. Three paired rounds
alternate execution order. Only forward and backward calls are timed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    import DeepGPR
    DeepGPR.native.loader.LIB_DIR = args.library.resolve()
    shape, nt, shots, pml = {
        '2d': ((96, 128, 1), 500, 4, 10),
        '3d': ((28, 32, 24), 100, 1, 4),
        'thin': ((2, 256, 1), 1000, 1, 0),
    }[args.case]
    mode = 3 if args.case == '3d' else 2
    eps_base = torch.linspace(4., 4.7, steps=shape[0]*shape[1]*shape[2]).reshape(shape)
    sigma_base = torch.full(shape, 3e-4)
    source = DeepGPR.ricker(3.5e8, nt, 2e-11, 3e-9).reshape(1, nt, 1)
    src = torch.zeros((shots, 1, 3), dtype=torch.int32)
    rec = torch.zeros((shots, 24, 3), dtype=torch.int32)
    src[:, 0, 0] = torch.linspace(0, shape[0]-1, shots).round().int()
    src[:, 0, 1] = 2
    rec[:, :, 0] = torch.linspace(0, shape[0]-1, 24).round().int()
    rec[:, :, 1] = 3
    if mode == 3:
        src[:, :, 2] = shape[2]//2
        rec[:, :, 2] = shape[2]//2
    rows = []
    for iteration in range(args.repeats + 1):
        eps, sigma = eps_base.clone().requires_grad_(), sigma_base.clone().requires_grad_()
        start = time.perf_counter()
        result = DeepGPR.compute(device='cpu', dx=.02, dt=2e-11,
            source_amplitudes=source, source_location=src, receiver_location=rec,
            eps_r=eps, sigma=sigma, pmlthick=pml, fdtd_order=args.order, mode=mode)
        forward = time.perf_counter() - start
        loss = result[-1].square().mean()
        start = time.perf_counter()
        loss.backward()
        backward = time.perf_counter() - start
        for tensor in (result[-1], eps.grad, sigma.grad):
            assert tensor is not None and torch.isfinite(tensor).all()
        if iteration:
            rows.append(dict(forward=forward, backward=backward, total=forward+backward))
        del result, eps, sigma, loss
        gc.collect()
    library = Path(DeepGPR.get_deepgpr_library_path('cpu'))
    data = dict(case=args.case, shape=shape, nt=nt, shots=shots, pml=pml,
                order=args.order, threads=os.environ['OMP_NUM_THREADS'],
                torch=torch.__version__, library=str(library),
                library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                rows=rows, median={k: statistics.median(r[k] for r in rows) for k in rows[0]})
    args.output.write_text(json.dumps(data, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--case', choices=('2d', '3d', 'thin'))
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--cases', nargs='+', choices=('2d', '3d', 'thin'), default=['2d', '3d', 'thin'])
    parser.add_argument('--threads', nargs='+', type=int, default=[2, 8])
    parser.add_argument('--orders', nargs='+', type=int, choices=(2, 4, 8), default=[2, 8])
    args = parser.parse_args()
    if min(args.repeats, args.rounds, *args.threads) < 1:
        parser.error('repeats, rounds and thread counts must be positive')
    if args.library:
        if args.case is None:
            parser.error('--case is required with --library')
        worker(args)
        return
    if not args.baseline or not args.candidate:
        parser.error('--baseline and --candidate are required')
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for case in args.cases:
        for threads in args.threads:
            for order in (args.orders if case != 'thin' else (2,)):
                paired = []
                for round_index in range(args.rounds):
                    result = {}
                    variants = ('baseline', 'candidate') if round_index % 2 == 0 else ('candidate', 'baseline')
                    for variant in variants:
                        out = args.output/f'{case}_t{threads}_o{order}_r{round_index}_{variant}.json'
                        print(out.name, flush=True)
                        subprocess.run([sys.executable, __file__, '--library', str(getattr(args, variant)),
                            '--case', case, '--order', str(order), '--repeats', str(args.repeats),
                            '--output', str(out)], check=True,
                            env={**os.environ, 'OMP_NUM_THREADS':str(threads), 'MKL_NUM_THREADS':str(threads)})
                        result[variant] = json.loads(out.read_text())['median']
                    paired.append(result)
                medians = {v:{k:statistics.median(r[v][k] for r in paired)
                              for k in ('forward','backward','total')} for v in ('baseline','candidate')}
                reduction = {k:100*(1-medians['candidate'][k]/medians['baseline'][k]) for k in medians['baseline']}
                summaries.append(dict(case=case, threads=threads, order=order,
                                      medians=medians, time_reduction_percent=reduction, paired_rounds=paired))
                (args.output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
                print(summaries[-1], flush=True)


if __name__ == '__main__':
    main()

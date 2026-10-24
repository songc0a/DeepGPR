"""Run finite, bitwise CPU A/B checks without replacing package libraries.

Usage: python tests/verify_cpu_index_decoding.py --baseline DIR --candidate DIR --output DIR
The library directories must contain freshly built deepgpr_cpu.so files.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = output / 'baseline_t1.json'
    if 1 not in args.threads:
        parser.error('--threads must include 1 to produce a serial reference')
    runs = [('baseline', 1)] + [(variant, threads) for threads in args.threads
                               for variant in ('baseline', 'candidate')
                               if (variant, threads) != ('baseline', 1)]
    for variant, threads in runs:
        name = f'{variant}_t{threads}'
        print(name, flush=True)
        command = [sys.executable, str(root/'tests/cpu_index_decoding_cases.py'),
                   str(output/f'{name}.json'), name,
                   '--library-dir', str(getattr(args, variant).resolve())]
        if name != 'baseline_t1':
            command += ['--reference', str(reference)]
        with (output/f'{name}.log').open('w') as log:
            subprocess.run(command, env={**os.environ, 'PYTHONPATH':str(root/'src'),
                                         'OMP_NUM_THREADS':str(threads)},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    print('All A/B comparisons passed', flush=True)


if __name__ == '__main__':
    main()

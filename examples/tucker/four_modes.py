"""The four on/off combinations of the two Tucker modules on the reduced 2D example.

| model Tucker | history sketch |
|---|---|
| off | off | original DeepGPR
| on  | off | Tucker model, exact adjoint
| off | on  | voxel model, sketched history
| on  | on  | both

Each mode runs a few Adam iterations of eps/sigma FWI on the reduced 2D FWI
example (``examples/2DFWImodel.npy``, see ``tests/tucker_cases.py``) and
prints the loss history, the history storage and the gradient error of the
first iteration against the exact gradient.

Run: ``PYTHONPATH=src python examples/tucker/four_modes.py [--iterations 10] [--3d]``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import tucker_cases  # noqa: E402
from DeepGPR.solver.sketched import WavefieldSketchStats  # noqa: E402
from DeepGPR.tucker import (  # noqa: E402
    TuckerMaterialModel,
    TuckerModelConfig,
    TuckerSketchConfig,
    cosine_similarity,
    relative_error,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--3d", dest="three_d", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.three_d:
        case = tucker_cases.case_3d(device)
        ranks = ((3, 8, 10), (2, 6, 8))
        sketch = TuckerSketchConfig(ranks=10, oversampling=6)
    else:
        case = tucker_cases.case_2d(device)
        ranks = ((12, 8), (8, 6))
        sketch = TuckerSketchConfig(ranks=16, oversampling=6)
    observed = tucker_cases.observed(case)
    scale = observed.square().sum()

    reference = None
    for use_model in (False, True):
        for use_sketch in (False, True):
            model = TuckerMaterialModel.from_models(
                case.eps_init,
                case.sigma_init,
                TuckerModelConfig(enabled=use_model, ranks=ranks[0], mapping="softplus", bounds=(1.0, None)),
                TuckerModelConfig(enabled=use_model, ranks=ranks[1], mapping="softplus", bounds=(0.0, None), scale=1e-3),
            )
            lr = 1e-3 if use_model else 2e-2  # Tucker needs a smaller Adam step (docs, Section 1)
            optimizer = torch.optim.Adam(model.parameter_groups(lr, lr))
            losses = []
            stats = WavefieldSketchStats()
            start = time.perf_counter()
            for iteration in range(args.iterations):
                optimizer.zero_grad()
                eps, sigma = model()
                eps.retain_grad()
                sigma.retain_grad()
                data = case.compute(
                    eps, sigma,
                    wavefield_sketch=sketch if use_sketch else None,
                    wavefield_sketch_stats=stats if use_sketch else None,
                )[-1]
                loss = 0.5 * (data - observed).square().sum() / scale
                loss.backward()
                if iteration == 0:
                    first = (eps.grad.detach().clone(), sigma.grad.detach().clone())
                optimizer.step()
                losses.append(float(loss))
            elapsed = time.perf_counter() - start
            name = f"model Tucker {'on ' if use_model else 'off'} | history sketch {'on ' if use_sketch else 'off'}"
            print(name, "| loss", " ".join(f"{v:.4g}" for v in losses), f"| {elapsed / args.iterations:.2f} s/it")
            report = model.parameter_report()
            print(f"    trainable values: eps {report['epsilon_r']['total']}, sigma {report['sigma']['total']}")
            if use_sketch:
                print(f"    history stored {stats.size.stored_bytes / 2**20:.2f} MiB of {stats.size.raw_bytes / 2**20:.2f} MiB (E+R)")
            if not use_model:
                if not use_sketch:
                    reference = first
                else:
                    for label, a, b in (("eps", first[0], reference[0]), ("sigma", first[1], reference[1])):
                        print(f"    first-iteration {label} gradient: rel. error {relative_error(a, b):.2e}, cosine {cosine_similarity(a, b):.7f}")


if __name__ == "__main__":
    main()

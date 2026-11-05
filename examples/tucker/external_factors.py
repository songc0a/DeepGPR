"""External factor matrices -> Tucker reconstruction -> DeepGPR FDTD -> gradients.

DeepGPR contains no neural network. Here plain differentiable tensors stand in
for the outputs of an external network (for example an implicit neural
representation built by the user): each factor is ``U = B @ theta`` with a
fixed coordinate basis ``B`` and a trainable ``theta``. The script

1. reconstructs ``eps_r`` with ``DeepGPR.tucker_reconstruct``,
2. runs the 3D FDTD forward model and an L2 misfit,
3. back-propagates through the native discrete adjoint,
4. checks that the core and every upstream tensor received a gradient and
   that the core gradient equals ``dL/deps ×_n U_n^T`` (chain rule),
5. optionally repeats it with the Tucker-sketched history.

Run: ``PYTHONPATH=src python examples/tucker/external_factors.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import DeepGPR  # noqa: E402
import tucker_cases  # noqa: E402
from DeepGPR.tucker import TuckerSketchConfig, relative_error, tucker_project  # noqa: E402


def coordinate_basis(n: int, degree: int, device) -> torch.Tensor:
    """Chebyshev polynomials of a normalised coordinate: (n, min(degree, n))."""
    degree = min(degree, n)
    x = torch.linspace(-1, 1, n, device=device)
    columns = [torch.ones_like(x), x]
    while len(columns) < degree:
        columns.append(2 * x * columns[-1] - columns[-2])
    return torch.stack(columns[:degree], dim=1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    case = tucker_cases.case_3d(device)
    observed = tucker_cases.observed(case)
    shape = case.eps_init.shape
    ranks = (3, 6, 8)
    degree = 24

    # Fit the stand-in "networks" to the starting model once (least squares in
    # the unconstrained softplus space), so the FWI starts from eps_init.
    target = case.eps_init - 1.0
    target = target + torch.log(-torch.expm1(-target.clamp_min(1e-6)))  # softplus^-1(eps - 1)
    bases = [coordinate_basis(n, degree, device) for n in shape]
    core0, factors0 = DeepGPR.tucker.hosvd(target, ranks)
    thetas = [torch.linalg.lstsq(b, f).solution.clone().requires_grad_(True) for b, f in zip(bases, factors0)]
    core = DeepGPR.tucker.tucker_project(target, [b @ t.detach() for b, t in zip(bases, thetas)])
    core = core.clone().requires_grad_(True)

    for sketch in (None, TuckerSketchConfig(ranks=10, oversampling=6)):
        for tensor in (core, *thetas):
            tensor.grad = None
        factors = [b @ t for b, t in zip(bases, thetas)]   # external code
        unconstrained = DeepGPR.tucker_reconstruct(core, factors)
        unconstrained.retain_grad()
        eps_r = 1.0 + F.softplus(unconstrained)
        print(f"eps_r range {float(eps_r.min()):.3f} .. {float(eps_r.max()):.3f}, "
              f"start-model fit {relative_error(eps_r, case.eps_init):.3e}")
        data = case.compute(eps_r, case.sigma_init, wavefield_sketch=sketch)[-1]
        loss = 0.5 * (data - observed).square().sum() / observed.square().sum()
        loss.backward()
        with torch.no_grad():
            expected = tucker_project(unconstrained.grad, [f.detach() for f in factors])
        label = "sketched history" if sketch else "exact history"
        print(f"[{label}] loss {float(loss):.6e}")
        print(f"  core grad norm {float(core.grad.norm()):.3e}, chain-rule check {relative_error(core.grad, expected):.2e}")
        for axis, theta in zip("xyz", thetas):
            print(f"  theta_{axis} grad norm {float(theta.grad.norm()):.3e}")
        assert all(t.grad is not None and float(t.grad.norm()) > 0 for t in (core, *thetas))


if __name__ == "__main__":
    main()

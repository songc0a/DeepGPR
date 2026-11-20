"""B3 gradients, LTS vs global: whole model and away from the source (GPU, float32).

The benchmark's relative gradient difference is dominated by the source node, whose
self-field (and hence dJ/d eps there) depends on the time step; this script masks it.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models3d as M
from DeepGPR.hybrid import HybridPropagator
CASE, T = "B3_sine", 6e-9
def grad(**kw):
    sol = M.build_hybrid(CASE, refinement=1, flux="central", **kw)
    nt = int(np.ceil(T / sol.dt)) + 1
    src, rec = M.antennas()
    w = torch.tensor(M.wavelet(nt, sol.dt), device="cuda")
    prop = HybridPropagator(sol, src, rec)
    eps, sig = (torch.tensor(a, dtype=torch.float64, device="cuda") for a in M.ground_model(CASE))
    e, s = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)
    d = prop(e, s, w[None]); (0.5 * (d ** 2).sum() * sol.dt).backward()
    return e.grad.cpu().numpy(), s.grad.cpu().numpy(), sol.lts, sol.dt
ge0, gs0, _, dt0 = grad(); ge1, gs1, p, dt1 = grad(lts="auto")
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(a))
print(f"global dt {dt0:.3e}, LTS p={p} dt {dt1:.3e}")
print(f"whole model: eps {rel(ge0, ge1):.4f} sigma {rel(gs0, gs1):.4f}")
src = M.antennas()[0][0, 0]; i, j, k = [int(round(v / M.DX)) for v in src]
m = np.ones(ge0.shape, bool); m[max(i - 3, 0):i + 4, max(j - 3, 0):j + 4, max(k - 3, 0):k + 4] = False
print(f"excluding 3 cells around the source: eps {rel(ge0[m], ge1[m]):.4f} sigma {rel(gs0[m], gs1[m]):.4f}")
print(f"norm fraction of the source neighbourhood (global): {np.linalg.norm(ge0[~m]) / np.linalg.norm(ge0):.4f}")
ib = int(round(M.BAND[1] / M.DX)); m2 = np.zeros(ge0.shape, bool); m2[ib + 1:] = True
print(f"below the band: eps {rel(ge0[m2], ge1[m2]):.4f} sigma {rel(gs0[m2], gs1[m2]):.4f}")

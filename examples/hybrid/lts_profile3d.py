"""Where the time of one LTS macro step goes (B3, p = 3, float32, eager, CUDA events)."""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models3d as M
from DeepGPR.hybrid import functional3d as fn3

sol = M.build_hybrid("B3_sine", refinement=1, flux="central", lts="auto")
st, c = sol.static, sol.coefs
d, y = st.dg, st.yee
state = fn3.zero_state(st)
src, rec = M.antennas()
src_ops = sol.point_ops(src, source=True)
w = torch.tensor(M.wavelet(400, sol.dt), device="cuda", dtype=sol.dtype)
for n in range(50):                                   # warm up and develop a field
    state = fn3.step(st, c, state, src_ops, w[n:n + 1])
torch.cuda.synchronize()
p = d.lts
print(f"B3 p={p} dt={sol.dt:.3e}; K={d.K} Np={d.Np}; Yee {y.shape}")

def timed(label, fn, reps=20):
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        out = fn()
    e1.record(); torch.cuda.synchronize()
    print(f"  {label:48s} {e0.elapsed_time(e1) / reps:7.3f} ms")
    return out

fe, fh = list(state[0:3]), list(state[3:6]); psh = list(state[12:18]); E, H, Baux, Bfd = state[18], state[19], state[21], state[23]
Eext = fn3.interface_trace(st, fe, True)
with torch.no_grad():
    timed("whole macro step (fn3.step)", lambda: fn3.step(st, c, state, src_ops, w[:1]))
    timed("global-style H half step (p=1 path pieces):", lambda: None, 1)
    T = timed("  terms (E-curl, with trace)", lambda: fn3.dg_operator(d, (c.e0, c.e1, c.e2, c.e3), E, H, Eext))
    zero = torch.zeros_like(E)
    Te = timed("  terms (H-curl of eta, no trace)", lambda: fn3.dg_operator(d, (c.h0, c.h1, c.h2, c.h3), E, zero, None))
    timed("  Yee fused h_update", lambda: y.cuda.h_update(fe, fh, psh, Bfd))
    timed("  Yee fused e_update", lambda: y.cuda.e_update(fh, fe, list(state[6:12]), state[22], c, None))
    cb0 = 2.0 * c.dg_cb / (1.0 + c.dg_ca)
    timed("  fused lts_z", lambda: d.cuda.lts_z(Te, E, cb0, 1.0 / p))
    timed("  fused lts_eta", lambda: d.cuda.lts_eta(T, E, E, 0.9, 0.3, 0.3))
    timed("  interface trace (mortar)", lambda: fn3.interface_trace(st, fe, True))
    timed("  H - ch cavg", lambda: H - d.ch[None, None] * E)
    if d.pml_idx is not None:
        u = d.upml_fine
        timed("  UPML p-fold (PML elements)", lambda: fn3._dg_upml_fold(d, p, E, H, H, Baux, u["h_hist"], u["h_new"], u["h_old"], -1.0))
    timed("  lts_h_half_step (whole)", lambda: fn3.lts_h_half_step(st, c, fe, fh, psh, Bfd, E, H, Baux))
    timed("  e_half_step (whole)", lambda: fn3.e_half_step(st, c, fe, fh, list(state[6:12]), state[22], E, H, state[20], Eext, True))
    # slab-only cost: run lts_h_half_step pieces with p sub-steps of slab ops only
    slabs = fn3.lts_slabs(st)
    def slab_ops():
        for sl in slabs:
            zs = [fe[k][:, sl.lo:sl.hi + 1] for k in range(3)]
            r = fn3._slab_curl(y, zs, fn3._fwd)
            q = fn3._slab_curl(y, r, fn3._bwd)
            z2 = [torch.where(sl.fine_e[k], zs[k] + q[k], zs[k]) for k in range(3)]
        return z2
    timed(f"  Yee slab ops (one sub-step's worth)", slab_ops)

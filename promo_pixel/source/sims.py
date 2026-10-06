"""Run the footage simulations with DeepGPR's native kernel. Usage: sims.py <stage>"""
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

import models
from dgpr_np import Sim, ricker

stage = sys.argv[1]
t0 = time.time()

if stage == "scene":
    # Three shots over the pixel-art subsurface; Ez movie per shot.
    ids = models.scene_ids()
    eps, sig = models.to_fields(ids)
    dx, dt, f0 = 0.01, 2e-11, 6e8
    nt, seg = 1650, 15
    w = ricker(f0, nt, dt, 1.2 / f0)
    movies = []
    for sx in models.SHOT_X:
        s = Sim(eps, sig, dx, dt, [[[sx, models.AIR - 2, 0]]], [[[sx + 4, models.AIR - 2, 0]]])
        fr = []
        for k in range(0, nt, seg):
            s.forward(w[k:k + seg])
            fr.append(s.ez()[0, :, :, 0].copy())
        movies.append(np.stack(fr))
        print("shot", sx, time.time() - t0, flush=True)
    np.savez_compressed("out_scene.npz", movies=np.stack(movies).astype(np.float16), dt=dt, seg=seg, dx=dx)

elif stage == "bscan":
    # Common-offset B-scan over the same scene on a 2 cm grid (96 traces).
    ids = models.scene_ids()
    ids2 = ids[::2, ::2]
    eps, sig = models.to_fields(ids2)
    dx, dt, f0 = 0.02, 4e-11, 6e8
    nt = 825
    w = ricker(f0, nt, dt, 1.2 / f0)
    xs = np.arange(1, 190, 2)[:95]
    ay = models.AIR // 2 - 1
    src = np.array([[[x, ay, 0]] for x in xs])
    rec = np.array([[[x + 2, ay, 0]] for x in xs])
    traces = []
    for b in range(0, len(xs), 16):
        s = Sim(eps, sig, dx, dt, src[b:b + 16], rec[b:b + 16])
        traces.append(s.forward(w)[:, :, 0])
        print("batch", b, time.time() - t0, flush=True)
    np.savez_compressed("out_bscan.npz", data=np.concatenate(traces), xs=xs * 2, dt=dt)

elif stage == "cube":
    # 3D forward run (mode 3 kernel): Ez slices through a half-space with a buried sphere.
    n, air = 72, 8
    eps = np.full((n, n, n), 6.0, np.float32)
    sig = np.full((n, n, n), 2e-3, np.float32)
    eps[:, :, :air] = 1.0
    sig[:, :, :air] = 0.0
    eps[:, :, 44:] = 12.0
    xx, yy, zz = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    ball = (xx - 50) ** 2 + (yy - 36) ** 2 + (zz - 30) ** 2 <= 8 ** 2
    eps[ball] = 1.0
    sig[ball] = 0.0
    dx, f0 = 0.02, 4e8
    dt = 3.5e-11
    nt, seg = 440, 4
    w = ricker(f0, nt, dt, 1.2 / f0)
    c = n // 2
    s = Sim(eps, sig, dx, dt, [[[c, c, air - 2]]], [[[c + 3, c, air - 2]]], pml=8, comp=0)
    top, sx_, sy_ = [], [], []
    for k in range(0, nt, seg):
        s.forward(w[k:k + seg])
        P = s.pml
        ex = s.fields[0][0, P[0]:P[0] + n, P[2]:P[2] + n, P[4]:P[4] + n]
        top.append(ex[:, :, air + 1].copy())
        sx_.append(ex[:, c, :].copy())      # x-z plane at y = c
        sy_.append(ex[c, :, :].copy())      # y-z plane at x = c
    print("3d", time.time() - t0, flush=True)
    np.savez_compressed("out_cube.npz", top=np.stack(top), xz=np.stack(sx_), yz=np.stack(sy_),
                        eps_xz=eps[:, c, :], eps_yz=eps[c, :, :], n=n, air=air)

elif stage == "fwi":
    # Dual-parameter (eps_r + sigma) FWI following the recipe of examples/2.2DFWI.ipynb:
    # sources along the top, receivers along the bottom, smoothed start model, Adam,
    # phase 1 updates eps_r only, phase 2 updates eps_r and sigma, L1 misfit + TV.
    k = int(sys.argv[2])                    # coarsening factor (1 = full resolution)
    n1, n2 = int(sys.argv[3]), int(sys.argv[4])
    nshot = int(sys.argv[5])
    tag = sys.argv[6] if len(sys.argv) > 6 else ""
    Et, St = models.fwi_true()
    Et, St = Et[::k, ::k].copy(), St[::k, ::k].copy()
    nx, ny = Et.shape
    dx, dt, f0 = 0.05 * k, 1e-10 * k, 2e8 / k
    nt = 600 if k == 1 else 700
    sxs = np.round(np.linspace(4 // k, nx - 1 - 4 // k, nshot)).astype(int)
    src = np.array([[[x, 1, 0]] for x in sxs])
    rec = np.array([[[x, ny - 2, 0] for x in range(nx)]] * nshot)
    w = ricker(f0, nt, dt, 1.0 / f0)
    obs = Sim(Et, St, dx, dt, src, rec).forward(w)
    sm = 20.0 / k
    E = gaussian_filter(Et, sm, mode="nearest").astype(np.float32)
    s_ms = (gaussian_filter(St, sm, mode="nearest") * 1e3).astype(np.float32)
    E0, S0 = E.copy(), s_ms.copy()
    mask = np.ones_like(E)
    mask[:, :3] = 0
    mask[:, -3:] = 0
    norm = 1.0 / float(np.abs(obs).sum())

    def tv_grad(m_):
        gx = np.sign(np.diff(m_, axis=0, append=m_[-1:, :]))
        gy = np.sign(np.diff(m_, axis=1, append=m_[:, -1:]))
        return -(gx - np.roll(gx, 1, 0)) - (gy - np.roll(gy, 1, 1))

    tv_ratio = 0.1
    hist_E, hist_S, losses = [E.copy()], [s_ms.copy()], []
    grads0, snap = None, None
    mA = [np.zeros_like(E), np.zeros_like(E)]
    vA = [np.zeros_like(E), np.zeros_like(E)]
    steps = [0, 0]
    for it in range(n1 + n2):
        both = it >= n1
        lr = (0.2, 0.0) if not both else (0.1, 0.1)       # sigma lr: 0.1 mS/m = 1e-4 S/m
        if it == n1:                                     # fresh optimiser for phase 2
            mA = [np.zeros_like(E), np.zeros_like(E)]
            vA = [np.zeros_like(E), np.zeros_like(E)]
            steps = [0, 0]
        sim = Sim(E, s_ms * 1e-3, dx, dt, src, rec)
        syn = sim.forward(w, grad=True)
        if snap is None:
            P = sim.pml
            snap = sim.E_saved[::6, nshot // 2, P[0]:P[0] + nx, P[2]:P[2] + ny, 0].copy()
        res = syn - obs
        loss = float(np.abs(res).sum()) * norm
        ge, gs = sim.backward(np.sign(res) * norm)
        ge = ge[:, :, 0]
        gs = gs[:, :, 0] * 1e-3
        if grads0 is None:
            grads0 = (ge.copy(), gs.copy())
        g = []
        for p, (gd, mdl) in enumerate(((ge, E), (gs, s_ms))):
            gt = tv_grad(mdl)
            wgt = np.linalg.norm(gd) / (np.linalg.norm(gt) + 1e-8) * tv_ratio
            g.append((gd + wgt * gt) * mask)
        b1, b2 = 0.9, 0.999
        for p in range(2):
            if lr[p] == 0.0:
                continue
            steps[p] += 1
            mA[p] = b1 * mA[p] + (1 - b1) * g[p]
            vA[p] = b2 * vA[p] + (1 - b2) * g[p] ** 2
            step = lr[p] * (mA[p] / (1 - b1 ** steps[p])) / (np.sqrt(vA[p] / (1 - b2 ** steps[p])) + 1e-30)
            if p == 0:
                E = np.clip(E - step, 1.0, None).astype(np.float32)
            else:
                s_ms = np.clip(s_ms - step, 0.0, None).astype(np.float32)
        losses.append(loss)
        hist_E.append(E.copy())
        hist_S.append(s_ms.copy())
        if it % 5 == 0 or it == n1 + n2 - 1:
            print(f"it {it} loss {loss:.4e} |deps| {np.abs(E - Et).mean():.3f} |dsig| {np.abs(s_ms - St * 1e3).mean():.3f} "
                  f"t {time.time() - t0:.0f}s", flush=True)
            np.savez_compressed(f"out_fwi{tag}.npz", E=np.stack(hist_E), S=np.stack(hist_S), loss=np.array(losses),
                                n1=n1, Et=Et, St=St * 1e3, E0=E0, S0=S0, g_eps=grads0[0], g_sig=grads0[1],
                                snap=snap, sxs=sxs)
print("done", stage, time.time() - t0)

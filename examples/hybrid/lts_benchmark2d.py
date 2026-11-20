"""Local time stepping of the 2D hybrid on the rugged model of ``rugged2d_study.py``.

Model, antennas, wavelet and the converged reference (``ref9`` of
``results/rugged2d/forward_gathers.npz``, written by ``rugged2d_study.py --part
forward``) are those of the rugged study. Two hybrids are compared with and without
local time stepping:

Both arms use central fluxes (the LTS needs them; the rugged study used alpha = 0.1) and
the terrain-following mesh (central fluxes on the Delaunay mesh excite spurious modes
with the antennas inside the band).

* ``r1``: Yee part on the 2.5 cm model grid, DG band h = 1.25 cm (Delaunay P1): the
  band's step is far below the Yee limit, so the LTS takes p ≈ 5-6 sub-steps;
* ``r3``: Yee part on 2.5/3 cm (the setting of the study): the Yee limit is close
  to the band's, p is small.

Reported: gather error against ``ref9`` (offsets >= 0.2 m, 14 shots), 14-shot run
time and peak memory, the LTS-global receiver difference, and the time of one
gradient (one shot, global: hand-written transpose; LTS: autograd transpose).

    python examples/hybrid/lts_benchmark2d.py [--nu 0.3] [--overlap 1] [--lts-cfl 0.9]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import rugged2d_study as R  # noqa: E402

from DeepGPR.hybrid import HybridPropagator, HybridTM2D  # noqa: E402

OUT = R.OUT
# Central fluxes on the Delaunay mesh excite spurious modes with the antennas inside the
# band (docs 5): the benchmark uses the terrain-following mesh, on which central fluxes
# are accurate (both arms).
MESH = "terrain"
MESH_OPTIONS = dict(h_y=R.DG_H)


def build(model, r, **kw):
    eps, sig = model
    return HybridTM2D(eps, sig, R.DX, surface=R.SURFACE, dg_band=R.BAND,
                      pmlthick=int(round(R.PML_M / R.DX)), dg_order=1, dg_h=R.DG_H,
                      mesh=MESH, mesh_options=MESH_OPTIONS, fdtd_refinement=r, device="cuda",
                      dtype=torch.float32, flux="central", **kw)


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, (torch.cuda.max_memory_allocated() - base) / 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--lts-cfl", type=float, default=0.9)
    ap.add_argument("--refine", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--skip-grad", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    g = np.load(OUT / "forward_gathers.npz")
    t_out, ref, keep = g["t"], g["ref9"], g["keep"]
    sel = np.broadcast_to(keep[:, None, :], ref.shape)
    model = R.true_model()
    src, rec = R.antennas()
    log = open(OUT / "lts_benchmark2d.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} 2D LTS benchmark nu={args.nu} overlap={args.overlap} "
        f"lts_cfl={args.lts_cfl}")
    table = {}
    for r in args.refine:
        rows = {}
        for name, kw in (("global", {}), ("lts", dict(lts="auto", lts_nu=args.nu, lts_overlap=args.overlap,
                                                       lts_cfl=args.lts_cfl))):
            sol = build(model, r, nshot=len(R.SRC_Y), **kw)
            nt = R.nt_of(sol.dt)
            res, secs, peak = timed(lambda: sol.run(R.wavelet(nt, sol.dt), src[:, None, :], rec))
            data = R.to_axis(res["receivers"].cpu().numpy(), sol.dt, t_out)
            err = float(np.linalg.norm((data - ref)[sel]) / np.linalg.norm(ref[sel]))
            row = dict(p=sol.lts, dt=sol.dt, dt_fine=sol.dt_fine, yee_limit=sol.yee_dt_limit(),
                       nt=nt, seconds=secs, ms_per_step=1e3 * secs / nt, peak_GB=peak, error=err,
                       triangles=sol.dg.K, yee=[sol.NX, sol.NY])
            say(f"[r={r}] {name}: p={sol.lts} dt={sol.dt:.3e} (fine {sol.dt_fine:.3e}, Yee limit "
                f"{sol.yee_dt_limit():.3e}) nt={nt} 14 shots {secs:.2f} s ({row['ms_per_step']:.3f} "
                f"ms/step) {peak:.2f} GB error vs ref9 {100 * err:.2f} %")
            rows[name] = dict(row, _data=data)
            del sol, res
            torch.cuda.empty_cache()
            if not args.skip_grad:
                one = build(model, r, nshot=1, **kw)
                nt = R.nt_of(one.dt)
                prop = HybridPropagator(one, src[:1, None, :], rec[:1])
                w = R.wavelet(nt, one.dt).to("cuda")
                e = torch.tensor(model[0], dtype=torch.float32, device="cuda").requires_grad_(True)
                s = torch.tensor(model[1], dtype=torch.float32, device="cuda").requires_grad_(True)

                def grad():
                    (0.5 * (prop(e, s, w[None]) ** 2).sum()).backward()
                    return e.grad.clone(), s.grad.clone()

                gr, gs, gm = timed(grad)
                rows[name].update(grad_seconds=gs, grad_ms_per_step=1e3 * gs / nt, grad_peak_GB=gm,
                                  _grad=gr)
                say(f"[r={r}] {name}: gradient (1 shot) {gs:.2f} s ({1e3 * gs / nt:.3f} ms/step) {gm:.2f} GB")
                del one, prop
                torch.cuda.empty_cache()
        a, b = rows["global"].pop("_data"), rows["lts"].pop("_data")
        diff = float(np.linalg.norm((a - b)[sel]) / np.linalg.norm(a[sel]))
        rows["receiver_diff"] = diff
        say(f"[r={r}] LTS vs global receivers {100 * diff:.2f} %, forward speed-up "
            f"{rows['global']['seconds'] / rows['lts']['seconds']:.2f}x")
        if not args.skip_grad:
            ga, gb = rows["global"].pop("_grad"), rows["lts"].pop("_grad")
            rows["gradient_diff"] = [float((x - y).norm() / x.norm()) for x, y in zip(ga, gb)]
            say(f"[r={r}] gradient LTS vs global (eps, sigma) {rows['gradient_diff']}, gradient speed-up "
                f"{rows['global']['grad_seconds'] / rows['lts']['grad_seconds']:.2f}x")
        table[f"r{r}"] = rows
        (OUT / "lts_benchmark2d.json").write_text(json.dumps(dict(args=vars(args), results=table),
                                                             indent=1, default=float))
    say("done")


if __name__ == "__main__":
    main()

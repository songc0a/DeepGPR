"""Figure: the grids of a Huygens subgrid below a DGTD surface band, with a field snapshot.

Vertical slice (x = depth, y) through the model of ``dgtd_subgrid_accuracy.py``:
main Yee grid lines (3 cm), the subgrid's lines (1 cm) over its whole extent,
the inner (IS, total field inside) and outer (OS) Huygens surfaces, the
subgrid CPML, the DG band (elements cut by the slice), the surface, and |Ez|
of the main grid and of the subgrid at one time.

    python examples/subgrid/layout_figure.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from DeepGPR.subgrid import Subgrid
from DeepGPR.subgrid import hybrid as hsub

OUT = Path(__file__).resolve().parent / "results" / "subgrid_layout.png"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    import dgtd_subgrid_accuracy as A

    torch.set_default_dtype(torch.float64)
    _, eps_t, sig = A.models()
    sol = hsub.HybridSubgridMaxwell3D(eps_t, sig, A.H, coarsen=3, surface=A.SURF, dg_band=A.BAND,
                                      pmlthick=8, device="cuda" if torch.cuda.is_available() else "cpu",
                                      subgrids=[Subgrid(A.BOX)])
    nt = 800
    sol.run(A.ricker(nt, sol.dt), A.SRC, A.REC)
    st = sol.state
    sg = sol.static.subs[0]
    dx, pml = sol.dx, sol.pml
    zc = 0.40
    kz = int(round(zc / dx)) + pml[4]
    ez = st[2][0, :, :, kz].cpu().numpy()
    b = hsub.HYB_STATE
    fz = sg.nb + int(round((zc / dx + pml[4] - sg.lo[2]) * sg.ratio))
    efine = st[b + 2][0, :, :, fz].cpu().numpy()
    NX, NY = ez.shape
    xm = (np.arange(NX) - pml[0]) * dx
    ym = (np.arange(NY) - pml[2]) * dx
    f = sg.ratio
    xf = sg.lo[0] - pml[0] + (np.arange(sg.yee.shape[0]) - sg.nb) / f
    yf = sg.lo[1] - pml[2] + (np.arange(sg.yee.shape[1]) - sg.nb) / f
    xf, yf = xf * dx, yf * dx

    fig, ax = plt.subplots(1, 2, figsize=(14, 7.5), constrained_layout=True)
    ground = xm[:, None] > A.BAND[1]
    vmax = float(np.percentile(np.abs(ez[ground.repeat(ez.shape[1], 1)]), 99.5))
    for a, zoom in zip(ax, (False, True)):
        a.pcolormesh(ym, xm, ez, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
        # the subgrid carries the total field only inside the IS (scattered field outside)
        mask = np.zeros_like(efine, dtype=bool)
        hi0 = sg.nb + (sg.hi[0] - sg.lo[0]) * f
        hi1 = sg.nb + (sg.hi[1] - sg.lo[1]) * f
        mask[sg.nb:hi0 + 1, sg.nb:hi1 + 1] = True
        a.pcolormesh(yf, xf, np.ma.masked_where(~mask, efine), cmap="RdBu_r", vmin=-vmax,
                     vmax=vmax, shading="nearest")
        # main grid lines
        segs = [[(y, xm[0]), (y, xm[-1])] for y in ym] + [[(ym[0], x), (ym[-1], x)] for x in xm]
        a.add_collection(LineCollection(segs, colors="0.3", linewidths=0.25, alpha=0.5))
        # fine grid lines (whole subgrid)
        segs = [[(y, xf[0]), (y, xf[-1])] for y in yf] + [[(yf[0], x), (yf[-1], x)] for x in xf]
        a.add_collection(LineCollection(segs, colors="tab:cyan", linewidths=0.2, alpha=0.5))
        lo_i, hi_i = sol.box_bounds(0, "inner")
        lo_o, hi_o = sol.box_bounds(0, "outer")
        a.add_patch(Rectangle((lo_i[1], lo_i[0]), hi_i[1] - lo_i[1], hi_i[0] - lo_i[0], fill=False,
                              ec="tab:green", lw=2.0, label="inner surface (IS)"))
        a.add_patch(Rectangle((lo_o[1], lo_o[0]), hi_o[1] - lo_o[1], hi_o[0] - lo_o[0], fill=False,
                              ec="tab:orange", lw=1.6, ls="--", label="outer surface (OS)"))
        p = 6 / f * dx
        a.add_patch(Rectangle((yf[0], xf[0]), yf[-1] - yf[0], xf[-1] - xf[0], fill=False, ec="c",
                              lw=1.2, label="subgrid extent"))
        a.add_patch(Rectangle((yf[0] + p, xf[0] + p), yf[-1] - yf[0] - 2 * p, xf[-1] - xf[0] - 2 * p,
                              fill=False, ec="c", lw=0.8, ls=":", label="subgrid CPML (inside)"))
        a.axhspan(A.BAND[0], A.BAND[1], color="tab:blue", alpha=0.18, label="DG band")
        yy = np.linspace(ym[0], ym[-1], 400)
        a.plot(yy, A.SURF(yy, np.full_like(yy, zc)), "k-", lw=1.5, label="surface")
        a.plot([p_[1] for p_ in A.REC], [p_[0] for p_ in A.REC], "v", color="k", ms=7)
        a.plot([A.SRC[0][1]], [A.SRC[0][0]], "*", color="r", ms=12)
        a.set_xlabel("y [m]")
        a.set_ylabel("x (depth) [m]")
        a.set_aspect("equal")
        if zoom:
            a.set_xlim(yf[0] - 0.05, yf[-1] + 0.05)
            a.set_ylim(xf[-1] + 0.05, xf[0] - 0.05)
            a.set_title("zoom: main 3 cm grid (grey), subgrid 1 cm (cyan)")
        else:
            a.set_xlim(ym[0], ym[-1])
            a.set_ylim(xm[-1], xm[0])
            a.set_title(f"Ez at t = {nt * sol.dt * 1e9:.1f} ns, slice z = {zc} m (DG band field "
                        "not shown)")
            a.legend(loc="lower left", fontsize=8, framealpha=0.8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

"""Phase 4 figures: surface representations, meshes and Ez snapshots in 3D.

* ``results/phase4_mesh_<case>.png`` - the ground surface as the Yee staircase
  (2 cm voxels, what native DeepGPR sees) and as the body-fitted triangles of
  the tetrahedral DGTD band; a vertical section ``z = 0.61 m`` through the
  source showing the tetrahedra cut by the plane, the Yee grid above and below
  the band, the true surface and the staircase.
* ``results/phase4_<case>.gif`` - ``Ez`` on the vertical section over time for
  staircase FDTD (native DeepGPR, 2 cm), the hybrid (P1 band + Yee at 2/3 cm)
  and the reference (native DeepGPR, 4 mm, cell-averaged surface), with the
  differences to the reference.

    python examples/hybrid/topography3d_figures.py [--cases B3_sine C3_irregular]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import models3d as M  # noqa: E402
from DeepGPR.hybrid.mesh3d import AIR, GROUND  # noqa: E402
from DeepGPR.hybrid.reference3d import FACE_VERTICES  # noqa: E402
from DeepGPR.hybrid.solver3d import FieldSampler3D  # noqa: E402

OUT = M.ROOT / "examples" / "hybrid" / "results"
INK, MUTED, GRID = "#1f2328", "#59636e", "#c8ccd1"
Z_PLANE = M.SRC_YZ[1]
FRAMES = np.arange(1.0, 12.01, 0.5) * 1e-9
DISP_H = 0.01


def physical(sol, pts_ext):
    return pts_ext - np.asarray(sol.origin)


def surface_triangles(sol):
    """Faces between air and ground tetrahedra (physical frame, physical region only)."""
    g, mesh = sol.geometry, sol.mesh
    ks, fs = np.nonzero((g.EToE != np.arange(g.K)[:, None]))
    other = g.EToE[ks, fs]
    sel = (mesh.region[ks] == AIR) & (mesh.region[other] == GROUND)
    tri = mesh.vertices[mesh.tets[ks[sel][:, None], np.array(FACE_VERTICES)[fs[sel]]]]
    tri = tri - np.asarray(sol.origin)
    w = np.array([(M.SHAPE[1] - 1) * M.DX, (M.SHAPE[2] - 1) * M.DX])
    inside = np.all((tri[..., 1] >= -1e-9) & (tri[..., 1] <= w[0] + 1e-9)
                    & (tri[..., 2] >= -1e-9) & (tri[..., 2] <= w[1] + 1e-9), axis=1)
    return tri[inside]


def staircase_quads(case, h=M.DX):
    """Voxel faces of the node-wise staircase (ground from x = (i0 - 1/2) h)."""
    n = M.SHAPE
    Y, Z = np.meshgrid(np.arange(n[1]) * h, np.arange(n[2]) * h, indexing="ij")
    d = M.CASES[case]["surface"](Y, Z)
    i0 = np.ceil(d / h - 1e-9)
    xt = (i0 - 0.5) * h
    quads = []
    for j in range(n[1]):
        for k in range(n[2]):
            y0, y1, z0, z1 = (j - 0.5) * h, (j + 0.5) * h, (k - 0.5) * h, (k + 0.5) * h
            x = xt[j, k]
            quads.append([(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)])
            if j + 1 < n[1] and xt[j + 1, k] != x:
                x2 = xt[j + 1, k]
                quads.append([(x, y1, z0), (x2, y1, z0), (x2, y1, z1), (x, y1, z1)])
            if k + 1 < n[2] and xt[j, k + 1] != x:
                x2 = xt[j, k + 1]
                quads.append([(x, y0, z1), (x2, y0, z1), (x2, y1, z1), (x, y1, z1)])
    return np.array(quads), xt


def section(sol, z0):
    """Polygons of the tetrahedra cut by ``z = z0`` (physical (x, y) coordinates)."""
    mesh = sol.mesh
    p = mesh.vertices[mesh.tets] - np.asarray(sol.origin)        # (K, 4, 3)
    s = p[..., 2] - z0
    hit = (s.min(1) < 0) & (s.max(1) > 0)
    polys = []
    for q in p[hit]:
        pts = []
        for a in range(4):
            for b in range(a + 1, 4):
                sa, sb = q[a, 2] - z0, q[b, 2] - z0
                if sa * sb < 0:
                    t = sa / (sa - sb)
                    pts.append(q[a] + t * (q[b] - q[a]))
        pts = np.array(pts)[:, :2]
        c = pts.mean(0)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
        polys.append(pts[:, ::-1])                                  # (y, x) for plotting
    return polys


def mesh_figure(case, sol):
    tri = surface_triangles(sol)
    quads, xt = staircase_quads(case)
    fig = plt.figure(figsize=(15, 9.5), facecolor="white")
    lim = dict(xlim=(0, 1.18), ylim=(0, 1.18), zlim=(0.36, 0.22))
    for i, (title, polys, xs) in enumerate((
            ("Yee staircase (native DeepGPR, 2 cm voxels)", quads, quads[..., 0].mean(1)),
            ("body-fitted DGTD surface (tetrahedron faces)", tri, tri[..., 0].mean(1)))):
        ax = fig.add_subplot(2, 3, 1 + i, projection="3d")
        verts = [[(v[1], v[2], v[0]) for v in poly] for poly in polys]
        cm = plt.cm.terrain_r((xs - 0.22) / 0.16)
        coll = Poly3DCollection(verts, facecolors=cm, edgecolors=(0.2, 0.2, 0.2, 0.25),
                                linewidths=0.15)
        ax.add_collection3d(coll)
        ax.set_xlim(*lim["xlim"])
        ax.set_ylim(*lim["ylim"])
        ax.set_zlim(*lim["zlim"])
        ax.set_xlabel("y (m)", fontsize=7)
        ax.set_ylabel("z (m)", fontsize=7)
        ax.set_zlabel("depth x (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=28, azim=-60)
        ax.set_title(title, fontsize=9, color=INK)
    ax = fig.add_subplot(2, 3, 3)
    ys = np.linspace(0, 1.18, 400)
    ax.plot(ys, M.CASES[case]["surface"](ys, np.full_like(ys, Z_PLANE)), color="#16a34a", lw=1.5,
            label="true surface")
    j = np.arange(M.SHAPE[1])
    k = int(round(Z_PLANE / M.DX))
    xs = xt[:, k]
    ax.step(np.concatenate([j * M.DX - 0.01, [j[-1] * M.DX + 0.01]]),
            np.concatenate([xs, [xs[-1]]]), where="post", color="#c2410c", lw=1.0,
            label="Yee staircase")
    ax.set_ylim(0.40, 0.18)
    ax.set_title(f"surface on the section z = {Z_PLANE:.2f} m", fontsize=9, color=INK)
    ax.set_xlabel("y (m)", fontsize=8)
    ax.set_ylabel("depth x (m)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    polys = section(sol, Z_PLANE)
    for c, (yl, xl, title) in enumerate((((0.0, 1.18), (0.78, 0.0), "vertical section z = 0.61 m"),
                                         ((0.40, 0.80), (0.48, 0.14), "zoom"))):
        ax = fig.add_subplot(2, 2, 3 + c)
        h = M.DX / 3 if sol.fdtd_refinement == 3 else M.DX
        step = 1 if c == 1 else 3
        segs = []
        for xg in np.arange(0, 0.78 + 1e-9, h * step):
            if not (M.BAND[0] < xg < M.BAND[1]):
                segs.append([(0, xg), (1.18, xg)])
        for yg in np.arange(0, 1.18 + 1e-9, h * step):
            segs.append([(yg, 0), (yg, M.BAND[0])])
            segs.append([(yg, M.BAND[1]), (yg, 0.78)])
        ax.add_collection(LineCollection(segs, colors=GRID, linewidths=0.4))
        ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="#1d4ed8",
                                         linewidths=0.35))
        ax.plot(ys, M.CASES[case]["surface"](ys, np.full_like(ys, Z_PLANE)), color="#16a34a", lw=1.4)
        ax.step(np.concatenate([j * M.DX - 0.01, [j[-1] * M.DX + 0.01]]),
                np.concatenate([xs, [xs[-1]]]), where="post", color="#c2410c", lw=1.0)
        for xb in M.BAND:
            ax.axhline(xb, color=INK, lw=0.8, ls="--")
        ax.plot([M.SRC_YZ[0]], [M.ANT_X], marker="*", color="#dc2626", ms=10)
        ax.set_xlim(*yl)
        ax.set_ylim(*xl)
        ax.set_aspect("equal")
        ax.set_title(f"{title}: DGTD tetrahedra (blue), Yee grid "
                     f"{'(every 3rd line) ' if step == 3 else ''}(grey), true surface (green), "
                     f"staircase (orange)", fontsize=8, color=INK)
        ax.set_xlabel("y (m)", fontsize=8)
        ax.set_ylabel("depth x (m)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{M.CASES[case]['label']} - surface representation and meshes", fontsize=12,
                 color=INK)
    fig.savefig(OUT / f"phase4_mesh_{case}.png", dpi=110)
    plt.close(fig)


def snapshots(case, sol):
    path = OUT / "phase4" / f"{case}_snapshots.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    xd = np.arange(0, 0.78 + 1e-9, DISP_H)
    yd = np.arange(0, 1.18 + 1e-9, DISP_H)
    X, Y = np.meshgrid(xd, yd, indexing="ij")
    out = {"x": xd, "y": yd}
    for tag, (eps, sig, h) in (("staircase", M.staircase_model(case) + (M.DX,)),
                               ("reference", M.averaged_model(case, M.DX / 5) + (M.DX / 5,))):
        frames, xg, yg = M.native_snapshots(eps, sig, h, FRAMES, Z_PLANE)
        out[tag] = np.stack([RegularGridInterpolator((xg, yg), f, bounds_error=False,
                                                     fill_value=0.0)((X, Y)) for f in frames])
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, Z_PLANE)], 1)
    sampler = FieldSampler3D(sol, pts)
    steps = {int(round(t / sol.dt)) - 1: i for i, t in enumerate(FRAMES)}
    frames = np.zeros((len(FRAMES),) + X.shape)

    def cb(n, s):
        if n in steps:
            frames[steps[n]] = sampler().cpu().numpy().reshape(X.shape)

    M.run_hybrid(sol, callback=cb, callback_every=1)
    out["hybrid"] = frames
    np.savez(path, **out)
    return out


def snapshot_gif(case, sol):
    s = snapshots(case, sol)
    x, y = s["x"], s["y"]
    ref = s["reference"]
    Yg, Xg = np.meshgrid(y, x)
    far = np.hypot(Xg - M.ANT_X, Yg - M.SRC_YZ[0]) > 0.15     # scale without the source near field
    vmax = np.percentile(np.abs(ref[:, far]), 99.7)
    surf = M.CASES[case]["surface"](y, np.full_like(y, Z_PLANE))
    k = int(round(Z_PLANE / M.DX))
    j = np.arange(M.SHAPE[1])
    d_st = M.CASES[case]["surface"](j * M.DX, np.full(j.size, k * M.DX))
    x_st = (np.ceil(d_st / M.DX - 1e-9) - 0.5) * M.DX
    cols = [("staircase", "staircase FDTD (native DeepGPR, 2 cm)"),
            ("hybrid", "hybrid: P1 DGTD band + Yee 2/3 cm"),
            ("reference", "reference: native DeepGPR 4 mm, averaged surface")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), facecolor="white")
    ims = []
    ext = (y[0], y[-1], x[-1], x[0])
    for c, (key, title) in enumerate(cols):
        for r in range(2):
            ax = axes[r, c]
            if r == 1 and key == "reference":
                ax.axis("off")
                continue
            data = s[key][0] if r == 0 else s[key][0] - ref[0]
            lim = vmax if r == 0 else vmax * 0.25
            im = ax.imshow(data, extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
            ax.plot(y, surf, color="#16a34a", lw=1.0)
            if key == "staircase":
                ax.step(np.concatenate([j * M.DX - 0.01, [j[-1] * M.DX + 0.01]]),
                        np.concatenate([x_st, [x_st[-1]]]), where="post", color="#c2410c", lw=0.8)
            if key == "hybrid":
                for xb in M.BAND:
                    ax.axhline(xb, color=INK, lw=0.6, ls="--")
            ax.plot([M.SRC_YZ[0]], [M.ANT_X], marker="*", color="#dc2626", ms=8)
            ax.set_xlim(y[0], y[-1])
            ax.set_ylim(x[-1], x[0])
            ax.set_title(title if r == 0 else f"{key} - reference (colour x4)", fontsize=8.5,
                         color=INK)
            ax.tick_params(labelsize=7, colors=MUTED)
            ims.append((im, key, r))
    title = fig.suptitle("", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(i):
        for im, key, r in ims:
            im.set_data(s[key][i] if r == 0 else s[key][i] - ref[i])
        title.set_text(f"{M.CASES[case]['label']} - Ez on the section z = {Z_PLANE:.2f} m, "
                       f"t = {FRAMES[i] * 1e9:.1f} ns")
        return []

    anim = FuncAnimation(fig, update, frames=len(FRAMES), interval=250)
    anim.save(OUT / f"phase4_{case}.gif", writer=PillowWriter(fps=4), dpi=75)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["B3_sine", "C3_irregular"])
    args = ap.parse_args()
    (OUT / "phase4").mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        print(case, flush=True)
        sol = M.build_hybrid(case, refinement=3)
        mesh_figure(case, sol)
        snapshot_gif(case, sol)
        del sol
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

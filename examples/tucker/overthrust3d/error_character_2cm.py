"""Character of the block-Tucker gradient error (2 cm, shot 5): wavenumber spectra and coherence with the gradient.

Spectra are DCT-II (even extension). A first version used a Tukey-windowed FFT, whose taper removed the top ~9
rows, where the gradient is strongest and least smooth, and so understated the high wavenumbers of the gradient.

    python error_character_2cm.py gradcheck_block_2cm error_character_2cm_v2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.fft import dct, dctn  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

DX = 0.02
CASES = {  # label: (tag, method)
    "tol 3 %": ("full_it30_shot5", "block0.03"),
    "tol 3 % + source region": ("full_it30_shot5_src0.1", "block0.03_src0.1"),
    "tol 1.5 % + source region": ("full_it30_shot5_src0.1", "block0.015_src0.1"),
}


def tukey(n: int, frac: float = 0.1) -> np.ndarray:
    w = np.ones(n)
    m = max(int(frac * n / 2), 1)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
    w[:m], w[-m:] = ramp, ramp[::-1]
    return w


def radial_spectrum(a: np.ndarray, window, edges: np.ndarray, kr: np.ndarray):
    """DCT-II power (even extension: no taper, no wrap-around jump, the surface rows are kept)."""
    p = dctn(a.astype(np.float32), type=2, norm="ortho", workers=16).astype(np.float64) ** 2
    s, _ = np.histogram(kr, bins=edges, weights=p.ravel())
    return s


def axis_spectrum(a: np.ndarray, axis: int) -> np.ndarray:
    return (dct(a.astype(np.float32), type=2, norm="ortho", axis=axis, workers=16).astype(np.float64) ** 2).sum(
        axis=tuple(i for i in range(3) if i != axis))


def main() -> None:
    src_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(exist_ok=False)
    acq = json.loads((src_dir.parent / "run16_250MHz" / "acquisition.json").read_text())
    sx, _, sz = (int(v) for v in acq["sources"][5])
    g = np.load(src_dir / "full_it30_shot5_shot05_exact.npy").astype(np.float64)
    nx, ny, nz = g.shape
    window = None
    kx, ky, kz = (np.arange(n) / (2 * n * DX) for n in (nx, ny, nz))  # DCT-II index m -> m / (2 N dx) cycles / m
    kr = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).ravel()
    edges = np.linspace(0, 25 * np.sqrt(3), 120)
    centers = 0.5 * (edges[1:] + edges[:-1])
    res = {"k_cycles_per_m": centers.tolist()}
    spec = {"exact gradient": radial_spectrum(g, window, edges, kr)}
    axis = {"exact gradient": [axis_spectrum(g, i) for i in range(3)]}
    # source column (horizontal distance <= 0.6 m) excluded for the "away from the source" statistics
    xx, zz = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    far = (np.hypot(xx - sx, zz - sz) > 30)[:, None, :]
    smooth = gaussian_filter(g, 1.5)
    high = g - smooth
    errors = {}
    for label, (tag, method) in CASES.items():
        e = np.load(src_dir / f"{tag}_shot05_{method}.npy").astype(np.float64) - g
        errors[label] = e
        spec[label] = radial_spectrum(e, window, edges, kr)
        axis[label] = [axis_spectrum(e, i) for i in range(3)]
        ef, gf = e * far, g * far

        def corr(a, b):
            return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

        tot = spec[label].sum()
        res[label] = dict(
            rel=float(np.linalg.norm(e) / np.linalg.norm(g)),
            rel_far=float(np.linalg.norm(ef) / np.linalg.norm(gf)),
            corr_err_grad=corr(e, g),
            corr_err_grad_far=corr(ef, gf),
            corr_err_highpass_far=corr(ef, high * far),
            share_err_above_12=float(spec[label][centers > 12].sum() / tot),
            share_err_above_6=float(spec[label][centers > 6].sum() / tot),
        )
    res["exact gradient"] = dict(
        share_above_12=float(spec["exact gradient"][centers > 12].sum() / spec["exact gradient"].sum()),
        share_above_6=float(spec["exact gradient"][centers > 6].sum() / spec["exact gradient"].sum()))
    a, b = errors["tol 3 % + source region"] * far, errors["tol 1.5 % + source region"] * far
    res["corr_err3_err15_far"] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    # ---- figure ----
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.3, left=0.05, right=0.985, top=0.88, bottom=0.07)
    colors = {"exact gradient": "#555555", "tol 3 %": "#d9822b", "tol 3 % + source region": "#8a4fd3",
              "tol 1.5 % + source region": "#2e9e6a"}
    ax = fig.add_subplot(gs[0, 0])
    ref = spec["exact gradient"].max()
    for label, s in spec.items():
        ax.semilogy(centers, s / ref, color=colors[label], lw=1.6 if label == "exact gradient" else 1.2, label=label)
    ax.set_xlabel("radial wavenumber |k| (cycles / m)")
    ax.set_ylabel("power (max of exact = 1)")
    ax.set_title("3-D radial DCT power spectra, it-30 model\n(whole ground, surface rows included)", fontsize=10)
    ax.set_ylim(1e-9, 2)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=7.5)
    ax = fig.add_subplot(gs[0, 1])
    for label, s in spec.items():
        if label == "exact gradient":
            continue
        ax.semilogy(centers, s / spec["exact gradient"], color=colors[label], label=label)
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_xlabel("radial wavenumber |k| (cycles / m)")
    ax.set_ylabel("error power / exact power")
    ax.set_title("error-to-gradient power ratio per wavenumber", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=7.5)
    names = ("x", "depth", "z")
    for i, ax_i in enumerate((1,)):
        ax = fig.add_subplot(gs[0, 2])
        k = np.arange(ny) / (2 * ny * DX)
        for label, s in axis.items():
            ax.semilogy(k, s[1] / axis["exact gradient"][1].max(), color=colors[label], label=label)
        for m in range(1, 4):
            ax.axvline(m / 0.64, color="#bbbbbb", lw=0.6, ls="--")
        ax.set_xlabel("depth wavenumber (cycles / m); dashed: multiples of 1 / (32 cells)")
        ax.set_ylabel("power (max of exact = 1)")
        ax.set_title("1-D DCT spectra along depth (block seams every 0.64 m)", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    ax = fig.add_subplot(gs[0, 3])
    k = np.arange(nx) / (2 * nx * DX)
    for label, s in axis.items():
        ax.semilogy(k, s[0] / axis["exact gradient"][0].max(), color=colors[label], label=label)
    ax.set_xlabel("x wavenumber (cycles / m)")
    ax.set_title("1-D DCT spectra along x", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    # zoomed sections away from the source: x 9-13 m, depth 0.4-2.4 m on the z plane of the source
    x0, x1, y0, y1 = 450, 650, 20, 120
    gsec = g[x0:x1, y0:y1, sz]
    c = float(np.percentile(np.abs(gsec), 99.5))
    extent = (x0 * DX, x1 * DX, y1 * DX, y0 * DX)
    panels = [("exact gradient", gsec, 1), ("tol 3 % + source region", None, 10), ("tol 1.5 % + source region", None, 10)]
    for j, (label, sec, amp) in enumerate(panels):
        if sec is None:
            sec = errors[label][x0:x1, y0:y1, sz]
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(sec.T * amp, extent=extent, aspect="auto", cmap="RdBu_r", vmin=-c, vmax=c, interpolation="nearest")
        ax.set_title(f"{label}{'' if amp == 1 else f' minus exact (x{amp})'}\n"
                     f"x 9-13 m, depth 0.4-2.4 m, z = {sz * DX:.2f} m", fontsize=9)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("depth (m)", fontsize=8)
        ax.axhline(0.64, color="k", lw=0.4, ls=":")
        ax.axhline(1.28, color="k", lw=0.4, ls=":")
        ax.axhline(1.92, color="k", lw=0.4, ls=":")
    ax = fig.add_subplot(gs[1, 3])
    sec = high[x0:x1, y0:y1, sz]
    ax.imshow(-sec.T * 10, extent=extent, aspect="auto", cmap="RdBu_r", vmin=-c, vmax=c, interpolation="nearest")
    ax.set_title("what a smoothing error would look like:\n-(exact - Gaussian(1.5 cells) of exact) (x10)", fontsize=9)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("depth (m)", fontsize=8)
    fig.suptitle("Overthrust 2 cm, shot 5, exact-FWI it-30 model: the block-Tucker gradient error is not a smoothed gradient.\n"
                 "Dotted lines in the sections: block seams (32 cells = 0.64 m). Away from the source = horizontal distance > 0.6 m.",
                 fontsize=10.5)
    fig.savefig(out / "error_character_2cm_it30.png", dpi=105)
    (out / "error_character_2cm_it30.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "k_cycles_per_m"}, indent=1))


if __name__ == "__main__":
    main()

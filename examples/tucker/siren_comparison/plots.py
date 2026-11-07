"""Figures 1-7 of the SIREN vs Tucker-SIREN study, generated only from result files.

Usage: ``python plots.py [--quick]`` -> ``results/<scale>/figures/*.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
SIZE_ORDER = ["50^3", "100^3", "120x120x100", "160^3", "200^3"]
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"siren": "#1f3a5f", "siren_chunk": "#4f79a8", "r8": "#c7522a", "r12": "#e5c185", "r16": "#74a892", "r24": "#008585", "dense": "#555555", "direct": "#b5838d"}


def load(path: Path) -> Dict:
    return json.loads(path.read_text()) if path.exists() else {}


def savefig(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# -- Figure 1 -----------------------------------------------------------------

def figure1(out: Path) -> None:
    """Schematic of the two parameterizations (data flow only, no data)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")

    def box(ax, x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", lw=0.8, alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(ax, x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"))

    ax = axes[0]
    ax.set_title("Standard SIREN", fontsize=10)
    box(ax, 0.1, 1.5, 2.0, 1.0, "all grid points\n(Nx Ny Nz) x 3", "#dfe7f2")
    box(ax, 2.9, 1.5, 2.2, 1.0, "SIREN\n3 -> 2 (L x W)", "#9fb7d6")
    box(ax, 5.9, 1.5, 1.6, 1.0, "logits\n(Nx Ny Nz) x 2", "#dfe7f2")
    box(ax, 8.1, 1.5, 1.8, 1.0, "mapping\neps_r, sigma", "#f2e2c4")
    for x0, x1 in ((2.1, 2.9), (5.1, 5.9), (7.5, 8.1)):
        arrow(ax, x0, 2.0, x1, 2.0)
    ax = axes[1]
    ax.set_title("Tucker-SIREN (per property: eps_r and sigma separately)", fontsize=10)
    for i, (label, y) in enumerate((("x: Nx", 3.0), ("y: Ny", 1.75), ("z: Nz", 0.5))):
        box(ax, 0.1, y, 1.3, 0.7, f"{label}\npoints", "#dfe7f2")
        box(ax, 1.9, y, 1.7, 0.7, "1D SIREN\n1 -> r", "#9fb7d6")
        box(ax, 4.1, y, 1.3, 0.7, f"U_{label[0]}\n{label[3:]} x r", "#dfe7f2")
        arrow(ax, 1.4, y + 0.35, 1.9, y + 0.35)
        arrow(ax, 3.6, y + 0.35, 4.1, y + 0.35)
        arrow(ax, 5.4, y + 0.35, 6.1, 2.1)
    box(ax, 4.1, -0.25, 1.3, 0.55, "core G\nr x r x r", "#c9d9c3")
    arrow(ax, 5.4, 0.0, 6.1, 1.9)
    box(ax, 6.1, 1.6, 1.9, 0.9, "DeepGPR\ntucker_reconstruct\nG x1 Ux x2 Uy x3 Uz", "#74a892")
    box(ax, 8.4, 1.6, 1.5, 0.9, "mapping\neps_r, sigma", "#f2e2c4")
    arrow(ax, 8.0, 2.05, 8.4, 2.05)
    fig.text(0.5, -0.02, "Both feed the same DeepGPR CUDA FDTD and discrete adjoint; gradients return through the mapping, the reconstruction and the networks.", ha="center", fontsize=8)
    savefig(fig, out, "fig1_architecture")


# -- Figures 2/3 ----------------------------------------------------------------

def _series(gen: Dict, key_fn, value_fn):
    series: Dict[str, Dict[str, float]] = {}
    for name, r in gen.items():
        if "spec" not in r:
            continue
        label = key_fn(name, r)
        if label is None:
            continue
        size = r["spec"]["size"]
        if r.get("oom"):
            series.setdefault(label, {})[size] = float("nan")
            continue
        if "cuda_event_seconds" not in r:
            continue
        series.setdefault(label, {})[size] = value_fn(r)
    return series


def _label(name, r):
    spec = r["spec"]
    if spec["group"] == "C_matched":
        return None
    if spec["method"] == "siren":
        return "SIREN 4x128" + (" (chunked)" if spec.get("chunk") else "")
    if spec["layers"] == 4:
        return f"T-SIREN 4x128 r{spec['rank']}"
    if spec["hidden"] == 32:
        return f"T-SIREN 2x32 r{spec['rank']}"
    return None


def _plot_series(ax, series, sizes, ylabel, title):
    x = np.arange(len(sizes))
    for label, values in sorted(series.items()):
        y = [values.get(s, float("nan")) for s in sizes]
        style = "-" if "SIREN 4x128" in label and "T-" not in label else ("--" if "4x128" in label else ":")
        ax.plot(x, y, style, marker="o", ms=3, lw=1.2, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=20)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)


def figure2_3(gen: Dict, out: Path) -> None:
    sizes = [s for s in SIZE_ORDER if any(r.get("spec", {}).get("size") == s for r in gen.values())]
    if not sizes:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    forward = _series(gen, _label, lambda r: (r["cuda_event_seconds"]["network"]["mean"] + r["cuda_event_seconds"]["reconstruction"]["mean"] + r["cuda_event_seconds"]["mapping"]["mean"]) * 1e3)
    backward = _series(gen, _label, lambda r: r["cuda_event_seconds"]["backward"]["mean"] * 1e3)
    total = _series(gen, _label, lambda r: r["cuda_event_seconds"]["total"]["mean"] * 1e3)
    _plot_series(axes[0], forward, sizes, "ms", "model generation (forward)")
    _plot_series(axes[1], backward, sizes, "ms", "backward")
    _plot_series(axes[2], total, sizes, "ms", "forward + backward + Adam step")
    axes[2].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Fig. 2  Model-generation time (CUDA events, mean of 20; missing = OOM)", fontsize=10)
    savefig(fig, out, "fig2_generation_time")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    alloc = _series(gen, _label, lambda r: r["peak_allocated_bytes"] / 2**20)
    reserved = _series(gen, _label, lambda r: r["peak_reserved_bytes"] / 2**20)
    _plot_series(axes[0], alloc, sizes, "MiB", "peak allocated")
    _plot_series(axes[1], reserved, sizes, "MiB", "peak reserved")
    axes[1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Fig. 3  GPU memory of one generation + backward + Adam step", fontsize=10)
    savefig(fig, out, "fig3_generation_memory")


# -- Figure 4 -------------------------------------------------------------------

def _fwi_groups(fwi: Dict, case: str) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for name, r in fwi.items():
        if "history" not in r or not name.startswith(f"{case}|"):
            continue
        label = name.split("|")[1]
        groups.setdefault(label, []).append(r)
    return groups


def figure4(fwi: Dict, out: Path) -> None:
    for case in ("3d", "2d"):
        groups = _fwi_groups(fwi, case)
        if not groups:
            continue
        labels = sorted(groups, key=lambda l: (l != "dense", l))
        parts = [("generation_s", "model generation", "#e5c185"), ("fdtd_forward_s", "FDTD forward", "#1f3a5f"), ("adjoint_s", "discrete adjoint", "#4f79a8"), ("network_backward_s", "network backward", "#c7522a"), ("optimizer_s", "optimizer", "#74a892")]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        bottom = np.zeros(len(labels))
        for key, text, color in parts:
            values = np.array([statistics.median(h[key] for r in groups[l] for h in r["history"][1:]) for l in labels])
            ax.bar(labels, values, bottom=bottom, color=color, label=text)
            bottom += values
        ax.set_ylabel("seconds per FWI iteration (median)")
        ax.set_title(f"Fig. 4  Time per FWI iteration ({case.upper()})", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        plt.xticks(rotation=25)
        savefig(fig, out, f"fig4_fwi_time_breakdown_{case}")


# -- Figure 5 -------------------------------------------------------------------

def figure5(rep: Dict, out: Path) -> None:
    rows = [(name, r) for name, r in rep.items() if "eps" in r and not r.get("spec", {}).get("lr_search")]
    if not rows:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for name, r in rows:
        color = COLORS["siren"] if name.startswith("siren") else (COLORS["direct"] if name.startswith("direct") else COLORS.get(name.split("_")[-1], "#74a892"))
        marker = "s" if name.startswith("siren") else ("^" if name.startswith("direct") else "o")
        p = r["parameters"]["total"]
        err = r["eps"]["rmse"]
        axes[0].scatter(p, err, color=color, marker=marker)
        axes[1].scatter(r["train_seconds"], err, color=color, marker=marker)
        if r.get("peak_allocated_bytes"):
            axes[2].scatter(r["peak_allocated_bytes"] / 2**20, err, color=color, marker=marker)
        label = name + (" (diverged)" if err > 1.0 else "")
        for ax in axes[:3]:
            ax.annotate(label, (p if ax is axes[0] else (r["train_seconds"] if ax is axes[1] else (r["peak_allocated_bytes"] or 0) / 2**20), err), fontsize=5)
    for ax, xl in zip(axes[:3], ("trainable parameters", "training time [s]", "peak GPU memory [MiB]")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("eps RMSE (subsurface)")
    for prefix, color, label in (("tsiren_2x32_r", COLORS["r16"], "Tucker-SIREN 2x32"), ("direct_tucker_r", COLORS["direct"], "direct Tucker (optimized)")):
        pts = sorted((int(n.split("_r")[-1]), r["eps"]["rmse"]) for n, r in rows if n.startswith(prefix) and n.split("_r")[-1].isdigit())
        if pts:
            axes[3].plot(*zip(*pts), "o-", color=color, label=label)
    siren = rep.get("siren_4x128", {})
    if "eps" in siren:
        axes[3].axhline(siren["eps"]["rmse"], color=COLORS["siren"], ls="--", label="SIREN 4x128")
    axes[3].set_xlabel("Tucker rank")
    axes[3].set_ylabel("eps RMSE")
    axes[3].legend(fontsize=7, frameon=False)
    fig.suptitle("Fig. 5  Representation of the true 3D model (eps_r; squares SIREN, circles Tucker-SIREN, triangles direct Tucker)", fontsize=10)
    savefig(fig, out, "fig5_representation")


# -- Figure 6 -------------------------------------------------------------------

def figure6(fwi: Dict, out: Path) -> None:
    for case in ("3d", "2d"):
        groups = _fwi_groups(fwi, case)
        if not groups:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
        for label in sorted(groups):
            runs = groups[label]
            color = None
            n = min(len(r["history"]) for r in runs)
            loss = np.array([[h["loss"] for h in r["history"][:n]] for r in runs])
            time_axis = np.cumsum([statistics.fmean(r["history"][i]["iteration_s"] for r in runs) for i in range(n)])
            mean = loss.mean(0)
            line, = axes[0].plot(np.arange(1, n + 1), mean, label=f"{label} (n={len(runs)})")
            color = line.get_color()
            if len(runs) > 1:
                axes[0].fill_between(np.arange(1, n + 1), loss.min(0), loss.max(0), color=color, alpha=0.15)
            axes[1].plot(time_axis, mean, color=color)
            pts = [(h["iteration"] + 1, h["errors"]["eps"]["rmse"], h["errors"]["eps"]["ssim"]) for h in runs[0]["history"] if "errors" in h]
            if pts:
                it = [p[0] for p in pts]
                rmse = np.array([[h["errors"]["eps"]["rmse"] for h in r["history"] if "errors" in h][: len(pts)] for r in runs])
                ssim_v = np.array([[h["errors"]["eps"]["ssim"] for h in r["history"] if "errors" in h][: len(pts)] for r in runs])
                axes[2].plot(it, rmse.mean(0), color=color)
                axes[3].plot(it, ssim_v.mean(0), color=color)
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[0].set_xlabel("iteration")
        axes[1].set_xlabel("wall time [s] (excl. pre-fit)")
        for ax, yl in zip(axes, ("data loss", "data loss", "eps RMSE", "eps SSIM")):
            ax.set_ylabel(yl)
        axes[2].set_xlabel("iteration")
        axes[3].set_xlabel("iteration")
        axes[0].legend(fontsize=6, frameon=False)
        fig.suptitle(f"Fig. 6  FWI convergence ({case.upper()}; mean over seeds, band = min-max)", fontsize=10)
        savefig(fig, out, f"fig6_fwi_convergence_{case}")


# -- Figure 7 -------------------------------------------------------------------

def figure7(fwi: Dict, results: Path, out: Path, quick: bool, protocol: str = "v1") -> None:
    sys.path.insert(0, str(HERE.parents[2] / "src"))
    sys.path.insert(0, str(HERE.parent))
    import benchmark_tucker as bt

    for case in ("3d", "2d"):
        run_dir = results / "fwi_runs" / case
        finals = sorted(run_dir.glob("*seed0_final.pt")) if run_dir.exists() else []
        if not finals:
            continue
        if case == "3d" and protocol == "v5":
            import bench_fwi

            truth = bench_fwi.build_case_large(quick, torch.device("cpu"))
        else:
            truth = bt.build_case(quick, "cpu") if case == "3d" else bt.build_case_2d("cpu", quick)
        initial = torch.load(run_dir / "initial.pt") if (run_dir / "initial.pt").exists() else {"eps": truth["eps_init"], "sigma": truth["sigma_init"]}
        panels = [("true", truth["eps"], truth["sigma"])]
        order = ["dense", "siren_4x128", "tsiren_2x32_r8", "tsiren_2x32_r12", "tsiren_2x32_r16", "tsiren_2x32_r24", "direct_tucker_r16"]
        for label in order:
            path = run_dir / f"{case}|{label}|seed0_final.pt"
            if path.exists():
                data = torch.load(path)
                panels.append((label, data["eps"], data["sigma"]))
        panels.append(("initial", initial["eps"], initial["sigma"]))
        tall = eps_ratio = panels[0][1].dim() == 2
        fig, axes = plt.subplots(2, len(panels), figsize=(1.7 * len(panels), 6.4 if tall else 4.6))
        eps_range = (float(truth["eps"].min()), float(truth["eps"].max()))
        sig = torch.log10(truth["sigma"].clamp_min(1e-6))
        sig_range = (float(sig.min()), float(sig.max()))
        for column, (label, eps, sigma) in enumerate(panels):
            # Display conventions of the example notebooks: 3D slice eps[x_mid] =
            # (y = depth, z), 3.3DFWI.ipynb; 2D imshow(eps) = (x vertical, y
            # horizontal), 2.2DFWI.ipynb (x is depth, boreholes at y = 0 and ny-1).
            if eps.dim() == 3:
                eps_slice, sigma_slice = eps[eps.shape[0] // 2], sigma[sigma.shape[0] // 2]
            else:
                eps_slice, sigma_slice = eps, sigma
            axes[0, column].imshow(eps_slice.numpy(), cmap="viridis", vmin=eps_range[0], vmax=eps_range[1], aspect="auto")
            axes[1, column].imshow(torch.log10(sigma_slice.clamp_min(1e-6)).numpy(), cmap="magma", vmin=sig_range[0], vmax=sig_range[1], aspect="auto")
            axes[0, column].set_title(label, fontsize=7)
            for row in (0, 1):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        axes[0, 0].set_ylabel("eps_r")
        axes[1, 0].set_ylabel("log10 sigma")
        fig.suptitle(f"Fig. 7  Reconstructions ({case.upper()}, seed 0{', central x slice' if case == '3d' else ''}; common colour scales)", fontsize=10)
        savefig(fig, out, f"fig7_reconstructions_{case}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--protocol", default="v1", choices=["v1", "v2", "v3", "v4", "v5"])
    args = parser.parse_args()
    base = "quick" if args.quick else "full"
    results = HERE / "results" / (base if args.protocol == "v1" else f"{base}_{args.protocol}")
    shared = HERE / "results" / base  # generation / representation results are protocol-independent
    out = results / "figures"
    fwi = load(results / "fwi.json")
    if args.protocol == "v1":  # protocol-independent figures are written once
        figure1(out)
        figure2_3(load(shared / "generation.json"), out)
        figure5(load(shared / "representation.json"), out)
    figure4(fwi, out)
    figure6(fwi, out)
    figure7(fwi, results, out, args.quick, args.protocol)
    print("figures in", out)


if __name__ == "__main__":
    main()

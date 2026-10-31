"""Aggregate statistics, figures and GIFs of ``amr_budget_study.py``.

    python examples/subgrid/amr_budget_study.py report

writes ``report.md`` and the figures into ``examples/subgrid/results/amr_budget/``.
Works on partial results (scenarios still running are left out).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import amr_budget_study as S

SUBJECT = "amr_fine"          # round 2: dynamic AMR v2 is the method under test
MARKERS = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "P", "M6": "X"}


def baselines(subject):
    return [k for k in S.METHODS if k != subject]


def original(res):
    return [n for n in res if n.split("_")[0] in S.ORIGINAL]


def fresh(res):
    return [n for n in res if n.split("_")[0] in S.FRESH]


def load(root: Path):
    out = {}
    for name in S.scenarios():
        p = root / name / "result.json"
        if p.exists():
            r = json.loads(p.read_text())
            if any("pe" in v for v in r.get("methods", {}).values()):
                out[name] = r
    return out


def final_pe(r, k, frame=S.N_FRAMES):
    v = r["methods"].get(k, {})
    return v.get("pe", {}).get(str(frame))


def paired(a, b, seed=0):
    """AMR values ``a`` against baseline values ``b`` (lower is better)."""
    from scipy.stats import binomtest, wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = np.log(a / b)
    n = len(r)
    out = dict(n=n, wins=int((r < 0).sum()), gm_ratio=float(np.exp(r.mean())),
               median_ratio=float(np.exp(np.median(r))))
    rng = np.random.default_rng(seed)
    m = r[rng.integers(0, n, (20000, n))].mean(1)
    out["ci"] = [float(np.exp(np.percentile(m, 2.5))), float(np.exp(np.percentile(m, 97.5)))]
    out["p_wilcoxon"] = float(wilcoxon(r).pvalue) if n >= 5 and np.any(r != 0) else float("nan")
    out["p_sign"] = float(binomtest(out["wins"], n, 0.5).pvalue) if n else float("nan")
    return out


def stats_table(res, names, metric="pe", frame=S.N_FRAMES, subject=SUBJECT):
    rows = {}
    for b in baselines(subject):
        a_, b_ = [], []
        for nm in names:
            r = res[nm]
            if metric == "pe":
                x, y = final_pe(r, subject, frame), final_pe(r, b, frame)
            elif metric == "pe_xf":
                x, y = r["methods"].get(subject, {}).get("pe_xf"), r["methods"].get(b, {}).get("pe_xf")
            else:
                x = r["methods"].get(subject, {}).get("metrics", {}).get(metric)
                y = r["methods"].get(b, {}).get("metrics", {}).get(metric)
            if x is not None and y is not None and x > 0 and y > 0:
                a_.append(x)
                b_.append(y)
        if len(a_) >= 2:
            rows[b] = paired(a_, b_)
    return rows


def fmt_stats(rows, title, subject=SUBJECT):
    lines = [f"**{title}**", "",
             f"| {S.LABELS[subject]} vs | scenarios | better | geo-mean ratio (it / baseline) [95 % CI] | "
             "median ratio | Wilcoxon p | sign-test p |", "|---|---|---|---|---|---|---|"]
    for b, v in rows.items():
        lines.append(f"| {S.LABELS[b]} | {v['n']} | {v['wins']} | {v['gm_ratio']:.3f} "
                     f"[{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | {v['median_ratio']:.3f} | "
                     f"{v['p_wilcoxon']:.3g} | {v['p_sign']:.3g} |")
    return lines + [""]


# --------------------------------------------------------------------------- figures
def _budget_ticks(a, marks):
    from matplotlib.ticker import NullFormatter, NullLocator
    t = [marks[k - 1] for k in S.PE_FRAMES]
    a.xaxis.set_minor_locator(NullLocator())
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xticks(t, [f"{v:.0f}" if v >= 10 else f"{v:.1f}" for v in t])

def fig_pe_vs_budget(res, root, plt):
    names = [n for n in res if n.endswith("_snr20")]
    freqs = sorted({res[n]["f0"] for n in names})
    models = [m for m in S.MODELS if any(n.startswith(m + "_") for n in names)]
    if not names:
        return
    fig, ax = plt.subplots(len(freqs), len(models), figsize=(3.6 * len(models), 3.0 * len(freqs)),
                           squeeze=False, constrained_layout=True, sharex=True)
    for i, f0 in enumerate(freqs):
        for j, m in enumerate(models):
            a = ax[i, j]
            nm = f"{m}_f{int(f0 / 1e6)}_snr20"
            if nm not in res:
                a.axis("off")
                continue
            r = res[nm]
            for k in S.METHODS:
                v = r["methods"].get(k, {})
                if "pe" not in v:
                    continue
                ks = sorted(v["pe"], key=float)
                a.loglog([r["marks"][int(q) - 1] for q in ks], [v["pe"][q] for q in ks],
                         marker="o", ms=3.5, color=S.COLORS[k], ls=S.STYLES.get(k, "-"), lw=1.6,
                         label=S.LABELS[k])
            a.axhline(r["evaluator_floor"], color="0.6", lw=0.7)
            _budget_ticks(a, r["marks"])
            a.set_title(f"{m} ({S.MODEL_TITLES[m]}), {f0 / 1e6:.0f} MHz", fontsize=9)
            if j == 0:
                a.set_ylabel("held-out PE")
            if i == len(freqs) - 1:
                a.set_xlabel("compute budget [s]")
            a.grid(True, which="major", color="0.9", lw=0.6)
    ax[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Held-out prediction error against compute budget (SNR 20 dB); grey line: evaluator floor")
    fig.savefig(root / "summary_pe_vs_budget.png", dpi=120)
    plt.close(fig)


def fig_paired(res, root, plt, subject=SUBJECT):
    names = [n for n in res if n.endswith("_snr20")] + [n for n in res if not n.endswith("_snr20")]
    bl = baselines(subject)
    nc = 4
    nr = int(math.ceil(len(bl) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.5 * nr), constrained_layout=True, squeeze=False)
    ax = ax.ravel()
    for a in ax[len(bl):]:
        a.axis("off")
    markers = MARKERS
    fcol = {200.0: "#4c72b0", 300.0: "#dd8452", 400.0: "#55a868"}
    for a, b in zip(ax, bl):
        lo, hi = 1e9, 0
        for nm in names:
            r = res[nm]
            x, y = final_pe(r, b), final_pe(r, subject)
            if x is None or y is None:
                continue
            filled = r["snr_db"] == 20
            c = fcol.get(r["f0"] / 1e6, "k")
            a.scatter(x, y, marker=markers[r["model"]], s=34, color=c if filled else "none",
                      edgecolor=c, lw=1.2)
            lo, hi = min(lo, x, y), max(hi, x, y)
        if hi > 0:
            g = np.array([lo / 1.3, hi * 1.3])
            a.plot(g, g, color="0.5", lw=0.8)
            a.fill_between(g, g * 0 + g[0] / 10, g, color="#d62728", alpha=0.05)
            a.set_xlim(*g)
            a.set_ylim(*g)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(f"PE, {S.LABELS[b]}")
        a.set_ylabel(f"PE, {S.LABELS[subject]}", fontsize=8)
        a.set_title("below the diagonal: it is better", fontsize=8)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mk, ls="", color="k", label=f"{m}") for m, mk in markers.items()]
    h += [Line2D([], [], marker="o", ls="", color=c, label=f"{f:.0f} MHz") for f, c in fcol.items()]
    h += [Line2D([], [], marker="o", ls="", mfc="none", color="k", label="SNR 10 / 30 dB")]
    ax[len(bl) - 1].legend(handles=h, fontsize=7, loc="lower right")
    fig.suptitle(f"Paired comparison at the same budget, {S.LABELS[subject]} (each point = one scenario)")
    fig.savefig(root / f"paired_scatter_{subject}.png", dpi=120)
    plt.close(fig)


def fig_ratio_by_budget(res, root, plt, names, tag, subject=SUBJECT):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    fr = [k / S.N_FRAMES for k in S.PE_FRAMES]
    drawn = False
    for b in baselines(subject):
        g, lo, hi = [], [], []
        for k in S.PE_FRAMES:
            st = stats_table(res, names, "pe", k, subject).get(b)
            g.append(st["gm_ratio"] if st else np.nan)
            lo.append(st["ci"][0] if st else np.nan)
            hi.append(st["ci"][1] if st else np.nan)
        if not np.isfinite(g).any():
            continue
        drawn = True
        ax.plot(fr, g, marker="o", color=S.COLORS[b], ls=S.STYLES.get(b, "-"), label=f"vs {S.LABELS[b]}")
        ax.fill_between(fr, lo, hi, color=S.COLORS[b], alpha=0.12)
    if not drawn:                 # the method has no results in this set yet
        plt.close(fig)
        return
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(fr, [f"T/{int(1 / f)}" if f < 1 else "T" for f in fr])
    ax.set_xlabel(f"compute budget (T = {S.BUDGET:.0f} s)")
    ax.set_ylabel(f"geo-mean PE ratio {S.LABELS[subject]} / baseline", fontsize=8)
    ax.set_title(f"{tag}: < 1 means {S.LABELS[subject]} is better (bands: bootstrap 95 % CI, "
                 f"{len(names)} scenarios)", fontsize=9)
    ax.legend(fontsize=7)
    fig.savefig(root / f"ratio_by_budget_{tag}_{subject}.png", dpi=120)
    plt.close(fig)


def fig_noise(res, root, plt):
    models = list(S.ORIGINAL)
    fig, ax = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.4), constrained_layout=True,
                           squeeze=False)
    any_ = False
    for j, m in enumerate(models):
        a = ax[0, j]
        for k in S.METHODS:
            xs, ys = [], []
            for snr in (10, 20, 30):
                nm = f"{m}_f300_snr{snr}"
                if nm in res and final_pe(res[nm], k) is not None:
                    xs.append(snr)
                    ys.append(final_pe(res[nm], k))
            if xs:
                any_ = True
                a.semilogy(xs, ys, marker="o", color=S.COLORS[k], ls=S.STYLES.get(k, "-"), label=S.LABELS[k])
        a.set_xlabel("SNR [dB]")
        a.set_title(f"{m} ({S.MODEL_TITLES[m]}), 300 MHz", fontsize=9)
        a.set_xticks([10, 20, 30])
        a.grid(True, color="0.9", lw=0.6)
    ax[0, 0].set_ylabel("held-out PE at the budget")
    ax[0, 0].legend(fontsize=7)
    if any_:
        fig.suptitle("Noise study: prediction error at the same budget against SNR")
        fig.savefig(root / "noise_trend.png", dpi=120)
    plt.close(fig)


def fig_heatmap(res, root, plt):
    names = list(res)
    kinds = list(S.METHODS)
    M = np.full((len(names), len(kinds)), np.nan)
    for i, nm in enumerate(names):
        for j, k in enumerate(kinds):
            v = final_pe(res[nm], k)
            if v is not None:
                M[i, j] = v
    best = np.nanmin(np.where(np.array(kinds)[None] == "oracle", np.nan, M), axis=1, keepdims=True)
    R = M / best
    fig, ax = plt.subplots(figsize=(1.25 * len(kinds) + 2.5, 0.36 * len(names) + 1.6), constrained_layout=True)
    im = ax.imshow(np.log2(R), cmap="Reds", vmin=0, vmax=3, aspect="auto")
    for i in range(len(names)):
        for j in range(len(kinds)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="w" if np.log2(R[i, j]) > 1.8 else "k",
                        fontweight="bold" if R[i, j] <= 1.0 + 1e-9 else "normal")
    ax.set_xticks(range(len(kinds)), [S.LABELS[k] for k in kinds], rotation=25, fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    fig.colorbar(im, ax=ax, label="log2(PE / best non-oracle PE)")
    ax.set_title("Held-out PE at the budget (bold: best without prior knowledge)", fontsize=9)
    fig.savefig(root / "heatmap_pe.png", dpi=120)
    plt.close(fig)


def fig_longrun(root, res, plt):
    lr = sorted(root.glob("longrun_*/result.json"))
    for p in lr:
        r = json.loads(p.read_text())
        fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
        for k in S.METHODS:
            v = r["methods"].get(k, {})
            if "pe" not in v:
                continue
            ks = sorted(v["pe"], key=float)
            ax.loglog([r["marks"][int(q) - 1] for q in ks], [v["pe"][q] for q in ks], marker="o",
                      color=S.COLORS[k], ls=S.STYLES.get(k, "-"), label=S.LABELS[k])
        ax.axhline(r["evaluator_floor"], color="0.6", lw=0.7)
        _budget_ticks(ax, r["marks"])
        ax.axvline(S.BUDGET, color="0.3", lw=0.8, ls=":")
        ax.text(S.BUDGET * 1.03, ax.get_ylim()[1] * 0.8, "main budget T", fontsize=7)
        ax.set_xlabel("compute budget [s]")
        ax.set_ylabel("held-out PE")
        ax.set_title(f"Long run ({r['budget']:.0f} s): {r['scenario']}", fontsize=9)
        ax.legend(fontsize=7)
        fig.savefig(root / f"{p.parent.name}.png", dpi=120)
        plt.close(fig)


FDTD1_TARGETS = ("fdtd3", "static_full", "amr_fine_once", "amr_fine")


def load_fdtd1(root, res):
    out = {}
    for nm in res:
        p = root / nm / "fdtd1_long.json"
        if p.exists():
            r = json.loads(p.read_text())
            if "pe" in r:
                out[nm] = r
    return out


def fig_fdtd1(root, res, f1, plt):
    """Time the uniform 1 cm solver needs to reach the PE other methods reach at T."""
    if not f1:
        return
    names = list(f1)
    fig, ax = plt.subplots(figsize=(max(8, 0.95 * len(names) + 2), 4.8), constrained_layout=True)
    w = 0.8 / len(FDTD1_TARGETS)
    for j, k in enumerate(FDTD1_TARGETS):
        xs, ys, hatch = [], [], []
        for i, nm in enumerate(names):
            t = f1[nm].get("time_to_reach", {}).get(k)
            if t is None:
                continue
            reached = t["seconds"] is not None
            xs.append(i + (j - (len(FDTD1_TARGETS) - 1) / 2) * w)
            ys.append(t["seconds"] if reached else f1[nm]["budget"])
            hatch.append(not reached)
        for x, y, hh in zip(xs, ys, hatch):
            ax.bar(x, y, w, color=S.COLORS[k], alpha=0.35 if hh else 1.0, hatch="//" if hh else None,
                   edgecolor=S.COLORS[k])
        ax.bar([], [], color=S.COLORS[k], label=f"PE of {S.LABELS[k]} at T")
    ax.axhline(S.BUDGET, color="k", lw=0.9, ls="--")
    ax.text(-0.5, S.BUDGET * 1.06, f"T = {S.BUDGET:.0f} s (budget of the other methods)", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)), names, rotation=35, fontsize=8, ha="right")
    ax.set_ylabel("time FDTD 1 cm needs [s]")
    ax.set_title("Uniform 1 cm FDTD: time to reach the held-out PE that each method reaches in "
                 f"{S.BUDGET:.0f} s (hatched: not reached within {S.FDTD1_BUDGET:.0f} s)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(root / "fdtd1_time_to_accuracy.png", dpi=120)
    plt.close(fig)


def fdtd1_lines(res, f1):
    if not f1:
        return []
    L = ["## Uniform 1 cm FDTD run to convergence (long budget)", "",
         f"The 1 cm solver ran for {S.FDTD1_BUDGET:.0f} s per scenario (same GPU, same settings); the "
         f"other methods had T = {S.BUDGET:.0f} s. 'time to reach X' = when the 1 cm run first reaches "
         "the held-out PE that method X reaches at T (log-linear between marks of "
         f"{S.FDTD1_BUDGET / S.N_FRAMES:.0f} s); speed-up = that time / T.", "",
         "| scenario | s / gradient | iterations | PE at T | PE at 3T | PE at 6T | PE at 12T | "
         + " | ".join(f"time to reach {S.LABELS[k]} (speed-up)" for k in FDTD1_TARGETS) + " |",
         "|---|---|---|---|---|---|---|" + "---|" * len(FDTD1_TARGETS)]
    for nm, r in f1.items():
        def pe_at(t):
            ks = [int(k) for k in r["pe"] if r["marks"][int(k) - 1] <= t + 1e-6]
            return r["pe"][str(max(ks))] if ks else 1.0
        cells = []
        for k in FDTD1_TARGETS:
            t = r.get("time_to_reach", {}).get(k)
            if t is None:
                cells.append("—")
            elif t["seconds"] is None:
                cells.append(f"> {r['budget']:.0f} s (> {r['budget'] / S.BUDGET:.0f}x)")
            else:
                cells.append(f"{t['seconds']:.0f} s ({t['seconds'] / S.BUDGET:.1f}x)")
        L.append(f"| {nm} | {r['seconds_per_evaluation']:.1f} | {r['iterations']} | {pe_at(S.BUDGET):.4f} | "
                 f"{pe_at(3 * S.BUDGET):.4f} | {pe_at(6 * S.BUDGET):.4f} | {pe_at(12 * S.BUDGET):.4f} | "
                 + " | ".join(cells) + " |")
    return L + [""]


def _box_edges(ax, bx, color, lw=0.8, ls="--"):
    (x0, x1), (y0, y1), (z0, z1) = bx
    cs = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    for i in range(8):
        for j in range(i + 1, 8):
            d = sum(1 for u, v in zip(cs[i], cs[j]) if u != v)
            if d == 1:
                ax.plot([cs[i][2], cs[j][2]], [cs[i][1], cs[j][1]], [cs[i][0], cs[j][0]], color=color,
                        lw=lw, ls=ls)


def iso_gif(root, name, r, plt, frames=30):
    """Rotating isosurfaces of the anomalies (the same thresholds for every model)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    ctx_e, ctx_s = S.model_arrays(r["model"])
    rs = S.roi_slices()
    kinds = [k for k in ("fdtd3", "static_full", "amr", "amr_fine", "amr_fine_once", "oracle")
             if (root / name / f"snap_{k}.npz").exists()]
    cols = [("true model", ctx_e[rs], ctx_s[rs], None)]
    for k in kinds:
        z = np.load(root / name / f"snap_{k}.npz")
        boxes = r["methods"][k].get("final_boxes") or r["methods"][k].get("boxes")
        cols.append((S.LABELS[k], z["eps"][-1], z["sig"][-1], boxes))
    de = ctx_e[rs] - S.EPS0
    ds = ctx_s[rs] - S.SIG0
    levels = []
    if de.max() > 0:
        levels.append(("eps", +1, 0.2 * de.max(), "#d62728", "eps_r high"))
    if de.min() < 0:
        levels.append(("eps", -1, 0.2 * -de.min(), "#1f77b4", "eps_r low"))
    if ds.max() > 0:
        levels.append(("sig", +1, 0.2 * ds.max(), "#2ca02c", "sigma high"))
    off = np.array([s.start for s in rs]) * S.H
    meshes = []
    for nm, e, s, boxes in cols:
        ms = []
        for par, sgn, lev, col, _ in levels:
            a = (e - S.EPS0) if par == "eps" else (s - S.SIG0)
            a = gaussian_filter(sgn * a.astype(np.float64), 0.7)
            if a.max() <= lev:
                continue
            v, f, _, _ = marching_cubes(a, lev, spacing=(S.H,) * 3)
            ms.append((v + off, f, col))
        meshes.append((nm, ms, boxes))
    n = len(meshes)
    nc = 3
    nr = int(math.ceil(n / nc))
    imgs = []
    for fr in range(frames):
        fig = plt.figure(figsize=(4.6 * nc, 3.9 * nr))
        for j, (nm, ms, boxes) in enumerate(meshes):
            ax = fig.add_subplot(nr, nc, j + 1, projection="3d")
            for v, f, col in ms:
                pc = Poly3DCollection(v[f][:, :, ::-1], facecolor=col, edgecolor="none", alpha=0.85)
                ax.add_collection3d(pc)
            _box_edges(ax, S.ROI, "0.6", lw=0.6, ls=":")
            for bx in boxes or []:
                _box_edges(ax, bx, "k")
            ax.set_xlim(S.ROI[2])
            ax.set_ylim(S.ROI[1])
            ax.set_zlim(S.ROI[0][1], S.ROI[0][0])
            ax.set_box_aspect((1, 1, 0.55))
            ax.view_init(elev=28, azim=-60 + 360 * fr / frames)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel("z", fontsize=8, labelpad=-12)
            ax.set_ylabel("y", fontsize=8, labelpad=-12)
            ax.set_zlabel("depth", fontsize=8, labelpad=-12)
            ax.set_title(nm, fontsize=10)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.9, wspace=0.0, hspace=0.05)
        lg = " | ".join(f"{lab}: {'red' if c == '#d62728' else 'blue' if c == '#1f77b4' else 'green'}"
                        for _, _, _, c, lab in levels)
        fig.suptitle(f"{name}: anomalies above 20 % of the true contrast after {r['budget']:.0f} s "
                     f"({lg}; dashed: subgrid boxes)", fontsize=9)
        imgs.append(S._fig_to_image(fig, 80))
        plt.close(fig)
    S.save_gif(imgs, root / name / "iso3d.gif", 120)


# --------------------------------------------------------------------------- report
def report(root: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = load(root)
    if not res:
        print("no results yet")
        return
    orig = original(res)
    main = [n for n in orig if n.endswith("_snr20")]
    fr = fresh(res)
    f1 = load_fdtd1(root, res)
    fig_pe_vs_budget(res, root, plt)
    for subj in (SUBJECT, "amr_fine_once"):
        fig_paired(res, root, plt, subj)
        if main:
            fig_ratio_by_budget(res, root, plt, main, "snr20", subj)
        fig_ratio_by_budget(res, root, plt, orig, "original", subj)
        if len(fr) >= 2:
            fig_ratio_by_budget(res, root, plt, fr, "fresh", subj)
    fig_noise(res, root, plt)
    fig_heatmap(res, root, plt)
    fig_longrun(root, res, plt)
    fig_fdtd1(root, res, f1, plt)
    for nm in res:
        g = root / nm / "iso3d.gif"
        if nm.endswith("_f300_snr20") and (not g.exists() or
                                           g.stat().st_mtime < (root / nm / "result.json").stat().st_mtime):
            try:
                iso_gif(root, nm, res[nm], plt)
            except Exception as exc:          # figures must not stop the report
                print(f"iso3d {nm} failed: {exc!r}")
    L = ["# Equal-budget FWI study: dynamic AMR vs static refinement vs uniform FDTD", "",
         f"Budget T = {S.BUDGET:.0f} s per inversion (wall clock, RTX 4090), {len(res)} scenarios "
         f"finished ({len(orig)} of round 1 on M1-M4, {len(fr)} on the fresh models M5/M6). Primary "
         "score: held-out prediction error PE (6 held-out shots, noise-free response, uniform 1 cm "
         "evaluator against the 0.5 cm reference data; 0 = perfect, 1 = start model).", "",
         "Round 2 adds AMR v2 ('finer partition'): indicator of the 3 cm grid alone at the current "
         f"model (boxes may shrink, move or vanish), Doerfler marking restricted to eta >= {S.GAMMA} "
         f"max eta, no a priori criteria of the current model, clustering efficiency "
         f"{S.EFFICIENCY_V2}, hysteresis {S.HYSTERESIS}. These settings were chosen on round-1 "
         "snapshots of M1-M4, so M1-M4 results of v2 are not independent of the choice; M5/M6 "
         "are fresh models that played no part in it.", ""]
    L += ["## Held-out PE at the budget", "",
          "| scenario | " + " | ".join(S.LABELS[k] for k in S.METHODS) + " | evaluator floor |",
          "|---|" + "---|" * (len(S.METHODS) + 1)]
    for nm, r in res.items():
        vals = {k: final_pe(r, k) for k in S.METHODS}
        ok = [v for k, v in vals.items() if v is not None and k != "oracle"]
        best = min(ok) if ok else None
        cells = []
        for k in S.METHODS:
            v = vals[k]
            cells.append("—" if v is None else (f"**{v:.4f}**" if v == best else f"{v:.4f}"))
        L.append(f"| {nm} | " + " | ".join(cells) + f" | {r['evaluator_floor']:.4f} |")
    L.append("")
    L += fdtd1_lines(res, f1)
    L.append("## Paired statistics (same budget)")
    L.append("")
    for subj in (SUBJECT, "amr_fine_once"):
        sets = (("fresh models M5/M6 (validation)", fr), ("round 1, SNR 20 dB", main),
                ("round 1, all 20 scenarios", orig))
        for tag, names in sets:
            if len(names) < 2:
                continue
            L += fmt_stats(stats_table(res, names, "pe", subject=subj), f"held-out PE at T, {tag}", subj)
            if subj == SUBJECT:
                for k in S.PE_FRAMES[:-1]:
                    L += fmt_stats(stats_table(res, names, "pe", k, subj),
                                   f"held-out PE at T/{S.N_FRAMES // k}, {tag}", subj)
            L += fmt_stats(stats_table(res, names, "pe_xf", subject=subj),
                           f"PE at 4/3 f0 (cross-frequency), {tag}", subj)
            L += fmt_stats(stats_table(res, names, "eps_err_smooth", subject=subj),
                           f"smoothed eps_r error, {tag}", subj)
            L += fmt_stats(stats_table(res, names, "sig_err_smooth", subject=subj),
                           f"smoothed sigma error, {tag}", subj)
    L += ["## Cost and layouts", "",
          "| scenario | method | iterations | gradients | s / gradient | peak GB | final fine nodes | boxes |",
          "|---|---|---|---|---|---|---|---|"]
    for nm, r in res.items():
        for k in S.METHODS:
            v = r["methods"].get(k)
            if not v or "error" in v:
                if v:
                    L.append(f"| {nm} | {k} | FAILED: {v['error']} | | | | | |")
                continue
            bx = v.get("final_boxes") or v.get("boxes") or []
            L.append(f"| {nm} | {k} | {v['iterations']} | {v['evaluations']} | "
                     f"{v['seconds_per_evaluation']:.2f} | {v['peak_GB']:.1f} | {v['final_fine_nodes']} | "
                     f"{[[[round(a, 2) for a in ax_] for ax_ in b] for b in bx]} |")
    L.append("")
    chk = root / "checks.json"
    if chk.exists():
        c = json.loads(chk.read_text())
        L += ["## Reference and modelling floors", "",
              "Background data, uniform 1 cm against the 0.5 cm reference (relative energy of the "
              "difference; one-sample shifts for the alignment check), and true-model response "
              "errors of each inversion solver on the training shots:", "",
              "| f0 | 1 cm vs 0.5 cm | shift -1 | shift +1 | model | FDTD 3 cm | FDTD 1 cm | static whole | oracle | evaluator floor |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for f, row in c.items():
            for m, fl in row["floors"].items():
                L.append(f"| {f} MHz | {row['bg_rel_diff_1cm_vs_05cm']:.4f} | {row['bg_rel_diff_shift-1']:.3f} | "
                         f"{row['bg_rel_diff_shift+1']:.3f} | {m} | {fl['fdtd3']:.4f} | {fl['fdtd1']:.4f} | "
                         f"{fl['static_full']:.4f} | {fl['oracle']:.4f} | {fl['evaluator_floor']:.4f} |")
        L.append("")
    ver = root / "verify.json"
    if ver.exists():
        from scipy.stats import spearmanr
        v = json.loads(ver.read_text())
        L += ["## Evaluator check (final PE with the 0.5 cm solver)", "",
              "| scenario | method | PE 0.5 cm | PE 1 cm |", "|---|---|---|---|"]
        for nm, rows in v.items():
            for k, x in rows.items():
                L.append(f"| {nm} | {k} | {x['pe_05cm']:.4f} | {x['pe_1cm'] if x['pe_1cm'] is None else round(x['pe_1cm'], 4)} |")
            ks = [k for k in rows if k != "true" and rows[k]["pe_1cm"] is not None]
            if len(ks) >= 3:
                rho = spearmanr([rows[k]["pe_05cm"] for k in ks], [rows[k]["pe_1cm"] for k in ks]).correlation
                L.append(f"| {nm} | Spearman rank correlation | {rho:.3f} | |")
        L.append("")
    L += ["## Figures", "",
          "* `summary_pe_vs_budget.png`, `ratio_by_budget_<set>_<method>.png`, "
          "`paired_scatter_<method>.png`, `heatmap_pe.png`, `noise_trend.png`, `longrun_*.png`, "
          "`fdtd1_time_to_accuracy.png`",
          "* per scenario with a long 1 cm run: `fdtd1_timeline.gif` (the 1 cm model against time "
          "next to the other methods' models at T), `fdtd1_vs_time.png`",
          "* per scenario (`<scenario>/`): `evolution.gif` (models of all methods at the same "
          "compute time), `depth_sweep.gif` (final models, depth slices), `iso3d.gif` "
          "(300 MHz / 20 dB: rotating anomaly isosurfaces), `curves.png`, `heldout_residuals.png`, "
          "`final_models.png`", ""]
    (root / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"wrote {root / 'report.md'}")


__all__ = ["report"]


if __name__ == "__main__":
    import sys
    report(Path(sys.argv[1]) if len(sys.argv) > 1 else S.ROOT)
    _ = math

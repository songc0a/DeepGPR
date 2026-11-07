"""Markdown tables of the SIREN vs Tucker-SIREN study, generated from result files.

Usage: ``python report.py [--quick]`` -> ``results/<scale>/report_tables.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
SIZES = ["50^3", "100^3", "120x120x100", "160^3", "200^3"]


def load(path: Path) -> Dict:
    return json.loads(path.read_text()) if path.exists() else {}


def f(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    if value == 0:
        return "0"
    if abs(value) >= 1e5 or abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.{digits}g}"


def ms(stats):
    return f"{stats['mean'] * 1e3:.2f} ± {stats['std'] * 1e3:.2f}"


def generation_tables(gen: Dict) -> List[str]:
    lines = ["## A. Model generation (one generation + backward + Adam step)", ""]
    lines.append(
        "| size | config | group | params | net fwd FLOPs | recon fwd FLOPs | network ms | reconstruction ms | mapping ms "
        "| backward ms | Adam ms | total ms (CUDA events) | wall ms | kernel ms | launches | graph MiB | peak alloc MiB | peak reserved MiB | Adam state KiB |"
    )
    lines.append("|" + "---|" * 19)
    for size in SIZES:
        for name, r in gen.items():
            if r.get("spec", {}).get("size") != size:
                continue
            spec = r["spec"]
            config = name.split("|")[1]
            if "cuda_event_seconds" not in r:
                lines.append(f"| {size} | {config} | {spec['group']} | - | {f(r.get('counts', {}).get('network_forward_flops'))} | - | OOM | | | | | | | | | | | | |")
                continue
            ev, c = r["cuda_event_seconds"], r["counts"]
            lines.append(
                f"| {size} | {config} | {spec['group']} | {r['parameters']['total']:,} | {f(c['network_forward_flops'])} | {f(c['reconstruction_forward_flops'])} | "
                f"{ms(ev['network'])} | {ms(ev['reconstruction'])} | {ms(ev['mapping'])} | {ms(ev['backward'])} | {ms(ev['optimizer'])} | {ms(ev['total'])} | "
                f"{r['wall_seconds']['mean'] * 1e3:.2f} | {r['profiler_cuda_kernel_seconds'] * 1e3:.2f} | {r['profiler_kernel_launches']} | "
                f"{r['graph_and_output_bytes'] / 2**20:.1f} | {r['peak_allocated_bytes'] / 2**20:.1f} | {r['peak_reserved_bytes'] / 2**20:.1f} | {r['optimizer_state_bytes'] / 1024:.0f} |"
            )
    lines.append("")
    lines.append("### A. Decomposed backward (mean ms of 10)")
    lines.append("")
    lines.append("| size | config | forward | backward reconstruction+mapping | backward network |")
    lines.append("|---|---|---|---|---|")
    for size in SIZES:
        for name, r in gen.items():
            if r.get("spec", {}).get("size") == size and "decomposed_backward_seconds" in r:
                d = r["decomposed_backward_seconds"]
                lines.append(f"| {size} | {name.split('|')[1]} | {d['forward']['mean'] * 1e3:.2f} | {d['backward_reconstruction_mapping']['mean'] * 1e3:.2f} | {d['backward_network']['mean'] * 1e3:.2f} |")
    lines.append("")
    return lines


def representation_tables(rep: Dict) -> List[str]:
    lines = ["## B. Representation of the true 3D model (120 x 120 x 100)", ""]
    lines.append("| method | params (cores) | lr | s/step | train s | peak MiB | eps RMSE | eps rel L2 | eps SSIM | log10 sigma RMSE | sigma SSIM | time to eps rel. L2 <= 5 % / 3 % / 2 % / 1 % |")
    lines.append("|" + "---|" * 12)
    for name, r in rep.items():
        if "eps" not in r:
            lines.append(f"| {name} | failed: {r.get('error', '')[:80]} |")
            continue
        p = r["parameters"]
        cores = p.get("cores")
        reach = r.get("time_to_eps_rel_l2", {})
        times = " / ".join(f(reach.get(k)) for k in ("0.05", "0.03", "0.02", "0.01"))
        lines.append(
            f"| {name}{' (lr search)' if r['spec'].get('lr_search') else ''} | {p['total']:,}{f' ({cores:,})' if cores else ''} | {r['spec']['lr']:g} | {r['seconds_per_step'] * 1e3:.1f} ms | "
            f"{r['train_seconds']:.0f} | {f((r.get('peak_allocated_bytes') or 0) / 2**20)} | {r['eps']['rmse']:.4f} | {r['eps']['rel_l2']:.4f} | {r['eps']['ssim']:.4f} | "
            f"{r['sigma_log10']['rmse']:.4f} | {r['sigma_log10']['ssim']:.4f} | {times} |"
        )
    lines.append("")
    return lines


def fwi_tables(fwi: Dict, search: Dict) -> List[str]:
    lines = []
    for case in ("3d", "2d"):
        groups: Dict[str, List[Dict]] = {}
        for name, r in fwi.items():
            if name.startswith(f"{case}|") and "history" in r:
                groups.setdefault(name.split("|")[1], []).append(r)
        if not groups:
            continue
        lines += [f"## C. FWI ({case.upper()})", ""]
        cost = {}
        for name, r in search.items():
            if name.startswith(f"{case}|search|") and "history" in r:
                label = name.split("|")[2]
                cost.setdefault(label, []).append(sum(h["iteration_s"] for h in r["history"]))
        lines.append(
            "| method | runs | params | lr (search cost s) | pre-fit s | iters | initial loss | final loss (mean ± sd) | eps RMSE start → end | eps SSIM end "
            "| zone eps RMSE end | log10 sigma RMSE end | gen ms | FDTD fwd s | adjoint s | net bwd ms | opt ms | iteration s | peak GiB |"
        )
        lines.append("|" + "---|" * 19)
        for label in sorted(groups, key=lambda l: (l != "dense", l)):
            runs = groups[label]
            hist = [r["history"] for r in runs]
            last = [h[-1] for h in hist]
            final = [h["loss"] for h in last]
            ends = [h["errors"] for h in last if "errors" in h]
            med = lambda key: statistics.median(x[key] for h in hist for x in h[1:])  # noqa: E731
            r0 = runs[0]
            lines.append(
                f"| {label} | {len(runs)} | {r0['parameters']['total']:,} | {r0['spec']['lr']:g} ({sum(cost.get(label, [0])):.0f}) | "
                f"{statistics.fmean(r['info']['prefit_seconds'] for r in runs):.1f} | {len(hist[0])} | {statistics.fmean(h[0]['loss'] for h in hist):.4g} | "
                f"{statistics.fmean(final):.4g} ± {statistics.pstdev(final):.2g} | "
                f"{statistics.fmean(r['info']['initial_errors']['eps']['rmse'] for r in runs):.3f} → {statistics.fmean(e['eps']['rmse'] for e in ends):.3f} | "
                f"{statistics.fmean(e['eps']['ssim'] for e in ends):.3f} | {statistics.fmean(e['eps_zone']['rmse'] for e in ends):.3f} | "
                f"{statistics.fmean(e['sigma_log']['rmse'] for e in ends):.3f} | {med('generation_s') * 1e3:.2f} | {med('fdtd_forward_s'):.3f} | {med('adjoint_s'):.3f} | "
                f"{med('network_backward_s') * 1e3:.2f} | {med('optimizer_s') * 1e3:.2f} | {med('iteration_s'):.3f} | {max(r['peak_allocated_bytes'] for r in runs) / 2**30:.2f} |"
            )
        lines.append("")
        # Time-to-threshold and fixed-budget views (cumulative FWI iteration
        # time; pre-fit and learning-rate search excluded, reported above).
        thresholds = (0.01, 0.003, 0.001) if case == "3d" else (0.03, 0.01, 0.003)
        budgets = (60.0, 120.0, 240.0) if case == "3d" else (10.0, 20.0, 45.0)
        lines.append(f"### C. FWI ({case.upper()}): time to reach a data loss, and best loss / eps RMSE within a time budget (mean over seeds)")
        lines.append("")
        header = " | ".join([f"t(loss <= {t:g}) s" for t in thresholds] + [f"loss @ {b:g} s" for b in budgets] + [f"eps RMSE @ {b:g} s" for b in budgets])
        lines.append(f"| method | {header} |")
        lines.append("|" + "---|" * (1 + 3 * len(budgets)))
        for label in sorted(groups, key=lambda l: (l != "dense", l)):
            reach_cells, loss_cells, rmse_cells = [], [], []
            for t in thresholds:
                times = []
                for r in groups[label]:
                    clock, hit = 0.0, None
                    for h in r["history"]:
                        clock += h["iteration_s"]
                        if h["loss"] <= t:
                            hit = clock
                            break
                    times.append(hit)
                reached = [x for x in times if x is not None]
                reach_cells.append(f"{statistics.fmean(reached):.0f} ({len(reached)}/{len(times)})" if reached else "not reached")
            for b in budgets:
                best_loss, best_rmse = [], []
                for r in groups[label]:
                    clock, lo, rm = 0.0, float("inf"), None
                    for h in r["history"]:
                        clock += h["iteration_s"]
                        if clock > b:
                            break
                        lo = min(lo, h["loss"])
                        if "errors" in h:
                            rm = h["errors"]["eps"]["rmse"]
                    best_loss.append(lo)
                    if rm is not None:
                        best_rmse.append(rm)
                loss_cells.append(f"{statistics.fmean(best_loss):.3g}")
                rmse_cells.append(f"{statistics.fmean(best_rmse):.3f}" if best_rmse else "-")
            lines.append(f"| {label} | " + " | ".join(reach_cells + loss_cells + rmse_cells) + " |")
        lines.append("")
    return lines


def v4_tables(fwi: Dict, search: Dict, search_sigma: Dict, protocol: str = "v4") -> List[str]:
    """Protocol v4: held-out validation misfit of every search candidate, and zone metrics."""
    lines = []
    for case in ("3d", "2d"):
        stage_a: Dict[str, List[str]] = {}
        for name, r in search.items():
            if name.startswith(f"{case}|search|"):
                label = name.split("|")[2]
                value = r.get("validation_loss")
                stage_a.setdefault(label, []).append(f"{r['spec']['lr']:g}: {f(value) if value is not None else 'failed'}")
        stage_b: Dict[str, List[str]] = {}
        for name, r in search_sigma.items():
            if name.startswith(f"{case}|search_sigma|"):
                label = name.split("|")[2]
                value = r.get("validation_loss")
                stage_b.setdefault(label, []).append(f"{r['spec']['sigma_scale']:g}: {f(value) if value is not None else 'failed'}")
        if not stage_a:
            continue
        chosen = {}
        for name, r in fwi.items():
            if name.startswith(f"{case}|") and "spec" in r:
                chosen[name.split("|")[1]] = (r["spec"]["lr"], r["spec"].get("sigma_scale"))
        lines += [f"### C. v4 hyper-parameter selection ({case.upper()}): validation misfit (held-out shots) after the search", ""]
        lines.append("| method | stage A, eps-only: lr: validation misfit | stage B, joint: sigma scale: validation misfit | chosen lr, sigma scale |")
        lines.append("|---|---|---|---|")
        for label in sorted(stage_a, key=lambda l: (l != "dense", l)):
            pick = chosen.get(label)
            lines.append(f"| {label} | {'; '.join(stage_a[label])} | {'; '.join(stage_b.get(label, []))} | {f'{pick[0]:g}, {pick[1]:g}' if pick else '-'} |")
        lines.append("")
        # Selection check (uses the truth, for reference only): model error after
        # the search for the candidate each criterion picks and for the best one.
        key = "eps_zone" if case == "3d" else "eps"
        lines += [f"### C. v4 selection check ({case.upper()}): {key} RMSE after the search of the candidate picked by each criterion", ""]
        lines.append("| method | training misfit (v3 rule): lr -> RMSE | validation misfit (v4): lr -> RMSE | lowest RMSE (truth): lr -> RMSE "
                     "| stage B validation pick: scale -> zone log10 sigma RMSE | stage B lowest sigma RMSE (truth): scale -> RMSE |")
        lines.append("|---|---|---|---|---|---|")
        for label in sorted(stage_a, key=lambda l: (l != "dense", l)):
            cands = {r["spec"]["lr"]: r for n, r in search.items() if n.startswith(f"{case}|search|{label}|") and "history" in r}
            if not cands:
                continue
            err = {lr: r["history"][-1]["errors"][key]["rmse"] for lr, r in cands.items()}
            train = {lr: statistics.fmean(h["loss"] for h in r["history"][-3:]) for lr, r in cands.items()}
            val = {lr: r.get("validation_loss", math.inf) for lr, r in cands.items()}
            cells = [f"{lr:g} -> {err[lr]:.3f}" for lr in (min(train, key=train.get), min(val, key=val.get), min(err, key=err.get))]
            scales = {r["spec"]["sigma_scale"]: r for n, r in search_sigma.items() if n.startswith(f"{case}|search_sigma|{label}|") and "history" in r}
            if scales:
                serr = {sc: r["history"][-1]["errors"]["sigma_log_zone"]["rmse"] for sc, r in scales.items()}
                sval = {sc: r.get("validation_loss", math.inf) for sc, r in scales.items()}
                cells += [f"{sc:g} -> {serr[sc]:.3f}" for sc in (min(sval, key=sval.get), min(serr, key=serr.get))]
            else:
                cells += ["-", "-"]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
        groups: Dict[str, List[Dict]] = {}
        for name, r in fwi.items():
            if name.startswith(f"{case}|") and "history" in r:
                groups.setdefault(name.split("|")[1], []).append(r)
        if not groups:
            continue
        zone_text = "whole model" if case == "2d" else ("top 1 m (25 cells), deep = 1-2 m" if protocol in ("v5", "v6") else "top 40 subsurface cells")
        lines += [f"### C. v4 model errors ({case.upper()}, mean over seeds; zone = {zone_text})", ""]
        lines.append("| method | zone eps RMSE start -> end | zone log10 sigma RMSE start -> end | deep eps RMSE start -> end | eps RMSE start -> end | log10 sigma RMSE start -> end |")
        lines.append("|---|---|---|---|---|---|")

        def pair(runs, key):
            if key not in runs[0]["info"]["initial_errors"]:
                return "-"
            start = statistics.fmean(r["info"]["initial_errors"][key]["rmse"] for r in runs)
            end = statistics.fmean(r["history"][-1]["errors"][key]["rmse"] for r in runs)
            return f"{start:.3f} -> {end:.3f}"

        for label in sorted(groups, key=lambda l: (l != "dense", l)):
            runs = groups[label]
            lines.append(f"| {label} | {pair(runs, 'eps_zone')} | {pair(runs, 'sigma_log_zone')} | {pair(runs, 'eps_deep')} | {pair(runs, 'eps')} | {pair(runs, 'sigma_log')} |")
        lines.append("")
    return lines


def oracle_tables(fwi: Dict, oracle: Dict) -> List[str]:
    """v4: data-selected runs vs the truth-selected reference (same searches)."""
    lines = []
    for case in ("3d", "2d"):
        rows = {}
        for source, results in (("data", fwi), ("oracle", oracle)):
            for name, r in results.items():
                if name.startswith(f"{case}|") and "history" in r:
                    rows.setdefault(name.split("|")[1], {}).setdefault(source, []).append(r)
        if not any("oracle" in v for v in rows.values()):
            continue
        lines += [f"### C. v4 reference ({case.upper()}): hyper-parameters selected by the data vs by the true model (oracle, reference only)", ""]
        lines.append("| method | data: lr, sigma scale | data: zone eps RMSE | data: zone log10 sigma RMSE | oracle: lr, sigma scale | oracle: zone eps RMSE | oracle: zone log10 sigma RMSE | oracle runs |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for label in sorted(rows, key=lambda l: (l != "dense", l)):
            cells = []
            for source in ("data", "oracle"):
                runs = rows[label].get(source)
                if not runs:
                    cells += ["-", "-", "-"]
                    continue
                spec = runs[0]["spec"]
                end = [r["history"][-1]["errors"] for r in runs]
                cells += [f"{spec['lr']:g}, {spec.get('sigma_scale', 1):g}", f"{statistics.fmean(e['eps_zone']['rmse'] for e in end):.3f}",
                          f"{statistics.fmean(e['sigma_log_zone']['rmse'] for e in end):.3f}" if "sigma_log_zone" in end[0] else "-"]
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {len(rows[label].get('oracle', []))} |")
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--protocol", default="v1", choices=["v1", "v2", "v3", "v4", "v5", "v6"])
    args = parser.parse_args()
    base = "quick" if args.quick else "full"
    results = HERE / "results" / (base if args.protocol == "v1" else f"{base}_{args.protocol}")
    shared = HERE / "results" / base
    results.mkdir(parents=True, exist_ok=True)
    lines = [f"# Tables generated by report.py (FWI protocol {args.protocol})", ""]
    lines += generation_tables(load(shared / "generation.json"))
    lines += representation_tables(load(shared / "representation.json"))
    lines += fwi_tables(load(results / "fwi.json"), load(results / "fwi_search.json"))
    if args.protocol in ("v4", "v5", "v6"):
        lines += v4_tables(load(results / "fwi.json"), load(results / "fwi_search.json"), load(results / "fwi_search_sigma.json"), args.protocol)
        lines += oracle_tables(load(results / "fwi.json"), load(results / "fwi_oracle.json"))
    (results / "report_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5]))


if __name__ == "__main__":
    main()

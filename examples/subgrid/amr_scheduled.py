"""Scheduler-driven dynamic refinement on the 3D sand-pit benchmark (``amr_sandpit3d.py``).

Adds the method ``amr_sched`` to a scenario of ``amr_sandpit3d.py``: the same
model, data, optimiser, marking (v3) and scoring as ``amr_dyn_v3``, but the
*decisions* come from :class:`DeepGPR.subgrid.schedule.Scheduler`:

* re-plan only when the misfit has stalled on the current layout (plateau test)
  and the dual-weighted-residual estimate of the modelling error is a sizeable
  fraction of the misfit (error balancing), instead of every 10 iterations;
* the fine-node budget of a re-plan follows from the remaining wall time and
  the calibrated cost model, instead of a fixed fraction of the whole region.

    python examples/subgrid/amr_scheduled.py M1 [--budget 2400] [--balance 0.25] [--plateau 0.02]

Writes ``results/sandpit3d/<name>/result.json`` (method ``amr_sched``) and the
snapshot file like the other methods, so ``amr_sandpit3d.py report`` picks it up.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import amr_sandpit3d as A  # noqa: E402

from DeepGPR.subgrid.adjoint import SubgridPropagator  # noqa: E402
from DeepGPR.subgrid.schedule import CostModel, Scheduler, TimeStepLadder  # noqa: E402

KIND = "amr_sched"


def signed_indicator(prop, eps, sig, w, residual, eps0, sig0):
    """``(|eta|, estimate)`` of the differential objective: the signed DWR estimate
    ``sum(eta(m) - eta(m0))`` is the model error of the current layout."""
    def signed(e, s):
        e = e.detach().clone().requires_grad_(True)
        d = prop(e, s.detach(), w)
        d.backward(residual)
        return prop.indicator_obj.signed().clone()

    eta = signed(eps, sig) - signed(eps0, sig0)
    ind = prop.indicator_obj
    est = float(eta.sum())
    e = eta.abs()
    sl = tuple(slice(ind.pml[2 * a], e.shape[a] - 1 - ind.pml[2 * a + 1]) for a in range(3))
    return e[sl].cpu().numpy(), est


def scheduled_fwi(ctx, budget, marks, log, args):
    e_bg, s_bg = A.background()
    E0, S0 = e_bg.to(A.DEV), s_bg.to(A.DEV)
    mask = torch.as_tensor(A.update_mask(), device=A.DEV)
    obs = (ctx.obs * ctx.mk[ctx.train]).cuda()
    mkc = ctx.mk[ctx.train].cuda()
    u_e = torch.zeros(A.SHAPE, dtype=A.DTYPE, device=A.DEV, requires_grad=True)
    u_s = torch.zeros(A.SHAPE, dtype=A.DTYPE, device=A.DEV, requires_grad=True)

    def mdl():
        return ((E0 + u_e * mask).clamp(1.0, 40.0), (S0 + A.SIG_SCALE * u_s * mask).clamp(0.0, 0.5))

    def new_opt():
        return torch.optim.LBFGS([u_e, u_s], lr=1.0, history_size=10, max_iter=1, line_search_fn="strong_wolfe")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    clock = A.Clock()
    layout = []
    clock.start()
    fn = A.Data("main", ctx.src, ctx.rec, ctx.train, boxes=layout)
    state = {"fn": fn, "evals": 0, "misfit": 1.0, "grad_seconds": []}
    opt = new_opt()
    cost = CostModel()
    duration = A.NT * A.DT
    nshots = len(ctx.train)
    sched = Scheduler(cost, budget, iterations=200, balance=args.balance, plateau=args.plateau,
                      window=args.window, min_cycle=args.min_cycle)
    sched.t0 = time.perf_counter()        # replaced by the pausing clock below

    def closure():
        opt.zero_grad()
        eps, sig = mdl()
        el, sl = eps.detach().requires_grad_(True), sig.detach().requires_grad_(True)
        misfit = 0.0
        t0 = time.perf_counter()
        for idx, d in state["fn"].batched(el, sl):
            part = ((d * mkc[idx] - obs[idx]) ** 2).sum() / ctx.norm
            part.backward()
            misfit += float(part)
        torch.cuda.synchronize()
        state["grad_seconds"].append(time.perf_counter() - t0)
        reg = A.TV * (A.tv_term(u_e, mask) + A.tv_term(u_s, mask))
        torch.autograd.backward([eps, sig, reg], [el.grad, sl.grad, torch.ones_like(reg)])
        state["evals"] += 1
        state["misfit"] = misfit
        return torch.tensor(misfit + float(reg))

    def calibrate():
        ladder = TimeStepLadder.from_solver(state["fn"].main_solver())
        measured = float(np.median(state["grad_seconds"][-3:]))
        scale = measured / cost.predict(ladder, duration, shots=nshots, gradient=True)
        cost.yee *= scale
        cost.dg *= scale
        cost.sub *= scale
        cost.overhead *= scale
        return ladder, measured

    def replan(n, est_only=False):
        with torch.no_grad():
            e, s = mdl()
        main = A.Data("main", ctx.src, ctx.rec, ctx.train)
        with torch.no_grad():
            d = main(e, s) * mkc
        ip = SubgridPropagator(main.main_solver(), ctx.src[ctx.train], ctx.rec[ctx.train], indicator=True)
        # dJ/dd of the normalised misfit J = |d - obs|^2 / norm, so that the DWR estimate
        # is in the units of the misfit (error balancing)
        eta, est = signed_indicator(ip, e, s, main.w, (d - obs) * mkc * (2.0 / ctx.norm), E0, S0)
        del ip, main
        return eta, est

    snaps, hist = {}, []
    pending = sorted(set(marks))
    prev = dict(t=0.0, it=0, misfit=1.0, boxes=[], eps=None, sig=None)
    it, n_replans, n_checks = 0, 0, 0
    decisions = []
    next_check = args.min_cycle
    while clock.elapsed() < budget:
        if it >= next_check and it > 0:
            # remaining-time bookkeeping of the scheduler follows the pausing clock
            sched.elapsed = lambda: clock.elapsed()
            # iterations planned for the whole run (a refined layout costs more per
            # gradient, so this is a target, not the extrapolated coarse-grid rate)
            sched.iterations = max(it + 1, args.target_iterations)
            n_checks += 1
            eta, est = replan(n_checks)
            dec = sched.decide(est, has_boxes=bool(layout))
            decisions.append(dict(it=it, t=clock.elapsed(), action=dec.action, reason=dec.reason,
                                  misfit=dec.misfit, estimate=est, ratio=dec.balance, plateau=dec.plateau))
            log(f"  [{KIND}] check {n_checks} at it {it}: misfit {dec.misfit:.4f}, estimate {est:+.4f} "
                f"(ratio {dec.balance:.2f}, plateau {dec.plateau}) -> {dec.action}: {dec.reason}")
            next_check = it + args.check_every
            if dec.action in ("refine", "coarsen"):
                ladder, measured = calibrate()
                spg = sched.seconds_per_gradient()
                max_nodes = cost.fine_node_budget(ladder, duration, spg, shots=nshots, time_ratio=A.R)
                cap = int(A.BUDGET * A.fine_nodes(ctx.full_boxes()))
                max_nodes = min(max_nodes, cap) if dec.action == "refine" else int(0.5 * A.fine_nodes(layout))
                new, info = A.mark_v3(eta, ctx.allowed, ctx.keep, max(max_nodes, 0))
                changed = sorted(map(str, new)) != sorted(map(str, layout))
                if changed and layout:
                    inside_new = np.zeros(eta.shape, dtype=bool)
                    for b in new:
                        sl = tuple(slice(int(round(lo / (A.R * A.H))), int(round(hi / (A.R * A.H))) + 1) for lo, hi in b)
                        inside_new[sl] = True
                    inside_old = np.zeros(eta.shape, dtype=bool)
                    for b in layout:
                        sl = tuple(slice(int(round(lo / (A.R * A.H))), int(round(hi / (A.R * A.H))) + 1) for lo, hi in b)
                        inside_old[sl] = True
                    cov = float(eta[inside_new & inside_old].sum() / max(eta[inside_new].sum(), 1e-30))
                    if cov >= A.HYST and A.fine_nodes(layout) <= 1.5 * max(A.fine_nodes(new), 1):
                        changed = False
                        log(f"  [{KIND}] kept the layout (coverage {cov:.2f})")
                log(f"  [{KIND}] re-plan: budget {max_nodes} fine nodes (affordable {spg:.1f} s/gradient, measured "
                    f"{measured:.1f} s), {info} -> {[[tuple(round(v, 3) for v in ax) for ax in b] for b in new]} "
                    f"({A.fine_nodes(new)} fine nodes), changed={changed}")
                if changed:
                    n_replans += 1
                    sched.last_change = sched.iteration
                    layout = new
                    state["fn"] = None
                    fn = None
                    torch.cuda.empty_cache()
                    fn = A.Data("main", ctx.src, ctx.rec, ctx.train, boxes=layout)
                    state["fn"] = fn
                    opt = new_opt()
        n0 = state["evals"]
        opt.step(closure)
        it += 1
        sched.record(state["misfit"])
        clock.pause()
        t = clock.acc
        with torch.no_grad():
            e, s = mdl()
            cur = dict(t=t, it=it, misfit=state["misfit"], boxes=[list(map(list, b)) for b in layout],
                       eps=e.float().cpu().numpy(), sig=s.float().cpu().numpy())
        while pending and pending[0] < t:
            snaps[pending.pop(0)] = prev
        hist.append(dict(t=t, it=it, misfit=state["misfit"], evals=state["evals"] - n0, boxes=cur["boxes"]))
        if it == 1 or it % 5 == 0:
            log(f"  [{KIND}] it {it:4d}: t {t:7.1f} s, misfit {state['misfit']:.4f}, {len(layout)} box(es), "
                f"{state['evals']} evals")
        prev = cur
        clock.start()
    clock.pause()
    for m in pending:
        snaps[m] = prev
    info = dict(iterations=it, evaluations=state["evals"], seconds=clock.acc,
                seconds_per_evaluation=clock.acc / max(1, state["evals"]),
                peak_GB=torch.cuda.max_memory_allocated() / 2 ** 30, final_boxes=layout,
                final_fine_nodes=A.fine_nodes(layout), replans=n_replans, checks=n_checks,
                decisions=decisions, cost_model=dict(yee=cost.yee, dg=cost.dg, sub=cost.sub,
                                                     overhead=cost.overhead))
    state["fn"] = None
    torch.cuda.empty_cache()
    return hist, snaps, info


def main():
    global KIND
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--budget", type=float, default=2400.0)
    ap.add_argument("--balance", type=float, default=0.25)
    ap.add_argument("--plateau", type=float, default=0.02)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-cycle", type=int, default=10)
    ap.add_argument("--check-every", type=int, default=5)
    ap.add_argument("--target-iterations", type=int, default=200,
                    help="iterations planned for the whole budget (sets the affordable cost per gradient)")
    ap.add_argument("--kind", default=KIND)
    args = ap.parse_args()
    KIND = args.kind
    out = A.ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = A.log_to(out / "log.txt")
    ctx = A.Context(args.name)
    path = out / "result.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res.setdefault("methods", {})
    res["budgets"] = dict(res.get("budgets", A.BUDGETS), **{KIND: args.budget})
    marks = sorted(set([m for m in A.MARKS if m <= args.budget] + [m for m in A.GIF_MARKS if m <= args.budget]))
    A.warm_up(ctx)
    log(f"[{KIND}] start, budget {args.budget:.0f} s, balance {args.balance}, plateau {args.plateau}, "
        f"window {args.window}, min_cycle {args.min_cycle}, check_every {args.check_every}, "
        f"target iterations {args.target_iterations}")
    try:
        hist, snaps, info = scheduled_fwi(ctx, args.budget, marks, log, args)
    except Exception as exc:
        log(f"[{KIND}] FAILED: {exc!r}\n{traceback.format_exc()}")
        res["methods"][KIND] = {"error": repr(exc)}
        path.write_text(json.dumps(res, indent=1))
        return
    e0 = A.background()[0].numpy()
    s0 = A.background()[1].numpy()
    for m in marks:
        if snaps[m]["eps"] is None:
            snaps[m] = dict(snaps[m], eps=e0, sig=s0, start=True)
    np.savez_compressed(out / f"snap_{KIND}.npz", eps=np.stack([snaps[m]["eps"] for m in marks]).astype(np.float16),
                        marks=np.array(marks), eps_final=snaps[marks[-1]]["eps"], sig_final=snaps[marks[-1]]["sig"])
    r = dict(history=hist, **info, marks=marks,
             snap_meta=[dict(mark=m, **{k: v for k, v in snaps[m].items() if k not in ("eps", "sig", "start")})
                        for m in marks])
    r["pe"], last = {}, None
    for m in marks:
        if m not in A.MARKS:
            continue
        ej, sj = snaps[m]["eps"], snaps[m]["sig"]
        if snaps[m].get("start"):
            r["pe"][str(m)] = 1.0
            continue
        if last is not None and np.array_equal(ej, last[0]) and np.array_equal(sj, last[1]):
            r["pe"][str(m)] = last[2]
            continue
        r["pe"][str(m)] = ctx.pe(ej, sj)
        last = (ej, sj, r["pe"][str(m)])
    r["metrics"] = A.model_metrics(ctx, snaps[marks[-1]]["eps"].astype(np.float64),
                                   snaps[marks[-1]]["sig"].astype(np.float64))
    res["methods"][KIND] = r
    log(f"[{KIND}] {info['iterations']} it, {info['evaluations']} evals, {info['seconds_per_evaluation']:.2f} s/eval, "
        f"{info['peak_GB']:.1f} GB, {info['replans']} re-plans in {info['checks']} checks, boxes {info['final_boxes']}; PE "
        + ", ".join(f"{float(k):.0f} s: {v:.4f}" for k, v in r["pe"].items()))
    for k in ("amr_dyn_v3", "amr_once_v3", "static_full", "coarse", "oracle", "fine"):
        if k in res["methods"] and "pe" in res["methods"][k]:
            log(f"    reference {k}: PE " + ", ".join(f"{float(m):.0f} s: {v:.4f}" for m, v in res["methods"][k]["pe"].items()))
    path.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()

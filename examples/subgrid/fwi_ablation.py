"""Controlled FWI experiments: what each part of ``DeepGPR.subgrid`` changes in the inversion.

Design principles (all experiments):

* observed data from an independent, more accurate solver (uniform grid at the model
  spacing h = 1 cm; with a DG surface band, the same band with 1 cm Yee), so every
  inversion carries its own modelling error. ``--data raw``: the data themselves (direct
  waves included). ``--data diff`` (default): differential data with a known baseline
  (target model minus the baseline model, as in monitoring with a reference survey); each
  solver predicts ``d(m) - d(m0)`` with its own baseline run. The pre-study shows why: on
  the 3 cm main grid the background modelling error of the raw data is 5-40 times the
  target signal, outside any refined box, so raw-data FWI on that grid is limited by the
  background, whatever is refined;
* acquisition with reflection and transmission aperture (sources and receivers on a
  plane above and a plane below the targets) and a frequency chosen by the pre-study
  so that the target signal exceeds the background modelling error;
* identical inversion settings for all solvers of an experiment (parameters on the
  model grid, update region, start model, L-BFGS, TV weight, iterations);
* model quality measured three ways: errors against the true model (point-wise and
  smoothed), integrated anomalies around each target, and the *held-out prediction
  error*: shots not used in the inversion are simulated with the accurate solver
  through the inverted model, ``PE = ||d(m_inv) - d_obs||^2 / ||d(m_0) - d_obs||^2``
  (0: perfect, 1: no better than the start model). PE rewards a model that is
  physically right, not one that fits the inversion's own solver;
* cost: seconds per gradient, gradient evaluations, peak GPU memory.

Sub-commands (results in ``examples/subgrid/results/ablation/``):

``prestudy``  data budget per frequency: target signal, background modelling error and
              target-response error of the 3 cm (with and without subgrid) and 5 cm grids
``exp1``      forward accuracy: main grid only / static subgrid / uniform 1 cm (upper bound)
``exp2``      exact discrete adjoint vs the main-grid adjoint used for the subgridded forward
``exp3``      grid sizes: main 5 cm against the 3 cm, 3 cm + subgrid and 1 cm runs of exp1
              (the planner's choice vs too coarse / too fine)
``exp4``      unknown target positions: main grid only, a box over the whole region,
              adjoint-driven AMR, oracle boxes (planner on the true model)
``exp5``      rugged surface: staircase FDTD, staircase + subgrid, DGTD band, DGTD band +
              subgrid (data from DG band + 1 cm Yee)
``report``    tables and figures from the saved results

    python examples/subgrid/fwi_ablation.py prestudy
    python examples/subgrid/fwi_ablation.py exp1 [--iters 30 --f0 3e8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from DeepGPR.subgrid import Subgrid, SubgridMaxwell3D, fine_source_samples
from DeepGPR.subgrid.adjoint import SubgridPropagator

OUT = Path(__file__).resolve().parent / "results" / "ablation"
DTYPE = torch.float32
H = 0.01


# --------------------------------------------------------------------------- benchmark
@dataclass
class Target:
    centre: Tuple[float, float, float]
    radius: float
    eps: Optional[float] = None       # None: background
    sig: Optional[float] = None


@dataclass
class Bench:
    shape: Tuple[int, int, int] = (76, 76, 76)
    eps0: float = 4.0
    sig0: float = 0.002
    targets: List[Target] = field(default_factory=lambda: [
        Target((0.42, 0.30, 0.375), 0.04, eps=16.0, sig=0.01),
        Target((0.42, 0.45, 0.375), 0.04, sig=0.1)])
    f0: float = 3e8
    t_max: float = 14e-9
    top: float = 0.10                  # source/receiver planes (depth x)
    bottom: float = 0.65
    roi: Tuple[Tuple[float, float], ...] = ((0.27, 0.57), (0.18, 0.57), (0.18, 0.57))
    box: Tuple[Tuple[float, float], ...] = ((0.33, 0.51), (0.24, 0.51), (0.27, 0.48))
    box5: Tuple[Tuple[float, float], ...] = ((0.35, 0.50), (0.25, 0.50), (0.25, 0.50))
    tv: float = 3e-6
    min_offset: float = 0.10

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


def models(b: Bench, targets=True):
    eps = np.full(b.shape, b.eps0)
    sig = np.full(b.shape, b.sig0)
    if targets:
        X, Y, Z = np.meshgrid(*[np.arange(n) * H for n in b.shape], indexing="ij")
        for t in b.targets:
            m = ((X - t.centre[0]) ** 2 + (Y - t.centre[1]) ** 2 + (Z - t.centre[2]) ** 2) <= t.radius ** 2
            if t.eps is not None:
                eps[m] = t.eps
            if t.sig is not None:
                sig[m] = t.sig
    return eps, sig


def acquisition(b: Bench):
    """9 shots (5 on the top plane, 4 on the bottom plane); 6 x 6 receivers on each plane.
    Shots 0, 2, 4, 6, 8, 5 train; 1, 3, 7 are held out."""
    lo, hi = 0.18, 0.57
    src = [(b.top, y, z) for y, z in ((0.25, 0.25), (0.25, 0.5), (0.375, 0.375), (0.5, 0.25),
                                     (0.5, 0.5))]
    src += [(b.bottom, y, z) for y, z in ((0.25, 0.375), (0.375, 0.25), (0.375, 0.5), (0.5, 0.375))]
    g = np.linspace(lo, hi, 6)
    rec = [(x, y, z) for x in (b.top, b.bottom) for y in g for z in g]
    src = np.array([[s] for s in src])
    rec = np.array([rec] * len(src))
    train = [0, 2, 4, 5, 6, 8]
    hold = [1, 3, 7]
    return src, rec, train, hold


def mute(src, rec, min_offset):
    off = np.linalg.norm(rec - src, axis=-1)
    return torch.as_tensor(off >= min_offset, dtype=DTYPE)[:, None, :]


def ricker(nt, dt, f0):
    t = np.arange(nt) * dt - 1.2 / f0
    a = (np.pi * f0 * t) ** 2
    return torch.as_tensor((1 - 2 * a) * np.exp(-a))


def main_dt(b: Bench, coarsen=3):
    return SubgridMaxwell3D(np.full(b.shape, b.eps0), np.full(b.shape, b.sig0), H, coarsen=coarsen,
                            pmlthick=10, cuda_kernels=False).dt


# --------------------------------------------------------------------------- solvers
def solver(kind: str, b: Bench, nshot: int, dt: float, boxes=None):
    """``coarse3``, ``sub3`` (ratio-3 box), ``coarse5``, ``fine``
    (uniform 1 cm, ``dt / 3``; the caller resamples the wavelet)."""
    e = torch.full(b.shape, b.eps0, dtype=DTYPE)
    s = torch.full(b.shape, b.sig0, dtype=DTYPE)
    if kind == "fine":
        return SubgridMaxwell3D(e, s, H, coarsen=1, pmlthick=30, dt=dt / 3, dipole_length=3 * H,
                                nshot=nshot, device="cuda", dtype=DTYPE)
    R = 5 if kind.endswith("5") else 3
    if boxes is None:
        boxes = ([b.box5] if R == 5 else [b.box]) if kind.startswith("sub") else []
    sgs = [Subgrid(bx, ratio=R, is_os_sep=2 if R == 5 else 3) for bx in boxes]
    return SubgridMaxwell3D(e, s, H, coarsen=R, subgrids=sgs, pmlthick=10 if R == 3 else 6,
                            dt=dt, dipole_length=3 * H, nshot=nshot, device="cuda", dtype=DTYPE)


class Prop:
    """Data of the shots ``shots`` on one solver kind (fine: resampled wavelet, decimated
    data, shots in batches to bound the checkpoint memory)."""

    def __init__(self, kind, b: Bench, src, rec, shots, dt, nt, boxes=None, batch=None):
        self.kind, self.b, self.nt, self.dt = kind, b, nt, dt
        self.shots = list(shots)
        self.batch = batch or (1 if kind == "fine" else len(self.shots))
        self.props = []
        for i in range(0, len(self.shots), self.batch):
            sh = self.shots[i:i + self.batch]
            sol = solver(kind, b, len(sh), dt, boxes)
            self.props.append((sh, SubgridPropagator(sol, src[sh], rec[sh])))
        w = ricker(nt, dt, b.f0)
        if kind == "fine":
            wf = fine_source_samples(w[None], 3)[0]
            wf = torch.cat([torch.zeros(1, dtype=wf.dtype), wf, torch.zeros(3, dtype=wf.dtype)])[:3 * nt]
            self.w = wf.to("cuda", DTYPE)
        else:
            self.w = w.to("cuda", DTYPE)

    def batched(self, eps, sig):
        """``(positions, data)`` per batch (differentiable); positions index ``shots``."""
        o = 0
        for sh, p in self.props:
            d = p(eps, sig, self.w)
            if self.kind == "fine":
                d = d[:, 2::3]
            yield slice(o, o + len(sh)), d
            o += len(sh)

    def __call__(self, eps, sig):
        return torch.cat([d for _, d in self.batched(eps, sig)], 0)


def batched(fn, eps, sig):
    """Per-batch data of a data function (one batch if it has no ``batched``)."""
    if hasattr(fn, "batched"):
        yield from fn.batched(eps, sig)
    else:
        yield slice(None), fn(eps, sig)


class Diff:
    """Differential data ``d(m) - d(m0)`` of a data function (known baseline ``m0``)."""

    def __init__(self, fn, b: Bench):
        self.fn = fn
        e0 = torch.full(b.shape, b.eps0, dtype=DTYPE, device="cuda")
        s0 = torch.full(b.shape, b.sig0, dtype=DTYPE, device="cuda")
        with torch.no_grad():
            self.base = [(i, d.detach()) for i, d in batched(fn, e0, s0)]

    def batched(self, eps, sig):
        for (i, d), (_, b0) in zip(batched(self.fn, eps, sig), self.base):
            yield i, d - b0

    def __call__(self, eps, sig):
        return torch.cat([d for _, d in self.batched(eps, sig)], 0)


def data_fn(fn, b: Bench, mode: str):
    return Diff(fn, b) if mode == "diff" else fn


def observed(mode, d_obs, d_bg):
    """Observed data of the inversion: raw accurate-solver data, or their differential
    with respect to the accurate solver's baseline (target response)."""
    return d_obs if mode == "raw" else d_obs - d_bg


def cached_fine(b: Bench, eps, sig, src, rec, dt, nt, tag):
    """Uniform-1 cm data of all shots (cached on disk)."""
    path = OUT / "cache" / f"fine_{b.key()}_{tag}.npz"
    if path.exists():
        return torch.as_tensor(np.load(path)["d"])
    path.parent.mkdir(parents=True, exist_ok=True)
    p = Prop("fine", b, src, rec, range(len(src)), dt, nt, batch=3)
    with torch.no_grad():
        d = p(torch.as_tensor(eps, dtype=DTYPE, device="cuda"),
              torch.as_tensor(sig, dtype=DTYPE, device="cuda")).cpu()
    np.savez_compressed(path, d=d.numpy())
    return d


# --------------------------------------------------------------------------- metrics
def roi_mask(b: Bench):
    m = np.zeros(b.shape, dtype=bool)
    m[tuple(slice(int(round(lo / H)) + 1, int(round(hi / H))) for lo, hi in b.roi)] = True
    return m


def model_metrics(b: Bench, eps, sig):
    from scipy.ndimage import gaussian_filter
    te, ts = models(b)
    mask = roi_mask(b)
    out = {}
    for name, a, x, m0 in (("eps", te, eps, b.eps0), ("sig", ts, sig, b.sig0)):
        out[f"{name}_err"] = float(np.linalg.norm((x - a)[mask]) / np.linalg.norm((a - m0)[mask]))
        A, B = gaussian_filter(a, 5), gaussian_filter(x, 5)
        out[f"{name}_err_smooth"] = float(np.linalg.norm((B - A)[mask]) / np.linalg.norm((A - m0)[mask]))
    X, Y, Z = np.meshgrid(*[np.arange(n) * H for n in b.shape], indexing="ij")
    for k, t in enumerate(b.targets):
        near = ((X - t.centre[0]) ** 2 + (Y - t.centre[1]) ** 2 + (Z - t.centre[2]) ** 2) <= (1.5 * t.radius) ** 2
        for name, a, x, m0 in (("eps", te, eps, b.eps0), ("sig", ts, sig, b.sig0)):
            ti = float((a - m0)[near].sum())
            if abs(ti) > 0:
                out[f"T{k}_int_d{name}"] = float((x - m0)[near].sum()) / ti
        out[f"T{k}_peak_eps"] = float(eps[near].max())
        out[f"T{k}_peak_sig"] = float(sig[near].max())
    return out


def prediction_error(b, eps, sig, src, rec, hold, dt, nt, d_obs, mk, d0_hold):
    """Held-out prediction error with the accurate solver: ``||d(m) - d_obs||^2 /
    ||d(m0) - d_obs||^2`` on the held-out shots (identical for raw and differential data,
    since both use the accurate solver's own baseline)."""
    p = Prop("fine", b, src, rec, hold, dt, nt, batch=3)
    with torch.no_grad():
        d = p(torch.as_tensor(eps, dtype=DTYPE, device="cuda"),
              torch.as_tensor(sig, dtype=DTYPE, device="cuda")).cpu() * mk[hold]
    obs = d_obs[hold] * mk[hold]
    return float(((d - obs) ** 2).sum() / ((d0_hold - obs) ** 2).sum()), d0_hold


def tv_term(u, m):
    gx = (u[1:] - u[:-1]) * (m[1:] & m[:-1])
    gy = (u[:, 1:] - u[:, :-1]) * (m[:, 1:] & m[:, :-1])
    gz = (u[:, :, 1:] - u[:, :, :-1]) * (m[:, :, 1:] & m[:, :, :-1])
    return torch.sqrt(gx[:, :-1, :-1] ** 2 + gy[:-1, :, :-1] ** 2 + gz[:-1, :-1, :] ** 2 + 1e-8).sum()


# --------------------------------------------------------------------------- FWI
class ApproxGradient(torch.autograd.Function):
    """Data from ``fwd`` (subgridded), gradient from ``adj`` (main grid only): the
    inconsistent gradient one gets without the subgrid's own adjoint."""

    @staticmethod
    def forward(ctx, fwd, adj, eps, sig):  # type: ignore[override]
        with torch.no_grad():
            d = fwd(eps, sig)
        ctx.adj = adj
        ctx.save_for_backward(eps, sig)
        return d

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        eps, sig = ctx.saved_tensors
        with torch.enable_grad():
            e = eps.detach().requires_grad_(True)
            s = sig.detach().requires_grad_(True)
            d = ctx.adj(e, s)
            d.backward(g)
        return None, None, e.grad, s.grad


def fwi(b: Bench, data_fn, d_obs, mk, iters, *, norm, label="", start=None, log=None,
        layout_cb=None):
    """L-BFGS on (eps, sigma) in the region of interest; ``data_fn(eps, sig)`` -> data of
    the training shots. ``layout_cb(cycle, eps, sig, residual)`` (AMR) may return a new
    ``data_fn`` every ``layout_cb.every`` iterations (L-BFGS restarted)."""
    E0 = torch.full(b.shape, b.eps0, dtype=DTYPE, device="cuda")
    S0 = torch.full(b.shape, b.sig0, dtype=DTYPE, device="cuda")
    Ei, Si = (E0, S0) if start is None else start
    mask = torch.as_tensor(roi_mask(b), device="cuda")
    obs = (d_obs * mk).cuda()
    mkc = mk.cuda()
    u_e = torch.zeros(b.shape, dtype=DTYPE, device="cuda", requires_grad=True)
    u_s = torch.zeros(b.shape, dtype=DTYPE, device="cuda", requires_grad=True)

    def mdl():
        return ((Ei + u_e * mask).clamp(min=1.0), (Si + 0.01 * u_s * mask).clamp(min=0.0))

    hist = {"misfit": [], "seconds": [], "evals": [], "failed_steps": 0}
    state = {"fn": data_fn, "evals": 0, "misfit": None}

    def new_opt():
        return torch.optim.LBFGS([u_e, u_s], lr=1.0, history_size=10, max_iter=1,
                                 line_search_fn="strong_wolfe")

    opt = new_opt()

    def closure():
        # backward per shot batch (bounded checkpoint memory), then once through the
        # parameterisation
        opt.zero_grad()
        eps, sig = mdl()
        e_l = eps.detach().requires_grad_(True)
        s_l = sig.detach().requires_grad_(True)
        misfit = 0.0
        for idx, d in batched(state["fn"], e_l, s_l):
            part = ((d * mkc[idx] - obs[idx]) ** 2).sum() / norm
            part.backward()
            misfit += float(part)
        reg = b.tv * (tv_term(u_e, mask) + tv_term(u_s, mask))
        torch.autograd.backward([eps, sig, reg], [e_l.grad, s_l.grad, torch.ones_like(reg)])
        state["evals"] += 1
        state["misfit"] = misfit
        return torch.tensor(misfit + float(reg))

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        e, s = mdl()
        hist["misfit0"] = sum(float(((d * mkc[i] - obs[i]) ** 2).sum()) / norm
                              for i, d in batched(state["fn"], e, s))
    t_total = time.perf_counter()
    for it in range(iters):
        if layout_cb is not None and it > 0 and it % layout_cb.every == 0:
            with torch.no_grad():
                e, s = mdl()
            fn = layout_cb(it // layout_cb.every, e, s)
            if fn is not None:
                state["fn"] = fn
                opt = new_opt()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n0 = state["evals"]
        before = [u_e.detach().clone(), u_s.detach().clone()]
        opt.step(closure)
        torch.cuda.synchronize()
        if torch.equal(before[0], u_e.detach()) and torch.equal(before[1], u_s.detach()):
            hist["failed_steps"] += 1           # line search found no decrease
        hist["misfit"].append(state["misfit"])
        hist["seconds"].append(time.perf_counter() - t0)
        hist["evals"].append(state["evals"] - n0)
        if log:
            log(f"  [{label}] it {it + 1:2d}: misfit {state['misfit']:.4f} "
                f"({hist['seconds'][-1]:.1f} s, {hist['evals'][-1]} evals)")
    hist["total_seconds"] = time.perf_counter() - t_total
    hist["peak_GB"] = torch.cuda.max_memory_allocated() / 2 ** 30
    hist["evaluations"] = state["evals"]
    hist["seconds_per_evaluation"] = hist["total_seconds"] / max(1, state["evals"])
    with torch.no_grad():
        e, s = mdl()
    return e.double().cpu().numpy(), s.double().cpu().numpy(), hist


# --------------------------------------------------------------------------- experiments
def load_res(name, b):
    """Results of an earlier run of the same experiment (kept when only some solvers are
    rerun), or a new record."""
    p = OUT / f"{name}.json"
    res = json.loads(p.read_text()) if p.exists() else {}
    res["bench"] = asdict(b)
    return res


def log_to(path):
    def log(msg):
        print(msg, flush=True)
        with open(path, "a") as f:
            f.write(msg + "\n")
    return log


def setup(b: Bench):
    src, rec, train, hold = acquisition(b)
    dt = main_dt(b, 3)
    nt = int(round(b.t_max / dt))
    mk = mute(src, rec, b.min_offset)
    e, s = models(b)
    d_obs = cached_fine(b, e, s, src, rec, dt, nt, "true")
    e0, s0 = models(b, targets=False)
    d_bg = cached_fine(b, e0, s0, src, rec, dt, nt, "bg")
    return src, rec, train, hold, dt, nt, mk, d_obs, d_bg


def signal_norm(d_obs, d_bg, mk, shots):
    """Energy of the target signal of ``shots`` (misfit normalisation: the background
    model's misfit on the accurate solver is 1)."""
    return float((((d_obs - d_bg)[shots] * mk[shots]) ** 2).sum())


def evaluate(b, name, eps, sig, hist, ctx, res, log):
    src, rec, train, hold, dt, nt, mk, d_obs, d_bg = ctx["setup"]
    pe, d0 = prediction_error(b, eps, sig, src, rec, hold, dt, nt, d_obs, mk, d_bg[hold] * mk[hold])
    ctx["d0_hold"] = d0
    met = model_metrics(b, eps, sig)
    row = dict(prediction_error=pe, **met,
               **{k: v for k, v in hist.items() if k not in ("misfit", "seconds", "evals")},
               misfit_history=hist["misfit"], final_misfit=hist["misfit"][-1])
    res[name] = row
    tk = " ".join(f"T{k}: eps {met.get(f'T{k}_int_deps', float('nan')):.2f} sig "
                  f"{met.get(f'T{k}_int_dsig', float('nan')):.2f}" for k in range(len(b.targets)))
    log(f"[{name}] misfit {hist['misfit0']:.4f} -> {hist['misfit'][-1]:.4f}, held-out PE {pe:.3f}, "
        f"eps err {met['eps_err']:.3f} (smooth {met['eps_err_smooth']:.3f}), sig err "
        f"{met['sig_err']:.3f} (smooth {met['sig_err_smooth']:.3f}); {tk}; "
        f"{row['seconds_per_evaluation']:.2f} s/grad x {row['evaluations']}, "
        f"{row['peak_GB']:.1f} GB, failed steps {row['failed_steps']}")
    np.savez_compressed(OUT / f"model_{ctx['exp']}_{name}.npz", eps=eps.astype(np.float32),
                        sig=sig.astype(np.float32))


def prestudy(args):
    OUT.mkdir(parents=True, exist_ok=True)
    log = log_to(OUT / "prestudy.log")
    rows = {}
    for f0 in args.freqs:
        b = Bench(f0=f0)
        src, rec, train, hold = acquisition(b)
        dt = main_dt(b, 3)
        nt = int(round(b.t_max / dt))
        mk = mute(src, rec, b.min_offset)
        e1, s1 = models(b)
        e0, s0 = models(b, targets=False)
        f1 = cached_fine(b, e1, s1, src, rec, dt, nt, "true") * mk
        f0d = cached_fine(b, e0, s0, src, rec, dt, nt, "bg") * mk
        sig_e = float(((f1 - f0d) ** 2).sum())
        row = {"target_signal_fraction": sig_e / float((f1 ** 2).sum())}
        for kind in ("coarse3", "sub3", "coarse5"):
            p = Prop(kind, b, src, rec, range(len(src)), dt, nt)
            with torch.no_grad():
                d1 = p(torch.as_tensor(e1, dtype=DTYPE, device="cuda"),
                       torch.as_tensor(s1, dtype=DTYPE, device="cuda")).cpu() * mk
                d0 = p(torch.as_tensor(e0, dtype=DTYPE, device="cuda"),
                       torch.as_tensor(s0, dtype=DTYPE, device="cuda")).cpu() * mk
            row[kind] = dict(
                background_error=float(((d0 - f0d) ** 2).sum()) / sig_e,
                target_response_error=float((((d1 - d0) - (f1 - f0d)) ** 2).sum()) / sig_e,
                true_model_floor=float(((d1 - f1) ** 2).sum()) / sig_e)
            del p
            torch.cuda.empty_cache()
        rows[f"{f0:.0e}"] = row
        log(f"f0 {f0:.2e}: target signal {row['target_signal_fraction']:.2e} of the data; "
            + "; ".join(f"{k}: bg {row[k]['background_error']:.3f}, resp {row[k]['target_response_error']:.3f}, "
                        f"floor {row[k]['true_model_floor']:.3f}" for k in ("coarse3", "sub3", "coarse5")))
    (OUT / "prestudy.json").write_text(json.dumps(rows, indent=1))


def exp1(args):
    """Forward accuracy: main grid only / static subgrid / uniform 1 cm."""
    b = Bench(f0=args.f0)
    OUT.mkdir(parents=True, exist_ok=True)
    log = log_to(OUT / f"exp1_{args.data}.log")
    ctx = {"setup": setup(b), "exp": f"exp1_{args.data}"}
    src, rec, train, hold, dt, nt, mk, d_obs, d_bg = ctx["setup"]
    norm = signal_norm(d_obs, d_bg, mk, train)
    res = load_res(f"{sys._getframe().f_code.co_name}_{args.data}", b)
    for kind in args.kinds:
        p = Prop(kind, b, src, rec, train, dt, nt)
        obs = observed(args.data, d_obs, d_bg)[train]
        e, s, hist = fwi(b, data_fn(p, b, args.data), obs, mk[train], args.iters, norm=norm,
                         label=kind, log=log)
        evaluate(b, kind, e, s, hist, ctx, res, log)
        del p
        torch.cuda.empty_cache()
        (OUT / f"exp1_{args.data}.json").write_text(json.dumps(res, indent=1))


def exp2(args):
    """Exact discrete adjoint vs main-grid adjoint for the subgridded forward model."""
    b = Bench(f0=args.f0)
    OUT.mkdir(parents=True, exist_ok=True)
    log = log_to(OUT / f"exp2_{args.data}.log")
    ctx = {"setup": setup(b), "exp": f"exp2_{args.data}"}
    src, rec, train, hold, dt, nt, mk, d_obs, d_bg = ctx["setup"]
    norm = signal_norm(d_obs, d_bg, mk, train)
    res = load_res(f"{sys._getframe().f_code.co_name}_{args.data}", b)
    sub = Prop("sub3", b, src, rec, train, dt, nt)
    coarse = Prop("coarse3", b, src, rec, train, dt, nt)
    # angle between the two gradients at the start model
    E0 = torch.full(b.shape, b.eps0, dtype=DTYPE, device="cuda").requires_grad_(True)
    S0 = torch.full(b.shape, b.sig0, dtype=DTYPE, device="cuda").requires_grad_(True)
    obs = (observed(args.data, d_obs, d_bg)[train] * mk[train]).cuda()
    sub = data_fn(sub, b, args.data)
    coarse = data_fn(coarse, b, args.data)
    grads = {}
    for name, fn in (("exact", sub), ("approx", lambda e, s: ApproxGradient.apply(sub, coarse, e, s))):
        E0.grad = S0.grad = None
        ((fn(E0, S0) * mk[train].cuda() - obs) ** 2).sum().backward()
        m = torch.as_tensor(roi_mask(b), device="cuda")
        grads[name] = (E0.grad[m].clone(), S0.grad[m].clone())
    for k, nm in enumerate(("eps", "sig")):
        a, c = grads["exact"][k], grads["approx"][k]
        res[f"gradient_cosine_{nm}"] = float((a * c).sum() / (a.norm() * c.norm()))
        res[f"gradient_rel_diff_{nm}"] = float((a - c).norm() / a.norm())
    log(f"start-model gradient: cos(exact, approx) eps {res['gradient_cosine_eps']:.3f}, sigma "
        f"{res['gradient_cosine_sig']:.3f}; rel. difference eps {res['gradient_rel_diff_eps']:.3f}, "
        f"sigma {res['gradient_rel_diff_sig']:.3f}")
    for name, fn in (("exact", sub), ("approx", lambda e, s: ApproxGradient.apply(sub, coarse, e, s))):
        e, s, hist = fwi(b, fn, observed(args.data, d_obs, d_bg)[train], mk[train], args.iters,
                         norm=norm, label=name, log=log)
        evaluate(b, name, e, s, hist, ctx, res, log)
        (OUT / f"exp2_{args.data}.json").write_text(json.dumps(res, indent=1))


def exp3(args):
    """Grid sizes: main 3 or 5 cm, with or without a subgrid (ratio 3 / 5 to 1 cm)."""
    b = Bench(f0=args.f0)
    OUT.mkdir(parents=True, exist_ok=True)
    log = log_to(OUT / f"exp3_{args.data}.log")
    ctx = {"setup": setup(b), "exp": f"exp3_{args.data}"}
    src, rec, train, hold, dt, nt, mk, d_obs, d_bg = ctx["setup"]
    norm = signal_norm(d_obs, d_bg, mk, train)
    res = load_res(f"{sys._getframe().f_code.co_name}_{args.data}", b)
    from DeepGPR.subgrid.adapt import plan_refinement
    e1, s1 = models(b)
    res["planner"] = {}
    for fm in (1.5, 2.5):
        plan = plan_refinement(e1, s1, H, f_max=fm * b.f0, ppw=10)
        res["planner"][f"f_max={fm}f0"] = {"coarsen": plan.coarsen,
                                           "boxes": [sg.box for sg in plan.subgrids],
                                           "summary": plan.summary()}
        log(f"planner (true model, f_max = {fm} f0): " + plan.summary().replace("\n", " | "))
    for kind in args.kinds:
        p = Prop(kind, b, src, rec, train, dt, nt)
        obs = observed(args.data, d_obs, d_bg)[train]
        e, s, hist = fwi(b, data_fn(p, b, args.data), obs, mk[train], args.iters, norm=norm,
                         label=kind, log=log)
        evaluate(b, kind, e, s, hist, ctx, res, log)
        del p
        torch.cuda.empty_cache()
        (OUT / f"exp3_{args.data}.json").write_text(json.dumps(res, indent=1))


def exp4_bench(f0):
    """Three targets spread over the region where boxes are admissible ((0.24-0.51) x
    (0.24-0.54)^2 m: a subgrid needs 7 main cells beyond its inner surface, and its outer
    surface must stay off the antenna planes); the update region is that region."""
    region = ((0.24, 0.51), (0.24, 0.54), (0.24, 0.54))
    return Bench(f0=f0, targets=[Target((0.33, 0.33, 0.33), 0.04, eps=16.0, sig=0.01),
                                 Target((0.42, 0.48, 0.45), 0.04, sig=0.1),
                                 Target((0.39, 0.33, 0.48), 0.035, eps=12.0)],
                 roi=region, box=region)


def exp4(args):
    """Unknown target positions: main grid only, whole-region box, AMR, oracle boxes."""
    from DeepGPR.subgrid.adapt import plan_refinement
    from DeepGPR.subgrid.amr import indicator, regrid
    b = exp4_bench(args.f0)
    OUT.mkdir(parents=True, exist_ok=True)
    log = log_to(OUT / f"exp4_{args.data}.log")
    ctx = {"setup": setup(b), "exp": f"exp4_{args.data}"}
    src, rec, train, hold, dt, nt, mk, d_obs, d_bg = ctx["setup"]
    norm = signal_norm(d_obs, d_bg, mk, train)
    res = load_res(f"{sys._getframe().f_code.co_name}_{args.data}", b)
    e1, s1 = models(b)
    # f_max = 1.5 f0 (most of the Ricker's energy): at 2.5 f0 the 3 cm grid itself is
    # under-resolved everywhere and the whole domain would be flagged
    oracle = plan_refinement(e1, s1, H, f_max=1.5 * b.f0, ppw=10, coarsen=3,
                             keep_points=np.concatenate([src.reshape(-1, 3), rec.reshape(-1, 3)]))
    res["oracle_boxes"] = [sg.box for sg in oracle.subgrids]
    log(f"oracle boxes: {res['oracle_boxes']}")
    main_n = tuple((n - 1) // 3 + 1 for n in b.shape)
    allowed = np.zeros(main_n, dtype=bool)
    allowed[tuple(slice(int(math.ceil(lo / 0.03)), int(hi / 0.03) + 1) for lo, hi in b.roi)] = True
    same = sorted(map(str, res["oracle_boxes"])) == [str(b.box)]
    if same:
        log("oracle boxes = the whole update region here: 'oracle' and 'whole' are one run")
    for kind in args.kinds:
        cb = None
        if kind == "oracle" and same and "whole" in args.kinds:
            continue
        if kind == "coarse3":
            p = Prop("coarse3", b, src, rec, train, dt, nt)
        elif kind == "whole":
            p = Prop("sub3", b, src, rec, train, dt, nt, boxes=[b.box])
        elif kind == "oracle":
            p = Prop("sub3", b, src, rec, train, dt, nt, boxes=[sg.box for sg in oracle.subgrids])
        else:                     # amr
            p = Prop("coarse3", b, src, rec, train, dt, nt)
            layout = {"boxes": [], "history": []}

            def cb(cycle, e, s, layout=layout):
                cur = Prop("sub3" if layout["boxes"] else "coarse3", b, src, rec, train, dt, nt,
                           boxes=layout["boxes"] or None)
                sh, pr = cur.props[0]
                ip = SubgridPropagator(pr.solver, src[train], rec[train], indicator=True)
                fn = data_fn(cur, b, args.data)
                with torch.no_grad():
                    d = fn(e, s) * mk[train].cuda()
                obs = (observed(args.data, d_obs, d_bg)[train] * mk[train]).cuda()
                kw = {}
                if args.data == "diff":       # DWR of the differential objective
                    kw = dict(eps0=torch.full(b.shape, b.eps0, dtype=DTYPE, device="cuda"),
                              sig0=torch.full(b.shape, b.sig0, dtype=DTYPE, device="cuda"))
                eta = indicator(ip, e, s, cur.w, (d - obs) * mk[train].cuda(), **kw)
                cur_sg = [Subgrid(bx) for bx in layout["boxes"]]
                rg = regrid(e.double().cpu().numpy(), s.double().cpu().numpy(), H, 3, eta,
                            allowed=allowed, theta=args.theta, f_max=1.5 * b.f0, ppw=10,
                            keep_points=np.concatenate([src.reshape(-1, 3), rec.reshape(-1, 3)]),
                            current=cur_sg)
                layout["boxes"] = [sg.box for sg in rg.plan.subgrids]
                layout["history"].append(layout["boxes"])
                boxes_txt = [[tuple(round(v, 2) for v in ax) for ax in bx] for bx in layout["boxes"]]
                log(f"  [amr] cycle {cycle}: {rg.info['marked_nodes']} marked -> boxes {boxes_txt}")
                if not rg.changed:
                    return None
                return data_fn(Prop("sub3", b, src, rec, train, dt, nt, boxes=layout["boxes"]), b,
                               args.data)

            cb.every = args.every
        e, s, hist = fwi(b, data_fn(p, b, args.data), observed(args.data, d_obs, d_bg)[train],
                         mk[train], args.iters, norm=norm, label=kind, log=log, layout_cb=cb)
        evaluate(b, kind, e, s, hist, ctx, res, log)
        if kind == "amr":
            res["amr"]["layouts"] = layout["history"]
        del p
        torch.cuda.empty_cache()
        (OUT / f"exp4_{args.data}.json").write_text(json.dumps(res, indent=1))


# --------------------------------------------------------------------------- exp5 (DG surface)
def exp5(args):
    """Rugged surface: staircase / staircase + subgrid / DG band / DG band + subgrid."""
    from DeepGPR.hybrid import HybridMaxwell3D, Surface3D
    from DeepGPR.subgrid.hybrid import HybridSubgridMaxwell3D
    OUT.mkdir(parents=True, exist_ok=True)
    log = log_to(OUT / f"exp5_{args.data}.log")
    surf = Surface3D.sine(0.12, 0.03, 0.5, 0.5)
    band = (0.06, 0.21)
    b = Bench(f0=args.f0, eps0=6.0, sig0=0.003, top=0.04, bottom=0.66, t_max=args.t_max,
              targets=[Target((0.495, 0.30, 0.375), 0.035, eps=18.0, sig=0.01),
                       Target((0.495, 0.45, 0.375), 0.035, sig=0.1)],
              roi=((0.40, 0.60), (0.18, 0.57), (0.18, 0.57)),
              box=((0.45, 0.54), (0.24, 0.51), (0.27, 0.48)))
    src, rec, train, hold = acquisition(b)
    mk = mute(src, rec, b.min_offset)
    sub_kw = dict(is_os_sep=2, pml_cells=4, time_ratio=1)

    def make(kind, shots, dt=None):
        e = torch.full(b.shape, b.eps0, dtype=DTYPE)
        s = torch.full(b.shape, b.sig0, dtype=DTYPE)
        sgs = [Subgrid(b.box, **sub_kw)] if kind.endswith("+sub") else []
        if kind.startswith("stair"):
            Y, Z = np.meshgrid(np.arange(b.shape[1]) * H, np.arange(b.shape[2]) * H, indexing="ij")
            air = torch.as_tensor((np.arange(b.shape[0]) * H)[:, None, None] < surf(Y, Z)[None])
            return SubgridMaxwell3D(torch.where(air, torch.ones_like(e), e),
                                    torch.where(air, torch.zeros_like(s), s), H, coarsen=3,
                                    subgrids=[Subgrid(b.box, is_os_sep=2, pml_cells=4)] if sgs else [],
                                    pmlthick=10, dt=dt, dipole_length=3 * H, nshot=len(shots),
                                    device="cuda", dtype=DTYPE)
        return HybridSubgridMaxwell3D(e, s, H, coarsen=3, surface=surf, dg_band=band, subgrids=sgs,
                                      pmlthick=10, nshot=len(shots), dt=dt, device="cuda", dtype=DTYPE)

    probe = make("dg", [0])
    dt = probe.dt
    del probe
    nt = int(round(b.t_max / dt))
    w = ricker(nt, dt, b.f0).to("cuda", DTYPE)

    def ref_data(targets, shots):
        e, s = models(b, targets)
        out = []
        for i in range(0, len(shots), 3):
            sh = shots[i:i + 3]
            sol = HybridMaxwell3D(torch.as_tensor(e), torch.as_tensor(s), H, surface=surf,
                                  dg_band=band, pmlthick=30, dg_h=3 * H, dt=dt, nshot=len(sh),
                                  device="cuda", dtype=DTYPE)
            out.append(3.0 * sol.run(w, src[sh], rec[sh])["receivers"].cpu())
            del sol
            torch.cuda.empty_cache()
        return torch.cat(out, 0)

    path = OUT / "cache" / f"exp5_ref_{b.key()}.npz"
    if path.exists():
        z = np.load(path)
        d_obs, d0 = torch.as_tensor(z["d"]), torch.as_tensor(z["d0"])
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        d_obs = ref_data(True, list(range(len(src))))
        d0 = ref_data(False, list(range(len(src))))
        np.savez_compressed(path, d=d_obs.numpy(), d0=d0.numpy())
    res = load_res(f"exp5_{args.data}", b)
    res.update({"dt": dt, "nt": nt})
    e1, s1 = models(b)
    E1 = torch.as_tensor(e1, dtype=DTYPE, device="cuda")
    S1 = torch.as_tensor(s1, dtype=DTYPE, device="cuda")
    sig_e = float((((d_obs - d0) * mk) ** 2).sum())
    log(f"exp5: dt {dt:.3e}, nt {nt}, target signal {sig_e / float(((d_obs * mk) ** 2).sum()):.2e}")
    for kind in args.kinds:
        sol = make(kind, train, dt)
        prop = SubgridPropagator(sol, src[train], rec[train], use_graph=kind.startswith("stair"))
        fn = data_fn(lambda e, s, prop=prop: prop(e, s, w), b, args.data)  # noqa: E731
        obs = observed(args.data, d_obs, d0)
        with torch.no_grad():
            floor = float((((fn(E1, S1).cpu() - obs[train]) * mk[train]) ** 2).sum()) / float(
                (((d0[train] - d_obs[train]) * mk[train]) ** 2).sum())
        log(f"[{kind}] true-model misfit ({args.data} data, relative to the target signal) {floor:.3f}")
        norm = float((((d_obs - d0)[train] * mk[train]) ** 2).sum())
        e, s, hist = fwi(b, fn, obs[train], mk[train], args.iters, norm=norm, label=kind, log=log)
        # held-out prediction with the reference solver
        hs = hold
        e_t, s_t = torch.as_tensor(e), torch.as_tensor(s)
        sol_r = HybridMaxwell3D(e_t.float(), s_t.float(), H, surface=surf, dg_band=band, pmlthick=30,
                                dg_h=3 * H, dt=dt, nshot=len(hs), device="cuda", dtype=DTYPE)
        dp = 3.0 * sol_r.run(w, src[hs], rec[hs])["receivers"].cpu()
        del sol_r
        pe = float((((dp - d_obs[hs]) * mk[hs]) ** 2).sum() / (((d0[hs] - d_obs[hs]) * mk[hs]) ** 2).sum())
        met = model_metrics(b, e, s)
        res[kind] = dict(true_model_floor=floor, prediction_error=pe, **met,
                         **{k: v for k, v in hist.items() if k not in ("misfit", "seconds", "evals")},
                         misfit_history=hist["misfit"], final_misfit=hist["misfit"][-1])
        log(f"[{kind}] misfit {hist['misfit0']:.4f} -> {hist['misfit'][-1]:.4f}, held-out PE {pe:.3f}, "
            f"eps err {met['eps_err']:.3f} (smooth {met['eps_err_smooth']:.3f}), sig err "
            f"{met['sig_err']:.3f}; T0 int deps {met.get('T0_int_deps', float('nan')):.2f}, "
            f"T1 int dsig {met.get('T1_int_dsig', float('nan')):.2f}; "
            f"{res[kind]['seconds_per_evaluation']:.1f} s/grad, {res[kind]['peak_GB']:.1f} GB")
        np.savez_compressed(OUT / f"model_exp5_{args.data}_{kind.replace('+', '_')}.npz", eps=e.astype(np.float32),
                            sig=s.astype(np.float32))
        del sol, prop
        torch.cuda.empty_cache()
        (OUT / f"exp5_{args.data}.json").write_text(json.dumps(res, indent=1))


# --------------------------------------------------------------------------- report
def _model_file(exp, kind):
    if exp.startswith("exp5"):
        return OUT / f"model_{exp}_{kind.replace('+', '_')}.npz"
    return OUT / f"model_{exp}_{kind}.npz"


def report(args):
    """Per experiment: training misfit, held-out prediction error, smoothed eps error and
    cost (``<exp>.png``), model slices through the targets (``<exp>_models.png``); over all
    experiments: cost against prediction error (``summary.png``) and ``report.md``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for exp in [f"exp{i}_{m}" for i in range(1, 6) for m in ("diff", "raw")]:
        p = OUT / f"{exp}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        kinds = [k for k, v in r.items() if isinstance(v, dict) and "prediction_error" in v]
        if not kinds:
            continue
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
        for k in kinds:
            v = r[k]
            ax[0].semilogy(np.arange(1, len(v["misfit_history"]) + 1), v["misfit_history"], label=k)
            rows.append((exp, k, v))
        ax[0].set_xlabel("iteration")
        ax[0].set_ylabel("training misfit")
        ax[0].legend()
        x = np.arange(len(kinds))
        ax[1].bar(x - 0.2, [r[k]["prediction_error"] for k in kinds], 0.4, label="held-out PE")
        ax[1].bar(x + 0.2, [r[k]["eps_err_smooth"] for k in kinds], 0.4, label="eps err (smoothed)")
        ax[1].set_xticks(x, kinds, rotation=20)
        ax[1].legend()
        ax[2].bar(x, [r[k]["seconds_per_evaluation"] for k in kinds])
        ax[2].set_xticks(x, kinds, rotation=20)
        ax[2].set_ylabel("s / gradient")
        fig.suptitle(exp)
        fig.savefig(OUT / f"{exp}.png", dpi=120)
        plt.close(fig)
        # model slices at the depth of the first target
        bd = r["bench"]
        b = Bench(**{k: v for k, v in bd.items() if k != "targets"})
        b.targets = [Target(tuple(t["centre"]), t["radius"], t["eps"], t["sig"]) for t in bd["targets"]]
        te, ts = models(b)
        ix = int(round(b.targets[0].centre[0] / H))
        cols = [("true", te, ts)]
        for k in kinds:
            f = _model_file(exp, k)
            if f.exists():
                z = np.load(f)
                cols.append((k, z["eps"], z["sig"]))
        emax = max([t.eps or b.eps0 for t in b.targets])
        smax = max([t.sig or b.sig0 for t in b.targets])
        fig, ax = plt.subplots(2, len(cols), figsize=(3.0 * len(cols), 5.6), constrained_layout=True)
        ext = [0, b.shape[2] * H, b.shape[1] * H, 0]
        for j, (name, e, sg) in enumerate(cols):
            ax[0, j].imshow(e[ix], vmin=b.eps0, vmax=emax, cmap="viridis", extent=ext)
            ax[1, j].imshow(1e3 * sg[ix], vmin=1e3 * b.sig0, vmax=1e3 * smax, cmap="magma", extent=ext)
            ax[0, j].set_title(f"{name}: eps_r")
            ax[1, j].set_title(f"{name}: sigma [mS/m]")
            for rr in range(2):
                ax[rr, j].set_xlabel("z [m]")
                ax[rr, j].set_ylabel("y [m]")
        fig.suptitle(f"{exp}: slice x = {ix * H:.2f} m")
        fig.savefig(OUT / f"{exp}_models.png", dpi=110)
        plt.close(fig)
    if rows:
        modes = [m for m in ("diff", "raw") if any(e.endswith(m) for e, _, _ in rows)]
        fig, axes = plt.subplots(1, len(modes), figsize=(7.5 * len(modes), 5.2), squeeze=False,
                                 constrained_layout=True)
        colors = {f"exp{i}": c for i, c in zip(range(1, 6), ("tab:blue", "tab:orange", "tab:green",
                                                              "tab:purple", "tab:red"))}
        for ax, mode in zip(axes[0], modes):
            for exp, k, v in rows:
                if not exp.endswith(mode):
                    continue
                e = exp.split("_")[0]
                ax.scatter(v["seconds_per_evaluation"], v["prediction_error"], s=45, color=colors[e])
                ax.annotate(f"{e}:{k}", (v["seconds_per_evaluation"], v["prediction_error"]),
                            fontsize=8, xytext=(4, 3), textcoords="offset points")
            ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
            ax.text(ax.get_xlim()[0] if False else 0.3, 1.05, "start model", fontsize=8, color="0.4")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("seconds per gradient (6 training shots)")
            ax.set_ylabel("held-out prediction error (accurate solver)")
            ax.set_title(f"{mode} data: cost against model quality")
        fig.savefig(OUT / "summary.png", dpi=120)
        plt.close(fig)
    lines = ["| exp | solver | held-out PE | train misfit | eps err (smooth) | sig err (smooth) | "
             "int dEps / dSig per target | s/grad | evals | GB | failed steps |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for exp, k, v in rows:
        ints = " ; ".join(f"{v.get(f'T{i}_int_deps', float('nan')):.2f} / {v.get(f'T{i}_int_dsig', float('nan')):.2f}"
                          for i in range(3) if f"T{i}_peak_eps" in v)
        lines.append(f"| {exp} | {k} | {v['prediction_error']:.3f} | {v['final_misfit']:.4f} | "
                     f"{v['eps_err']:.3f} ({v['eps_err_smooth']:.3f}) | {v['sig_err']:.3f} "
                     f"({v['sig_err_smooth']:.3f}) | {ints} | {v['seconds_per_evaluation']:.2f} | "
                     f"{v['evaluations']} | {v['peak_GB']:.1f} | {v.get('failed_steps', '')} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("prestudy", "exp1", "exp2", "exp3", "exp4", "exp5", "report"))
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--f0", type=float, default=3e8)
    p.add_argument("--freqs", type=float, nargs="+", default=[2e8, 3e8, 4e8])
    p.add_argument("--kinds", nargs="+", default=None)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--t_max", type=float, default=14e-9)
    p.add_argument("--data", choices=("diff", "raw"), default="diff",
                   help="differential data (known baseline, default) or raw data")
    args = p.parse_args()
    defaults = {"exp1": ["coarse3", "sub3", "fine"], "exp3": ["coarse5"],
                "exp4": ["coarse3", "whole", "amr", "oracle"],
                "exp5": ["stair", "stair+sub", "dg", "dg+sub"]}
    if args.kinds is None:
        args.kinds = defaults.get(args.cmd, [])
    {"prestudy": prestudy, "exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4, "exp5": exp5,
     "report": report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

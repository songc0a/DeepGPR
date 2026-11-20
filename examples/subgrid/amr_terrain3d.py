"""Dynamic mesh and time-step scheduling for 3D joint eps/sigma FWI on rugged terrain
(paper step b).

The deep variant of the model of ``examples/hybrid/fwi3d_lts.py`` (same surface
statistics, band, flux, acquisition pattern, data generator and optimiser), deep
enough for Huygens subgrids under the DG band (a box needs 7 main cells of padding
below the band's two-cell gap):

* 2 cm model grid, 1.0 m deep, 0.8 m x 0.8 m; surface 0.17 m +- 7 cm (11 x 11
  samples, sine pattern + random roughness, PCHIP); DG band 0.08-0.28 m (P1, flux
  0.1 with the split penalty under LTS); ground eps_r 5 / 3 mS/m; an eps_r 7.5
  sphere and, 24 cm away, a 40 mS/m sphere (radius 8 cm, centres 0.60 m deep); a
  known reflector eps_r 8 below 0.86 m; 3 x 3 sources, 7 x 7 receivers 4 cm below the
  top, 400 MHz, 16 ns; data from the native 4 mm staircase solver.
* Time-step ladder: Yee 2 cm main grid at its own step (LTS macro step), the DG band
  sub-cycled (``lts="auto"``), subgrids at 2/3 cm with time ratio 3.

Methods (same stages and optimiser as fwi3d_lts.py: 8 L-BFGS iterations eps only at
250 MHz, then 24 joint):

* ``fine``: uniform Yee 2/3 cm everywhere (``fdtd_refinement = 3``) with LTS - the
  accurate and expensive reference;
* ``coarse``: Yee 2 cm, no subgrids;
* ``static``: one subgrid box over the whole admissible region from the start;
* ``sched``: the scheduler (:class:`DeepGPR.subgrid.schedule.Scheduler`) decides from
  the misfit history and the dual-weighted-residual estimate of the modelling error
  (coarse main grid, ``--estimate absolute|differential``, ``--indicator-shots``)
  when to refine; with ``--urgent u`` (default 0.5) a check runs at the start of each
  stage and refines at once when the estimate is at least ``u`` of the misfit,
  otherwise checks start ``--min-cycle`` iterations into each stage and wait for a
  misfit plateau.
  Where to refine follows from where the indicator lies: when more than
  ``--global-threshold`` of its mass is outside the region subgrid boxes can cover
  (above the band, near the surface and the antennas), the layout switches to the
  uniform fine grid (global refinement); otherwise boxes are planned under the
  fine-node budget of the remaining time.

The uniform fine reference (``fine``) uses global stepping (hand-written transpose,
the fastest gradient of the fine level); ``coarse``/``static``/``sched`` use LTS on
the 2 cm main grid.

    python examples/subgrid/amr_terrain3d.py observed
    python examples/subgrid/amr_terrain3d.py diagnose
    python examples/subgrid/amr_terrain3d.py fine | coarse | static | sched [--budget 7200]
    python examples/subgrid/amr_terrain3d.py evaluate     # equal wall time, fine solver
    python examples/subgrid/amr_terrain3d.py report

Results in ``examples/subgrid/results/amr_terrain3d/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples" / "hybrid"))
sys.path.insert(0, str(ROOT / "src"))

import fwi3d_lts as F  # noqa: E402

from DeepGPR import apply_filter  # noqa: E402
from DeepGPR.hybrid import Surface3D, resample  # noqa: E402
from DeepGPR.subgrid import Subgrid  # noqa: E402
from DeepGPR.subgrid.adapt import subgrid_padding  # noqa: E402
from DeepGPR.subgrid.adjoint import SubgridPropagator  # noqa: E402
from DeepGPR.subgrid.hybrid import HybridSubgridMaxwell3D  # noqa: E402
from DeepGPR.subgrid.schedule import CostModel, Scheduler, TimeStepLadder, subgrid_fine_nodes  # noqa: E402

OUT = ROOT / "examples" / "subgrid" / "results" / "amr_terrain3d"
R = 3                       # subgrid ratio = coarsening of the fine model grid
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SUB = dict(is_os_sep=3, pml_cells=6)


def configure():
    """Deep variant of the fwi3d_lts.py model (patches its module globals)."""
    F.OUT = OUT
    F.SHAPE = (50, 40, 40)
    F.T_MAX = 16e-9
    F.LAYER_X = 0.86
    F.SPHERES = [(0.60, 0.28, 0.40, 0.08, 7.5, None), (0.60, 0.52, 0.40, 0.08, None, 0.04)]
    ey = np.linspace(0.0, 0.78, 11)
    Y, Z = np.meshgrid(ey, ey, indexing="ij")
    elev = (0.05 * np.sin(2 * np.pi * Y / 0.45) * np.cos(2 * np.pi * Z / 0.55)
            + 0.025 * np.random.default_rng(3).uniform(-1.0, 1.0, Y.shape))
    F.SURFACE = Surface3D.from_grid(ey, ey, 0.17 - elev)
    F.SRC_YZ = [(y, z) for y in (0.16, 0.40, 0.64) for z in (0.15, 0.39, 0.63)]
    F.REC_YZ = [(round(y, 2), round(z, 2)) for y in np.arange(0.08, 0.69, 0.10)
                for z in np.arange(0.07, 0.68, 0.10)]
    yy, zz = np.meshgrid(np.arange(F.SHAPE[1]) * F.DX, np.arange(F.SHAPE[2]) * F.DX, indexing="ij")
    s = F.SURFACE(yy, zz)
    assert F.BAND[0] < s.min() and s.max() < F.BAND[1], (s.min(), s.max())
    OUT.mkdir(parents=True, exist_ok=True)


FINE = None          # fine model shape ((n - 1) R + 1)
_IDX = None          # nearest 2 cm pixel of every fine node


def _setup_fine():
    global FINE, _IDX
    FINE = tuple((n - 1) * R + 1 for n in F.SHAPE)
    _IDX = [torch.as_tensor(np.clip(np.floor(np.arange(m) / R + 0.5).astype(int), 0, n - 1),
                            device=DEV) for m, n in zip(FINE, F.SHAPE)]


def up(t: torch.Tensor) -> torch.Tensor:
    """2 cm model -> fine model grid (2/3 cm), nearest pixel (differentiable); the
    hybrid's block average of the result is the 2 cm model again."""
    return t.index_select(0, _IDX[0]).index_select(1, _IDX[1]).index_select(2, _IDX[2])


def log_to(path):
    fh = open(path, "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log


# --------------------------------------------------------------------------- layouts
class Layout:
    """Hybrid (2 cm Yee + DG band, LTS) with Huygens subgrids ``boxes`` (2/3 cm);
    shot-by-shot propagators sharing the solver."""

    def __init__(self, boxes=(), indicator=False):
        self.boxes = [tuple(map(tuple, b)) for b in boxes]
        e0, s0 = (torch.tensor(a, dtype=torch.float32, device=DEV) for a in F.background())
        self.sol = HybridSubgridMaxwell3D(
            up(e0), up(s0), F.DX / R, coarsen=R, surface=F.SURFACE, dg_band=F.BAND,
            subgrids=[Subgrid(b, ratio=R, **SUB) for b in self.boxes], pmlthick=F.PML,
            flux=F.FLUX, lts="auto", nshot=1, device=DEV, dtype=torch.float32)
        src, rec = F.antennas()
        self.props = [SubgridPropagator(self.sol, src[i:i + 1], rec[i:i + 1], indicator=indicator)
                      for i in range(len(F.SRC_YZ))]
        self.nt = int(np.ceil(F.T_MAX / self.sol.dt)) + 1
        self.t_prop = ((torch.arange(self.nt, dtype=torch.float64) + 1) * self.sol.dt).to(DEV)
        self.t_out = F.time_axis().to(DEV)
        self.w = torch.tensor(F.wavelet(self.nt, self.sol.dt), dtype=torch.float32, device=DEV)

    def __call__(self, eps2, sig2, i):
        d = self.props[i](up(eps2), up(sig2), self.w[None])
        return resample(d, self.t_prop.to(d.dtype), self.t_out.to(d.dtype))

    def fine_nodes(self) -> int:
        return int(sum(int(np.prod(sg.yee.shape)) for sg in self.sol.static.subs))

    def ladder(self):
        return TimeStepLadder.from_solver(self.sol)


class FineLayout:
    """Uniform Yee 2/3 cm (hybrid, ``fdtd_refinement = 3``, global stepping): the
    global refinement level of the scheduler, same interface as :class:`Layout`."""

    boxes = ["uniform 2/3 cm"]

    def __init__(self):
        self.sol = F.build_solver("global", 1, DEV)
        self.fw = F.Forward(self.sol, DEV)

    def __call__(self, eps2, sig2, i):
        return self.fw(eps2, sig2, i)

    def fine_nodes(self) -> int:
        return int(np.prod(self.sol.static.yee.shape))

    def ladder(self):
        return TimeStepLadder.from_solver(self.sol)


def allowed_mask():
    """Main nodes (2 cm model grid) a box may cover: below the band's gap plus the
    subgrid padding, and the padding away from the lateral and bottom edges."""
    pad = subgrid_padding(R, SUB["is_os_sep"], SUB["pml_cells"])
    i_b = int(round(F.BAND[1] / F.DX))
    m = np.zeros(F.SHAPE, dtype=bool)
    lo_x, hi_x = i_b + 2 + pad + 1, F.SHAPE[0] - 3 - pad - 1
    lo_y, hi_y = 1 + pad + 1, F.SHAPE[1] - 2 - pad - 1
    lo_z, hi_z = 1 + pad + 1, F.SHAPE[2] - 2 - pad - 1
    m[lo_x:hi_x + 1, lo_y:hi_y + 1, lo_z:hi_z + 1] = True
    return m


def whole_box():
    """The largest admissible box (static method)."""
    a = allowed_mask()
    idx = np.nonzero(a)
    return tuple((float(i.min() * F.DX), float(i.max() * F.DX)) for i in idx)


# --------------------------------------------------------------------------- indicator
def dwr_indicator(eps2, sig2, eps0, sig0, dobs, mute, norm, shots, differential=False):
    """``(|eta|, estimate)``: the signed DWR estimate of the coarse main grid's
    modelling error in the misfit (no subgrids), summed over ``shots`` and scaled to
    all shots, in misfit units. ``differential``: of the anomaly response only
    (``eta(m) - eta(m0)``, as in amr_scheduled.py), which does not see the error of
    the background response."""
    lay = Layout((), indicator=True)
    eta = None
    for i in shots:
        with torch.enable_grad():
            e = eps2.detach().clone().requires_grad_(True)
            d = lay(e, sig2.detach(), i)
            J = 0.5 * (((d * mute[i:i + 1]) - dobs[i:i + 1]) ** 2).sum() / norm
            res = torch.autograd.grad(J, d)[0].detach()

        def signed(e_, s_):
            e_ = e_.detach().clone().requires_grad_(True)
            dd = lay(e_, s_.detach(), i)
            dd.backward(res)
            return lay.props[i].indicator_obj.signed().clone()

        part = signed(eps2, sig2) - signed(eps0, sig0) if differential else signed(eps2, sig2)
        eta = part if eta is None else eta + part
    eta = eta * (len(F.SRC_YZ) / len(shots))
    est = float(eta.sum())
    ind = lay.props[shots[-1]].indicator_obj
    e = eta.abs()
    sl = tuple(slice(ind.pml[2 * a], e.shape[a] - 1 - ind.pml[2 * a + 1]) for a in range(3))
    del lay
    torch.cuda.empty_cache()
    return e[sl].cpu().numpy(), est


# --------------------------------------------------------------------------- inversion
def run_amr(method, budget, args, log):
    mute = (F.mute_mask()[:, None, :] * F.gain()[None, :, None]).to(DEV)    # mute and time gain
    dobs = F.observed_data(DEV).to(DEV, torch.float32) * mute
    fs = 1.0 / F.DT_OUT
    eps0, sig0 = (torch.tensor(a, dtype=torch.float32, device=DEV) for a in F.background())
    true_e, true_s = (torch.tensor(a, dtype=torch.float32) for a in F.true_ground())
    gmask = F.ground_mask().to(DEV)
    tm = F.ground_mask()
    sph = F.sphere_masks()
    me = torch.zeros(F.SHAPE, device=DEV, requires_grad=True)
    ms = torch.zeros(F.SHAPE, device=DEV, requires_grad=True)
    hist = {k: [] for k in ("misfit", "eps_err", "sig_err", "seconds", "evals", "fine_nodes",
                            "eps_in_eps_sphere", "sig_in_sig_sphere", "sig_in_eps_sphere",
                            "eps_in_sig_sphere", "eps", "sig")}
    decisions, grad_seconds = [], []
    n_eval = [0]
    boxes = [whole_box()] if method == "static" else []
    lay = Layout(boxes)
    log(f"[{method}] layout {lay.boxes}, fine nodes {lay.fine_nodes()}, p={lay.sol.hyb.lts}, "
        f"dt={lay.sol.dt:.3e}, nt={lay.nt}")

    def models():
        return (torch.clamp(eps0 + me * gmask, min=1.0),
                torch.clamp(sig0 + F.OMEGA_EPS0 * ms * gmask, min=0.0))

    def err(m, t, m0):
        return float((m - t)[tm].norm() / (t - m0.cpu())[tm].norm())

    def filtered(a, cutoff):
        return a if cutoff is None else apply_filter(a, fs, cutoff)

    def objective(cutoff, backward=False, scale=1.0):
        norm = float((filtered(dobs, cutoff) ** 2).sum())
        total = 0.0
        t0 = time.perf_counter()
        for i in range(len(F.SRC_YZ)):
            eps, sig = models()
            d = lay(eps, sig, i) * mute[i:i + 1]
            J = 0.5 * ((filtered(d, cutoff) - filtered(dobs[i:i + 1], cutoff)) ** 2).sum() / norm
            if backward:
                (scale * J).backward()
            total += float(J)
        if backward:
            torch.cuda.synchronize()
            grad_seconds.append(time.perf_counter() - t0)
            n_eval[0] += 1
        return total

    def record(J):
        eps, sig = (t.detach().cpu() for t in models())
        hist["misfit"].append(float(J))
        hist["eps_err"].append(err(eps, true_e, eps0))
        hist["sig_err"].append(err(sig, true_s, sig0))
        hist["eps_in_eps_sphere"].append(float(eps[sph[0]].mean()))
        hist["sig_in_sig_sphere"].append(float(sig[sph[1]].mean()))
        hist["sig_in_eps_sphere"].append(float(sig[sph[0]].mean()))
        hist["eps_in_sig_sphere"].append(float(eps[sph[1]].mean()))
        hist["eps"].append(eps.numpy().astype(np.float32))
        hist["sig"].append(sig.numpy().astype(np.float32))
        hist["seconds"].append(time.time() - t_start)
        hist["evals"].append(n_eval[0])
        hist["fine_nodes"].append(lay.fine_nodes())
        log(f"  [{method}] epoch {len(hist['misfit']) - 1:3d}  misfit {J:.4e}  eps err "
            f"{hist['eps_err'][-1]:.3f}  sig err {hist['sig_err'][-1]:.3f}  eps@eps-sphere "
            f"{hist['eps_in_eps_sphere'][-1]:.2f}  sig@sig-sphere {1e3 * hist['sig_in_sig_sphere'][-1]:.1f}"
            f" mS/m  boxes {len(lay.boxes)} ({lay.fine_nodes()} fine nodes)  evals {n_eval[0]}  "
            f"t {hist['seconds'][-1]:.0f} s")

    def save():
        np.savez(OUT / f"run_{method}.npz", **{k: np.array(v) for k, v in hist.items()},
                 true_eps=true_e.numpy(), true_sig=true_s.numpy(), mask=tm.numpy())
        meta = {k: hist[k] for k in hist if k not in ("eps", "sig")}
        meta.update(method=method, decisions=decisions, grad_seconds=grad_seconds,
                    final_boxes=lay.boxes, budget=budget)
        (OUT / f"run_{method}.json").write_text(json.dumps(meta, indent=1, default=float))

    cost = CostModel()
    n_iter = sum(s[0] for s in F.STAGES)
    sched = Scheduler(cost, budget, n_iter, balance=args.balance, plateau=args.plateau,
                      window=args.window, min_cycle=args.min_cycle,
                      urgent=args.urgent if args.urgent > 0 else None)
    t_start = time.time()
    sched.t0 = time.perf_counter()
    with torch.no_grad():
        j0 = objective(None)
    record(j0)
    sched.record(j0)                # the misfit of the initial layout (check before iteration 1)
    it = 0
    for st_i, (iters, joint, cutoff) in enumerate(F.STAGES):
        params = [me, ms] if joint else [me]

        def new_opt():
            return torch.optim.LBFGS(params, lr=1.0, max_iter=1, history_size=10,
                                     line_search_fn="strong_wolfe")
        opt = new_opt()
        with torch.no_grad():
            j_ref = objective(cutoff)
        # first check of a stage: at its start when the urgent rule is on (a layout whose
        # own error dominates the residual is refined before any iteration), else after
        # min_cycle iterations
        next_check = it if args.urgent > 0 else it + args.min_cycle
        for _ in range(iters):
            if method == "sched" and it >= next_check and not isinstance(lay, FineLayout):
                next_check = it + args.check_every
                with torch.no_grad():
                    eps, sig = models()
                norm = float((dobs ** 2).sum())
                t0 = time.perf_counter()
                eta, est = dwr_indicator(eps, sig, eps0, sig0, dobs, mute, norm,
                                         args.indicator_shots,
                                         differential=args.estimate == "differential")
                t_ind = time.perf_counter() - t0
                allowed = allowed_mask()
                out_frac = float(eta[~allowed].sum() / max(eta.sum(), 1e-30))
                dec = sched.decide(est, has_boxes=bool(lay.boxes))
                rec = dict(it=it, t=time.time() - t_start, action=dec.action, reason=dec.reason,
                           misfit=dec.misfit, estimate=est, ratio=dec.balance,
                           plateau=dec.plateau, indicator_seconds=t_ind,
                           indicator_outside_boxes=out_frac)
                log(f"  [{method}] check at it {it}: misfit {dec.misfit:.4e}, estimate {est:+.4e} "
                    f"(ratio {dec.balance:.3f}, plateau {dec.plateau}) -> {dec.action}: {dec.reason} "
                    f"(indicator {t_ind:.0f} s, {100 * out_frac:.0f} % of it outside the box region)")
                if dec.action == "refine" and out_frac > args.global_threshold \
                        and not isinstance(lay, FineLayout):
                    log(f"  [{method}] global refinement: the modelling error lies mostly where "
                        f"boxes cannot go -> uniform 2/3 cm")
                    rec.update(action="refine-global")
                    lay = FineLayout()
                    sched.last_change = sched.iteration
                    opt = new_opt()
                    with torch.no_grad():
                        j_ref = objective(cutoff)
                elif dec.action in ("refine", "coarsen") and not isinstance(lay, FineLayout):
                    ladder = lay.ladder()
                    # calibrate the cost model on the measured gradients of this layout
                    # (none yet before the first iteration: keep the default rates)
                    measured = float(np.median(grad_seconds[-3:])) if grad_seconds else float("nan")
                    if grad_seconds:
                        scale = measured / cost.predict(ladder, F.T_MAX, shots=len(F.SRC_YZ),
                                                        gradient=True)
                        for k in ("yee", "dg", "sub", "overhead"):
                            setattr(cost, k, getattr(cost, k) * scale)
                    with torch.no_grad():
                        ef, sf = (up(t).cpu().numpy().astype(np.float64) for t in models())
                    current = [Subgrid(b, ratio=R, **SUB) for b in lay.boxes]
                    rg = sched.plan(ef, sf, F.DX / R, R, eta, ladder=ladder, duration=F.T_MAX,
                                    shots=len(F.SRC_YZ), allowed=allowed, current=current,
                                    f_max=1.0e9, ratio=R, pmlthick=F.PML, **SUB)
                    new = [tuple(map(tuple, np.round(np.asarray(sg.box, dtype=float), 6)))
                           for sg in rg.plan.subgrids]
                    rec.update(new_boxes=new, changed=bool(rg.changed),
                               fine_nodes_planned=sum(subgrid_fine_nodes(rg.plan.boxes_cells, R,
                                                                         SUB["is_os_sep"],
                                                                         SUB["pml_cells"])),
                               measured_gradient_seconds=measured)
                    log(f"  [{method}] plan: {new} (changed {rg.changed}, measured {measured:.1f} s/gradient)")
                    if rg.changed:
                        try:
                            lay_new = Layout(new)
                        except Exception as exc:   # box outside the admissible region
                            log(f"  [{method}] layout rejected: {exc}")
                            lay_new = None
                        if lay_new is not None:
                            lay = lay_new
                            sched.last_change = sched.iteration
                            opt = new_opt()
                            with torch.no_grad():
                                j_ref = objective(cutoff)
                decisions.append(rec)
            def closure():
                opt.zero_grad()
                J = objective(cutoff, backward=True, scale=1.0 / j_ref) / j_ref
                return torch.tensor(J)
            opt.step(closure)
            it += 1
            with torch.no_grad():
                Jf = objective(None)
            sched.record(Jf)
            record(Jf)
            save()
    save()


def run_fine(log, epochs=None):
    """Uniform Yee 2/3 cm, global stepping (fwi3d_lts.run on this model)."""
    log("[fine] start")
    F.run("global", 1, DEV, epochs=epochs, tag="fine")


def diagnose(log):
    mute = F.mute_mask()[:, None, :] * F.gain()[None, :, None]
    obs_t = F.observed_data(DEV) * mute
    obs_b = F.observed_data(DEV, background_model=True) * mute
    n = lambda a: float(a.norm())  # noqa: E731
    out = {"anomaly_signal": n(obs_t - obs_b) / n(obs_t)}
    log(f"anomaly signal {out['anomaly_signal']:.4f}")
    for name, boxes in (("coarse", []), ("static", [whole_box()])):
        lay = Layout(boxes)
        t0 = time.time()
        res = {}
        for tag, model in (("true", F.true_ground()), ("background", F.background())):
            e, s = (torch.tensor(a, dtype=torch.float32, device=DEV) for a in model)
            with torch.no_grad():
                res[tag] = torch.cat([lay(e, s, i) for i in range(len(F.SRC_YZ))]).cpu().float() * mute
        out[name] = dict(modelling_error=n(res["true"] - obs_t) / n(obs_t),
                         own_anomaly_signal=n(res["true"] - res["background"]) / n(res["true"]),
                         fine_nodes=lay.fine_nodes(), boxes=lay.boxes, p=lay.sol.hyb.lts,
                         dt=lay.sol.dt, forward_seconds=(time.time() - t0) / 2)
        log(f"{name}: {out[name]}")
        del lay
        torch.cuda.empty_cache()
    F.FLUX = 0.1
    sol = F.build_solver("lts", 1, DEV)
    fw = F.Forward(sol, DEV)
    res = {}
    t0 = time.time()
    for tag, model in (("true", F.true_ground()), ("background", F.background())):
        e, s = (torch.tensor(a, dtype=torch.float32, device=DEV) for a in model)
        with torch.no_grad():
            res[tag] = torch.cat([fw(e, s, i) for i in range(len(F.SRC_YZ))]).cpu().float() * mute
    out["fine"] = dict(modelling_error=n(res["true"] - obs_t) / n(obs_t),
                       own_anomaly_signal=n(res["true"] - res["background"]) / n(res["true"]),
                       p=sol.lts, dt=sol.dt, forward_seconds=(time.time() - t0) / 2)
    log(f"fine: {out['fine']}")
    (OUT / "diagnose.json").write_text(json.dumps(out, indent=1, default=float))


EVAL_MIN = (10, 20, 40, 60, 90, 120, 180, 240)     # wall times [min] of the comparison


def evaluate(log):
    """Equal-wall-time comparison with one accurate solver: the model every method
    holds at the times ``EVAL_MIN`` (its last epoch before that time; after its end
    its final model) is simulated with the uniform 2/3 cm hybrid (the ``fine``
    level, modelling error 2.1 %) and scored against the observed data. The methods'
    own misfits are not comparable: each includes the modelling error of its own
    solver (27 % on the coarse grid). Results in ``evaluate.json``."""
    mute = (F.mute_mask()[:, None, :] * F.gain()[None, :, None]).to(DEV)
    dobs = F.observed_data(DEV).to(DEV, torch.float32) * mute
    norm = float((dobs ** 2).sum())
    lay = FineLayout()
    path = OUT / "evaluate.json"
    res = json.loads(path.read_text()) if path.exists() else {}

    def fine_misfit(eps, sig):
        J = 0.0
        with torch.no_grad():
            for i in range(len(F.SRC_YZ)):
                d = lay(torch.as_tensor(eps, device=DEV), torch.as_tensor(sig, device=DEV), i)
                J += 0.5 * float(((d * mute[i:i + 1] - dobs[i:i + 1]) ** 2).sum()) / norm
        return J

    for m in ("fine", "coarse", "static", "sched"):
        p = OUT / f"run_{m}.npz"
        if not p.exists():
            continue
        r = np.load(p)
        minutes = np.asarray(r["seconds"]) / 60
        picks = {str(t): max(int(np.searchsorted(minutes, t, side="right")) - 1, 0)
                 for t in EVAL_MIN}
        picks["final"] = len(minutes) - 1
        out = res.setdefault(m, {})
        cache = {}
        for key, k in picks.items():
            if key in out and out[key]["epoch"] == k:
                continue
            if k not in cache:
                cache[k] = fine_misfit(r["eps"][k], r["sig"][k])
            out[key] = dict(epoch=k, minutes=float(minutes[k]), misfit_fine=cache[k],
                            data_error=float(np.sqrt(2.0 * cache[k])),
                            own_misfit=float(r["misfit"][k]), eps_err=float(r["eps_err"][k]),
                            sig_err=float(r["sig_err"][k]),
                            eps_in_eps_sphere=float(r["eps_in_eps_sphere"][k]),
                            sig_in_sig_sphere=float(r["sig_in_sig_sphere"][k]))
            log(f"[evaluate] {m} at {key} min: epoch {k} ({minutes[k]:.1f} min), data error "
                f"{100 * out[key]['data_error']:.2f} % (own misfit {out[key]['own_misfit']:.3e}, "
                f"fine {cache[k]:.3e}), eps err {out[key]['eps_err']:.3f}, "
                f"sig err {out[key]['sig_err']:.3f}")
            path.write_text(json.dumps(res, indent=1))
    path.write_text(json.dumps(res, indent=1))


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {}
    for m in ("fine", "coarse", "static", "sched"):
        p = OUT / f"run_{m}.npz"
        if p.exists():
            runs[m] = np.load(p)
    colors = {"fine": "#1d4ed8", "coarse": "#c2410c", "static": "#0f766e", "sched": "#7c3aed"}
    pe = OUT / "evaluate.json"
    ev = json.loads(pe.read_text()) if pe.exists() else {}
    fig, axes = plt.subplots(1, 4 if ev else 3, figsize=(20 if ev else 15, 4.2), facecolor="white")
    for m, r in runs.items():
        t = np.array(r["seconds"]) / 60
        axes[0].semilogy(t, r["misfit"] / r["misfit"][0], color=colors[m], label=m)
        axes[1].plot(t, r["eps_err"], color=colors[m], label=m)
        axes[2].plot(t, r["sig_err"], color=colors[m], label=m)
        if m in ev:
            pts = sorted({(v["minutes"], v["data_error"]) for v in ev[m].values()})
            axes[3].plot([a for a, _ in pts], [100 * b for _, b in pts], "o-", color=colors[m],
                         label=m)
    titles = ["own misfit / initial (own solver)", "eps_r model error", "sigma model error",
              "data error with the fine solver [%]"]
    for a, title in zip(axes, titles):
        a.set_title(title)
        a.set_xlabel("wall time [min]")
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=120)
    plt.close(fig)
    lines = ["# AMR + LTS scheduling on rugged terrain (deep model)", ""]
    p = OUT / "diagnose.json"
    if p.exists():
        lines += ["```json", p.read_text(), "```", ""]
    lines += ["| method | epochs | evals | final misfit/initial | eps_r error | sigma error | "
              "eps_r in eps sphere | sigma in sigma sphere [mS/m] | wall time [min] | final fine nodes |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in runs.items():
        fn = int(r["fine_nodes"][-1]) if "fine_nodes" in r.files else -1
        lines.append(f"| {m} | {len(r['misfit']) - 1} | {int(r['evals'][-1])} | "
                     f"{r['misfit'][-1] / r['misfit'][0]:.3f} | {r['eps_err'][-1]:.3f} | "
                     f"{r['sig_err'][-1]:.3f} | {r['eps_in_eps_sphere'][-1]:.2f} | "
                     f"{1e3 * r['sig_in_sig_sphere'][-1]:.1f} | {r['seconds'][-1] / 60:.1f} | {fn} |")
    if ev:
        keys = [str(t) for t in EVAL_MIN]
        lines += ["", "Equal wall time (model held at that time; data error with the fine solver, "
                  "eps_r error, sigma error):", "",
                  "| method | " + " | ".join(f"{k} min" for k in keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for m in runs:
            if m not in ev:
                continue
            cells = []
            for k in keys:
                v = ev[m].get(k)
                cells.append("–" if v is None else
                             f"{100 * v['data_error']:.1f} % / {v['eps_err']:.3f} / {v['sig_err']:.3f}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("observed", "diagnose", "fine", "coarse", "static", "sched",
                                     "evaluate", "report"))
    ap.add_argument("--budget", type=float, default=7200.0)
    ap.add_argument("--balance", type=float, default=0.01)
    ap.add_argument("--plateau", type=float, default=0.02)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--min-cycle", type=int, default=4)
    ap.add_argument("--check-every", type=int, default=3)
    ap.add_argument("--indicator-shots", type=int, nargs="+", default=[0, 4, 8])
    ap.add_argument("--global-threshold", type=float, default=0.5)
    ap.add_argument("--urgent", type=float, default=0.5,
                    help="refine at once when |estimate| >= urgent * misfit (0: off, the "
                         "plateau/min-cycle rules only; the first, stopped sched run)")
    ap.add_argument("--estimate", choices=("absolute", "differential"), default="absolute")
    ap.add_argument("--epochs", type=int, nargs=2, default=None,
                    help="iterations of the two stages (default 8 24)")
    args = ap.parse_args()
    configure()
    _setup_fine()
    F.FLUX = 0.1
    log = log_to(OUT / "log.txt")
    if args.epochs is not None:
        F.STAGES = tuple((n, j, c) for n, (_, j, c) in zip(args.epochs, F.STAGES))
    if args.what == "observed":
        F.observed_data(DEV)
        F.observed_data(DEV, background_model=True)
    elif args.what == "diagnose":
        diagnose(log)
    elif args.what == "fine":
        run_fine(log, args.epochs)
    elif args.what == "evaluate":
        evaluate(log)
    elif args.what == "report":
        report()
    else:
        try:
            run_amr(args.what, args.budget, args, log)
        except Exception as exc:
            log(f"[{args.what}] FAILED: {exc!r}\n{traceback.format_exc()}")
            raise


if __name__ == "__main__":
    main()

"""Equal-budget 3D FWI study: dynamic AMR against static local refinement and uniform FDTD.

Question: at the same compute budget (wall-clock seconds of the inversion on one GPU),
does adjoint-driven dynamic subgridding (``DeepGPR.subgrid.amr``) give better models than
static local refinement and than traditional uniform-grid FDTD, across models,
frequencies and noise levels?

Design (fixed before any result was seen):

* domain 0.75 x 1.2 x 1.2 m (depth x, lateral y, z), background eps_r 4, sigma 2 mS/m;
  four target models (``MODELS``): an off-centre compact pair of spheres, a water pipe and
  an air void far apart, a dipping conductive plate and a boulder, four random inclusions
  (fixed seed);
* acquisition on two planes (x = 0.10 and 0.65 m): 13 shots (9 top, 4 bottom), 8 x 8
  receivers per plane, offsets < 10 cm muted; 7 shots invert, 6 held-out shots at new
  positions measure generalisation;
* observed data from an independent solver: uniform 0.5 cm Yee grid with the model defined
  on that grid (no inverse crime for any inversion solver, including uniform 1 cm);
  differential data (target response with a known baseline survey) plus band-limited
  Gaussian noise at a given SNR (energy of the training-shot response over noise energy);
* centre frequencies 200 / 300 / 400 MHz (Ricker), SNR 20 dB; noise study at 300 MHz with
  30 and 10 dB;
* seven inversion solvers, identical parameterisation (eps_r and sigma on the 1 cm model
  grid in the update region), start model, L-BFGS, TV weight and budget:

  ``fdtd3``          uniform 3 cm FDTD (traditional, the main grid)
  ``fdtd1``          uniform 1 cm FDTD (traditional, fine everywhere)
  ``static_full``    3 cm + one 1 cm subgrid over the whole update region (static, no prior)
  ``static_centre``  3 cm + one 1 cm subgrid at the survey centre with the fine-node count
                     of AMR's final layout (static, no prior, same cost per gradient)
  ``amr``            dynamic AMR: 3 cm for 10 iterations, then every 10 iterations the
                     differential DWR indicator + a priori criteria re-plan the boxes
  ``amr_once``       one-shot adaptive: AMR's first re-plan, then the boxes stay fixed
                     (isolates the value of re-adapting)
  ``oracle``         3 cm + the planner's boxes on the true model (prior knowledge;
                     reference, not a competitor)

* every solver runs until its clock passes the budget ``T``; the clock covers everything
  the method needs (solver construction, CUDA graphs, baseline run, gradients, line
  searches, AMR indicators and re-planning) and pauses only for bookkeeping. The model
  scored at time ``t`` is the last iterate completed by ``t`` (no method is scored on
  work done after the budget);
* scores (pre-registered; primary first): held-out prediction error
  ``PE = ||R(m) - R_true||^2 / ||R_true||^2`` on the six held-out shots (noise-free
  responses ``R``, evaluated with a uniform 1 cm solver against the 0.5 cm data; its own
  floor is reported), the same at 4/3 of the centre frequency (cross-frequency
  generalisation), smoothed model errors and correlations of eps_r and sigma in the update
  region;
* statistics over scenarios: wins, geometric-mean PE ratio AMR / baseline with a
  bootstrap 95 % interval, Wilcoxon signed-rank test; at budget fractions 1/8, 1/4, 1/2, 1
  (so a crossover at small or large budgets is visible); one long run (6 T) shows the
  asymptotic order.

Sub-commands (results in ``examples/subgrid/results/amr_budget/``)::

    python examples/subgrid/amr_budget_study.py list               # scenario names
    python examples/subgrid/amr_budget_study.py data               # reference data, checks
    python examples/subgrid/amr_budget_study.py scenario M1_f300_snr20
    python examples/subgrid/amr_budget_study.py longrun
    python examples/subgrid/amr_budget_study.py verify             # 0.5 cm evaluator check
    python examples/subgrid/amr_budget_study.py report

``examples/subgrid/run_amr_budget_study.sh`` runs everything in order.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fwi_ablation as A  # noqa: E402

from DeepGPR.subgrid import Subgrid, SubgridMaxwell3D, fine_source_samples  # noqa: E402
from DeepGPR.subgrid.adapt import plan_refinement  # noqa: E402
from DeepGPR.subgrid.adjoint import SubgridPropagator  # noqa: E402
from DeepGPR.subgrid.amr import indicator, regrid  # noqa: E402

ROOT = Path(__file__).resolve().parent / "results" / "amr_budget"
OUT = ROOT
DTYPE = torch.float32
H = 0.01
H_REF = 0.005
SHAPE = (76, 121, 121)
EPS0, SIG0 = 4.0, 0.002
TOP, BOTTOM = 0.10, 0.65
ROI = ((0.24, 0.51), (0.24, 0.96), (0.24, 0.96))   # update region = admissible boxes
TV = 1.5e-6
SIG_SCALE = 0.01
EVERY = 10                                           # AMR cycle [iterations]
THETA = 0.5
BUDGET = 150.0
N_FRAMES = 16
PE_FRAMES = (2, 4, 8, 16)                            # budget fractions 1/8, 1/4, 1/2, 1
FREQS = (3e8, 4e8, 2e8)
METHODS = ("fdtd3", "fdtd1", "static_full", "oracle", "amr", "amr_once", "static_centre",
           "amr_fine", "amr_fine_once")
LABELS = {"fdtd3": "FDTD 3 cm", "fdtd1": "FDTD 1 cm", "static_full": "static, whole region",
          "static_centre": "static, centre box", "amr_once": "adaptive once v1",
          "amr": "dynamic AMR v1", "oracle": "oracle boxes (prior)",
          "amr_fine": "dynamic AMR v2 (fine)", "amr_fine_once": "adaptive once v2 (fine)"}
COLORS = {"fdtd3": "#7f7f7f", "fdtd1": "#8c564b", "static_full": "#1f77b4",
          "static_centre": "#17becf", "amr_once": "#ff7f0e", "amr": "#d62728", "oracle": "#2ca02c",
          "amr_fine": "#9467bd", "amr_fine_once": "#e377c2"}
STYLES = {"oracle": "--", "fdtd1": ":", "fdtd3": ":", "amr": "-.", "amr_once": "-."}
# AMR v2 (round 2, "finer partition"): indicator of the main grid alone at the current model
# (defined everywhere, so boxes can shrink, move or vanish), Doerfler marking restricted
# to eta >= GAMMA max(eta) (maximum strategy), no a priori criteria of the current model,
# tighter clustering, hysteresis against re-plans that would barely change the layout.
# Chosen on M1-M4 snapshots of round 1 (probe); M5/M6 are fresh models for validation.
GAMMA = 0.2
EFFICIENCY_V2 = 0.85
HYSTERESIS = 0.9
FDTD1_BUDGET = 1800.0                                # long run of the uniform 1 cm solver


# --------------------------------------------------------------------------- models
def _sphere(c, r, eps=None, sig=None):
    return dict(kind="ellipsoid", c=c, a=(r, r, r), eps=eps, sig=sig)


def random_cluster(seed=7, n=4):
    rng = np.random.default_rng(seed)
    bodies = []
    while len(bodies) < n:
        c = (rng.uniform(0.31, 0.44), rng.uniform(0.34, 0.86), rng.uniform(0.34, 0.86))
        a = tuple(rng.uniform(0.025, 0.045, 3))
        if any(np.linalg.norm(np.subtract(c, b["c"])) < max(a) + max(b["a"]) + 0.03 for b in bodies):
            continue
        eps = rng.uniform(1.5, 2.5) if rng.random() < 0.35 else rng.uniform(8.0, 16.0)
        sig = float(10 ** rng.uniform(-3.0, -1.3))
        bodies.append(dict(kind="ellipsoid", c=tuple(map(float, c)), a=tuple(map(float, a)),
                           eps=float(eps), sig=sig))
    return bodies


MODELS = {
    # compact pair, off the survey centre
    "M1": [_sphere((0.36, 0.40, 0.78), 0.045, eps=16.0, sig=0.01),
           _sphere((0.42, 0.53, 0.80), 0.04, sig=0.1)],
    # water-filled pipe (along z) and an air void far from it
    "M2": [dict(kind="cylinder", c=(0.36, 0.36), r=0.03, span=(0.36, 0.84), eps=20.0, sig=0.05),
           _sphere((0.44, 0.82, 0.48), 0.05, eps=1.0, sig=0.0)],
    # dipping conductive plate and a boulder
    "M3": [dict(kind="plate", c=(0.40, 0.72, 0.70), dip=25.0, half=(0.12, 0.10, 0.02), eps=10.0,
                sig=0.03),
           dict(kind="ellipsoid", c=(0.33, 0.36, 0.38), a=(0.04, 0.06, 0.05), eps=8.0, sig=0.004)],
    # four random inclusions
    "M4": random_cluster(),
    # round 2: fresh random models (not used to choose any setting)
    "M5": random_cluster(seed=11),
    "M6": random_cluster(seed=23),
}
ORIGINAL = ("M1", "M2", "M3", "M4")
FRESH = ("M5", "M6")
MODEL_TITLES = {"M1": "compact sphere pair", "M2": "pipe + air void", "M3": "dipping plate + boulder",
                "M4": "random inclusions", "M5": "random inclusions (fresh)",
                "M6": "random inclusions (fresh)"}


def body_mask(b, X, Y, Z):
    if b["kind"] == "ellipsoid":
        c, a = b["c"], b["a"]
        return ((X - c[0]) / a[0]) ** 2 + ((Y - c[1]) / a[1]) ** 2 + ((Z - c[2]) / a[2]) ** 2 <= 1.0
    if b["kind"] == "cylinder":          # axis along z
        c = b["c"]
        return (((X - c[0]) ** 2 + (Y - c[1]) ** 2) <= b["r"] ** 2) & (Z >= b["span"][0]) & (Z <= b["span"][1])
    if b["kind"] == "plate":             # normal tilted by ``dip`` from x towards y
        d = math.radians(b["dip"])
        c = b["c"]
        px, py, pz = X - c[0], Y - c[1], Z - c[2]
        n = px * math.cos(d) + py * math.sin(d)
        u = -px * math.sin(d) + py * math.cos(d)
        hu, hv, ht = b["half"]
        return (np.abs(n) <= ht) & (np.abs(u) <= hu) & (np.abs(pz) <= hv)
    raise ValueError(b["kind"])


def grid_shape(h):
    return tuple(int(round((n - 1) * H / h)) + 1 for n in SHAPE)


def model_arrays(name, h=H, targets=True):
    shp = grid_shape(h)
    eps = np.full(shp, EPS0)
    sig = np.full(shp, SIG0)
    if targets and name is not None:
        X, Y, Z = np.meshgrid(*[np.arange(n) * h for n in shp], indexing="ij")
        for b in MODELS[name]:
            m = body_mask(b, X, Y, Z)
            if b.get("eps") is not None:
                eps[m] = b["eps"]
            if b.get("sig") is not None:
                sig[m] = b["sig"]
    return eps, sig


def body_masks(name):
    X, Y, Z = np.meshgrid(*[np.arange(n) * H for n in SHAPE], indexing="ij")
    return [body_mask(b, X, Y, Z) for b in MODELS[name]]


def roi_slices():
    return tuple(slice(int(round(lo / H)) + 1, int(round(hi / H))) for lo, hi in ROI)


def roi_mask():
    m = np.zeros(SHAPE, dtype=bool)
    m[roi_slices()] = True
    return m


def embed(crop, bg):
    full = np.full(SHAPE, bg, dtype=np.float32)
    full[roi_slices()] = crop
    return full


# --------------------------------------------------------------------------- acquisition
def acquisition():
    top = [(y, z) for y in (0.3, 0.6, 0.9) for z in (0.3, 0.6, 0.9)]
    bot = [(y, z) for y in (0.45, 0.75) for z in (0.45, 0.75)]
    src = [(TOP, y, z) for y, z in top] + [(BOTTOM, y, z) for y, z in bot]
    g = np.linspace(0.15, 1.05, 8)
    rec = [(x, y, z) for x in (TOP, BOTTOM) for y in g for z in g]
    src = np.array([[s] for s in src])
    rec = np.array([rec] * len(src))
    train = [0, 2, 4, 6, 8, 9, 12]      # top corners + centre, bottom diagonal
    hold = [1, 3, 5, 7, 10, 11]         # top edge mid-points, bottom anti-diagonal
    return src, rec, train, hold


def t_max(f0):
    return 9e-9 + 2.2 / f0


def bench(f0):
    return A.Bench(shape=SHAPE, eps0=EPS0, sig0=SIG0, f0=f0, t_max=t_max(f0), top=TOP,
                   bottom=BOTTOM, roi=ROI, box=ROI, targets=[], tv=TV)


def main_dt():
    return 0.95 / (299792458.0 * math.sqrt(3.0) / (3 * H))


def n_steps(f0):
    return int(round(t_max(f0) / main_dt()))


def ref_wavelet(w):
    """Main-rate Ricker samples ``w`` -> 0.5 cm grid samples at ``dt / 6`` with the main
    grid's half-step source delay: ``J((k + 1/2) dt / 6) = W((2k - 5) / 12)`` (band-limited
    interpolation)."""
    up = fine_source_samples(w[None].double(), 12)[0]
    k = np.arange(6 * w.numel())
    idx = 2 * k - 5
    ok = torch.as_tensor((idx >= 0) & (idx < up.numel()))
    return torch.where(ok, up[np.clip(idx, 0, up.numel() - 1)], torch.zeros((), dtype=up.dtype))


def reference_data(name, f0, shots, src, rec, targets=True, batch=3, nt=None):
    """Uniform 0.5 cm data (receivers at the main ``dt``; ``nt`` samples, default the
    record length of ``f0``), cached on disk."""
    nt = n_steps(f0) if nt is None else int(nt)
    tag = f"{name if targets else 'bg'}_f{int(f0 / 1e6)}_nt{nt}_" + "-".join(map(str, shots))
    path = ROOT / "cache" / f"ref_{tag}.npz"
    if path.exists():
        return torch.as_tensor(np.load(path)["d"])
    path.parent.mkdir(parents=True, exist_ok=True)
    e, s = model_arrays(name, H_REF, targets)
    e, s = torch.as_tensor(e, dtype=DTYPE), torch.as_tensor(s, dtype=DTYPE)
    dt = main_dt()
    w6 = ref_wavelet(A.ricker(nt, dt, f0))
    out = []
    for i in range(0, len(shots), batch):
        sh = shots[i:i + batch]
        sol = SubgridMaxwell3D(e, s, H_REF, coarsen=1, pmlthick=40, dt=dt / 6, dipole_length=3 * H,
                               nshot=len(sh), device="cuda", dtype=DTYPE)
        d = sol.run(w6, src[sh], rec[sh])["receivers"][:, 5::6].cpu()
        out.append(d)
        del sol
        torch.cuda.empty_cache()
    d = torch.cat(out, 0)
    np.savez_compressed(path, d=d.numpy())
    return d


def plan_frequency(f0):
    """``f_max`` of the a priori criteria (oracle boxes, AMR's model criteria): 1.5 f0,
    capped below the frequency at which the main grid resolves the background with 10
    points per wavelength. Above it (400 MHz) the whole domain would be flagged, which the
    planner treats as unrefinable, and the oracle would get no box."""
    f_bg = 299792458.0 / (math.sqrt(EPS0) * 10 * 3 * H)
    return min(1.5 * f0, 0.95 * f_bg)


def band_noise(shape, f0, dt, seed):
    """White Gaussian noise shaped by the Ricker amplitude spectrum (source band)."""
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(shape)
    f = np.fft.rfftfreq(shape[1], dt)
    amp = (f / f0) ** 2 * np.exp(-(f / f0) ** 2)
    spec = np.fft.rfft(n, axis=1) * (amp / amp.max())[None, :, None]
    return np.fft.irfft(spec, n=shape[1], axis=1)


# --------------------------------------------------------------------------- scenarios
def scenarios(fresh=True):
    """Round 1: M1-M4 at 300 / 400 / 200 MHz (20 dB) and 300 MHz at 10 / 30 dB; round 2
    adds the fresh models M5, M6 at 300 / 400 MHz (20 dB)."""
    out = []
    for f0 in FREQS:
        for m in ORIGINAL:
            out.append(f"{m}_f{int(f0 / 1e6)}_snr20")
    for snr in (10, 30):
        for m in ORIGINAL:
            out.append(f"{m}_f300_snr{snr}")
    if fresh:
        out += [f"{m}_f{f}_snr20" for f in (300, 400) for m in FRESH]
    return out


def fdtd1_scenarios():
    """Scenarios of the long uniform 1 cm runs (all 20 dB ones; 300 / 400 MHz first)."""
    sc = [s for s in scenarios() if s.endswith("_snr20")]
    return [s for s in sc if "_f200_" not in s] + [s for s in sc if "_f200_" in s]


def parse(name):
    m, f, s = name.split("_")[:3]
    return m, float(f[1:]) * 1e6, float(s[3:])


class Context:
    """Data, masks and evaluators of one scenario."""

    def __init__(self, name):
        self.name = name
        self.model, self.f0, self.snr = parse(name)
        self.b = bench(self.f0)
        self.src, self.rec, self.train, self.hold = acquisition()
        self.dt, self.nt = main_dt(), n_steps(self.f0)
        self.mk = A.mute(self.src, self.rec, 0.10)
        allsh = list(range(len(self.src)))
        d_true = reference_data(self.model, self.f0, allsh, self.src, self.rec)
        d_bg = reference_data(None, self.f0, allsh, self.src, self.rec, targets=False)
        self.resp = d_true - d_bg                                   # clean target response
        sig_e = float(((self.resp[self.train] * self.mk[self.train]) ** 2).sum())
        seed = zlib.crc32(name.encode()) & 0xFFFFFFFF
        noise = band_noise(tuple(self.resp.shape), self.f0, self.dt, seed)
        noise *= math.sqrt(10 ** (-self.snr / 10) * sig_e /
                           float(((noise[self.train] * self.mk[self.train].numpy()) ** 2).sum()))
        self.obs = (self.resp + torch.as_tensor(noise, dtype=self.resp.dtype))[self.train]
        self.norm = float(((self.obs * self.mk[self.train]) ** 2).sum())
        self.noise_seed = seed
        self.true_e, self.true_s = model_arrays(self.model)
        self.f_xf = 4.0 / 3.0 * self.f0
        self.f_plan = plan_frequency(self.f0)
        # same record length as the centre frequency (the evaluators share ``nt``)
        dx_true = reference_data(self.model, self.f_xf, self.hold, self.src, self.rec, nt=self.nt)
        dx_bg = reference_data(None, self.f_xf, self.hold, self.src, self.rec, targets=False,
                               nt=self.nt)
        self.resp_xf = dx_true - dx_bg
        self.keep = np.concatenate([self.src.reshape(-1, 3), self.rec.reshape(-1, 3)])
        main_n = tuple((n - 1) // 3 + 1 for n in SHAPE)
        self.allowed = np.zeros(main_n, dtype=bool)
        self.allowed[tuple(slice(int(math.ceil(lo / 0.03 - 1e-9)), int(hi / 0.03 + 1e-9) + 1)
                           for lo, hi in ROI)] = True
        self._eval = {}

    def prop(self, kind, boxes=None):
        """Differential data function of the training shots."""
        if kind == "fdtd1":
            p = A.Prop("fine", self.b, self.src, self.rec, self.train, self.dt, self.nt, batch=1)
        elif not boxes:
            p = A.Prop("coarse3", self.b, self.src, self.rec, self.train, self.dt, self.nt)
        else:
            p = A.Prop("sub3", self.b, self.src, self.rec, self.train, self.dt, self.nt,
                       boxes=[tuple(map(tuple, bx)) for bx in boxes])
        return p, A.Diff(p, self.b)

    def oracle_boxes(self):
        plan = plan_refinement(self.true_e, self.true_s, H, f_max=self.f_plan, ppw=10, coarsen=3,
                               keep_points=self.keep)
        return [list(map(list, sg.box)) for sg in plan.subgrids]

    def evaluator(self, xf=False):
        """Uniform 1 cm differential data of the held-out shots (centre or 4/3 frequency)."""
        if xf not in self._eval:
            b = bench(self.f_xf if xf else self.f0)
            p = A.Prop("fine", b, self.src, self.rec, self.hold, self.dt, self.nt, batch=len(self.hold))
            self._eval[xf] = A.Diff(p, b)
        return self._eval[xf]

    def pe(self, eps, sig, xf=False, return_data=False):
        fn = self.evaluator(xf)
        mk = self.mk[self.hold]
        with torch.no_grad():
            d = fn(torch.as_tensor(eps, dtype=DTYPE, device="cuda"),
                   torch.as_tensor(sig, dtype=DTYPE, device="cuda")).cpu() * mk
        obs = (self.resp_xf if xf else self.resp[self.hold]) * mk
        pe = float(((d - obs) ** 2).sum() / (obs ** 2).sum())
        return (pe, d) if return_data else pe


def fine_nodes(boxes):
    """Subgrid nodes of ratio-3 boxes (main-cell extent ``n``: ``3 n + 37`` per axis)."""
    tot = 0
    for bx in boxes:
        cells = [int(round(hi / 0.03)) - int(round(lo / 0.03)) for lo, hi in bx]
        tot += int(np.prod([3 * c + 37 for c in cells]))
    return tot


def centre_box(target_nodes):
    """Box over the whole update depth, centred laterally at the survey centre, whose
    fine-node count is closest to ``target_nodes``."""
    lo_x, hi_x = ROI[0]
    c = 20                                   # 0.60 m
    best = None
    for a in range(1, 13):
        for b_ in range(a, a + 2):
            bx = [[lo_x, hi_x], [(c - a) * 0.03, (c + b_) * 0.03], [(c - a) * 0.03, (c + b_) * 0.03]]
            if bx[1][0] < ROI[1][0] - 1e-9 or bx[1][1] > ROI[1][1] + 1e-9:
                continue
            n = fine_nodes([bx])
            if best is None or abs(n - target_nodes) < abs(best[0] - target_nodes):
                best = (n, bx)
    return [best[1]]


# --------------------------------------------------------------------------- budgeted FWI
class Clock:
    def __init__(self):
        self.acc, self.t0 = 0.0, None

    def start(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def pause(self):
        torch.cuda.synchronize()
        self.acc += time.perf_counter() - self.t0
        self.t0 = None

    def elapsed(self):
        if self.t0 is None:
            return self.acc
        torch.cuda.synchronize()
        return self.acc + time.perf_counter() - self.t0


def budget_fwi(ctx: Context, kind, budget, marks, log, boxes=None, every=EVERY):
    """L-BFGS on (eps, sigma) until the method's clock passes ``budget``.

    Returns the history, the snapshot of each mark (last iterate completed by the mark)
    and cost figures. ``kind`` ``amr`` / ``amr_once`` re-plan the boxes every ``every``
    iterations (``amr_once``: only once)."""
    dynamic = kind in ("amr", "amr_once", "amr_fine", "amr_fine_once")
    v2 = kind.startswith("amr_fine")
    E0 = torch.full(SHAPE, EPS0, dtype=DTYPE, device="cuda")
    S0 = torch.full(SHAPE, SIG0, dtype=DTYPE, device="cuda")
    mask = torch.as_tensor(roi_mask(), device="cuda")
    obs = (ctx.obs * ctx.mk[ctx.train]).cuda()
    mkc = ctx.mk[ctx.train].cuda()
    rs = roi_slices()
    u_e = torch.zeros(SHAPE, dtype=DTYPE, device="cuda", requires_grad=True)
    u_s = torch.zeros(SHAPE, dtype=DTYPE, device="cuda", requires_grad=True)

    def mdl():
        return ((E0 + u_e * mask).clamp(min=1.0), (S0 + SIG_SCALE * u_s * mask).clamp(min=0.0))

    def new_opt():
        return torch.optim.LBFGS([u_e, u_s], lr=1.0, history_size=10, max_iter=1,
                                 line_search_fn="strong_wolfe")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    clock = Clock()
    clock.start()
    layout = [] if dynamic else list(boxes or [])
    prop, fn = ctx.prop(kind, layout)
    state = {"fn": fn, "evals": 0, "misfit": 1.0}
    opt = new_opt()

    def closure():
        opt.zero_grad()
        eps, sig = mdl()
        e_l = eps.detach().requires_grad_(True)
        s_l = sig.detach().requires_grad_(True)
        misfit = 0.0
        for idx, d in A.batched(state["fn"], e_l, s_l):
            part = ((d * mkc[idx] - obs[idx]) ** 2).sum() / ctx.norm
            part.backward()
            misfit += float(part)
        reg = TV * (A.tv_term(u_e, mask) + A.tv_term(u_s, mask))
        torch.autograd.backward([eps, sig, reg], [e_l.grad, s_l.grad, torch.ones_like(reg)])
        state["evals"] += 1
        state["misfit"] = misfit
        return torch.tensor(misfit + float(reg))

    def replan(cycle):
        with torch.no_grad():
            e, s = mdl()
        if v2:            # main grid alone: the indicator is defined everywhere
            cp, cfn = ctx.prop("fdtd3")
            sol, w = cp.props[0][1].solver, cp.w
            with torch.no_grad():
                d = cfn(e, s) * mkc
        else:             # v1: the current (subgridded) solver
            sol, w = prop.props[0][1].solver, prop.w
            with torch.no_grad():
                d = state["fn"](e, s) * mkc
        ip = SubgridPropagator(sol, ctx.src[ctx.train], ctx.rec[ctx.train], indicator=True)
        eta = indicator(ip, e, s, w, (d - obs) * mkc, eps0=E0, sig0=S0)
        del ip
        cur = [Subgrid(bx) for bx in layout]
        if v2:
            cp = cfn = None
            rg = regrid(e.double().cpu().numpy(), s.double().cpu().numpy(), H, 3, eta,
                        allowed=ctx.allowed, theta=THETA, f_max=ctx.f_plan, ppw=10,
                        keep_points=ctx.keep, current=cur, use_model_criteria=False, gamma=GAMMA,
                        persistent=False, hysteresis=HYSTERESIS, efficiency=EFFICIENCY_V2)
        else:
            rg = regrid(e.double().cpu().numpy(), s.double().cpu().numpy(), H, 3, eta,
                        allowed=ctx.allowed, theta=THETA, f_max=ctx.f_plan, ppw=10,
                        keep_points=ctx.keep, current=cur)
        new = [list(map(list, sg.box)) for sg in rg.plan.subgrids]
        extra = (f", covered {rg.info['covered_fraction']:.2f}" if "covered_fraction" in rg.info else "")
        log(f"  [{kind}] re-plan {cycle}: {rg.info['marked_nodes']} marked nodes -> "
            f"{[[tuple(round(v, 2) for v in ax) for ax in bx] for bx in new]} ({fine_nodes(new)} fine "
            f"nodes{extra}){'' if rg.changed else ' (unchanged)'}")
        return new, rg.changed

    snaps, hist = {}, []
    pending = sorted(marks)
    prev = dict(t=0.0, it=0, misfit=1.0, boxes=[], eps=np.full(tuple(s.stop - s.start for s in rs), EPS0,
                                                                np.float32))
    prev["sig"] = np.full_like(prev["eps"], SIG0)
    it, n_replans = 0, 0
    while clock.elapsed() < budget:
        if dynamic and it > 0 and it % every == 0 and (not kind.endswith("_once") or n_replans == 0):
            n_replans += 1
            new, changed = replan(n_replans)
            if changed:
                layout = new
                fn = prop = None
                state["fn"] = None
                torch.cuda.empty_cache()
                prop, fn = ctx.prop(kind, layout)
                state["fn"] = fn
                opt = new_opt()
        n0 = state["evals"]
        opt.step(closure)
        it += 1
        clock.pause()
        t = clock.acc
        with torch.no_grad():
            e, s = mdl()
            cur = dict(t=t, it=it, misfit=state["misfit"], boxes=[list(map(list, bx)) for bx in layout],
                       eps=e[rs].float().cpu().numpy(), sig=s[rs].float().cpu().numpy())
        while pending and pending[0] < t:
            snaps[pending.pop(0)] = prev
        hist.append(dict(t=t, it=it, misfit=state["misfit"], evals=state["evals"] - n0,
                         boxes=cur["boxes"]))
        if it == 1 or it % 5 == 0:
            log(f"  [{kind}] it {it:3d}: t {t:6.1f} s, misfit {state['misfit']:.4f}, "
                f"{len(layout)} box(es), {state['evals']} evals")
        prev = cur
        clock.start()
    clock.pause()
    for m in pending:                    # only marks at the very end of the budget
        snaps[m] = prev
    info = dict(iterations=it, evaluations=state["evals"], seconds=clock.acc,
                seconds_per_evaluation=clock.acc / max(1, state["evals"]),
                peak_GB=torch.cuda.max_memory_allocated() / 2 ** 30, final_boxes=layout,
                final_fine_nodes=fine_nodes(layout), replans=n_replans)
    fn = prop = opt = None
    torch.cuda.empty_cache()
    return hist, snaps, info


# --------------------------------------------------------------------------- metrics
def model_metrics(ctx: Context, eps, sig):
    from scipy.ndimage import binary_dilation, gaussian_filter
    m = roi_mask()
    out = {}
    for nm, x, a, m0 in (("eps", eps, ctx.true_e, EPS0), ("sig", sig, ctx.true_s, SIG0)):
        out[f"{nm}_err"] = float(np.linalg.norm((x - a)[m]) / np.linalg.norm((a - m0)[m]))
        As, Bs = gaussian_filter(a, 3), gaussian_filter(x, 3)
        out[f"{nm}_err_smooth"] = float(np.linalg.norm((Bs - As)[m]) / np.linalg.norm((As - m0)[m]))
        u, v = (x - m0)[m], (a - m0)[m]
        out[f"{nm}_corr"] = float(np.corrcoef(u, v)[0, 1]) if u.std() > 0 else 0.0
    for k, bm in enumerate(body_masks(ctx.model)):
        near = binary_dilation(bm, iterations=2)
        for nm, x, a, m0 in (("eps", eps, ctx.true_e, EPS0), ("sig", sig, ctx.true_s, SIG0)):
            ti = float((a - m0)[near].sum())
            if abs(ti) > 1e-12:
                out[f"T{k}_int_d{nm}"] = float((x - m0)[near].sum()) / ti
    return out


# --------------------------------------------------------------------------- scenario
def log_to(path):
    def log(msg):
        print(msg, flush=True)
        with open(path, "a") as f:
            f.write(msg + "\n")
    return log


def solver_floors(ctx: Context, oracle):
    """True-model response error of each inversion solver on the training shots."""
    mk = ctx.mk[ctx.train]
    te = torch.as_tensor(ctx.true_e, dtype=DTYPE, device="cuda")
    ts = torch.as_tensor(ctx.true_s, dtype=DTYPE, device="cuda")
    resp = ctx.resp[ctx.train] * mk
    fl = {}
    for kind, boxes in (("fdtd3", None), ("fdtd1", None), ("static_full", [list(map(list, ROI))]),
                        ("oracle", oracle)):
        p, fn = ctx.prop(kind, boxes)
        with torch.no_grad():
            d = fn(te, ts).cpu() * mk
        fl[kind] = float(((d - resp) ** 2).sum() / (resp ** 2).sum())
        del p, fn
        torch.cuda.empty_cache()
    fl["oracle_boxes"] = oracle
    fl["evaluator_floor"] = ctx.pe(ctx.true_e, ctx.true_s)
    return fl


def refresh_floors(ctx: Context, oracle, log):
    """Recompute the floors in ``checks.json`` of this scenario's model and frequency when
    they were computed with other oracle boxes (outside every timed run)."""
    path = ROOT / "checks.json"
    if not path.exists():
        return
    chk = json.loads(path.read_text())
    key = f"{int(ctx.f0 / 1e6)}"
    old = chk.get(key, {}).get("floors", {}).get(ctx.model)
    if key not in chk or (old is not None and old.get("oracle_boxes") == oracle):
        return
    fl = solver_floors(ctx, oracle)
    chk[key]["floors"][ctx.model] = fl
    path.write_text(json.dumps(chk, indent=1))
    log(f"floors refreshed ({ctx.model}, {key} MHz, oracle {oracle}): " +
        ", ".join(f"{k} {v:.4f}" for k, v in fl.items() if not isinstance(v, list)))


def warm_up(ctx):
    p, fn = ctx.prop("static_full", [[[0.30, 0.42], [0.42, 0.57], [0.42, 0.57]]])
    e = torch.full(SHAPE, EPS0, dtype=DTYPE, device="cuda", requires_grad=True)
    s = torch.full(SHAPE, SIG0, dtype=DTYPE, device="cuda")
    (fn(e, s) ** 2).sum().backward()
    del p, fn
    torch.cuda.empty_cache()


def run_scenario(name, budget=BUDGET, methods=METHODS, out=None, every=EVERY, figures=True):
    out = Path(out or OUT / name)
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    ctx = Context(name)
    path = out / "result.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res.update(dict(scenario=name, model=ctx.model, f0=ctx.f0, snr_db=ctx.snr, budget=budget,
                    noise_seed=ctx.noise_seed, bodies=MODELS[ctx.model], train=ctx.train,
                    hold=ctx.hold, dt=ctx.dt, nt=ctx.nt, every=every, theta=THETA, tv=TV))
    res.setdefault("methods", {})
    marks = [budget * k / N_FRAMES for k in range(1, N_FRAMES + 1)]
    res["marks"] = marks
    log(f"=== {name}: f0 {ctx.f0 / 1e6:.0f} MHz, SNR {ctx.snr:.0f} dB, nt {ctx.nt}, budget {budget:.0f} s")
    oracle = ctx.oracle_boxes()
    res["oracle_boxes"] = oracle
    res["f_plan"] = ctx.f_plan
    refresh_floors(ctx, oracle, log)
    warm_up(ctx)
    for kind in methods:
        if kind in res["methods"] and "error" not in res["methods"][kind] and \
                (out / f"snap_{kind}.npz").exists():
            log(f"[{kind}] done before, skipped")
            continue
        boxes = None
        if kind == "static_full":
            boxes = [list(map(list, ROI))]
        elif kind == "oracle":
            boxes = oracle
        elif kind == "static_centre":
            amr = res["methods"].get("amr", {})
            if "final_fine_nodes" not in amr:
                log("[static_centre] needs the AMR run; skipped")
                continue
            boxes = centre_box(max(amr["final_fine_nodes"], 1))
        log(f"[{kind}] start; boxes {boxes}")
        try:
            hist, snaps, info = budget_fwi(ctx, kind, budget, marks, log, boxes=boxes, every=every)
        except Exception as exc:              # keep the chain going; the failure is recorded
            log(f"[{kind}] FAILED: {exc!r}\n{traceback.format_exc()}")
            res["methods"][kind] = {"error": repr(exc)}
            path.write_text(json.dumps(res, indent=1))
            torch.cuda.empty_cache()
            continue
        if boxes is not None:
            info["boxes"] = boxes
        res["methods"][kind] = dict(history=hist, **info,
                                    snap_meta=[dict(mark=m, **{k: v for k, v in snaps[m].items()
                                                               if k not in ("eps", "sig")})
                                               for m in marks])
        np.savez_compressed(out / f"snap_{kind}.npz", eps=np.stack([snaps[m]["eps"] for m in marks]),
                            sig=np.stack([snaps[m]["sig"] for m in marks]))
        log(f"[{kind}] {info['iterations']} it, {info['evaluations']} evals, "
            f"{info['seconds_per_evaluation']:.2f} s/eval, {info['peak_GB']:.1f} GB, final boxes "
            f"{info['final_boxes']}")
        path.write_text(json.dumps(res, indent=1))
    evaluate(ctx, res, out, log)
    path.write_text(json.dumps(res, indent=1))
    if figures:
        try:
            scenario_figures(ctx, res, out)
        except Exception as exc:
            log(f"figures FAILED: {exc!r}\n{traceback.format_exc()}")
    return res


def evaluate(ctx: Context, res, out, log):
    """PE at the budget fractions of ``PE_FRAMES``, cross-frequency PE and model metrics at
    the final budget, held-out data of the first held-out shot for the figures."""
    rs = roi_slices()
    if "evaluator_floor" not in res:
        res["evaluator_floor"] = ctx.pe(ctx.true_e, ctx.true_s)
        res["evaluator_floor_xf"] = ctx.pe(ctx.true_e, ctx.true_s, xf=True)
        log(f"evaluator floor (true model, 1 cm vs 0.5 cm): PE {res['evaluator_floor']:.4f}, "
            f"cross-frequency {res['evaluator_floor_xf']:.4f}")
    preds = {}
    for kind, r in res["methods"].items():
        if "error" in r or not (out / f"snap_{kind}.npz").exists():
            continue
        if "pe" in r and "pe_xf" in r and (out / "heldout_pred.npz").exists():
            continue
        z = np.load(out / f"snap_{kind}.npz")
        r["pe"] = {}
        last = None
        for k in PE_FRAMES:
            ej, sj = z["eps"][k - 1], z["sig"][k - 1]
            if np.all(ej == np.float32(EPS0)) and np.all(np.abs(sj - SIG0) < 1e-8):
                r["pe"][str(k)] = 1.0            # the start model predicts no response
                if k == N_FRAMES:
                    preds[kind] = np.zeros_like((ctx.resp[ctx.hold[0]]).numpy())
                continue
            if last is not None and k != N_FRAMES and np.array_equal(ej, last[0]) and \
                    np.array_equal(sj, last[1]):
                r["pe"][str(k)] = last[2]        # no iterate completed in between
                continue
            e, s = embed(ej, EPS0), embed(sj, SIG0)
            if k == N_FRAMES:
                r["pe"][str(k)], d = ctx.pe(e, s, return_data=True)
                preds[kind] = d[0].numpy()
            else:
                r["pe"][str(k)] = ctx.pe(e, s)
            last = (ej, sj, r["pe"][str(k)])
        e, s = embed(z["eps"][-1], EPS0), embed(z["sig"][-1], SIG0)
        r["pe_xf"] = ctx.pe(e, s, xf=True)
        r["metrics"] = model_metrics(ctx, e.astype(np.float64), s.astype(np.float64))
        log(f"[{kind}] PE " + ", ".join(f"{float(k) / N_FRAMES:.3g}T: {r['pe'][str(k)]:.4f}"
                                        for k in PE_FRAMES)
            + f"; PE at 4/3 f0 {r['pe_xf']:.4f}; eps err (smooth) {r['metrics']['eps_err_smooth']:.3f}, "
              f"sig err (smooth) {r['metrics']['sig_err_smooth']:.3f}")
    if preds:
        old = dict(np.load(out / "heldout_pred.npz")) if (out / "heldout_pred.npz").exists() else {}
        old.update(preds)
        old["true"] = (ctx.resp[ctx.hold[0]] * ctx.mk[ctx.hold[0]]).numpy()
        np.savez_compressed(out / "heldout_pred.npz", **old)
    del rs


# --------------------------------------------------------------------------- figures
def display_slices(ctx: Context):
    a = np.abs(ctx.true_e - EPS0) / EPS0 + np.abs(ctx.true_s - SIG0) / 0.02
    ix = int(np.argmax(a.sum((1, 2))))
    iz = int(np.argmax(a.sum((0, 1))))
    return ix, iz


def _norms(ctx: Context):
    from matplotlib.colors import TwoSlopeNorm
    out = []
    for a, m0 in ((ctx.true_e, EPS0), (1e3 * ctx.true_s, 1e3 * SIG0)):
        lo, hi = float(a.min()), float(a.max())
        span = max(hi - m0, m0 - lo, 1e-9)
        lo = min(lo, m0 - 0.25 * span)
        hi = max(hi, m0 + 0.25 * span)
        out.append(TwoSlopeNorm(vmin=lo, vcenter=m0, vmax=hi))
    return out


def _draw_boxes(ax, boxes, view, pos):
    """Box outlines on a slice: ``view`` "h" (y-z plane at depth ``pos``) or "v" (x-y plane
    at z = ``pos``)."""
    for bx in boxes or []:
        (x0, x1), (y0, y1), (z0, z1) = bx
        if view == "h" and x0 - 1e-9 <= pos <= x1 + 1e-9:
            ax.plot([z0, z1, z1, z0, z0], [y0, y0, y1, y1, y0], color="k", ls="--", lw=0.9)
        if view == "v" and z0 - 1e-9 <= pos <= z1 + 1e-9:
            ax.plot([y0, y1, y1, y0, y0], [x0, x0, x1, x1, x0], color="k", ls="--", lw=0.9)


def _panel_grid(ctx, columns, title, ix, iz):
    """Figure with rows eps(h), sigma(h), eps(v), sigma(v) and ``columns`` (name, eps,
    sig, boxes, note)."""
    import matplotlib.pyplot as plt
    ne, ns = _norms(ctx)
    n = len(columns)
    fig, ax = plt.subplots(4, n, figsize=(1.75 * n + 0.9, 7.4), constrained_layout=True)
    ext_h = [0, SHAPE[2] * H, SHAPE[1] * H, 0]
    ext_v = [0, SHAPE[1] * H, SHAPE[0] * H, 0]
    ims = []
    for j, (nm, e, s, boxes, note) in enumerate(columns):
        ims.append(ax[0, j].imshow(e[ix], norm=ne, cmap="RdBu_r", extent=ext_h))
        ims.append(ax[1, j].imshow(1e3 * s[ix], norm=ns, cmap="PuOr_r", extent=ext_h))
        ax[2, j].imshow(e[:, :, iz], norm=ne, cmap="RdBu_r", extent=ext_v)
        ax[3, j].imshow(1e3 * s[:, :, iz], norm=ns, cmap="PuOr_r", extent=ext_v)
        for r in range(4):
            _draw_boxes(ax[r, j], boxes, "h" if r < 2 else "v", ix * H if r < 2 else iz * H)
            ax[r, j].set_xticks([])
            ax[r, j].set_yticks([])
        ax[0, j].set_title(nm + (f"\n{note}" if note else ""), fontsize=8)
        for r in range(4):
            ax[r, j].set_xlim(0.15, 1.05)
            ax[r, j].set_ylim((1.05, 0.15) if r < 2 else (0.60, 0.15))
    ax[0, 0].set_ylabel(f"eps_r, x = {ix * H:.2f} m", fontsize=8)
    ax[1, 0].set_ylabel(f"sigma [mS/m], x = {ix * H:.2f} m", fontsize=8)
    ax[2, 0].set_ylabel(f"eps_r, z = {iz * H:.2f} m", fontsize=8)
    ax[3, 0].set_ylabel(f"sigma, z = {iz * H:.2f} m", fontsize=8)
    fig.colorbar(ims[0], ax=ax[0, :], shrink=0.8, pad=0.01)
    fig.colorbar(ims[1], ax=ax[1, :], shrink=0.8, pad=0.01)
    fig.colorbar(ims[0], ax=ax[2, :], shrink=0.8, pad=0.01)
    fig.colorbar(ims[1], ax=ax[3, :], shrink=0.8, pad=0.01)
    fig.suptitle(title, fontsize=10)
    return fig


def _fig_to_image(fig, dpi):
    from PIL import Image
    fig.set_dpi(dpi)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = Image.frombuffer("RGBA", (w, h), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
    return img


def save_gif(frames, path, ms):
    frames = [f.quantize(colors=200, method=2) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)


def scenario_figures(ctx: Context, res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = [k for k in METHODS if k in res["methods"] and "error" not in res["methods"][k]
             and (out / f"snap_{k}.npz").exists()]
    snaps = {k: np.load(out / f"snap_{k}.npz") for k in kinds}
    ix, iz = display_slices(ctx)
    marks = res["marks"]
    head = f"{ctx.model} ({MODEL_TITLES[ctx.model]}), {ctx.f0 / 1e6:.0f} MHz, SNR {ctx.snr:.0f} dB"
    # 1) evolution at equal compute time
    frames = []
    for j in [-1] + list(range(len(marks))):
        cols = [("true model", ctx.true_e, ctx.true_s, None, "")]
        for k in kinds:
            r = res["methods"][k]
            if j < 0:
                e = np.full(SHAPE, EPS0, np.float32)
                s = np.full(SHAPE, SIG0, np.float32)
                boxes, note = [], "start"
            else:
                e, s = embed(snaps[k]["eps"][j], EPS0), embed(snaps[k]["sig"][j], SIG0)
                meta = r["snap_meta"][j]
                boxes = meta["boxes"] if k in ("amr", "amr_once") else r.get("boxes")
                pe = r.get("pe", {}).get(str(j + 1))
                note = f"it {meta['it']}, misfit {meta['misfit']:.3f}" + (f"\nPE {pe:.3f}" if pe is not None else "")
            cols.append((LABELS[k], e, s, boxes, note))
        t = 0.0 if j < 0 else marks[j]
        fig = _panel_grid(ctx, cols, f"{head} | compute time {t:5.1f} s of {res['budget']:.0f} s "
                                     "(same for every method)", ix, iz)
        frames.append(_fig_to_image(fig, 72))
        plt.close(fig)
    frames += [frames[-1]] * 4
    save_gif(frames, out / "evolution.gif", 700)
    frames[-1].save(out / "final_models.png")
    # 2) depth sweep of the final (equal-budget) models
    finals = [("true model", ctx.true_e, ctx.true_s)] + [
        (LABELS[k], embed(snaps[k]["eps"][-1], EPS0), embed(snaps[k]["sig"][-1], SIG0)) for k in kinds]
    ne, ns = _norms(ctx)
    rs = roi_slices()
    frames = []
    ext_h = [0, SHAPE[2] * H, SHAPE[1] * H, 0]
    for i in range(rs[0].start, rs[0].stop):
        n = len(finals)
        fig, ax = plt.subplots(2, n, figsize=(1.75 * n + 0.9, 3.9), constrained_layout=True)
        for c, (nm, e, s) in enumerate(finals):
            a = ax[0, c].imshow(e[i], norm=ne, cmap="RdBu_r", extent=ext_h)
            b = ax[1, c].imshow(1e3 * s[i], norm=ns, cmap="PuOr_r", extent=ext_h)
            ax[0, c].set_title(nm, fontsize=8)
            for r in range(2):
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                ax[r, c].set_xlim(0.15, 1.05)
                ax[r, c].set_ylim(1.05, 0.15)
        ax[0, 0].set_ylabel("eps_r", fontsize=8)
        ax[1, 0].set_ylabel("sigma [mS/m]", fontsize=8)
        fig.colorbar(a, ax=ax[0, :], shrink=0.8, pad=0.01)
        fig.colorbar(b, ax=ax[1, :], shrink=0.8, pad=0.01)
        fig.suptitle(f"{head} | final models after {res['budget']:.0f} s | depth x = {i * H:.2f} m",
                     fontsize=10)
        frames.append(_fig_to_image(fig, 72))
        plt.close(fig)
    save_gif(frames + frames[::-1], out / "depth_sweep.gif", 160)
    # 3) curves: PE and training misfit against compute time
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for k in kinds:
        r = res["methods"][k]
        tt = [h_["t"] for h_ in r["history"]]
        ax[0].semilogy(tt, [h_["misfit"] for h_ in r["history"]], color=COLORS[k],
                       ls=STYLES.get(k, "-"), label=LABELS[k])
        if "pe" in r:
            ks = sorted(r["pe"], key=float)
            ax[1].loglog([marks[int(q) - 1] for q in ks], [r["pe"][q] for q in ks], "o",
                         color=COLORS[k], ls=STYLES.get(k, "-"), label=LABELS[k], ms=5)
    ax[1].axhline(res["evaluator_floor"], color="0.5", lw=0.8)
    from matplotlib.ticker import NullFormatter, NullLocator
    tk = [marks[k - 1] for k in PE_FRAMES]
    ax[1].xaxis.set_minor_locator(NullLocator())
    ax[1].xaxis.set_minor_formatter(NullFormatter())
    ax[1].set_xticks(tk, [f"{v:.0f}" if v >= 10 else f"{v:.1f}" for v in tk])
    ax[1].text(marks[0], res["evaluator_floor"] * 1.05, "evaluator floor", fontsize=7, color="0.4")
    ax[0].set_xlabel("compute time [s]")
    ax[0].set_ylabel("training misfit (noisy data)")
    ax[1].set_xlabel("compute budget [s]")
    ax[1].set_ylabel("held-out prediction error")
    ax[0].legend(fontsize=7)
    x = np.arange(len(kinds))
    ax[2].bar(x - 0.2, [res["methods"][k].get("pe", {}).get(str(N_FRAMES), np.nan) for k in kinds], 0.4,
              color=[COLORS[k] for k in kinds], label="held-out PE")
    ax[2].bar(x + 0.2, [res["methods"][k].get("pe_xf", np.nan) for k in kinds], 0.4,
              color=[COLORS[k] for k in kinds], alpha=0.45, hatch="//", label="PE at 4/3 f0")
    ax[2].set_xticks(x, [LABELS[k] for k in kinds], rotation=25, fontsize=8)
    ax[2].set_yscale("log")
    ax[2].legend(fontsize=7)
    ax[2].set_title(f"at the budget ({res['budget']:.0f} s)")
    fig.suptitle(head)
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)
    # 4) held-out shot: true response and residuals
    hp = out / "heldout_pred.npz"
    if hp.exists():
        z = np.load(hp)
        names = [k for k in kinds if k in z.files]
        true = z["true"]
        v = float(np.percentile(np.abs(true), 99.5))
        fig, ax = plt.subplots(1, len(names) + 1, figsize=(2.2 * (len(names) + 1), 4.2), sharey=True,
                               constrained_layout=True)
        tt = np.arange(true.shape[0]) * ctx.dt * 1e9
        ext = [0, true.shape[1], tt[-1], 0]
        ax[0].imshow(true, aspect="auto", cmap="gray", vmin=-v, vmax=v, extent=ext)
        ax[0].set_title("true response\n(held-out shot)", fontsize=8)
        ax[0].set_ylabel("time [ns]")
        for c, k in enumerate(names, start=1):
            rr = z[k] - true
            ax[c].imshow(rr, aspect="auto", cmap="gray", vmin=-v, vmax=v, extent=ext)
            ax[c].set_title(f"{LABELS[k]}\nresidual, rel. {np.sum(rr ** 2) / np.sum(true ** 2):.3f}",
                            fontsize=8)
            ax[c].set_xlabel("receiver")
        fig.suptitle(f"{head}: held-out shot {ctx.hold[0]}, residuals on the same scale")
        fig.savefig(out / "heldout_residuals.png", dpi=110)
        plt.close(fig)


# --------------------------------------------------------------------------- commands
def cmd_data(args):
    """All reference data, the alignment check of the 0.5 cm reference against the 1 cm
    solver, and the modelling floor of every solver on the true models."""
    ROOT.mkdir(parents=True, exist_ok=True)
    log = log_to(ROOT / "data.log")
    src, rec, train, hold = acquisition()
    allsh = list(range(len(src)))
    dt = main_dt()
    for name in scenarios():
        m, f0, _ = parse(name)
        for xf in (False, True):
            f = 4 / 3 * f0 if xf else f0
            sh = hold if xf else allsh
            t0 = time.perf_counter()
            reference_data(m, f, sh, src, rec, nt=n_steps(f0))
            reference_data(None, f, sh, src, rec, targets=False, nt=n_steps(f0))
            log(f"reference data {m} {f / 1e6:.0f} MHz ({len(sh)} shots): {time.perf_counter() - t0:.0f} s")
    chk_path = ROOT / "checks.json"
    chk = json.loads(chk_path.read_text()) if chk_path.exists() else {}
    mk = A.mute(src, rec, 0.10)
    for f0 in FREQS:
        key = f"{int(f0 / 1e6)}"
        if key in chk:
            continue
        b = bench(f0)
        nt = n_steps(f0)
        ref_bg = reference_data(None, f0, allsh, src, rec, targets=False) * mk
        p = A.Prop("fine", b, src, rec, train, dt, nt, batch=len(train))
        with torch.no_grad():
            d1 = p(torch.full(SHAPE, EPS0, dtype=DTYPE, device="cuda"),
                   torch.full(SHAPE, SIG0, dtype=DTYPE, device="cuda")).cpu() * mk[train]
        r = ref_bg[train]
        row = {"bg_rel_diff_1cm_vs_05cm": float(((d1 - r) ** 2).sum() / (r ** 2).sum())}
        for sh_ in (-1, 1):            # one main sample earlier / later: alignment check
            rr = torch.roll(r, sh_, dims=1)
            row[f"bg_rel_diff_shift{sh_:+d}"] = float(((d1 - rr) ** 2).sum() / (r ** 2).sum())
        del p
        torch.cuda.empty_cache()
        row["floors"] = {}
        for m in MODELS:
            ctx = Context(f"{m}_f{key}_snr20")
            fl = solver_floors(ctx, ctx.oracle_boxes())
            row["floors"][m] = fl
            log(f"{m} {key} MHz true-model response error: " +
                ", ".join(f"{k} {v:.4f}" for k, v in fl.items() if not isinstance(v, list)))
        chk[key] = row
        log(f"{key} MHz: background 1 cm vs 0.5 cm {row['bg_rel_diff_1cm_vs_05cm']:.4f} "
            f"(shifted by -1 / +1 sample: {row['bg_rel_diff_shift-1']:.3f} / {row['bg_rel_diff_shift+1']:.3f})")
        chk_path.write_text(json.dumps(chk, indent=1))


def pe_series(ctx: Context, z, frames):
    """PE of the snapshots ``frames`` (1-based marks), identical consecutive models
    evaluated once, the start model without a simulation (PE 1)."""
    out, last = {}, None
    for k in frames:
        ej, sj = z["eps"][k - 1], z["sig"][k - 1]
        if np.all(ej == np.float32(EPS0)) and np.all(np.abs(sj - SIG0) < 1e-8):
            out[str(k)] = 1.0
            continue
        if last is not None and np.array_equal(ej, last[0]) and np.array_equal(sj, last[1]):
            out[str(k)] = last[2]
            continue
        out[str(k)] = ctx.pe(embed(ej, EPS0), embed(sj, SIG0))
        last = (ej, sj, out[str(k)])
    return out


def time_to_reach(times, pes, target):
    """First time at which a PE curve (marks ``times``) reaches ``target`` (log-linear
    interpolation between marks; None if never)."""
    prev = (0.0, 1.0)
    for t, v in zip(times, pes):
        if v <= target:
            if prev[1] <= target:
                return prev[0]
            a = (math.log(prev[1]) - math.log(target)) / max(math.log(prev[1]) - math.log(v), 1e-12)
            return prev[0] + a * (t - prev[0])
        prev = (t, v)
    return None


def cmd_fdtd1long(name, budget=FDTD1_BUDGET):
    """Uniform 1 cm FDTD run long enough to converge (``budget`` seconds), PE at every
    mark, and the time it needs to reach the accuracy the other methods reach at ``T``."""
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    ctx = Context(name)
    path = out / "fdtd1_long.json"
    marks = [budget * k / N_FRAMES for k in range(1, N_FRAMES + 1)]
    if path.exists() and (out / "snap_fdtd1_long.npz").exists():
        r = json.loads(path.read_text())
    else:
        log(f"=== {name}: uniform 1 cm FDTD, long run {budget:.0f} s")
        warm_up(ctx)
        hist, snaps, info = budget_fwi(ctx, "fdtd1", budget, marks, log)
        np.savez_compressed(out / "snap_fdtd1_long.npz", eps=np.stack([snaps[m]["eps"] for m in marks]),
                            sig=np.stack([snaps[m]["sig"] for m in marks]))
        r = dict(budget=budget, marks=marks, history=hist, **info,
                 snap_meta=[dict(mark=m, **{k: v for k, v in snaps[m].items() if k not in ("eps", "sig")})
                            for m in marks])
        path.write_text(json.dumps(r, indent=1))
    if "pe" not in r:
        z = np.load(out / "snap_fdtd1_long.npz")
        r["pe"] = pe_series(ctx, z, range(1, N_FRAMES + 1))
        e, s = embed(z["eps"][-1], EPS0), embed(z["sig"][-1], SIG0)
        r["pe_xf"] = ctx.pe(e, s, xf=True)
        r["metrics"] = model_metrics(ctx, e.astype(np.float64), s.astype(np.float64))
        path.write_text(json.dumps(r, indent=1))
    res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
    tt = [r["marks"][int(k) - 1] for k in sorted(r["pe"], key=int)]
    pp = [r["pe"][k] for k in sorted(r["pe"], key=int)]
    r["time_to_reach"] = {}
    for k, v in res.get("methods", {}).items():
        target = v.get("pe", {}).get(str(N_FRAMES))
        if target is not None and k != "fdtd1":
            r["time_to_reach"][k] = dict(target=target, seconds=time_to_reach(tt, pp, target))
    path.write_text(json.dumps(r, indent=1))
    log(f"[fdtd1 long] {r['iterations']} it, {r['evaluations']} gradients, {r['seconds_per_evaluation']:.1f} "
        f"s/gradient; PE " + ", ".join(f"{t:.0f} s: {v:.4f}" for t, v in zip(tt, pp) if t in
                                       (marks[1], marks[3], marks[7], marks[15]))
        + "; time to reach the PE others reach at T = " + ", ".join(
            f"{k} {'>' + str(int(budget)) if x['seconds'] is None else int(round(x['seconds']))} s"
            for k, x in r["time_to_reach"].items()))
    try:
        fdtd1_figures(ctx, res, r, out)
    except Exception as exc:
        log(f"fdtd1 figures FAILED: {exc!r}\n{traceback.format_exc()}")


def fdtd1_figures(ctx: Context, res, r, out):
    """``fdtd1_timeline.gif``: the 1 cm model against time next to the other methods'
    models at the main budget; ``fdtd1_vs_time.png``: PE against time of every method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(out / "snap_fdtd1_long.npz")
    ix, iz = display_slices(ctx)
    head = f"{ctx.model} ({MODEL_TITLES[ctx.model]}), {ctx.f0 / 1e6:.0f} MHz, SNR {ctx.snr:.0f} dB"
    refs = [k for k in ("fdtd3", "static_full", "amr_fine_once", "amr_fine") if k in res.get("methods", {})
            and (out / f"snap_{k}.npz").exists()]
    ref_cols = []
    for k in refs:
        zz = np.load(out / f"snap_{k}.npz")
        m = res["methods"][k]
        pe = m.get("pe", {}).get(str(N_FRAMES))
        boxes = m.get("final_boxes") or m.get("boxes")
        ref_cols.append((f"{LABELS[k]}\n(at T = {res['budget']:.0f} s)", embed(zz["eps"][-1], EPS0),
                         embed(zz["sig"][-1], SIG0), boxes, "" if pe is None else f"PE {pe:.3f}"))
    frames = []
    for j in [-1] + list(range(N_FRAMES)):
        if j < 0:
            e, s, note, t = np.full(SHAPE, EPS0, np.float32), np.full(SHAPE, SIG0, np.float32), "start", 0.0
        else:
            e, s = embed(z["eps"][j], EPS0), embed(z["sig"][j], SIG0)
            meta = r["snap_meta"][j]
            t = r["marks"][j]
            note = f"it {meta['it']}, PE {r['pe'][str(j + 1)]:.3f}"
        cols = [("true model", ctx.true_e, ctx.true_s, None, ""),
                (f"FDTD 1 cm\n(at t = {t:.0f} s)", e, s, None, note)] + ref_cols
        fig = _panel_grid(ctx, cols, f"{head}\nuniform 1 cm FDTD after {t:5.0f} s "
                                     f"({r['seconds_per_evaluation']:.0f} s / gradient); the others after "
                                     f"T = {res.get('budget', BUDGET):.0f} s", ix, iz)
        frames.append(_fig_to_image(fig, 72))
        plt.close(fig)
    save_gif(frames + [frames[-1]] * 4, out / "fdtd1_timeline.gif", 700)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    tt = [r["marks"][int(k) - 1] for k in sorted(r["pe"], key=int)]
    ax.loglog(tt, [r["pe"][k] for k in sorted(r["pe"], key=int)], marker="o", color=COLORS["fdtd1"],
              ls=":", label=f"{LABELS['fdtd1']} (long run)")
    for k in METHODS:
        m = res.get("methods", {}).get(k, {})
        if "pe" not in m or k == "fdtd1":
            continue
        ks = sorted(m["pe"], key=int)
        ax.loglog([res["marks"][int(q) - 1] for q in ks], [m["pe"][q] for q in ks], marker="o", ms=4,
                  color=COLORS[k], ls=STYLES.get(k, "-"), label=LABELS[k])
        x = r.get("time_to_reach", {}).get(k, {}).get("seconds")
        if x is not None and k in ("amr_fine", "amr_fine_once", "fdtd3", "static_full"):
            ax.plot([x], [m["pe"][str(N_FRAMES)]], marker="x", color=COLORS[k], ms=9, mew=2)
    ax.axvline(res.get("budget", BUDGET), color="0.3", lw=0.8, ls=":")
    ax.axhline(res.get("evaluator_floor", 0) or 1e-4, color="0.6", lw=0.7)
    ax.set_xlabel("compute time [s]")
    ax.set_ylabel("held-out PE")
    ax.set_title(f"{head}: x = time at which FDTD 1 cm reaches that method's PE at T", fontsize=9)
    ax.legend(fontsize=7)
    fig.savefig(out / "fdtd1_vs_time.png", dpi=120)
    plt.close(fig)


def cmd_longrun(args):
    name = args.name or scenarios()[0]
    out = ROOT / f"longrun_{name}"
    run_scenario(name, budget=args.budget or 6 * BUDGET,
                 methods=("fdtd3", "fdtd1", "static_full", "oracle", "amr"), out=out)


def cmd_verify(args):
    """Final-budget PE of the 300 MHz / 20 dB scenarios with the 0.5 cm solver (the
    evaluator of the main runs is uniform 1 cm)."""
    log = log_to(ROOT / "verify.log")
    path = ROOT / "verify.json"
    ver = json.loads(path.read_text()) if path.exists() else {}
    for name in [s for s in scenarios() if s.endswith("_f300_snr20")]:
        rp = ROOT / name / "result.json"
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        ctx = Context(name)
        ver.setdefault(name, {})
        mk = ctx.mk[ctx.hold]
        obs = ctx.resp[ctx.hold] * mk
        if "true" not in ver[name]:
            d = refsim(*model_arrays(ctx.model, H_REF), ctx) * mk
            ver[name]["true"] = {"pe_05cm": float(((d - obs) ** 2).sum() / (obs ** 2).sum()),
                                 "pe_1cm": res.get("evaluator_floor")}
        for kind, r in res["methods"].items():
            if "error" in r or kind in ver[name] or not (ROOT / name / f"snap_{kind}.npz").exists():
                continue
            z = np.load(ROOT / name / f"snap_{kind}.npz")
            e = embed(z["eps"][-1], EPS0)
            s = embed(z["sig"][-1], SIG0)
            e5, s5 = to_ref_grid(e), to_ref_grid(s)
            d = refsim(e5, s5, ctx) * mk
            pe = float(((d - obs) ** 2).sum() / (obs ** 2).sum())
            ver[name][kind] = {"pe_05cm": pe, "pe_1cm": r.get("pe", {}).get(str(N_FRAMES))}
            log(f"{name} {kind}: PE 0.5 cm {pe:.4f}, 1 cm {ver[name][kind]['pe_1cm']}")
            path.write_text(json.dumps(ver, indent=1))


def to_ref_grid(a):
    """1 cm model -> 0.5 cm grid (trilinear; exact at the common nodes)."""
    t = torch.as_tensor(np.asarray(a, dtype=np.float32))[None, None]
    return torch.nn.functional.interpolate(t, size=grid_shape(H_REF), mode="trilinear",
                                           align_corners=True)[0, 0].numpy()


def refsim(e5, s5, ctx: Context, batch=3):
    """Differential held-out data of a 0.5 cm model."""
    dt, nt = ctx.dt, ctx.nt
    w6 = ref_wavelet(A.ricker(nt, dt, ctx.f0))
    bg = reference_data(None, ctx.f0, list(range(len(ctx.src))), ctx.src, ctx.rec, targets=False)
    out = []
    for i in range(0, len(ctx.hold), batch):
        sh = ctx.hold[i:i + batch]
        sol = SubgridMaxwell3D(torch.as_tensor(e5, dtype=DTYPE), torch.as_tensor(s5, dtype=DTYPE), H_REF,
                               coarsen=1, pmlthick=40, dt=dt / 6, dipole_length=3 * H, nshot=len(sh),
                               device="cuda", dtype=DTYPE)
        out.append(sol.run(w6, ctx.src[sh], ctx.rec[sh])["receivers"][:, 5::6].cpu())
        del sol
        torch.cuda.empty_cache()
    return torch.cat(out, 0) - bg[ctx.hold]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("list", "list_fdtd1", "data", "scenario", "fdtd1long", "longrun",
                                   "verify", "report", "smoke"))
    p.add_argument("name", nargs="?")
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--methods", default=None)
    args = p.parse_args()
    torch.backends.cudnn.benchmark = False
    if args.cmd == "list":
        print("\n".join(scenarios()))
    elif args.cmd == "list_fdtd1":
        print("\n".join(fdtd1_scenarios()))
    elif args.cmd == "fdtd1long":
        cmd_fdtd1long(args.name, budget=args.budget or FDTD1_BUDGET)
    elif args.cmd == "data":
        cmd_data(args)
    elif args.cmd == "scenario":
        run_scenario(args.name, budget=args.budget or BUDGET,
                     methods=tuple(args.methods.split(",")) if args.methods else METHODS)
    elif args.cmd == "smoke":
        # every code path on a short budget (results in results/amr_budget/smoke)
        run_scenario(args.name or "M1_f300_snr20", budget=args.budget or 25.0,
                     methods=tuple(args.methods.split(",")) if args.methods else METHODS,
                     out=ROOT / "smoke", every=3)
    elif args.cmd == "longrun":
        cmd_longrun(args)
    elif args.cmd == "verify":
        cmd_verify(args)
    elif args.cmd == "report":
        import amr_budget_report
        amr_budget_report.report(ROOT)


if __name__ == "__main__":
    sys.exit(main())

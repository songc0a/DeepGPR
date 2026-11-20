"""Sand-pit-like 3D benchmark: adaptive Huygens subgridding against uniform grids.

Question: does error-driven adaptive refinement reach the accuracy of a uniform fine grid in
less time, or a better accuracy than uniform / static grids in the same time, for a GPR
problem like the indoor sand pit (MALA 500 MHz, metal pipes, voids, pit floor)?

Design (fixed before the results):

* model grid h = 0.75 cm, frame (depth, x along the profile, y across): 0.675 x 1.2 x 0.74 m;
  air above the surface (0.0675 m), sand eps_r 4.6 / 1 mS/m, pit floor below 0.48 m
  (eps_r 7, 10 mS/m); targets: metal pipes (PEC, along y), air voids, water-filled pipes,
  dielectric bodies; models M1, M2 (designed), M3, M4 (random; M4 is validation only);
* surface acquisition 1.5 cm above the sand: 15 transmitters (5 x 3), 44 receivers
  (11 x 4), horizontal dipoles along y (like the MALA antennas), Ricker 450 MHz; 9 shots
  invert, 6 held-out shots at new positions;
* observed data from an independent uniform 0.375 cm grid; differential data (known
  background: air, sand, floor) plus Ricker-band noise, SNR 20 dB;
* methods (identical parameterisation on the 0.75 cm model grid, L-BFGS, TV, start =
  background):
    coarse        uniform 2.25 cm (main grid)
    fine          uniform 0.75 cm (the accuracy limit; expensive)
    static_full   2.25 cm + one 0.75 cm box over the whole update region
    oracle        2.25 cm + boxes on the true targets (prior knowledge, reference)
    amr_once_v1   one re-plan after 10 coarse iterations, Doerfler marking (round-1 method)
    amr_once_v3   one re-plan, v3 marking: every local peak of the main-grid DWR indicator
                  gets its own region (eta >= LOCAL x its peak), fine-node budget
    amr_dyn_v3    v3 marking, re-planned every 10 iterations (hysteresis 0.9)
* supplementary runs (after the first round): coarse_long = uniform 2.25 cm to 3600 s (its
  converged accuracy); amr_dyn_v3_long = amr_dyn_v3 continued from its 2400 s state to
  10800 s (same layout, fresh L-BFGS history; the restart is on the clock);
* every method runs on its own wall clock (setup, baseline, gradients, line searches,
  indicators, re-plans) to its budget; the model scored at time t is the last iterate
  completed by t; held-out PE (noise-free responses) with a uniform 0.75 cm evaluator at
  fixed time marks -> PE(t) curves, equal-time and equal-accuracy (time-to-reach) tables.

    python examples/subgrid/amr_sandpit3d.py data M1
    python examples/subgrid/amr_sandpit3d.py run M1 [--methods ... --smoke]
    python examples/subgrid/amr_sandpit3d.py report
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

from DeepGPR.subgrid import Subgrid, SubgridMaxwell3D, fine_source_samples
from DeepGPR.subgrid.adapt import plan_refinement
from DeepGPR.subgrid.adjoint import SubgridPropagator
from DeepGPR.subgrid.amr import indicator, regrid

ROOT = Path(__file__).resolve().parent / "results" / "sandpit3d"
DTYPE = torch.float32
DEV = torch.device("cuda")
C0 = 299792458.0
H = 0.0075                     # model grid = subgrid spacing = uniform fine grid
R = 3                          # main grid 2.25 cm
H_REF = H / 2                  # data generator 0.375 cm
SHAPE = (91, 160, 100)         # (depth, x, y) model nodes (n - 1 divisible by 3)
NA = 9                         # surface at row 9 (0.0675 m of air above)
EPS0, SIG0 = 4.6, 1e-3
FLOOR, EPS_FLOOR, SIG_FLOOR = 0.48, 7.0, 0.01
ANT_ROW = NA - 2               # antennas 1.5 cm above the sand
F0, T_MAX = 450e6, 16e-9
COMP = 1                       # horizontal dipoles along y
DT = 0.95 / (C0 * math.sqrt(3.0) / (R * H))       # main dt
NT = int(round(T_MAX / DT))
SNR_DB = 20.0
TV = 1.0e-6
SIG_SCALE = 0.01
EVERY = 10
PEC_SIGMA = 1e6
GAMMA_PEAK, LOCAL, BUDGET, HYST = 0.1, 0.3, 0.7, 0.9    # budget: x fine nodes of the whole-region box
BUDGETS = {"coarse": 1200, "static_full": 1200, "oracle": 1200, "amr_once_v1": 1200,
           "amr_once_v3": 2400, "amr_dyn_v3": 2400, "fine": 10800,
           "coarse_long": 3600, "amr_dyn_v3_long": 10800}
MARKS = [75, 150, 300, 600, 1200, 2400, 3600, 4800, 7200, 10800]
GIF_MARKS = [75 * k for k in range(1, 17)]            # 75 .. 1200 s
METHODS = ("coarse", "static_full", "oracle", "amr_once_v1", "amr_once_v3", "amr_dyn_v3", "fine",
           "coarse_long", "amr_dyn_v3_long")
RESUMED = {"amr_dyn_v3_long": "amr_dyn_v3"}          # continued from the end state of that run
LABELS = {"coarse": "uniform 2.25 cm", "fine": "uniform 0.75 cm (fine)", "static_full": "static box, whole region",
          "oracle": "oracle boxes (prior)", "amr_once_v1": "adaptive once v1", "amr_once_v3": "adaptive once v3",
          "amr_dyn_v3": "dynamic v3", "coarse_long": "uniform 2.25 cm (long run)",
          "amr_dyn_v3_long": "dynamic v3 (continued to 3 h)"}
COLORS = {"coarse": "#7f7f7f", "fine": "#8c564b", "static_full": "#1f77b4", "oracle": "#2ca02c",
          "amr_once_v1": "#ff7f0e", "amr_once_v3": "#e377c2", "amr_dyn_v3": "#d62728",
          "coarse_long": "#4d4d4d", "amr_dyn_v3_long": "#8b0000"}
STYLES = {"oracle": "--", "fine": ":", "coarse": ":", "coarse_long": "-."}


# --------------------------------------------------------------------------- models
def _pipe(x, d, r, eps=None, sig=None, pec=False, y=(0.22, 0.52)):
    return dict(kind="pipe", x=x, d=d, r=r, y=y, eps=eps, sig=sig, pec=pec)


def _sphere(x, y, d, r, eps=None, sig=None, pec=False):
    return dict(kind="sphere", x=x, y=y, d=d, r=r, eps=eps, sig=sig, pec=pec)


def _box(x, y, d, half, eps=None, sig=None, pec=False):
    return dict(kind="box", x=x, y=y, d=d, half=half, eps=eps, sig=sig, pec=pec)


def random_model(seed):
    rng = np.random.default_rng(seed)
    out = []
    for xs in (0.28, 0.62, 0.96):
        x = xs + rng.uniform(-0.04, 0.04)
        d = rng.uniform(0.18, 0.34)
        k = rng.integers(4)
        if k == 0:
            out.append(_pipe(x, d, rng.uniform(0.015, 0.03), pec=True))
        elif k == 1:
            out.append(_sphere(x, rng.uniform(0.3, 0.44), d, rng.uniform(0.035, 0.05), eps=1.0, sig=0.0))
        elif k == 2:
            out.append(_pipe(x, d, rng.uniform(0.02, 0.03), eps=20.0, sig=0.05))
        else:
            out.append(_sphere(x, rng.uniform(0.3, 0.44), d, rng.uniform(0.035, 0.05), eps=9.0, sig=0.005))
    return out


MODELS = {
    "M1": [_pipe(0.28, 0.20, 0.02, pec=True), _sphere(0.62, 0.37, 0.27, 0.045, eps=1.0, sig=0.0),
           _pipe(0.96, 0.30, 0.025, eps=20.0, sig=0.05)],
    "M2": [_pipe(0.30, 0.18, 0.015, pec=True), _box(0.62, 0.37, 0.24, (0.04, 0.04, 0.04), eps=1.0, sig=0.0),
           _pipe(0.93, 0.33, 0.02, pec=True)],
    "M3": random_model(5),
    "M4": random_model(17),        # validation only
}
TITLES = {"M1": "metal pipe, void, water pipe", "M2": "two metal pipes and a void box",
          "M3": "random (seed 5)", "M4": "random (seed 17, validation)"}


def grid_shape(h):
    return tuple(int(round((n - 1) * H / h)) + 1 for n in SHAPE)


def coords(h):
    s = grid_shape(h)
    d = (np.arange(s[0]) - NA * H / h) * h          # depth below the sand surface
    return np.meshgrid(d, np.arange(s[1]) * h, np.arange(s[2]) * h, indexing="ij")


def body_mask(b, D, X, Y):
    if b["kind"] == "pipe":
        return ((X - b["x"]) ** 2 + (D - b["d"]) ** 2 <= b["r"] ** 2) & (Y >= b["y"][0]) & (Y <= b["y"][1])
    if b["kind"] == "sphere":
        return (X - b["x"]) ** 2 + (Y - b["y"]) ** 2 + (D - b["d"]) ** 2 <= b["r"] ** 2
    hx, hy, hd = b["half"]
    return (np.abs(X - b["x"]) <= hx) & (np.abs(Y - b["y"]) <= hy) & (np.abs(D - b["d"]) <= hd)


def model_arrays(name, h=H, targets=True):
    D, X, Y = coords(h)
    eps = np.where(D < 0, 1.0, np.where(D < FLOOR, EPS0, EPS_FLOOR))
    sig = np.where(D < 0, 0.0, np.where(D < FLOOR, SIG0, SIG_FLOOR))
    if targets and name:
        for b in MODELS[name]:
            m = body_mask(b, D, X, Y)
            if b["pec"]:
                sig = np.where(m, PEC_SIGMA, sig)
            else:
                eps = np.where(m, b["eps"], eps)
                sig = np.where(m, b["sig"], sig)
    return eps, sig


def update_mask():
    D, X, Y = coords(H)
    return (D >= 0.03) & (D <= 0.45) & (X >= 0.10) & (X <= 1.10) & (Y >= 0.10) & (Y <= 0.64)


def allowed_main():
    """Main nodes where refinement may be marked (inside the update region)."""
    n = tuple((s - 1) // R + 1 for s in SHAPE)
    D, X, Y = np.meshgrid((np.arange(n[0]) * R - NA) * H, np.arange(n[1]) * R * H, np.arange(n[2]) * R * H,
                          indexing="ij")
    return (D >= 0.08) & (D <= 0.45) & (X >= 0.12) & (X <= 1.08) & (Y >= 0.12) & (Y <= 0.62)


# --------------------------------------------------------------------------- acquisition
def acquisition():
    a = ANT_ROW * H
    txs = [(a, x, y) for x in (0.2, 0.4, 0.6, 0.8, 1.0) for y in (0.2, 0.37, 0.54)]
    rxs = [(a, x, y) for x in np.linspace(0.1, 1.1, 11) for y in (0.15, 0.30, 0.45, 0.60)]
    src = np.array([[t] for t in txs])
    rec = np.array([rxs] * len(txs))
    train = [i for i, t in enumerate(txs) if round(t[1], 2) in (0.2, 0.6, 1.0)]
    hold = [i for i, t in enumerate(txs) if round(t[1], 2) in (0.4, 0.8)]
    return src, rec, train, hold


def mute(src, rec, min_offset=0.10):
    off = np.linalg.norm(rec - src, axis=-1)
    return torch.as_tensor(off >= min_offset, dtype=DTYPE)[:, None, :]


def ricker(nt, dt, f0=F0):
    t = np.arange(nt) * dt - 1.2 / f0
    a = (np.pi * f0 * t) ** 2
    return torch.as_tensor((1 - 2 * a) * np.exp(-a))


def resampled(w, r):
    """Main-rate samples -> grid with dt / r, with the main grid's half-step source delay."""
    if r == 1:
        return w
    up = fine_source_samples(w[None].double(), 2 * r if r % 2 == 0 else r)[0]
    k = np.arange(r * w.numel())
    # J((k + 1/2) dt / r) = W((k + 1/2) / r - 1/2): index k - (r - 1)/2 on the dt / r lattice
    # (odd r) or 2k - (r - 1) on the dt / (2r) lattice (even r)
    idx = k - (r - 1) // 2 if r % 2 else 2 * k - (r - 1)
    ok = torch.as_tensor((idx >= 0) & (idx < up.numel()))
    return torch.where(ok, up[np.clip(idx, 0, up.numel() - 1)], torch.zeros((), dtype=up.dtype))


# --------------------------------------------------------------------------- data
def reference_data(name, shots, src, rec, targets=True, batch=3):
    """Uniform 0.375 cm data at the main dt (cached)."""
    tag = f"{name if targets else 'bg'}_" + "-".join(map(str, shots))
    path = ROOT / "cache" / f"ref_{tag}.npz"
    if path.exists():
        return torch.as_tensor(np.load(path)["d"])
    path.parent.mkdir(parents=True, exist_ok=True)
    e, s = model_arrays(name, H_REF, targets)
    e, s = torch.as_tensor(e, dtype=DTYPE), torch.as_tensor(s, dtype=DTYPE)
    w6 = resampled(ricker(NT, DT), 6)
    out = []
    for i in range(0, len(shots), batch):
        sh = shots[i:i + batch]
        sol = SubgridMaxwell3D(e, s, H_REF, coarsen=1, pmlthick=40, dt=DT / 6, dipole_length=R * H,
                               component=COMP, nshot=len(sh), device=DEV, dtype=DTYPE)
        out.append(sol.run(w6, src[sh], rec[sh])["receivers"][:, 5::6].cpu())
        del sol
        torch.cuda.empty_cache()
    d = torch.cat(out, 0)
    np.savez_compressed(path, d=d.numpy())
    return d


def band_noise(shape, dt, seed):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(shape)
    f = np.fft.rfftfreq(shape[1], dt)
    amp = (f / F0) ** 2 * np.exp(-(f / F0) ** 2)
    return np.fft.irfft(np.fft.rfft(n, axis=1) * (amp / amp.max())[None, :, None], n=shape[1], axis=1)


# --------------------------------------------------------------------------- solvers
class Data:
    """Differential data d(m) - d(m_bg) of a set of shots on one solver kind."""

    def __init__(self, kind, src, rec, shots, boxes=(), batch=None):
        self.kind, self.shots = kind, list(shots)
        self.batch = batch or (1 if kind == "fine" else len(self.shots))
        self.props = []
        e, s = background()
        for i in range(0, len(self.shots), self.batch):
            sh = self.shots[i:i + self.batch]
            if kind == "fine":
                sol = SubgridMaxwell3D(e, s, H, coarsen=1, pmlthick=20, dt=DT / R, dipole_length=R * H,
                                       component=COMP, nshot=len(sh), device=DEV, dtype=DTYPE)
            else:
                sol = SubgridMaxwell3D(e, s, H, coarsen=R, subgrids=[Subgrid(tuple(map(tuple, b))) for b in boxes],
                                       pmlthick=10, dt=DT, dipole_length=R * H, component=COMP, nshot=len(sh),
                                       device=DEV, dtype=DTYPE)
            self.props.append((sh, SubgridPropagator(sol, src[sh], rec[sh])))
        w = ricker(NT, DT)
        if kind == "fine":
            self.w = resampled(w, R).to(DEV, DTYPE)
        else:
            self.w = w.to(DEV, DTYPE)
        with torch.no_grad():
            self.base = [d.detach() for _, d in self._raw(e.to(DEV), s.to(DEV))]

    def _raw(self, eps, sig):
        o = 0
        for sh, p in self.props:
            d = p(eps, sig, self.w)
            if self.kind == "fine":
                d = d[:, R - 1::R][:, :NT]
            yield slice(o, o + len(sh)), d
            o += len(sh)

    def batched(self, eps, sig):
        for (i, d), b0 in zip(self._raw(eps, sig), self.base):
            yield i, d - b0

    def __call__(self, eps, sig):
        return torch.cat([d for _, d in self.batched(eps, sig)], 0)

    def main_solver(self):
        return self.props[0][1].solver


def background():
    e, s = model_arrays(None, H, targets=False)
    return torch.as_tensor(e, dtype=DTYPE), torch.as_tensor(s, dtype=DTYPE)


def fine_nodes(boxes):
    tot = 0
    for b in boxes:
        cells = [int(round(hi / (R * H))) - int(round(lo / (R * H))) for lo, hi in b]
        tot += int(np.prod([R * c + 37 for c in cells]))
    return tot


MAIN_NODES = int(np.prod([(s - 1) // R + 1 + 20 for s in SHAPE]))


# --------------------------------------------------------------------------- refinement
def plan_from_flags(flags, keep):
    z = np.zeros(SHAPE)
    plan = plan_refinement(z + 1.0, z, H, 1.0, ppw=1e-9, coarsen=R, extra_flags=flags, keep_points=keep,
                           efficiency=0.85)
    return [[list(map(float, ax)) for ax in sg.box] for sg in plan.subgrids]


def mark_v3(eta, allowed, keep, budget_nodes, peak_frac=GAMMA_PEAK, local=LOCAL):
    """Every local peak of the (smoothed) indicator above peak_frac x max gets the region
    where eta >= local x its own peak; weakest peaks dropped until the layout fits the
    fine-node budget ``budget_nodes``."""
    from scipy.ndimage import gaussian_filter, label, maximum_filter
    e = gaussian_filter(np.where(allowed, eta, 0.0), 1.0)
    if e.max() <= 0:
        return [], {"peaks": 0}
    pk = np.argwhere((e == maximum_filter(e, size=5)) & (e >= peak_frac * e.max()) & allowed)
    pk = sorted(pk.tolist(), key=lambda p: -e[tuple(p)])
    regions = []
    covered = np.zeros(e.shape, dtype=bool)
    for p in pk:
        if covered[tuple(p)]:
            continue
        lab, _ = label(e >= local * e[tuple(p)])
        comp = (lab == lab[tuple(p)]) & allowed
        covered |= comp
        regions.append((float(e[tuple(p)]), comp))
    while regions:
        flags = np.zeros(e.shape, dtype=bool)
        for _, c in regions:
            flags |= c
        boxes = plan_from_flags(flags, keep)
        if fine_nodes(boxes) <= budget_nodes:
            return boxes, {"peaks": len(regions), "marked": int(flags.sum()), "fine_nodes": fine_nodes(boxes)}
        regions = regions[:-1]
    return [], {"peaks": 0}


# --------------------------------------------------------------------------- scenario context
class Context:
    def __init__(self, name):
        self.name = name
        self.src, self.rec, self.train, self.hold = acquisition()
        self.mk = mute(self.src, self.rec)
        allsh = list(range(len(self.src)))
        d_true = reference_data(name, allsh, self.src, self.rec)
        d_bg = reference_data(name, allsh, self.src, self.rec, targets=False)
        self.resp = d_true - d_bg
        sig_e = float(((self.resp[self.train] * self.mk[self.train]) ** 2).sum())
        seed = zlib.crc32(name.encode()) & 0xFFFFFFFF
        noise = band_noise(tuple(self.resp.shape), DT, seed)
        noise *= math.sqrt(10 ** (-SNR_DB / 10) * sig_e / float(((noise[self.train] * self.mk[self.train].numpy()) ** 2).sum()))
        self.obs = (self.resp + torch.as_tensor(noise, dtype=self.resp.dtype))[self.train]
        self.norm = float(((self.obs * self.mk[self.train]) ** 2).sum())
        self.true_e, self.true_s = model_arrays(name)
        self.keep = np.concatenate([self.src.reshape(-1, 3), self.rec.reshape(-1, 3)])
        self.allowed = allowed_main()
        self._eval = None

    def evaluator(self):
        if self._eval is None:
            self._eval = Data("fine", self.src, self.rec, self.hold, batch=len(self.hold))
        return self._eval

    def pe(self, eps, sig, return_data=False):
        fn = self.evaluator()
        mk = self.mk[self.hold]
        with torch.no_grad():
            d = fn(torch.as_tensor(eps, dtype=DTYPE, device=DEV), torch.as_tensor(sig, dtype=DTYPE, device=DEV)).cpu() * mk
        obs = self.resp[self.hold] * mk
        pe = float(((d - obs) ** 2).sum() / (obs ** 2).sum())
        return (pe, d) if return_data else pe

    def oracle_boxes(self):
        from scipy.ndimage import binary_dilation
        e0, s0 = model_arrays(None)
        m = (np.abs(self.true_e - e0) > 0.1) | (np.abs(self.true_s - s0) > 1e-3)
        n = tuple((s - 1) // R + 1 for s in SHAPE)
        mm = np.zeros(n, dtype=bool)
        idx = np.argwhere(m)
        mm[tuple((idx // R).T)] = True
        return plan_from_flags(binary_dilation(mm, iterations=1) & self.allowed, self.keep)

    def full_boxes(self):
        return plan_from_flags(self.allowed, self.keep)


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


def tv_term(u, m):
    gx = (u[1:] - u[:-1]) * (m[1:] & m[:-1])
    gy = (u[:, 1:] - u[:, :-1]) * (m[:, 1:] & m[:, :-1])
    gz = (u[:, :, 1:] - u[:, :, :-1]) * (m[:, :, 1:] & m[:, :, :-1])
    return torch.sqrt(gx[:, :-1, :-1] ** 2 + gy[:-1, :, :-1] ** 2 + gz[:-1, :-1, :] ** 2 + 1e-8).sum()


def budget_fwi(ctx: Context, kind, budget, marks, log, boxes=None, every=EVERY, resume=None):
    """``resume``: dict(u_e, u_s, layout, t, it, n_replans) continues a finished run from that
    state (clock, iteration count and layout carried over, new L-BFGS history)."""
    base = RESUMED.get(kind, kind[:-5] if kind.endswith("_long") else kind)
    dynamic = base.startswith("amr")
    e_bg, s_bg = background()
    E0, S0 = e_bg.to(DEV), s_bg.to(DEV)
    mask = torch.as_tensor(update_mask(), device=DEV)
    obs = (ctx.obs * ctx.mk[ctx.train]).cuda()
    mkc = ctx.mk[ctx.train].cuda()
    u_e = torch.zeros(SHAPE, dtype=DTYPE, device=DEV, requires_grad=True)
    u_s = torch.zeros(SHAPE, dtype=DTYPE, device=DEV, requires_grad=True)

    def mdl():
        return ((E0 + u_e * mask).clamp(1.0, 40.0), (S0 + SIG_SCALE * u_s * mask).clamp(0.0, 0.5))

    def new_opt():
        return torch.optim.LBFGS([u_e, u_s], lr=1.0, history_size=10, max_iter=1, line_search_fn="strong_wolfe")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    clock = Clock()
    layout = [] if dynamic else list(boxes or [])
    it, n_replans = 0, 0
    if resume:
        with torch.no_grad():
            u_e.copy_(resume["u_e"].to(DEV, DTYPE))
            u_s.copy_(resume["u_s"].to(DEV, DTYPE))
        layout = [list(map(list, b)) for b in resume["layout"]]
        it, n_replans, clock.acc = resume["it"], resume["n_replans"], resume["t"]
    clock.start()
    fn = Data("fine" if base == "fine" else "main", ctx.src, ctx.rec, ctx.train, boxes=layout)
    state = {"fn": fn, "evals": 0, "misfit": 1.0}
    opt = new_opt()

    def closure():
        opt.zero_grad()
        eps, sig = mdl()
        el, sl = eps.detach().requires_grad_(True), sig.detach().requires_grad_(True)
        misfit = 0.0
        for idx, d in state["fn"].batched(el, sl):
            part = ((d * mkc[idx] - obs[idx]) ** 2).sum() / ctx.norm
            part.backward()
            misfit += float(part)
        reg = TV * (tv_term(u_e, mask) + tv_term(u_s, mask))
        torch.autograd.backward([eps, sig, reg], [el.grad, sl.grad, torch.ones_like(reg)])
        state["evals"] += 1
        state["misfit"] = misfit
        return torch.tensor(misfit + float(reg))

    def replan(n):
        with torch.no_grad():
            e, s = mdl()
        main = Data("main", ctx.src, ctx.rec, ctx.train)          # main grid alone: eta defined everywhere
        with torch.no_grad():
            d = main(e, s) * mkc
        ip = SubgridPropagator(main.main_solver(), ctx.src[ctx.train], ctx.rec[ctx.train], indicator=True)
        eta = indicator(ip, e, s, main.w, (d - obs) * mkc, eps0=E0, sig0=S0)
        del ip, main
        if base == "amr_once_v1":
            f_bg = C0 / (math.sqrt(EPS0) * 10 * R * H)
            rg = regrid(e.double().cpu().numpy(), s.double().cpu().numpy(), H, R, eta, allowed=ctx.allowed,
                        theta=0.5, f_max=min(1.5 * F0, 0.95 * f_bg), ppw=10, keep_points=ctx.keep, current=[])
            new = [[list(map(float, ax)) for ax in sg.box] for sg in rg.plan.subgrids]
            info = {"marked": rg.info["marked_nodes"]}
        else:
            new, info = mark_v3(eta, ctx.allowed, ctx.keep, BUDGET * fine_nodes(ctx.full_boxes()))
        log(f"  [{kind}] re-plan {n}: {info} -> {[[tuple(round(v, 3) for v in ax) for ax in b] for b in new]} "
            f"({fine_nodes(new)} fine nodes)")
        return new, eta

    snaps, hist = {}, []
    pending = sorted(set(marks))
    prev = dict(t=0.0, it=0, misfit=1.0, boxes=[], eps=None, sig=None)
    if resume:
        with torch.no_grad():
            e, s = mdl()
        prev = dict(t=resume["t"], it=it, misfit=resume.get("misfit", 1.0), boxes=layout,
                    eps=e.float().cpu().numpy(), sig=s.float().cpu().numpy())
    while clock.elapsed() < budget:
        if dynamic and it > 0 and it % every == 0 and (base == "amr_dyn_v3" or n_replans == 0):
            n_replans += 1
            new, eta = replan(n_replans)
            changed = sorted(map(str, new)) != sorted(map(str, layout))
            if changed and base == "amr_dyn_v3" and layout:
                # hysteresis: keep the layout when its boxes already hold most of the new marks
                inside_new = np.zeros(eta.shape, dtype=bool)
                for b in new:
                    sl = tuple(slice(int(round(lo / (R * H))), int(round(hi / (R * H))) + 1) for lo, hi in b)
                    inside_new[sl] = True
                inside_old = np.zeros(eta.shape, dtype=bool)
                for b in layout:
                    sl = tuple(slice(int(round(lo / (R * H))), int(round(hi / (R * H))) + 1) for lo, hi in b)
                    inside_old[sl] = True
                cov = float(eta[inside_new & inside_old].sum() / max(eta[inside_new].sum(), 1e-30))
                if cov >= HYST and fine_nodes(layout) <= 1.5 * max(fine_nodes(new), 1):
                    changed = False
                    log(f"  [{kind}] kept the layout (coverage {cov:.2f})")
            if changed:
                layout = new
                state["fn"] = None
                fn = None
                torch.cuda.empty_cache()
                fn = Data("main", ctx.src, ctx.rec, ctx.train, boxes=layout)
                state["fn"] = fn
                opt = new_opt()
        n0 = state["evals"]
        opt.step(closure)
        it += 1
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
            log(f"  [{kind}] it {it:4d}: t {t:7.1f} s, misfit {state['misfit']:.4f}, {len(layout)} box(es), "
                f"{state['evals']} evals")
        prev = cur
        clock.start()
    clock.pause()
    for m in pending:
        snaps[m] = prev
    info = dict(iterations=it, evaluations=state["evals"], seconds=clock.acc,
                seconds_per_evaluation=clock.acc / max(1, state["evals"]),
                peak_GB=torch.cuda.max_memory_allocated() / 2 ** 30, final_boxes=layout,
                final_fine_nodes=fine_nodes(layout), replans=n_replans)
    fn = opt = None
    state["fn"] = None
    torch.cuda.empty_cache()
    return hist, snaps, info


# --------------------------------------------------------------------------- run / evaluate
def log_to(path):
    def log(msg):
        print(msg, flush=True)
        with open(path, "a") as f:
            f.write(msg + "\n")
    return log


def warm_up(ctx):
    d = Data("main", ctx.src, ctx.rec, ctx.train[:2], boxes=ctx.full_boxes())
    e, s = background()
    el = e.to(DEV).requires_grad_(True)
    (d(el, s.to(DEV)) ** 2).sum().backward()
    del d
    torch.cuda.empty_cache()


def model_metrics(ctx, eps, sig):
    from scipy.ndimage import gaussian_filter
    m = update_mask()
    pec = ctx.true_s > 1.0
    out = {}
    for nm, x, a, b0 in (("eps", eps, ctx.true_e, background()[0].numpy()), ("sig", sig, np.minimum(ctx.true_s, 0.5),
                                                                              background()[1].numpy())):
        mm = m & ~pec if nm == "eps" else m
        A, B = gaussian_filter(a, 2), gaussian_filter(x, 2)
        out[f"{nm}_err_smooth"] = float(np.linalg.norm((B - A)[mm]) / max(np.linalg.norm((A - b0)[mm]), 1e-12))
    return out


def resume_state(res, out, base):
    """End state of a finished run: the iterate scored at its last mark."""
    old = res["methods"][base]
    z = np.load(out / f"snap_{base}.npz")
    last = old["snap_meta"][-1]
    e0, s0 = (x.numpy() for x in background())
    mask = update_mask()
    u_e = np.where(mask, z["eps_final"] - e0, 0.0)
    u_s = np.where(mask, (z["sig_final"] - s0) / SIG_SCALE, 0.0)
    resume = dict(u_e=torch.as_tensor(u_e), u_s=torch.as_tensor(u_s), layout=last["boxes"], t=float(last["t"]),
                  it=int(last["it"]), n_replans=int(old.get("replans", 0)), misfit=float(last["misfit"]))
    return old, z, resume


def run(name, methods, smoke=False):
    out = ROOT / ("smoke_" + name if smoke else name)
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    ctx = Context(name)
    path = out / "result.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res.update(dict(scenario=name, title=TITLES[name], bodies=MODELS[name], dt=DT, nt=NT, snr_db=SNR_DB,
                    budgets=BUDGETS, marks=MARKS, gif_marks=GIF_MARKS))
    res.setdefault("methods", {})
    res["oracle_boxes"] = ctx.oracle_boxes()
    res["full_boxes"] = ctx.full_boxes()
    if "evaluator_floor" not in res:
        res["evaluator_floor"] = ctx.pe(ctx.true_e, ctx.true_s)
    log(f"=== {name} ({TITLES[name]}): main {tuple((s - 1) // R + 1 for s in SHAPE)}, nt {NT}, oracle "
        f"{res['oracle_boxes']}, full {res['full_boxes']}, evaluator floor {res['evaluator_floor']:.4f}")
    path.write_text(json.dumps(res, indent=1))
    warm_up(ctx)
    scale = 0.05 if smoke else 1.0
    for kind in methods:
        if kind in res["methods"] and "error" not in res["methods"][kind] and (out / f"snap_{kind}.npz").exists():
            log(f"[{kind}] done before, skipped")
            continue
        budget = BUDGETS[kind] * scale
        marks = sorted(set([m * scale for m in MARKS if m * scale <= budget] + [m * scale for m in GIF_MARKS
                                                                                if m * scale <= budget]))
        boxes = {"static_full": res["full_boxes"], "oracle": res["oracle_boxes"]}.get(kind)
        resume, old = None, None
        if kind in RESUMED:
            old, zold, resume = resume_state(res, out, RESUMED[kind])
            marks = [m for m in marks if m > resume["t"]]
        log(f"[{kind}] start, budget {budget:.0f} s, boxes {boxes}" +
            (f", resumed from {RESUMED[kind]} at {resume['t']:.0f} s / it {resume['it']} with {resume['layout']}"
             if resume else ""))
        try:
            hist, snaps, info = budget_fwi(ctx, kind, budget, marks, log, boxes=boxes, every=3 if smoke else EVERY,
                                           resume=resume)
            if resume:      # carry the scored marks, history and evaluations of the first part over
                for j, m in enumerate(old["marks"]):
                    meta = {k: v for k, v in old["snap_meta"][j].items() if k != "mark"}
                    snaps[m] = dict(meta, eps=zold["eps"][j].astype(np.float32), sig=None, old=True)
                head = [h for h in old["history"] if h["it"] <= resume["it"]]
                hist = head + hist
                info["evaluations"] += sum(h["evals"] for h in head)
                info["seconds_per_evaluation"] = info["seconds"] / max(1, info["evaluations"])
                marks = sorted(set(list(old["marks"]) + marks))
        except Exception as exc:
            log(f"[{kind}] FAILED: {exc!r}\n{traceback.format_exc()}")
            res["methods"][kind] = {"error": repr(exc)}
            path.write_text(json.dumps(res, indent=1))
            torch.cuda.empty_cache()
            continue
        e0 = background()[0].numpy()
        s0 = background()[1].numpy()
        for m in marks:
            if snaps[m]["eps"] is None and not snaps[m].get("old"):
                snaps[m] = dict(snaps[m], eps=e0, sig=s0, start=True)
        stack_e = np.stack([snaps[m]["eps"] for m in marks]).astype(np.float16)       # figures only
        np.savez_compressed(out / f"snap_{kind}.npz", eps=stack_e, marks=np.array(marks),
                            eps_final=snaps[marks[-1]]["eps"], sig_final=snaps[marks[-1]]["sig"])
        r = dict(history=hist, **info, marks=marks,
                 snap_meta=[dict(mark=m, **{k: v for k, v in snaps[m].items() if k not in ("eps", "sig", "start", "old")})
                            for m in marks])
        if resume:
            r["resumed_from"] = dict(kind=RESUMED[kind], t=resume["t"], it=resume["it"])
        # PE at every scoring mark (identical consecutive models evaluated once)
        r["pe"], last = {}, None
        for j, m in enumerate(marks):
            if m not in [x * scale for x in MARKS]:
                continue
            ej, sj = snaps[m]["eps"], snaps[m]["sig"]
            if snaps[m].get("old"):
                r["pe"][str(m)] = old["pe"][str(m)]
                continue
            if snaps[m].get("start"):
                r["pe"][str(m)] = 1.0
                continue
            if last is not None and np.array_equal(ej, last[0]) and np.array_equal(sj, last[1]):
                r["pe"][str(m)] = last[2]
                continue
            r["pe"][str(m)] = ctx.pe(ej, sj)
            last = (ej, sj, r["pe"][str(m)])
        r["metrics"] = model_metrics(ctx, snaps[marks[-1]]["eps"].astype(np.float64),
                                     snaps[marks[-1]]["sig"].astype(np.float64))
        res["methods"][kind] = r
        log(f"[{kind}] {info['iterations']} it, {info['evaluations']} evals, {info['seconds_per_evaluation']:.2f} s/eval, "
            f"{info['peak_GB']:.1f} GB, boxes {info['final_boxes']}; PE " +
            ", ".join(f"{float(k):.0f} s: {v:.4f}" for k, v in r["pe"].items()))
        path.write_text(json.dumps(res, indent=1))
    try:
        figures(name, out, res)
    except Exception as exc:
        log(f"figures FAILED: {exc!r}\n{traceback.format_exc()}")


# --------------------------------------------------------------------------- figures / report
def time_to_reach(times, pes, target):
    prev = (0.0, 1.0)
    for t, v in zip(times, pes):
        if v <= target:
            if prev[1] <= target:
                return prev[0]
            a = (math.log(prev[1]) - math.log(target)) / max(math.log(prev[1]) - math.log(v), 1e-12)
            return prev[0] + a * (t - prev[0])
        prev = (t, v)
    return None


def pe_curve(r):
    ks = sorted(r.get("pe", {}), key=float)
    return [float(k) for k in ks], [r["pe"][k] for k in ks]


def figures(name, out, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    kinds = [k for k in METHODS if k in res["methods"] and "error" not in res["methods"][k]]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for k in kinds:
        t, p = pe_curve(res["methods"][k])
        ax.loglog(t, p, marker="o", ms=4, color=COLORS[k], ls=STYLES.get(k, "-"), label=LABELS[k])
    ax.axhline(res["evaluator_floor"], color="0.6", lw=0.7)
    ax.set_xlabel("compute time [s]")
    ax.set_ylabel("held-out prediction error")
    ax.set_title(f"{name} ({TITLES[name]})")
    ax.grid(True, which="major", color="0.9")
    ax.legend(fontsize=8)
    fig.savefig(out / "pe_vs_time.png", dpi=110)
    plt.close(fig)
    # evolution GIF: x-z slice through the middle of the pipes (y = 0.37 m) at equal compute times
    D, X, _ = coords(H)
    jy = int(round(0.37 / H))
    ext = [0, (SHAPE[1] - 1) * H, D[-1, 0, 0], D[0, 0, 0]]
    e0 = background()[0].numpy()
    kinds = [k for k in kinds if not k.endswith("_long")]          # GIF covers 0-1200 s: first round only
    snaps = {k: np.load(out / f"snap_{k}.npz") for k in kinds if (out / f"snap_{k}.npz").exists()}
    frames = []
    for tm in [0] + GIF_MARKS:
        te, ts = model_arrays(name)
        cols = [("true model (PEC shown as 8)", np.where(ts > 1.0, 8.0, te), None, "")]
        for k in kinds:
            z = snaps.get(k)
            if z is None:
                continue
            ms = list(z["marks"])
            idx = [i for i, m in enumerate(ms) if m <= tm + 1e-6]
            if tm == 0 or not idx:
                cols.append((LABELS[k], e0, [], "start"))
                continue
            i = idx[-1]
            meta = res["methods"][k]["snap_meta"][i]
            cols.append((LABELS[k], z["eps"][i].astype(np.float32), meta["boxes"], f"it {meta['it']}"))
        n = len(cols)
        fig, axs = plt.subplots(n, 1, figsize=(9, 1.55 * n + 0.6), constrained_layout=True)
        for a, (lab, e, boxes, note) in zip(axs, cols):
            a.imshow(np.clip(e[:, :, jy], 1, 12), aspect="auto", cmap="RdBu_r", vmin=1.0, vmax=8.2, extent=ext)
            for b in boxes or []:
                (d0, d1), (x0, x1), (y0, y1) = b
                if y0 <= 0.37 <= y1:
                    a.plot([x0, x1, x1, x0, x0], [d0 - NA * H, d0 - NA * H, d1 - NA * H, d1 - NA * H, d0 - NA * H],
                           "k--", lw=0.8)
            a.set_ylim(0.5, -0.07)
            a.set_ylabel(lab, fontsize=7, rotation=0, ha="right")
            a.set_yticks([])
            a.text(0.01, 0.05, note, transform=a.transAxes, fontsize=7)
        axs[-1].set_xlabel("x [m]")
        fig.suptitle(f"{name}: eps_r (x-z slice at y = 0.37 m) after {tm:.0f} s of compute", fontsize=9)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frames.append(Image.frombuffer("RGBA", (w, h), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
                      .quantize(colors=200, method=2))
        plt.close(fig)
    frames += [frames[-1]] * 4
    frames[0].save(out / "evolution.gif", save_all=True, append_images=frames[1:], duration=600, loop=0)


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {}
    for name in MODELS:
        p = ROOT / name / "result.json"
        if p.exists():
            res[name] = json.loads(p.read_text())
    if not res:
        print("no results")
        return
    L = ["# Sand-pit-like 3D benchmark: adaptive subgridding vs uniform grids", "",
         "Held-out PE (noise-free, 0.75 cm evaluator against 0.375 cm data) at fixed compute times; "
         "time-to-reach = when the uniform fine grid first reaches a method's PE at 1200 s.", ""]
    hdr = "| scenario | method | " + " | ".join(f"PE @ {m} s" for m in MARKS) + " | s/grad | final boxes (fine nodes) |"
    L += [hdr, "|---|---|" + "---|" * (len(MARKS) + 2)]
    tt_rows = []
    for name, r in res.items():
        f_t, f_p = pe_curve(r["methods"].get("fine", {}))
        for k in METHODS:
            v = r["methods"].get(k)
            if not v or "error" in v:
                continue
            cells = [f"{v['pe'][str(m)]:.4f}" if str(m) in v.get("pe", {}) else (f"{v['pe'][str(float(m))]:.4f}"
                     if str(float(m)) in v.get("pe", {}) else "—") for m in MARKS]
            L.append(f"| {name} | {LABELS[k]} | " + " | ".join(cells) +
                     f" | {v['seconds_per_evaluation']:.1f} | {len(v['final_boxes'])} ({v['final_fine_nodes']}) |")
            t, p = pe_curve(v)
            p1200 = dict(zip(t, p)).get(1200.0)
            if k != "fine" and p1200 is not None and f_t:
                tt = time_to_reach(f_t, f_p, p1200)
                tt_rows.append((name, k, p1200, tt))
    L += ["", "| scenario | method | PE at 1200 s | uniform fine needs | speed-up |", "|---|---|---|---|---|"]
    for name, k, p1200, tt in tt_rows:
        L.append(f"| {name} | {LABELS[k]} | {p1200:.4f} | {'not within 10800 s' if tt is None else f'{tt:.0f} s'} | "
                 f"{'> 9x' if tt is None else f'{tt / 1200:.1f}x'} |")
    (ROOT / "report.md").write_text("\n".join(L) + "\n")
    fig, ax = plt.subplots(1, len(res), figsize=(5.5 * len(res), 4.6), constrained_layout=True, squeeze=False)
    for a, (name, r) in zip(ax[0], res.items()):
        for k in METHODS:
            v = r["methods"].get(k)
            if not v or "error" in v:
                continue
            t, p = pe_curve(v)
            a.loglog(t, p, marker="o", ms=3, color=COLORS[k], ls=STYLES.get(k, "-"), label=LABELS[k])
        a.set_title(f"{name} ({TITLES[name]})", fontsize=9)
        a.set_xlabel("compute time [s]")
        a.grid(True, color="0.9")
    ax[0, 0].set_ylabel("held-out PE")
    ax[0, 0].legend(fontsize=7)
    fig.savefig(ROOT / "summary_pe_vs_time.png", dpi=110)
    print("\n".join(L))


# --------------------------------------------------------------------------- three claims
def _snap_eps(name, kind, mark):
    z = np.load(ROOT / name / f"snap_{kind}.npz")
    ms = np.asarray(z["marks"], dtype=float)
    i = int(np.argmin(np.abs(ms - mark)))
    return (z["eps"][i].astype(np.float32) if abs(ms[i] - mark) < 1e-6 else None), i


_TCR_CACHE = {}


def target_clutter(name, eps):
    """min over targets of max |eps - eps_bg| near the target / 99.5th percentile of |eps - eps_bg|
    away from all targets (>= 5 cm) in the update region; higher = cleaner image."""
    from scipy.ndimage import binary_dilation
    if name not in _TCR_CACHE:
        D, X, Y = coords(H)
        um = update_mask()
        bodies = [body_mask(b, D, X, Y) for b in MODELS[name]]
        near = binary_dilation(np.any(bodies, 0), iterations=7)
        _TCR_CACHE[name] = (um & ~near, [binary_dilation(b, iterations=2) & um for b in bodies],
                            model_arrays(None)[0])
    away, near_t, bg = _TCR_CACHE[name]
    de = np.abs(eps - bg)
    return float(min(de[m].max() for m in near_t) / np.percentile(de[away], 99.5))


def pe_iter_curve(r):
    """(iterations, PE) at the scored marks (start model excluded)."""
    its = {float(m["mark"]): m["it"] for m in r["snap_meta"]}
    pts = sorted((its[float(k)], v) for k, v in r["pe"].items() if float(k) in its and its[float(k)] > 0)
    return [p[0] for p in pts], [p[1] for p in pts]


def at_iteration(r, it):
    x, y = pe_iter_curve(r)
    if not x or it < x[0] or it > x[-1]:
        return None
    return float(np.exp(np.interp(np.log(it), np.log(x), np.log(y))))


def claims(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {n: json.loads((ROOT / n / "result.json").read_text()) for n in MODELS if (ROOT / n / "result.json").exists()}
    floors = {n: json.loads((ROOT / f"check_{n}.json").read_text()) for n in res if (ROOT / f"check_{n}.json").exists()}
    names = list(res)
    pick = lambda r, k: k if k in r["methods"] and "error" not in r["methods"][k] else None
    ok = lambda r, *ks: next((k for k in ks if pick(r, k)), None)
    L = ["# Three comparisons (sand-pit-like 3D benchmark)", "",
         "PE = held-out prediction error (6 unseen shots, uniform 0.75 cm evaluator against 0.375 cm data, "
         "noise-free; floor ~0.01). TCR = target-to-clutter ratio of the eps_r image (weakest target / "
         "99.5th percentile away from targets; higher = cleaner).", ""]

    # ---- 1. adaptive vs fixed local refinement (equal time)
    one = ("amr_dyn_v3", "amr_once_v3", "oracle", "static_full")
    L += ["## 1. Adaptive refinement vs fixed refinement (same compute time)", "",
          "Fixed = boxes on the true targets (needs prior knowledge) or one box over the whole update region.", "",
          "| model | method | PE @ 600 s | PE @ 1200 s | TCR @ 1200 s | fine nodes | s / gradient | peak GB |",
          "|---|---|---|---|---|---|---|---|"]
    tcr1 = {}
    for n in names:
        r = res[n]
        for k in one:
            v = r["methods"].get(k)
            if not v or "error" in v:
                continue
            e, _ = _snap_eps(n, k, 1200.0)
            tcr1[(n, k)] = target_clutter(n, e) if e is not None else float("nan")
            L.append(f"| {n} | {LABELS[k]} | {v['pe'].get('600.0', float('nan')):.4f} | {v['pe'].get('1200.0', float('nan')):.4f} | "
                     f"{tcr1[(n, k)]:.2f} | {v['final_fine_nodes']} | {v['seconds_per_evaluation']:.1f} | {v['peak_GB']:.1f} |")
    # ---- 2. subgrid vs coarse (accuracy)
    ck = lambda r: ok(r, "coarse_long", "coarse")
    ak = lambda r: ok(r, "amr_dyn_v3_long", "amr_dyn_v3")
    L += ["", "## 2. Subgridding vs the coarse grid (accuracy)", "",
          "Forward floor = response error of the TRUE model on each solver (training shots). "
          "Inversion compared at equal iteration counts and at the end of each run.", "",
          "| model | forward floor coarse / static box / oracle boxes / fine | PE @ 50 it coarse / dynamic | "
          "PE @ 100 it | PE @ 200 it | final PE coarse (it, s) | final PE dynamic (it, s) | TCR @ ~100 it coarse / dynamic | "
          "final TCR coarse / dynamic |", "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = res[n]
        c, a = ck(r), ak(r)
        if not (c and a):
            continue
        fl = floors.get(n, {})
        cells = []
        for it in (50, 100, 200):
            pc, pa = at_iteration(r["methods"][c], it), at_iteration(r["methods"][a], it)
            cells.append(f"{pc:.3f} / {pa:.3f}" if pc and pa else "—")
        fin = []
        for k in (c, a):
            v = r["methods"][k]
            lastm = max(v["pe"], key=float)
            fin.append(f"{v['pe'][lastm]:.4f} ({v['snap_meta'][-1]['it']}, {float(lastm):.0f})")
        t100 = []
        for k in (c, a):
            meta = r["methods"][k]["snap_meta"]
            j = int(np.argmin([abs(m["it"] - 100) for m in meta]))
            e, _ = _snap_eps(n, k, float(meta[j]["mark"]))
            t100.append(target_clutter(n, e) if e is not None else float("nan"))
        tf = [target_clutter(n, np.load(ROOT / n / f"snap_{k}.npz")["eps_final"]) for k in (c, a)]
        L.append(f"| {n} | {fl.get('coarse', float('nan')):.3f} / {fl.get('static_full', float('nan')):.3f} / "
                 f"{fl.get('oracle', float('nan')):.3f} / {fl.get('fine', float('nan')):.3f} | " + " | ".join(cells) +
                 f" | {fin[0]} | {fin[1]} | {t100[0]:.2f} / {t100[1]:.2f} | {tf[0]:.2f} / {tf[1]:.2f} |")
    # ---- 3. subgrid vs full refinement (speed)
    L += ["", "## 3. Subgridding vs uniform full refinement (time to the same accuracy)", "",
          "For each scored time t of the dynamic run: PE reached, the time the uniform 0.75 cm grid needs for "
          "that PE, and the speed-up.", "",
          "| model | t [s] | dynamic PE | uniform 0.75 cm PE at t | uniform 0.75 cm needs | speed-up |", "|---|---|---|---|---|---|"]
    speed, lower = {}, set()
    for n in names:
        r = res[n]
        a = ak(r)
        if not (a and pick(r, "fine")):
            continue
        f_t, f_p = pe_curve(r["methods"]["fine"])
        fpe = dict(zip(f_t, f_p))
        for t, p in zip(*pe_curve(r["methods"][a])):
            if t < 300:
                continue
            tt = time_to_reach(f_t, f_p, p)
            speed[(n, t)] = None if tt is None else tt / t
            tmax = max(f_t)
            L.append(f"| {n} | {t:.0f} | {p:.4f} | {fpe.get(t, float('nan')):.4f} | "
                     f"{f'not within {tmax:.0f} s' if tt is None else f'{tt:.0f} s'} | "
                     f"{f'> {tmax / t:.1f}x' if tt is None else f'{tt / t:.1f}x'} |")
            if tt is None:
                speed[(n, t)] = tmax / t          # lower bound, drawn hatched
                lower.add((n, t))
    (ROOT / "claims.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # ---- figures (one per claim; PE curves on top, a second measure below)
    nm = len(names)

    def curves(ax, r, kinds, x="time"):
        for k in kinds:
            v = r["methods"].get(k)
            if not v or "error" in v:
                continue
            t, p = pe_curve(v) if x == "time" else pe_iter_curve(v)
            if x == "time":
                keep = [i for i, q in enumerate(p) if q < 0.999]
                t, p = [t[i] for i in keep], [p[i] for i in keep]
            ax.loglog(t, p, marker="o", ms=3.5, lw=1.6, color=COLORS[k], ls=STYLES.get(k, "-"), label=LABELS[k])
        ax.grid(True, which="major", color="0.9")

    def bars(ax, groups, series, values, ylabel, log=False):
        w = 0.8 / len(series)
        for j, k in enumerate(series):
            ys = [values.get((g, k), np.nan) for g in groups]
            ax.bar(np.arange(len(groups)) + (j - (len(series) - 1) / 2) * w, ys, w * 0.92, color=COLORS.get(k, "0.5"),
                   label=LABELS.get(k, k))
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(True, axis="y", color="0.9")

    # claim 1
    fig = plt.figure(figsize=(4.2 * nm, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, nm, height_ratios=[1.3, 1])
    for j, n in enumerate(names):
        ax = fig.add_subplot(gs[0, j])
        curves(ax, res[n], one)
        ax.set_xlim(60, 2600)
        ax.set_title(f"{n}", fontsize=10)
        ax.set_xlabel("compute time [s]")
        if j == 0:
            ax.set_ylabel("held-out PE (lower = better)")
            ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, :])
    bars(ax, names, one, tcr1, "TCR at 1200 s (higher = cleaner image)")
    ax.legend(fontsize=7, ncol=4)
    fig.suptitle("1. Adaptive vs fixed refinement at equal compute time", fontsize=11)
    fig.savefig(ROOT / "claim1_adaptive_vs_fixed.png", dpi=110)
    plt.close(fig)
    # claim 2
    fig = plt.figure(figsize=(4.2 * nm, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, nm, height_ratios=[1.3, 1])
    for j, n in enumerate(names):
        ax = fig.add_subplot(gs[0, j])
        curves(ax, res[n], [k for k in (ck(res[n]), ak(res[n]), "fine") if k], x="iter")
        ax.set_title(f"{n}", fontsize=10)
        ax.set_xlabel("L-BFGS iterations")
        if j == 0:
            ax.set_ylabel("held-out PE (lower = better)")
            ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, :])
    fv = {(n, k): floors[n][k] for n in floors for k in floors[n]}
    bars(ax, [n for n in names if n in floors], ["coarse", "static_full", "oracle", "fine"], fv,
         "true-model forward error (lower = better)", log=True)
    ax.legend(fontsize=7, ncol=4)
    fig.suptitle("2. Subgridding vs the coarse grid: accuracy per iteration and forward accuracy", fontsize=11)
    fig.savefig(ROOT / "claim2_subgrid_vs_coarse.png", dpi=110)
    plt.close(fig)
    # claim 3
    fig = plt.figure(figsize=(4.2 * nm, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, nm, height_ratios=[1.3, 1])
    for j, n in enumerate(names):
        ax = fig.add_subplot(gs[0, j])
        curves(ax, res[n], [k for k in (ak(res[n]), "fine") if k])
        ax.set_title(f"{n}", fontsize=10)
        ax.set_xlabel("compute time [s]")
        if j == 0:
            ax.set_ylabel("held-out PE (lower = better)")
            ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, :])
    ts = sorted({t for (_, t) in speed})
    sv = {(n, f"{t:.0f} s"): (v if v is not None else np.nan) for (n, t), v in speed.items()}
    w = 0.8 / max(1, len(ts))
    cm = plt.get_cmap("Reds")
    for i, t in enumerate(ts):
        ys = [sv.get((n, f"{t:.0f} s"), np.nan) for n in names]
        xs = np.arange(nm) + (i - (len(ts) - 1) / 2) * w
        bs = ax.bar(xs, ys, w * 0.92, color=cm(0.3 + 0.6 * i / max(1, len(ts) - 1)), label=f"PE of dynamic v3 at {t:.0f} s")
        for b_, n in zip(bs, names):
            if (n, t) in lower:
                b_.set_hatch("//")
                b_.set_edgecolor("k")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(range(nm))
    ax.set_xticklabels(names)
    ax.set_ylabel("uniform 0.75 cm time / dynamic time\nto reach the same PE (hatched: lower bound)")
    ax.grid(True, axis="y", color="0.9")
    ax.legend(fontsize=7, ncol=min(5, len(ts)))
    fig.suptitle("3. Subgridding vs uniform full refinement: time to the same accuracy", fontsize=11)
    fig.savefig(ROOT / "claim3_subgrid_vs_uniform.png", dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("data", "run", "report", "check", "claims"))
    p.add_argument("name", nargs="?", default="M1")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "data":
        Context(a.name)
    elif a.cmd == "run":
        run(a.name, a.methods.split(","), smoke=a.smoke)
    elif a.cmd == "report":
        report(a)
    elif a.cmd == "check":
        check(a.name)
    elif a.cmd == "claims":
        claims(a)


def check(name):
    """Forward floors: true-model response error of each solver against the 0.375 cm data."""
    ctx = Context(name)
    te = torch.as_tensor(ctx.true_e, dtype=DTYPE, device=DEV)
    ts = torch.as_tensor(ctx.true_s, dtype=DTYPE, device=DEV)
    resp = ctx.resp[ctx.train] * ctx.mk[ctx.train]
    out = {}
    for kind, boxes in (("coarse", []), ("static_full", ctx.full_boxes()), ("oracle", ctx.oracle_boxes()), ("fine", [])):
        t0 = time.time()
        fn = Data("fine" if kind == "fine" else "main", ctx.src, ctx.rec, ctx.train, boxes=boxes)
        with torch.no_grad():
            d = fn(te, ts).cpu() * ctx.mk[ctx.train]
        out[kind] = float(((d - resp) ** 2).sum() / (resp ** 2).sum())
        print(f"{name} {kind}: true-model response error {out[kind]:.4f} ({time.time() - t0:.0f} s), boxes {boxes}",
              flush=True)
        del fn
        torch.cuda.empty_cache()
    (ROOT / f"check_{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())

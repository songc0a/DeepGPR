r"""Side-effect-free time step of the 3D hybrid DGTD-FDTD scheme.

State (24 tensors)::

    Ex, Ey, Ez, Hx, Hy, Hz                      Yee fields (nshot, NX, NY, NZ)
    psi_e[6], psi_h[6]                          Yee CPML auxiliaries (one per curl term, CPML slabs only)
    E, H                                        DG fields (nshot, 3, K, Np)
    D, B                                        DG UPML auxiliaries (nshot, 3, Kp, Np)
    D_fd, B_fd                                  Yee UPML auxiliaries (nshot, n_upml)

Curl terms are numbered as in :data:`TERMS`: ``(curl F)_c = sum_j sign_j d_a F_f``
over the two terms ``j = (c, a, f, sign)`` of component ``c``. In the Yee grid
every term that lies in a CPML along its derivative axis ``a`` is stretched
exactly as in DeepGPR's ``cpml_e``/``cpml_h``: ``d -> d + (RA - 1) d + RB psi``,
``psi <- RE psi - RF d``.

In the hybrid the side absorbers (y and z faces) are a uniaxial PML in
auxiliary-equation form (Gedney, IEEE Trans. Antennas Propag. 44(12), 1996,
doi:10.1109/8.546249), in the DG band and in the Yee grid, with damping rates
``s_a = sigma_a / eps0`` (DeepGPR's quartic profile) and, per component ``c``
(cyclic ``c+1``, ``c+2``)::

    dD_c/dt + s_(c+1) D_c = (curl_D H)_c
    d(eps E_c)/dt + s_(c+2) eps E_c + sigma E_c = dD_c/dt + s_c D_c

(likewise ``B``/``H`` with ``mu`` and ``-curl_D E``), integrated with the
Crank-Nicolson rule inside the leapfrog step. The DG curl, including its
fluxes, is the same operator as outside the PML. (Stretching the individual DG
derivatives like the Yee CPML is unstable on tetrahedra: the semi-discrete
operator has eigenvalues with positive real part; and a stretched Yee CPML next
to the DG band destabilises the coupled scheme; see docs/HYBRID_DGTD.md.) The
x faces keep DeepGPR's CPML; pure-Yee mode is DeepGPR's scheme.

The Yee part reproduces DeepGPR's native second-order 3D update
(``update_h_cpu``, ``cpml_h_cpu``, ``update_e_cpu``, ``cpml_e_cpu``,
``inject_sources_cpu`` in ``lib/deepgpr_cpu.c``) on full-grid arrays with
per-component activity masks. A DGTD band between the E planes ``i_T`` and
``i_B`` removes the planes strictly between them; on the interface planes the
tangential ``Ey``/``Ez`` and the normal ``Hx`` carry half weight and the
x-derivatives of ``Hy``/``Hz`` use the ghost rows ``2 H_b - H_inner`` with the
tangential DG trace ``H_b`` (summation-by-parts closure, as in 2D).

The step is used for forward modelling, the CFL estimate and (through
reverse-mode differentiation of exactly this function) the discrete adjoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..config.constants import PEC_CONDUCTIVITY_THRESHOLD, VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from .coupling import SparseOp
from .functional import PointOps

#: Curl terms ``(component, derivative axis, differentiated component, sign)``.
TERMS = ((0, 1, 2, 1.0), (0, 2, 1, -1.0), (1, 2, 0, 1.0), (1, 0, 2, -1.0),
         (2, 0, 1, 1.0), (2, 1, 0, -1.0))
STATE_SIZE = 24


@dataclass
class YeeStatic3D:
    spacing: Tuple[float, float, float]
    shape: Tuple[int, int, int]
    act: Tuple[torch.Tensor, ...]          # (ex, ey, ez, hx, hy, hz) bool masks
    chr: torch.Tensor                      # dt / mu on the field grid
    pe: Tuple[Optional[tuple], ...]        # per axis (A, B, E, F) for the E update
    ph: Tuple[Optional[tuple], ...]        # per axis (A, B, E, F) for the H update
    weights: Tuple[torch.Tensor, ...]      # energy weights (ex, ey, ez, hx, hy, hz)
    upml: Optional[dict] = None            # side UPML: compact indices and static factors
    cuda: Optional[object] = None          # :class:`~DeepGPR.hybrid.cuda3d.YeeCudaOps`
    upml_fine: Optional[dict] = None       # the same UPML data for the LTS sub-step dt/p


@dataclass
class DGStatic3D:
    K: int
    Np: int
    Nfp: int
    DT: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]   # Dr^T, Ds^T, Dt^T
    LIFTT: torch.Tensor
    R: torch.Tensor            # (3, 3, K, 1): R[a, q] = d r_q / d x_a
    n: torch.Tensor            # (3, K, 4 Nfp) outward unit normals
    fscale: torch.Tensor       # (K, 4 Nfp)
    vmapM: torch.Tensor
    vmapP: torch.Tensor
    m_int: torch.Tensor        # face-type masks per face (K, 4)
    m_pec: torch.Tensor
    m_abc: torch.Tensor
    m_if: torch.Tensor
    has_abc: bool
    alpha: float
    ch: torch.Tensor           # (K, 1) dt / mu
    neighbour: torch.Tensor    # (K, 4)
    mass: torch.Tensor         # (Np, Np) reference mass matrix
    J: torch.Tensor            # (K, 1)
    slotP: torch.Tensor        # (K * 4 Nfp) mirror face-node slot (own slot on boundaries)
    fmask: torch.Tensor        # (4 Nfp) element-local node of each face slot
    alpha_if: float = 0.0      # interface penalty weight (0 = none)
    pml_idx: Optional[torch.Tensor] = None   # (Kp,) elements with UPML
    upml: Optional[dict] = None   # rates s_c, s_c1, s_c2 and static D/B/H factors (3, Kp, Np)
    cuda: Optional[object] = None   # :class:`~DeepGPR.hybrid.cuda3d.DGCudaOps` when available
    lts: int = 1               # local time stepping: DG sub-steps per Yee step (1 = off)
    lts_w: Optional[Tuple[float, ...]] = None   # quadrature weights of the sub-step curls
    lts_overlap: int = 1       # Yee planes beyond the interface plane in the fine set
    lts_coef: Optional[dict] = None   # :func:`lts_coefficients` (stabilised recursion)
    upml_fine: Optional[dict] = None  # UPML factors for the LTS sub-step dt/p (p-fold update)
    lts_split: bool = False    # LTS with a penalty flux: central recursion + split damping
    lts_pen_k: int = 1         # explicit sub-steps of the split penalty damping per macro step


@dataclass
class Coupling3D:
    row: int                   # Yee E plane of the interface
    ghost: int                 # Hy/Hz row that receives the ghost values
    inner: int                 # Hy/Hz row inside the Yee region next to it
    slots: torch.Tensor        # interface face-node slots in the (K * 4 Nfp) layout
    nodes: torch.Tensor        # flat DG node indices of the same face nodes
    to_dg: Tuple[object, object]   # (Ey plane -> E*_y, Ez plane -> E*_z) at the slots
    to_fd: Tuple[object, object]   # (H_z DG -> Hz_b on Ey nodes, H_y DG -> Hy_b on Ez nodes)


@dataclass
class StepStatic3D:
    nshot: int
    yee: Optional[YeeStatic3D]
    dg: Optional[DGStatic3D]
    couplings: List[Coupling3D] = field(default_factory=list)
    dtype: torch.dtype = torch.float64
    device: torch.device = torch.device("cpu")


@dataclass
class PointOps3D(PointOps):
    comp: int = 2


@dataclass
class StepCoefs3D:
    """Material-dependent coefficients: Yee ``ca``/``cb``, DG ``ca``/``cb`` and the
    face coefficients ``b0..b3`` of the E-flux (``e*``) and H-flux (``h*``)
    (see :func:`flux_coefficients`)."""

    fd_ca: Optional[torch.Tensor] = None
    fd_cb: Optional[torch.Tensor] = None
    dg_ca: Optional[torch.Tensor] = None
    dg_cb: Optional[torch.Tensor] = None
    e0: Optional[torch.Tensor] = None
    e1: Optional[torch.Tensor] = None
    e2: Optional[torch.Tensor] = None
    e3: Optional[torch.Tensor] = None
    h0: Optional[torch.Tensor] = None
    h1: Optional[torch.Tensor] = None
    h2: Optional[torch.Tensor] = None
    h3: Optional[torch.Tensor] = None
    u_hist: Optional[torch.Tensor] = None   # DG UPML E update factors (3, Kp, Np)
    u_new: Optional[torch.Tensor] = None
    u_old: Optional[torch.Tensor] = None
    g_if: Optional[torch.Tensor] = None     # interface penalty gamma = alpha_if Y-/2 (K, 4)
    fu_hist: Optional[torch.Tensor] = None  # Yee UPML E update factors (n_upml,)
    fu_new: Optional[torch.Tensor] = None
    fu_old: Optional[torch.Tensor] = None
    u_hist_f: Optional[torch.Tensor] = None   # the same factors for the LTS sub-step dt/p
    u_new_f: Optional[torch.Tensor] = None
    u_old_f: Optional[torch.Tensor] = None
    fu_hist_f: Optional[torch.Tensor] = None
    fu_new_f: Optional[torch.Tensor] = None
    fu_old_f: Optional[torch.Tensor] = None
    pe1: Optional[torch.Tensor] = None    # LTS split penalty: H-flux jump coefficient (K, 4)
    ph1: Optional[torch.Tensor] = None    # E-flux jump coefficient (K, 4)
    ph3: Optional[torch.Tensor] = None    # E-flux PEC penalty (K, 4)

    NAMES = ("fd_ca", "fd_cb", "dg_ca", "dg_cb", "e0", "e1", "e2", "e3", "h0", "h1", "h2", "h3",
             "g_if", "u_hist", "u_new", "u_old", "fu_hist", "fu_new", "fu_old",
             "u_hist_f", "u_new_f", "u_old_f", "fu_hist_f", "fu_new_f", "fu_old_f",
             "pe1", "ph1", "ph3")

    def tensors(self) -> List[torch.Tensor]:
        return [getattr(self, n) for n in self.NAMES if getattr(self, n) is not None]

    def replaced(self, tensors: Sequence[torch.Tensor]) -> "StepCoefs3D":
        it = iter(tensors)
        return StepCoefs3D(**{n: (next(it) if getattr(self, n) is not None else None)
                              for n in self.NAMES})


# --------------------------------------------------------------------------- static
def _view(axis: int, v: torch.Tensor) -> torch.Tensor:
    shape = [1, 1, 1]
    shape[axis] = -1
    return v.reshape(shape)


def cpml_axis_profiles(n: int, low: int, high: int, c_low, c_high, spacing: float,
                       electric: bool, axis: int, dtype, device):
    """``(A, B, E, F, idx)`` of one axis from DeepGPR's ``(4, 1, p)`` face
    coefficients (``A = (RA - 1)/d``, ``F = RF/d``) on the planes ``idx`` of the
    CPML slabs only (the auxiliaries are stored there); ``None`` without CPML."""
    if low <= 0 and high <= 0:
        return None
    vals = torch.zeros((4, n), dtype=torch.float64)
    rows_all = []
    if low > 0:
        c = c_low[:, 0].detach().cpu().to(torch.float64)
        rows = torch.arange(1, low + 1) if electric else torch.arange(0, low)
        i1 = (low - rows) if electric else (low - 1 - rows)
        vals[:, rows] = c[:, i1]
        vals[0, rows] -= 1.0
        rows_all.append(rows)
    if high > 0:
        c = c_high[:, 0].detach().cpu().to(torch.float64)
        rows = torch.arange(n - 1 - high, n - 1)
        vals[:, rows] = c[:, rows - (n - 1 - high)]
        vals[0, rows] -= 1.0
        rows_all.append(rows)
    idx = torch.unique(torch.cat(rows_all))
    vals = vals[:, idx]
    A, B, E, Fm = vals[0] / spacing, vals[1], vals[2], vals[3] / spacing
    return tuple(_view(axis, t.to(dtype=dtype, device=device)) for t in (A, B, E, Fm)) + (
        idx.to(device),)


def yee_static_3d(shape, spacing, pml, pml_coefficients, chr, band, dtype, device) -> YeeStatic3D:
    """Static Yee data. ``pml_coefficients``: the 12 tensors ``(x01, x02, xm1, xm2,
    y01, ..., zm2)`` of :func:`DeepGPR.solver.pml.build_pml_coeffs` ``[6:]``;
    ``band``: ``None`` or ``(i_T, i_B)`` E planes."""
    NX, NY, NZ = shape
    pe, ph = [], []
    for a in range(3):
        lo, hi = int(pml[2 * a]), int(pml[2 * a + 1])
        c = pml_coefficients[4 * a: 4 * a + 4]
        pe.append(cpml_axis_profiles(shape[a], lo, hi, c[0], c[2], spacing[a], True, a, dtype, device))
        ph.append(cpml_axis_profiles(shape[a], lo, hi, c[1], c[3], spacing[a], False, a, dtype, device))
    inner = lambda n: torch.arange(n) > 0  # noqa: E731
    lower = lambda n: torch.arange(n) < n - 1  # noqa: E731
    both = lambda n: inner(n) & lower(n)  # noqa: E731
    rows_int = both(NX)            # Ey, Ez, Hx planes (integer x)
    rows_half = lower(NX)          # Ex, Hy, Hz planes (x + 1/2)
    w_int = rows_int.to(torch.float64)
    w_half = rows_half.to(torch.float64)
    if band is not None:
        top, bottom = band
        idx = torch.arange(NX)
        rows_int = rows_int & ((idx <= top) | (idx >= bottom))
        rows_half = rows_half & ((idx <= top - 1) | (idx >= bottom))
        w_int = rows_int.to(torch.float64)
        w_int[top] = 0.5
        w_int[bottom] = 0.5
        w_half = rows_half.to(torch.float64)
    m = lambda a, b, c: (a[:, None, None] & b[None, :, None] & c[None, None, :])  # noqa: E731
    act = (m(rows_half, both(NY), both(NZ)), m(rows_int, lower(NY), both(NZ)),
           m(rows_int, both(NY), lower(NZ)), m(rows_int, lower(NY), lower(NZ)),
           m(rows_half, both(NY), lower(NZ)), m(rows_half, lower(NY), both(NZ)))
    act = tuple(t.to(device) for t in act)
    rw = (w_half, w_int, w_int, w_int, w_half, w_half)
    weights = tuple((a.to(torch.float64) * w.to(device)[:, None, None]).to(dtype)
                    for a, w in zip(act, rw))
    return YeeStatic3D(tuple(float(s) for s in spacing), (NX, NY, NZ), act, chr.to(dtype),
                       tuple(pe), tuple(ph), weights)


def yee_upml_3d(y: YeeStatic3D, sigma_fn, mu_r_pad, dt, dtype, device) -> dict:
    """Compact side-UPML data of the Yee grid.

    ``sigma_fn(axis, coords)`` returns the UPML conductivity [S/m] along ``axis``
    (1 = y, 2 = z) at extended coordinates; every component uses its staggered
    position. Per component the active nodes with a nonzero rate form a flat
    index set; E uses ``(s_c, s_c2)`` (material factors are built per step from
    ``eps``/``sigma``), H the static factors."""
    NX, NY, NZ = y.shape
    sp = y.spacing
    mu = (VACUUM_PERMEABILITY * mu_r_pad.detach().to(torch.float64).cpu()).reshape(-1)
    out = {"e": [], "h": [], "offsets_e": [0], "offsets_h": [0]}
    for kind in ("e", "h"):
        for c in range(3):
            # staggered offsets (in cells) of component c along (x, y, z)
            off = [0.0, 0.0, 0.0]
            if kind == "e":
                off[c] = 0.5
            else:
                off = [0.5, 0.5, 0.5]
                off[c] = 0.0
            ys = (torch.arange(NY, dtype=torch.float64) + off[1]) * sp[1]
            zs = (torch.arange(NZ, dtype=torch.float64) + off[2]) * sp[2]
            rate = [torch.zeros((NY, NZ), dtype=torch.float64),
                    torch.as_tensor(sigma_fn(1, ys.numpy()))[:, None].expand(NY, NZ) / VACUUM_PERMITTIVITY,
                    torch.as_tensor(sigma_fn(2, zs.numpy()))[None, :].expand(NY, NZ) / VACUUM_PERMITTIVITY]
            active = y.act[c if kind == "e" else 3 + c].cpu()
            hot = ((rate[1] > 0) | (rate[2] > 0))[None].expand(NX, NY, NZ) & active
            idx = torch.nonzero(hot.reshape(-1)).reshape(-1)
            jk = idx % (NY * NZ)
            r = [rt.reshape(-1)[jk] for rt in rate]
            s_c, s_c1, s_c2 = r[c], r[(c + 1) % 3], r[(c + 2) % 3]
            q1 = 0.5 * dt * s_c1
            ent = dict(idx=idx, a1=(1 - q1) / (1 + q1), b1=dt / (1 + q1), s_c=s_c, s_c2=s_c2)
            if kind == "h":
                m = mu[idx]
                den = m / dt + 0.5 * m * s_c2
                ent.update(hist=(m / dt - 0.5 * m * s_c2) / den, new=(1 / dt + 0.5 * s_c) / den,
                           old=(1 / dt - 0.5 * s_c) / den)
            ent = {k: (v.to(device) if k == "idx" else v.to(dtype=dtype, device=device))
                   for k, v in ent.items()}
            out[kind].append(ent)
            out["offsets_" + kind].append(out["offsets_" + kind][-1] + idx.numel())
    return out


def lts_coefficients(p: int, nu: float = 0.0) -> dict:
    r"""Coefficients of the (stabilised) leapfrog local time stepping with ``p``
    sub-steps and stabilisation ``nu`` (Grote, Michel & Sauter, Math. Comp. 90,
    2021, doi:10.1090/mcom/3650; ``nu = 0``: Diaz & Grote 2009).

    With the Chebyshev polynomials ``T_k`` at ``delta = 1 + nu / p^2``,
    ``omega = 2 T_p'(delta) / T_p(delta)``, ``beta_k = T_(k-1)(delta) / T_(k+1)(delta)``
    and ``beta_(k+1/2) = T_k(delta) / T_(k+1)(delta)``, the inner recursion on the
    fine unknowns (``tau = dt / p``, ``r`` the H rate, ``q`` the E rate) is ::

        eta_0 = -tau c_0 r(z_0),                     c_0 = p^2 / (omega delta)
        z_(k+1) = z_k + tau q(eta_k)
        eta_(k+1) = beta_(k+1) eta_k - tau c_(k+1) r(z_(k+1)),   c_k = (2 p^2 / omega) beta_(k+1/2)

    and ``H^(n+1/2) = H^(n-1/2) + (2/p) sum_k eta_k = H^(n-1/2) - dt sum_j w_j r(z_j)``.
    Returns ``{"w": (p,), "c0": float, "c": (p,), "beta": (p,)}`` (``c[0]``, ``beta[0]``
    unused). For ``nu = 0``: ``c_0 = 1/2``, ``c_k = beta_k = 1``, ``w_0 = 1/p``,
    ``w_j = 2 (p - j) / p^2``.
    """
    p = int(p)
    if p < 1:
        raise ValueError("The number of local sub-steps must be a positive integer.")
    nu = float(nu)
    if nu < 0:
        raise ValueError("The stabilisation parameter must be non-negative.")
    import math as _m
    delta = 1.0 + nu / p ** 2
    theta = _m.acosh(delta)
    T = lambda k: _m.cosh(k * theta)  # noqa: E731
    if theta > 0:
        omega = 2.0 * p * _m.sinh(p * theta) / (_m.sinh(theta) * _m.cosh(p * theta))
    else:
        omega = 2.0 * p ** 2
    c0 = p ** 2 / (omega * delta)
    c = [0.0] + [(2.0 * p ** 2 / omega) * T(k) / T(k + 1) for k in range(1, p)]
    beta = [0.0] + [T(k - 1) / T(k + 1) for k in range(1, p)]
    # w_j = (2 / p^2) sum_(k >= j) A_kj with A_jj = c_j (c_0 for j = 0), A_kj = beta_k A_(k-1)j
    w = []
    for j in range(p):
        a = c0 if j == 0 else c[j]
        total = a
        for k in range(j + 1, p):
            a = beta[k] * a
            total += a
        w.append(2.0 * total / p ** 2)
    return {"w": tuple(w), "c0": c0, "c": tuple(c), "beta": tuple(beta), "nu": nu}


def lts_weights(p: int, nu: float = 0.0) -> Tuple[float, ...]:
    """Weights ``w_j`` of the sub-step curls (see :func:`lts_coefficients`)."""
    return lts_coefficients(p, nu)["w"]


def dg_static_3d(geometry, face_tags, mu_r, dt, alpha, pml_sigma, dtype, device,
                 alpha_if: float = 0.0, lts: int = 1, lts_overlap: int = 1,
                 lts_nu: float = 0.0) -> DGStatic3D:
    """Static DG operators; ``pml_sigma``: per axis ``(K, Np)`` UPML conductivity
    profiles [S/m] (``None`` entries or ``None`` for no PML); ``alpha_if``: weight
    of the dissipative DG-Yee interface penalty; ``lts``: number of local DG
    sub-steps per Yee step (:func:`dg_h_update`)."""
    import numpy as np

    from .dgtd3d import FACE_ABC, FACE_INTERFACE, FACE_INTERIOR, FACE_PEC

    g, ref = geometry, geometry.ref
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)  # noqa: E731
    ti = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long, device=device)  # noqa: E731
    K, Np, Nfp = g.K, ref.np_, ref.nfp
    rep = lambda a: np.repeat(a, Nfp, axis=1)  # noqa: E731
    tags = np.asarray(face_tags)
    if np.any(tags[~g.boundary] != FACE_INTERIOR) or np.any(tags[g.boundary] == FACE_INTERIOR):
        raise ValueError("Face tags do not match the mesh boundary.")
    tagf = rep(tags)
    R = np.transpose(g.rst_x, (2, 1, 0))[..., None]     # (a, q, K, 1)
    n = np.stack([rep(g.normals[:, :, a]) for a in range(3)])
    if np.any(tags == FACE_INTERFACE):
        nif = n[:, tagf == FACE_INTERFACE]
        if np.max(np.abs(nif[1:])) > 1e-9:
            raise ValueError("Interface faces must be horizontal (normal along x).")
    mu = VACUUM_PERMEABILITY * torch.as_tensor(mu_r, dtype=torch.float64)
    pml_idx, upml, upml_fine = None, None, None
    if pml_sigma is not None and any(s is not None for s in pml_sigma):
        rate = [np.zeros((K, Np)) if s is None else np.asarray(s) / VACUUM_PERMITTIVITY
                for s in pml_sigma]
        active = np.nonzero(np.any(np.stack(rate) > 0, axis=(0, 2)))[0]
        if active.size:
            pml_idx = ti(active)
            r = [v[active] for v in rate]
            s_c = np.stack([r[c] for c in range(3)])
            s_c1 = np.stack([r[(c + 1) % 3] for c in range(3)])
            s_c2 = np.stack([r[(c + 2) % 3] for c in range(3)])
            mu_p = (mu.numpy()[active])[None, :, None]

            def upml_dict(dt_):
                q1 = 0.5 * dt_ * s_c1
                den = mu_p / dt_ + 0.5 * mu_p * s_c2
                return dict(s_c=t(s_c), s_c2=t(s_c2), a1=t((1 - q1) / (1 + q1)),
                            b1=t(dt_ / (1 + q1)), h_hist=t((mu_p / dt_ - 0.5 * mu_p * s_c2) / den),
                            h_new=t((1 / dt_ + 0.5 * s_c) / den), h_old=t((1 / dt_ - 0.5 * s_c) / den))

            upml = upml_dict(dt)
            if int(lts) > 1:
                upml_fine = upml_dict(dt / int(lts))
    # Mirror slots: the neighbour's face slot holding the same node.
    NF = 4 * Nfp
    slot = np.arange(K * NF).reshape(K, 4, Nfp)
    cand = (g.EToE[:, :, None] * NF + g.EToF[:, :, None] * Nfp)[..., None] + np.arange(Nfp)
    vm = g.vmapM.reshape(-1)
    hit = vm[cand] == g.vmapP[..., None]                     # (K, 4, Nfp, Nfp)
    slotP = np.where(g.boundary[:, :, None], slot,
                     np.take_along_axis(cand, np.argmax(hit, axis=-1)[..., None], -1)[..., 0])
    if not np.all(hit.any(-1) | g.boundary[:, :, None]):
        raise RuntimeError("Mirror face slots not found.")
    ch = (dt / mu)[:, None]
    return DGStatic3D(
        K, Np, Nfp, (t(ref.Dr.T), t(ref.Ds.T), t(ref.Dt.T)), t(ref.LIFT.T), t(R), t(n),
        t(rep(g.sJ / g.J[:, None])), ti(g.vmapM.reshape(-1)), ti(g.vmapP.reshape(-1)),
        t(tags == FACE_INTERIOR), t(tags == FACE_PEC), t(tags == FACE_ABC),
        t(tags == FACE_INTERFACE), bool(np.any(tags == FACE_ABC)), float(alpha),
        ch.to(dtype=dtype, device=device), ti(g.EToE), t(ref.M), t(g.J[:, None]),
        ti(slotP.reshape(-1)), ti(ref.fmask.reshape(-1)), float(alpha_if), pml_idx, upml,
        lts=int(lts), lts_w=lts_weights(lts, lts_nu), lts_overlap=int(lts_overlap),
        lts_coef=lts_coefficients(lts, lts_nu), upml_fine=upml_fine,
        lts_split=int(lts) > 1 and float(alpha) > 0)


# --------------------------------------------------------------------------- coefficients
def yee_coefficients_3d(eps_r_pad, sigma_pad, dt):
    e_term = VACUUM_PERMITTIVITY * eps_r_pad / dt
    s_term = 0.5 * sigma_pad
    ea = e_term + s_term
    ca, cb = (e_term - s_term) / ea, 1.0 / ea
    pec = (sigma_pad > PEC_CONDUCTIVITY_THRESHOLD).detach()
    zero = torch.zeros_like(ca)
    return torch.where(pec, zero, ca), torch.where(pec, zero, cb)


def flux_coefficients(ds: DGStatic3D, eps_r, mu_r, alpha: Optional[float] = None,
                      alpha_if: Optional[float] = None):
    """Face coefficients of the E- and H-fluxes from element materials (``alpha``/
    ``alpha_if`` override the static penalty weights; ``0`` gives the central,
    energy-conserving fluxes used inside the local time stepping).

    ``dF = b0 [F] + b1 (n x [G]) + b2 F- + b3 (n x G-) + X`` with ``b2`` split
    into the x and the y/z components (``(2, K, 4)``); every coefficient is
    constant on a face (``(K, 4)``)::

        E-flux: b0 = m_int (1/2 + alpha (Y+/(Y- + Y+) - 1/2)), b1 =  m_int alpha/(Y- + Y+),
                b2 = -m_pec - m_abc/2 (- m_if for y/z),          b3 = -m_abc/(2 Y-)
        H-flux: b0 = m_int (1/2 + alpha (Z+/(Z- + Z+) - 1/2)), b1 = -m_int alpha/(Z- + Z+),
                b2 = -m_abc/2,                                   b3 = m_pec alpha/Z- + m_abc/(2 Z-) + g

    with the interface penalty ``g = m_if alpha_if Y-/2`` (returned as the third
    item): on interface faces ``H* - H- = -g n x (P E_FD - E-)`` (the ``P E_FD``
    part enters as ``X``), matched on the Yee side (see :func:`step`).
    """
    Y = torch.sqrt(VACUUM_PERMITTIVITY * eps_r / (VACUUM_PERMEABILITY * mu_r))
    nf = ds.neighbour.shape[1]
    Ym = Y[:, None].expand(-1, nf)          # per face (affine elements: constant on a face)
    Yp = Y[ds.neighbour]
    Zm, Zp = 1.0 / Ym, 1.0 / Yp
    a = ds.alpha if alpha is None else float(alpha)
    a_if = ds.alpha_if if alpha_if is None else float(alpha_if)
    mi, mp, ma, mf = ds.m_int, ds.m_pec, ds.m_abc, ds.m_if
    e0 = mi * (0.5 + a * (Yp / (Ym + Yp) - 0.5))
    e1 = mi * a / (Ym + Yp)
    e2x = -mp - 0.5 * ma
    e2 = torch.stack([e2x, e2x - mf])
    e3 = -ma / (2.0 * Ym)
    h0 = mi * (0.5 + a * (Zp / (Zm + Zp) - 0.5))
    h1 = -mi * a / (Zm + Zp)
    h2x = -0.5 * ma
    h2 = torch.stack([h2x, h2x])
    g = mf * (0.5 * a_if) * Ym
    h3 = mp * a / Zm + ma / (2.0 * Zm) + g
    return (e0, e1, e2, e3), (h0, h1, h2, h3), g


def dg_coefficients_3d(ds: DGStatic3D, eps_r, sigma, mu_r, dt):
    eps = VACUUM_PERMITTIVITY * eps_r
    ea = eps / dt + 0.5 * sigma
    ca = ((eps / dt - 0.5 * sigma) / ea)[:, None]
    cb = (1.0 / ea)[:, None]
    return (ca, cb) + flux_coefficients(ds, eps_r, mu_r)


def upml_coefficients(ds: DGStatic3D, eps_r, sigma, dt):
    """Material-dependent UPML E-update factors ``(hist, new, old)`` ``(3, Kp, Np)``."""
    u = ds.upml
    idx = ds.pml_idx
    eps = (VACUUM_PERMITTIVITY * eps_r[idx])[None, :, None]
    sig = sigma[idx][None, :, None]
    den = eps / dt + 0.5 * eps * u["s_c2"] + 0.5 * sig
    return ((eps / dt - 0.5 * eps * u["s_c2"] - 0.5 * sig) / den,
            (1.0 / dt + 0.5 * u["s_c"]) / den, (1.0 / dt - 0.5 * u["s_c"]) / den)


def yee_upml_coefficients(u, eps_r_pad, sigma_pad, dt):
    """Concatenated Yee UPML E-update factors ``(hist, new, old)`` (material dependent)."""
    parts = [[], [], []]
    for ent in u["e"]:
        eps = VACUUM_PERMITTIVITY * eps_r_pad.reshape(-1)[ent["idx"]]
        sig = sigma_pad.reshape(-1)[ent["idx"]]
        den = eps / dt + 0.5 * eps * ent["s_c2"] + 0.5 * sig
        parts[0].append((eps / dt - 0.5 * eps * ent["s_c2"] - 0.5 * sig) / den)
        parts[1].append((1.0 / dt + 0.5 * ent["s_c"]) / den)
        parts[2].append((1.0 / dt - 0.5 * ent["s_c"]) / den)
    return tuple(torch.cat(p) for p in parts)


def build_coefficients(st: StepStatic3D, fields, dt) -> StepCoefs3D:
    """Coefficients from ``(eps_fd_pad, sig_fd_pad, dg_eps, dg_sig, dg_mu)``."""
    eps_fd, sig_fd, dg_eps, dg_sig, dg_mu = fields
    c = StepCoefs3D()
    if st.yee is not None:
        c.fd_ca, c.fd_cb = yee_coefficients_3d(eps_fd, sig_fd, dt)
        if st.yee.upml is not None:
            c.fu_hist, c.fu_new, c.fu_old = yee_upml_coefficients(st.yee.upml, eps_fd, sig_fd, dt)
    if st.dg is not None:
        c.dg_ca, c.dg_cb, (c.e0, c.e1, c.e2, c.e3), (c.h0, c.h1, c.h2, c.h3), g = \
            dg_coefficients_3d(st.dg, dg_eps, dg_sig, dg_mu, dt)
        if st.dg.alpha_if > 0 and st.couplings:
            c.g_if = g
        if st.dg.pml_idx is not None:
            c.u_hist, c.u_new, c.u_old = upml_coefficients(st.dg, dg_eps, dg_sig, dt)
        p = int(getattr(st.dg, "lts", 1))
        if p > 1 and getattr(st.dg, "lts_split", False):
            # penalty flux with LTS: the recursion and the updates use the central
            # fluxes, the dissipative jump terms act as a split damping (lts_damp)
            ce, chh, _ = flux_coefficients(st.dg, dg_eps, dg_mu, alpha=0.0)
            fe_, fh_, _ = flux_coefficients(st.dg, dg_eps, dg_mu)
            (c.e0, c.e1, c.e2, c.e3), (c.h0, c.h1, c.h2, c.h3) = ce, chh
            keep = lts_penalty_faces(st.dg)
            c.pe1 = (fe_[1] - ce[1]) * keep
            c.ph1 = (fh_[1] - chh[1]) * keep
            c.ph3 = (fh_[3] - chh[3]) * keep
        if p > 1:
            # the UPML of the fine set is integrated with p sub-steps dt/p per macro step
            if st.dg.pml_idx is not None:
                c.u_hist_f, c.u_new_f, c.u_old_f = upml_coefficients(st.dg, dg_eps, dg_sig, dt / p)
            if st.yee is not None and st.yee.upml_fine is not None:
                c.fu_hist_f, c.fu_new_f, c.fu_old_f = yee_upml_coefficients(
                    st.yee.upml_fine, eps_fd, sig_fd, dt / p)
    return c


# --------------------------------------------------------------------------- state
def zero_state(st: StepStatic3D) -> Tuple[torch.Tensor, ...]:
    S = st.nshot
    kw = dict(dtype=st.dtype, device=st.device)
    empty = torch.zeros(0, **kw)
    out: List[torch.Tensor] = []
    y = st.yee
    if y is not None:
        out += [torch.zeros((S,) + y.shape, **kw) for _ in range(6)]
        for prof in (y.pe, y.ph):
            for (_, a, _, _) in TERMS:
                if prof[a] is None:
                    out.append(empty)
                else:
                    shape = list((S,) + y.shape)
                    shape[a + 1] = prof[a][4].numel()          # CPML slab planes only
                    out.append(torch.zeros(shape, **kw))
    else:
        out += [empty] * 18
    d = st.dg
    if d is not None:
        out += [torch.zeros((S, 3, d.K, d.Np), **kw) for _ in range(2)]
        if d.pml_idx is not None:
            out += [torch.zeros((S, 3, d.pml_idx.numel(), d.Np), **kw) for _ in range(2)]
        else:
            out += [empty, empty]
    else:
        out += [empty] * 4
    if y is not None and y.upml is not None:
        out += [torch.zeros((S, y.upml["offsets_e"][-1]), **kw),
                torch.zeros((S, y.upml["offsets_h"][-1]), **kw)]
    else:
        out += [empty, empty]
    return tuple(out)


# --------------------------------------------------------------------------- Yee
def _fwd(f, axis):
    """``f[i+1] - f[i]`` along spatial ``axis`` (0..2), zero in the last plane."""
    dim = axis + 1
    n = f.shape[dim]
    d = f.narrow(dim, 1, n - 1) - f.narrow(dim, 0, n - 1)
    pad = [0, 0] * (3 - axis)
    pad[-1] = 1
    return F.pad(d, pad)


def _bwd(f, axis):
    """``f[i] - f[i-1]`` along spatial ``axis``, zero in the first plane."""
    dim = axis + 1
    n = f.shape[dim]
    d = f.narrow(dim, 1, n - 1) - f.narrow(dim, 0, n - 1)
    pad = [0, 0] * (3 - axis)
    pad[-2] = 1
    return F.pad(d, pad)


def _stretch(d, spacing, prof, psi, axis):
    """Stretched difference ``d/h + (RA - 1) d/h + RB psi`` on the CPML slabs."""
    if prof is None:
        return d / spacing, psi
    A, B, E, Fm, idx = prof
    dim = axis + 1
    dp = d.index_select(dim, idx)
    return (d / spacing).index_add(dim, idx, A * dp + B * psi), E * psi - Fm * dp


def _yee_curl(y: YeeStatic3D, fields, profiles, psis, diff):
    rhs = [0.0, 0.0, 0.0]
    new_psi = list(psis)
    for j, (c, a, f, sign) in enumerate(TERMS):
        s, new_psi[j] = _stretch(diff(fields[f], a), y.spacing[a], profiles[a], psis[j], a)
        rhs[c] = rhs[c] + sign * s
    return rhs, new_psi


# --------------------------------------------------------------------------- DG
def _traces(d: DGStatic3D, f):
    S = f.shape[0]
    flat = f.reshape(S, 3, -1)
    shape = (S, 3, d.K, 4 * d.Nfp)
    return flat[:, :, d.vmapM].view(shape), flat[:, :, d.vmapP].view(shape)


def _face_nodes(d: DGStatic3D, t):
    """Per-face values ``(..., K, 4)`` -> per face node ``(..., K, 4 Nfp)``."""
    # expand + reshape, not repeat_interleave (which synchronises with the device)
    return t.unsqueeze(-1).expand(*t.shape, d.Nfp).reshape(*t.shape[:-1], -1)


def _cross(n, v):
    """``n x v`` with ``n (3, K, L)`` and ``v (S, 3, K, L)``."""
    return torch.stack([n[1] * v[:, 2] - n[2] * v[:, 1],
                        n[2] * v[:, 0] - n[0] * v[:, 2],
                        n[0] * v[:, 1] - n[1] * v[:, 0]], dim=1)


def dg_flux(d: DGStatic3D, b, F, G, X=None):
    """``F* - F-`` at the face nodes: ``b0 [F] + b1 (n x [G]) + b2 F- + b3 (n x G-) + X``."""
    b0, b1, b2, b3 = (_face_nodes(d, t) for t in b)
    Fm, Fp = _traces(d, F)
    Gm, Gp = _traces(d, G)
    out = b0 * (Fp - Fm) + b1 * _cross(d.n, Gp - Gm) + b3 * _cross(d.n, Gm)
    out = out + torch.cat([b2[0] * Fm[:, :1], b2[1] * Fm[:, 1:]], dim=1)
    if X is not None:
        out = out + torch.cat([torch.zeros_like(Fm[:, :1]), X], dim=1)
    return out


def dg_terms(d: DGStatic3D, f, df):
    """The six DG derivatives ``D_a f_c`` of :data:`TERMS` (``df = f* - f-``)."""
    drst = [torch.matmul(f, DT) for DT in d.DT]               # (S, 3, K, Np)
    out = []
    for (_, a, comp, _) in TERMS:
        grad = d.R[a, 0] * drst[0][:, comp] + d.R[a, 1] * drst[1][:, comp] + d.R[a, 2] * drst[2][:, comp]
        out.append(grad + torch.matmul(d.fscale * d.n[a] * df[:, comp], d.LIFTT))
    return out


def dg_operator(d: DGStatic3D, b, F, G, X=None):
    """The six DG curl terms ``(S, 6, K, Np)`` of ``F`` with the flux ``b`` (fused CUDA
    kernel with an exact transpose when available, else PyTorch; same formulas)."""
    if d.cuda is not None and F.is_cuda:
        return d.cuda.terms(F, G, X, *b)
    return torch.stack(dg_terms(d, F, dg_flux(d, b, F, G, X)), dim=1)


def _dg_curl(T):
    """``curl_D`` from the six terms (TERMS alternate + / -)."""
    return T[:, 0::2] - T[:, 1::2]


def _upml(idx, aux, curl, f_old, f_new, a1, b1, hist, new, old):
    """UPML update of ``f`` in the PML elements: ``aux <- a1 aux + b1 curl`` and
    ``f = hist f_old + new aux_new - old aux_old`` (replaces the plain update)."""
    aux_new = a1 * aux + b1 * curl.index_select(2, idx)
    fp = hist * f_old.index_select(2, idx) + new * aux_new - old * aux
    return f_new.index_copy(2, idx, fp), aux_new


def _yee_upml(u, kind, aux, rhs, f_old, f_new, coefs=None):
    """Side-UPML update of the Yee components (compact index sets per component)."""
    S = aux.shape[0]
    off = u["offsets_" + kind]
    new_aux = []
    out = []
    for c, ent in enumerate(u[kind]):
        a, b = off[c], off[c + 1]
        if b == a:
            out.append(f_new[c])
            new_aux.append(aux[:, a:b])
            continue
        idx = ent["idx"]
        r = rhs[c].reshape(S, -1)[:, idx]
        old = aux[:, a:b]
        if kind == "e":
            nw = ent["a1"] * old + ent["b1"] * r
            hist, new, oldf = coefs[0][a:b], coefs[1][a:b], coefs[2][a:b]
        else:
            nw = ent["a1"] * old - ent["b1"] * r
            hist, new, oldf = ent["hist"], ent["new"], ent["old"]
        val = hist * f_old[c].reshape(S, -1)[:, idx] + new * nw - oldf * old
        shape = f_new[c].shape
        out.append(f_new[c].reshape(S, -1).index_copy(1, idx, val).view(shape))
        new_aux.append(nw)
    return out, torch.cat(new_aux, dim=1)


def trace_op(st: StepStatic3D) -> SparseOp:
    """The mortar products ``to_dg`` of all couplings and both components as one
    :class:`~DeepGPR.hybrid.coupling.SparseOp`: input the interface planes stacked
    ``(S, ncp, 2, NY NZ)`` in the order (coupling, (y, z)), output the compact trace
    ``(S, 2 nX)`` (``(S, 2, nX)`` in coupling order); cached on the static."""
    def make():
        NYZ = st.yee.shape[1] * st.yee.shape[2]
        nX = sum(int(cp.slots.numel()) for cp in st.couplings)
        rows, cols, vals, off = [], [], [], 0
        for i, cp in enumerate(st.couplings):
            for comp in (0, 1):
                op = cp.to_dg[comp]
                rows.append(op.rows + comp * nX + off)
                cols.append(op.cols + (i * 2 + comp) * NYZ)
                vals.append(op.vals)
            off += int(cp.slots.numel())
        return SparseOp(torch.cat(rows), torch.cat(cols), torch.cat(vals),
                        (2 * nX, 2 * len(st.couplings) * NYZ), vals[0].dtype, vals[0].device)
    return _cached(st, "trace_op", make)


def ghost_op(st: StepStatic3D) -> SparseOp:
    """The mortar products ``to_fd`` of all couplings as one :class:`SparseOp`: input
    the flat DG H-like tensor ``(S, 3 K Np)``, output ``(S, ncp, 2, NY NZ)`` with
    (``Hz`` on the Ey nodes, ``Hy`` on the Ez nodes) per coupling; cached."""
    def make():
        d = st.dg
        KN, NYZ = d.K * d.Np, st.yee.shape[1] * st.yee.shape[2]
        rows, cols, vals = [], [], []
        for i, cp in enumerate(st.couplings):
            for j, comp in ((0, 2), (1, 1)):           # to_fd[0]: DG Hz, to_fd[1]: DG Hy
                op = cp.to_fd[j]
                rows.append(op.rows + (i * 2 + j) * NYZ)
                cols.append(comp * KN + cp.nodes.index_select(0, op.cols))
                vals.append(op.vals)
        return SparseOp(torch.cat(rows), torch.cat(cols), torch.cat(vals),
                        (2 * len(st.couplings) * NYZ, 3 * KN), vals[0].dtype, vals[0].device)
    return _cached(st, "ghost_op", make)


def interface_trace(st: StepStatic3D, fe, compact: bool):
    """``E* = P fe|rows`` (projected Yee trace, y and z) at the interface face slots:
    compact ``(S, 2, nX)`` in coupling order for the fused kernels, otherwise
    dense ``(S, 2, K, 4 Nfp)`` (zero outside the interface)."""
    d, S = st.dg, st.nshot
    if compact:
        planes = torch.stack([fe[1 + comp][:, cp.row].reshape(S, -1)
                              for cp in st.couplings for comp in (0, 1)], dim=1)
        return trace_op(st)(planes.reshape(S, -1)).view(S, 2, -1)
    ys = [cp.to_dg[0](fe[1][:, cp.row].reshape(S, -1)) for cp in st.couplings]
    zs = [cp.to_dg[1](fe[2][:, cp.row].reshape(S, -1)) for cp in st.couplings]
    L = d.K * 4 * d.Nfp
    ey = torch.zeros((S, L), dtype=fe[1].dtype, device=fe[1].device)
    ez = torch.zeros_like(ey)
    for cp, a, b in zip(st.couplings, ys, zs):
        ey = ey.index_copy(1, cp.slots, a)
        ez = ez.index_copy(1, cp.slots, b)
    return torch.stack([ey, ez], dim=1).view(S, 2, d.K, 4 * d.Nfp)


def interface_slots(st: StepStatic3D):
    """``(face, n_x)`` of the interface face slots in coupling order (cached)."""
    def make():
        d = st.dg
        slots = torch.cat([cp.slots for cp in st.couplings])
        return (torch.div(slots, d.Nfp, rounding_mode="floor"),
                d.n[0].reshape(-1).index_select(0, slots))
    return _cached(st, "xslots", make)


def interface_x(st: StepStatic3D, c: StepCoefs3D, Eext, compact: bool):
    """External input of the H-flux, ``X_h = g n_x (E*_z, -E*_y)`` (layout of ``Eext``)."""
    if compact:
        face, n0 = interface_slots(st)
        gn = c.g_if.reshape(-1).index_select(0, face) * n0
    else:
        gn = _face_nodes(st.dg, c.g_if) * st.dg.n[0]
    return torch.stack([gn * Eext[:, 1], -gn * Eext[:, 0]], dim=1)


def _cached(st: StepStatic3D, key, make):
    """Small index tensors cached on the static, so that the step issues no
    host-to-device copy (which would synchronise every step)."""
    cache = st.__dict__.setdefault("_index_cache", {})
    if key not in cache:
        cache[key] = make()
    return cache[key]


def interface_penalty_rows(st: StepStatic3D, c: StepCoefs3D, fe, E):
    """Yee side of the interface penalty, ``(S, n_rows, 2, NY, NZ)`` for the y/z
    components of the interface planes: ``(2/dx) [C^T(g E-_c)/dA - C^T(g)/dA E_c]``."""
    y, S = st.yee, st.nshot
    NY, NZ = y.shape[1], y.shape[2]
    scale = 2.0 / y.spacing[0]
    Ef = E.reshape(S, 3, -1)
    g_all = c.g_if.reshape(-1)                      # per face (K * 4)
    faces = _cached(st, "faces", lambda: [torch.div(cp.slots, st.dg.Nfp, rounding_mode="floor")
                                          for cp in st.couplings])
    rows = []
    for cp, fc in zip(st.couplings, faces):
        g = g_all.index_select(0, fc)
        comps = []
        for comp, lat in ((1, 0), (2, 1)):
            em = Ef[:, comp].index_select(1, cp.nodes)
            pen = cp.to_fd[lat](g * em) - cp.to_fd[lat](g[None]) * fe[comp][:, cp.row].reshape(S, -1)
            comps.append(scale * pen.view(S, NY, NZ))
        rows.append(torch.stack(comps, dim=1))
    return torch.stack(rows, dim=1)


def _yee_interface_penalty(st: StepStatic3D, c: StepCoefs3D, rhs, fe, E):
    """Add :func:`interface_penalty_rows` to the E-update right-hand sides."""
    pen = interface_penalty_rows(st, c, fe, E)
    rows = _cached(st, "rows", lambda: torch.as_tensor([cp.row for cp in st.couplings],
                                                       device=E.device))
    rhs = list(rhs)
    for comp in (1, 2):
        rhs[comp] = rhs[comp].index_add(1, rows, pen[:, :, comp - 1])
    return rhs


# --------------------------------------------------------------------------- DG H update
def _curl_terms_h(d: DGStatic3D, c: StepCoefs3D, E, H, Eext):
    """The six curl terms of the H update (``E`` with the frozen penalty partner ``H``
    and the frozen projected Yee trace ``Eext``)."""
    return dg_operator(d, (c.e0, c.e1, c.e2, c.e3), E, H, Eext)


def dg_h_update(d: DGStatic3D, c: StepCoefs3D, E, H, Baux, Eext, T=None):
    """``H^(n+1/2)`` (and the UPML auxiliary) of the DG band from the curl terms ``T``
    of ``E^n`` (computed here when not given): ``H - (dt/mu) curl_D E`` plus the UPML
    recursion in the PML elements."""
    if T is None:
        T = _curl_terms_h(d, c, E, H, Eext)
    if d.cuda is not None and E.is_cuda:
        return d.cuda.update_h(T, H, Baux)
    curl = _dg_curl(T)
    H_new = H - d.ch * curl
    if d.pml_idx is not None:
        u = d.upml
        H_new, Baux = _upml(d.pml_idx, Baux, -curl, H, H_new, u["a1"], u["b1"],
                            u["h_hist"], u["h_new"], u["h_old"])
    return H_new, Baux


# --------------------------------------------------------------------------- local time stepping
def _pen_coefs(c: StepCoefs3D, kind: str):
    """Face coefficients ``(b0, b1, b2, b3)`` of the dissipative part of the flux: the
    jump penalty on ``H`` (``kind="h"``, H-flux) or on ``E`` with the PEC penalty (``"e"``)."""
    z, z2 = torch.zeros_like(c.e0), torch.zeros_like(c.e2)
    return (z, c.pe1, z2, z) if kind == "h" else (z, c.ph1, z2, c.ph3)


def lts_penalty_faces(d: DGStatic3D) -> torch.Tensor:
    """``(K, 4)`` mask of the faces that carry the split penalty: every face except
    those of a side-UPML element or of its neighbour (the damping inside the UPML,
    whose own update is a damped recursion, destabilises the scheme for penalty
    weights above about 0.1; the UPML absorbs the grid-scale modes there)."""
    keep = torch.ones(d.neighbour.shape, dtype=d.ch.dtype, device=d.ch.device)
    if d.pml_idx is not None:
        pm = torch.zeros(d.K, dtype=torch.bool, device=d.ch.device)
        pm[d.pml_idx] = True
        keep = keep * (~(pm[:, None] | pm[d.neighbour])).to(keep.dtype)
    return keep


def lts_penalty_rate(d: DGStatic3D, c: StepCoefs3D, F, kind: str):
    """``B F`` of the split penalty damping (positive semi-definite in the energy inner
    product): ``(dt/mu) curl P_H(H)`` for ``kind="h"``, ``-(dt/eps) curl P_E(E)`` for
    ``"e"`` (``P``: the jump terms of the penalty flux, ``dt`` the macro step)."""
    curl = _dg_curl(dg_operator(d, _pen_coefs(c, kind), torch.zeros_like(F), F, None))
    if kind == "h":
        return d.ch[None, None] * curl
    cb0 = 2.0 * c.dg_cb / (1.0 + c.dg_ca)                     # (K, 1): dt/eps
    return -cb0[None, None] * curl


def lts_damp(d: DGStatic3D, c: StepCoefs3D, F_old, F_cons, kind: str):
    r"""Split penalty damping of a band update with local time stepping (Strang):
    ``F_new = A (F_old + D/2) + D/2`` with the conservative increment ``D = F_cons -
    F_old`` (LTS recursion with central fluxes) and ``A = (I - B/k)^k``,
    ``k = d.lts_pen_k``.

    For the leapfrog pair ``h' = a h - b tau w e``, ``e' = e + tau w h'`` the scheme is
    stable iff ``b Q < 2 (1 + a)`` (``Q``: the effective ``tau^2 w^2`` of the LTS
    polynomial, ``< 4`` for the conservative recursion). An explicit penalty at the
    macro step (``a = 1 - tau d``, ``b = 1``) needs ``Q < 4 - 2 tau d`` and fails for
    the strongly damped grid-scale modes, whose ``Q`` the recursion pushes towards
    4; the split form has ``b = (1 + a)/2`` and is stable for ``Q < 4`` whenever
    ``|a| < 1`` (``k`` sub-steps with ``rho(B)/k < 2``), like Crank-Nicolson.
    """
    k = max(1, int(d.lts_pen_k))
    m = 0.5 * (F_old + F_cons)
    for _ in range(k):
        m = m - (1.0 / k) * lts_penalty_rate(d, c, m, kind)
    return m + 0.5 * (F_cons - F_old)


@dataclass
class LTSSlab:
    """Yee planes around one interface that belong to the fine set of the local
    time stepping: planes ``lo..hi`` (inclusive) of the full grid; ``fine_e``/``fine_h``
    are per-component masks ``(1, n, NY, NZ)`` (fine planes, active nodes); ``upml`` lists, per H component,
    the compact side-UPML entries ``(positions in the aux vector, local flat
    indices in the slab, b1, new)`` lying on the fine H planes."""
    cp: Coupling3D
    lo: int
    hi: int
    fine_e: Tuple[torch.Tensor, ...]
    fine_h: Tuple[torch.Tensor, ...]
    upml_h: list                              # side-UPML H nodes on the fine planes
    upml_e: list                              # side-UPML E nodes on the fine planes
    planes: Optional[torch.Tensor] = None     # full-grid plane indices lo..hi
    ghost: Optional[torch.Tensor] = None      # local index of the ghost plane, (1,)
    row: Optional[torch.Tensor] = None        # local index of the interface plane, (1,)
    cuda: Optional[dict] = None               # kernel data (cuda3d.YeeCudaOps.slab_data)


def lts_slabs(st: StepStatic3D):
    """The :class:`LTSSlab` of every coupling (cached on the static)."""
    def make():
        y, d = st.yee, st.dg
        b = int(getattr(d, "lts_overlap", 1))
        if b < 0:
            raise ValueError("lts_overlap must be a non-negative integer.")
        NX, NY, NZ = y.shape
        out = []
        for cp in st.couplings:
            rho = cp.row
            top = cp.ghost == rho
            if top:
                lo, hi = rho - b - 1, rho
                e_planes = range(rho - b, rho + 1)        # Ey, Ez
                ex_planes = range(rho - b - 1, rho)       # Ex (position i + 1/2)
                hx_planes = range(rho - b, rho + 1)
                hyz_planes = range(rho - b - 1, rho + 1)  # includes the ghost plane rho
            else:
                lo, hi = rho - 1, rho + b + 1
                e_planes = range(rho, rho + b + 1)
                ex_planes = range(rho, rho + b + 1)
                hx_planes = range(rho, rho + b + 1)
                hyz_planes = range(rho - 1, rho + b + 1)  # includes the ghost plane rho - 1
            if lo < 1 or hi > NX - 2:
                raise ValueError("The local time stepping overlap reaches the x boundary; "
                                 "reduce lts_overlap or move the band.")
            n = hi - lo + 1

            def mask(planes):
                m = torch.zeros(n, dtype=torch.bool)
                for i in planes:
                    m[i - lo] = True
                return m.to(st.device).view(1, n, 1, 1)

            # plane masks intersected with the activity masks (PEC boundary nodes and
            # the nodes inside the band are never part of the fine set)
            fine_e = tuple(mask(pl) & y.act[k][lo:hi + 1][None]
                           for k, pl in enumerate((ex_planes, e_planes, e_planes)))
            fine_h = tuple(mask(pl) & y.act[3 + k][lo:hi + 1][None]
                           for k, pl in enumerate((hx_planes, hyz_planes, hyz_planes)))
            ups_h, ups_e = [], []
            if y.upml is not None:
                if y.upml_fine is None:
                    raise RuntimeError("Local time stepping with a side UPML needs the "
                                       "sub-step UPML data (upml_fine) on the Yee static.")
                for kind, fine_m, store in (("h", fine_h, ups_h), ("e", fine_e, ups_e)):
                    off = y.upml["offsets_" + kind]
                    for k, ent in enumerate(y.upml[kind]):
                        idx = ent["idx"]
                        entf = y.upml_fine[kind][k]
                        if not torch.equal(idx, entf["idx"]):
                            raise RuntimeError("UPML index sets differ between dt and dt/p.")
                        plane = torch.div(idx, NY * NZ, rounding_mode="floor")
                        # fine masks are (1, n, NY, NZ): node (plane, j, k) -> flat local index
                        sel = torch.nonzero((plane >= lo) & (plane <= hi)).reshape(-1)
                        local = idx[sel] - lo * NY * NZ
                        keep = fine_m[k].reshape(-1)[local]
                        sel, local = sel[keep], local[keep]
                        if sel.numel():
                            fac = dict(a1=entf["a1"][sel], b1=entf["b1"][sel])
                            if kind == "h":
                                fac.update(hist=entf["hist"][sel], new=entf["new"][sel],
                                           old=entf["old"][sel])
                            store.append((k, off[k] + sel, idx[sel], local, fac))
            out.append(LTSSlab(cp, lo, hi, fine_e, fine_h, ups_h, ups_e,
                               torch.arange(lo, hi + 1, device=st.device),
                               torch.tensor([cp.ghost - lo], device=st.device),
                               torch.tensor([rho - lo], device=st.device)))
        return out
    return _cached(st, "lts_slabs", make)


def _fold_upml(f, aux, rate, fac, p: int, sign: float):
    """``p`` Crank-Nicolson sub-steps of the UPML pair ``(aux, f)`` with the constant
    rate ``rate``: ``aux <- a1 aux + sign b1 rate``, ``f <- hist f + new aux_new - old aux``."""
    for _ in range(p):
        nw = fac["a1"] * aux + sign * fac["b1"] * rate
        f = fac["hist"] * f + fac["new"] * nw - fac["old"] * aux
        aux = nw
    return f, aux


def _dg_upml_fold(d: DGStatic3D, p: int, rate, f_old, f_new, aux, hist, new, old, sign: float,
                  terms: bool = False):
    """Overwrite the PML elements of ``f_new`` (plain update with the macro step) by the
    ``p``-fold sub-step UPML update from ``f_old``/``aux`` with the averaged rate (the
    curl ``(S, 3, K, Np)`` or, ``terms=True``, the six curl terms); PyTorch path, the
    CUDA step does this inside ``DGCudaOps.lts_update``."""
    idx = d.pml_idx
    curl = _dg_curl(rate) if terms else rate
    u = d.upml_fine
    fac = dict(a1=u["a1"], b1=u["b1"], hist=hist, new=new, old=old)
    fp, aux_new = _fold_upml(f_old.index_select(2, idx), aux, curl.index_select(2, idx), fac, p,
                             sign)
    return f_new.index_copy(2, idx, fp), aux_new


def _slab_curl(y: YeeStatic3D, fields, diff):
    """Plain Yee curl (no CPML) of slab tensors ``(S, n, NY, NZ)``."""
    rhs = [0.0, 0.0, 0.0]
    for (c_, a, f, sign) in TERMS:
        rhs[c_] = rhs[c_] + sign * diff(fields[f], a) / y.spacing[a]
    return rhs


def _lts_trace(st: StepStatic3D, planes, compact: bool):
    """Projected trace ``E*`` from the interface planes ``planes[i] = (Ey, Ez)`` of
    coupling ``i`` (layout of :func:`interface_trace`)."""
    d, S = st.dg, st.nshot
    if compact:
        x = torch.stack([t.reshape(S, -1) for pl in planes for t in pl], dim=1)
        return trace_op(st)(x.reshape(S, -1)).view(S, 2, -1)
    ys = [cp.to_dg[0](ey.reshape(S, -1)) for cp, (ey, _) in zip(st.couplings, planes)]
    zs = [cp.to_dg[1](ez.reshape(S, -1)) for cp, (_, ez) in zip(st.couplings, planes)]
    L = d.K * 4 * d.Nfp
    ey = torch.zeros((S, L), dtype=ys[0].dtype, device=ys[0].device)
    ez = torch.zeros_like(ey)
    for cp, a, b in zip(st.couplings, ys, zs):
        ey = ey.index_copy(1, cp.slots, a)
        ez = ez.index_copy(1, cp.slots, b)
    return torch.stack([ey, ez], dim=1).view(S, 2, d.K, 4 * d.Nfp)


def lts_h_half_step(st: StepStatic3D, c: StepCoefs3D, fe, fh, psh, Bfd, E, H, Baux):
    r"""H half step of the coupled system with local time stepping (``p = dg.lts > 1``).

    Explicit local time stepping of Diaz & Grote (SIAM J. Sci. Comput. 31(3),
    2009, doi:10.1137/070709414) in the first-order variables of the leapfrog.
    The fine set ``P`` holds the DG unknowns and the Yee planes of
    :func:`lts_slabs` (the interface plane and ``lts_overlap`` planes beyond it,
    the overlap that makes the method stable for the full range of ``p``); the
    rest of the Yee grid is coarse. With ``r(e) = M_h^-1 K^T e`` (the H rates of
    the lossless scheme) and ``q(h) = M_e^-1 K h`` (its E rates), one macro step
    ``dt`` replaces the H half step of the fine unknowns by ``p`` sub-steps
    ``dt/p`` of ::

        z_0 = e^n,   eta_(1/2) = -(dt/2p) r(z_0)
        z_(m+1) = z_m + (dt/p) q(eta_(m+1/2)),   eta_(m+3/2) = eta_(m+1/2) - (dt/p) r(z_(m+1))
        h^(n+1/2) = h^(n-1/2) - dt sum_j w_j r(z_j)       (w of :func:`lts_weights`)

    where in ``r`` the coarse unknowns are frozen at ``e^n`` (``z`` and ``eta`` live on
    the fine set; ``q`` includes the mortar and the ghost planes). On the fine E
    unknowns this is exactly Diaz & Grote's ``e^(n+1) = 2 z_p - e^(n-1)``, the coarse
    unknowns take the ordinary leapfrog step, so for central fluxes without
    losses the coupled scheme conserves their modified energy, and ``dt`` only
    has to satisfy the Yee limit while ``dt/p`` satisfies the band's. The
    dissipative parts (conductivity, the UPML of the band and of the side
    columns, the penalty fluxes, the interface penalty) act once per macro step
    on the averaged rates: the plain update is evaluated and the fine H
    unknowns are corrected by ``-(dt/mu) (r_avg - r_0)`` (exactly also through the
    linear UPML recursion). Returns ``(fh, psh, Bfd, H, Baux, Eext, compact)``
    with the ghost planes not yet written.
    """
    y, d, S = st.yee, st.dg, st.nshot
    p = int(d.lts)
    lc = d.lts_coef if d.lts_coef is not None else lts_coefficients(p)
    w, c0, ck, beta = lc["w"], lc["c0"], lc["c"], lc["beta"]
    inv_p = 1.0 / p
    compact = d.cuda is not None and E.is_cuda
    slabs = lts_slabs(st)
    NY, NZ = y.shape[1], y.shape[2]
    # ---- plain macro update of the Yee grid and the standard rates r_0 on the slabs
    fused = y.cuda is not None and fe[0].is_cuda
    if fused:
        fh1, psh1, Bfd1 = y.cuda.h_update(fe, fh, psh, Bfd)
    else:
        rhs, psh1 = _yee_curl(y, fe, y.ph, psh, _fwd)
        fh1 = [torch.where(y.act[3 + k], fh[k] - y.chr * rhs[k], fh[k]) for k in range(3)]
        Bfd1 = Bfd
        if y.upml is not None:
            fh1, Bfd1 = _yee_upml(y.upml, "h", Bfd, rhs, fh, fh1)
    # ---- fine-set recursion (slab kernels of cuda3d when available: one launch per
    #      slab and rate evaluation instead of ~70 elementwise PyTorch ops)
    fused_slab = fused and hasattr(y.cuda, "slab_r")
    sds = [y.cuda.slab_data(sl) for sl in slabs] if fused_slab else None
    e_slab = [[fe[k][:, sl.lo:sl.hi + 1] for k in range(3)] for sl in slabs]
    cb0 = 2.0 * c.fd_cb / (1.0 + c.fd_ca)                     # dt/eps (0 on inactive nodes)
    cb0_s = [cb0[sl.lo:sl.hi + 1][None] for sl in slabs]
    chr_s = [[y.chr[sl.lo:sl.hi + 1][None] * y.act[3 + k][sl.lo:sl.hi + 1][None]
              for k in range(3)] for sl in slabs]
    cb0_dg = 2.0 * c.dg_cb / (1.0 + c.dg_ca)                  # (K, 1): dt/eps per element
    ch_dg = d.ch[None, None]
    z_s = [[t for t in es] for es in e_slab]                  # z on the slabs (fine planes)
    z_dg = E
    hb = (c.h0, c.h1, c.h2, c.h3)
    zero_dg = torch.zeros_like(E)
    fused_dg = d.cuda is not None and E.is_cuda and hasattr(d.cuda, "lts_z")

    def trace_of(z_s):
        """Projected trace of the interface planes of ``z``."""
        tr = [(zs[1][:, sl.cp.row - sl.lo], zs[2][:, sl.cp.row - sl.lo])
              for zs, sl in zip(z_s, slabs)]
        return _lts_trace(st, tr, compact)

    def slab_r(i, z, eta, ravg, first, last, bm, bk, wm, r0=None):
        """``(r, eta_new, ravg_new, corr)`` of slab ``i`` from the H rates ``r =
        curl_fwd(z)``: ``r`` on the first sub-step only, ``eta_new = bm eta - bk/p chr r``
        (not on the last), ``ravg_new = ravg + wm r`` (``eta``/``ravg`` ``None`` on the
        first); ``corr`` (the fused kernel, last sub-step with ``r0``): the correction
        ``-chr fine_h (ravg_new - r0)`` of the fine H planes."""
        if fused_slab:
            return y.cuda.slab_r(sds[i], z, eta, ravg, first, last, bm, bk * inv_p, wm, r0)
        r = _slab_curl(y, z, _fwd)
        if first:
            ravg_new = [wm * rk for rk in r]
            eta_new = [(-bk * inv_p * chr_s[i][k]) * r[k] for k in range(3)]
        else:
            ravg_new = [ravg[k] + wm * r[k] for k in range(3)]
            eta_new = None if last else [bm * eta[k] - (bk * inv_p * chr_s[i][k]) * r[k]
                                         for k in range(3)]
        return (r if first else None), eta_new, ravg_new, None

    def slab_z(i, eta, z):
        """``z + (dt/p eps) curl_bwd(eta)`` on the fine E nodes of slab ``i``."""
        if fused_slab:
            return y.cuda.slab_z(sds[i], eta, z, cb0_s[i], inv_p)
        q = _slab_curl(y, eta, _bwd)
        sl = slabs[i]
        return [torch.where(sl.fine_e[k], z[k] + (inv_p * cb0_s[i]) * q[k], z[k])
                for k in range(3)]

    # DG recursion variables: eta (H-like increments) and the weighted curl average
    # cavg = sum_j w_j curl(T(z_j)); fused kernels (csrc/lts3d.cu) or PyTorch
    def dg_eta(T, eta, cavg, bm, bk, wm):
        if fused_dg:
            return d.cuda.lts_eta(T, eta, cavg, bm, bk * inv_p, wm)
        curl = _dg_curl(T)
        if eta is None:
            return (-bk * inv_p * ch_dg) * curl, wm * curl
        return bm * eta - (bk * inv_p * ch_dg) * curl, cavg + wm * curl

    def dg_z(Te, z):
        if fused_dg:
            return d.cuda.lts_z(Te, z, cb0_dg, inv_p)
        return z + (inv_p * cb0_dg[None, None]) * _dg_curl(Te)

    # operator + recursion update in one kernel pass (epilogue of the terms kernel)
    fused_epi = fused_dg and hasattr(d.cuda, "terms_eta")
    eb = (c.e0, c.e1, c.e2, c.e3)

    def eta_step(z_dg, Eext, eta, cavg, bm, bk, wm):
        if fused_epi:
            return d.cuda.terms_eta(z_dg, H, Eext, *eb, eta, cavg, bm, bk * inv_p, wm)
        return dg_eta(_curl_terms_h(d, c, z_dg, H, Eext), eta, cavg, bm, bk, wm)

    def z_step(eta_dg, z):
        if fused_epi:
            return d.cuda.terms_z(eta_dg, zero_dg, None, *hb, z, cb0_dg, inv_p)
        return dg_z(dg_operator(d, hb, eta_dg, zero_dg, None), z)

    Eext0 = trace_of(z_s)
    eta_dg, cavg = eta_step(z_dg, Eext0, None, None, 0.0, c0, w[0])
    r0, eta_s, r_avg = [], [], []
    for i in range(len(slabs)):
        r_i, eta_i, avg_i, _ = slab_r(i, z_s[i], None, None, True, False, 0.0, c0, w[0])
        r0.append(r_i)
        eta_s.append(eta_i)
        r_avg.append(avg_i)
    corr = [None] * len(slabs)
    gop = ghost_op(st)
    for m in range(1, p):
        # ghost planes of eta from the DG eta (one mortar product for all couplings),
        # then the lossless E rates
        g = gop(eta_dg.reshape(S, -1)).view(S, len(slabs), 2, NY, NZ)
        for i, sl in enumerate(slabs):
            inner = sl.cp.inner - sl.lo
            hz = 2.0 * g[:, i, 0] - eta_s[i][2][:, inner]
            hy = 2.0 * g[:, i, 1] - eta_s[i][1][:, inner]
            eta_s[i][2] = eta_s[i][2].index_copy(1, sl.ghost, hz[:, None])
            eta_s[i][1] = eta_s[i][1].index_copy(1, sl.ghost, hy[:, None])
        z_dg = z_step(eta_dg, z_dg)
        for i in range(len(slabs)):
            z_s[i] = slab_z(i, eta_s[i], z_s[i])
        eta_dg, cavg = eta_step(z_dg, trace_of(z_s), eta_dg, cavg, beta[m], ck[m], w[m])
        last = m == p - 1
        for i in range(len(slabs)):
            _, eta_s[i], r_avg[i], corr[i] = slab_r(i, z_s[i], eta_s[i], r_avg[i], False, last,
                                                    beta[m], ck[m], w[m], r0[i] if last else None)
    # ---- DG: H - (dt/mu) cavg; PML elements by p sub-steps of the UPML with cavg
    if fused_dg:
        u = d.upml_fine or {}
        H_new, Baux_new = d.cuda.lts_update(0, cavg, H, Baux, None, None, u.get("h_hist"),
                                            u.get("h_new"), u.get("h_old"), p)
    else:
        H_new = H - ch_dg * cavg
        Baux_new = Baux
        if d.pml_idx is not None:
            u = d.upml_fine
            H_new, Baux_new = _dg_upml_fold(d, p, cavg, H, H_new, Baux, u["h_hist"], u["h_new"],
                                            u["h_old"], -1.0)
    if d.lts_split:
        H_new = lts_damp(d, c, H, H_new, "h")
    # ---- Yee: correct the fine H planes by -(dt/mu) (r_avg - r_0); UPML nodes of the fine
    #      planes by p sub-steps of the UPML from the old state with the averaged rate
    # (fh1 and Bfd1 are fresh outputs of the plain update: corrected in place, which
    #  avoids a copy of the full Yee arrays per correction)
    fh1 = list(fh1)
    for i, sl in enumerate(slabs):
        n_sl = sl.hi - sl.lo + 1
        for k in range(3):
            if corr[i] is None:
                delta = torch.where(sl.fine_h[k], r_avg[i][k] - r0[i][k],
                                    torch.zeros_like(r0[i][k]))
                corr_k = -chr_s[i][k] * delta
            else:
                corr_k = corr[i][k]
            fh1[k].narrow(1, sl.lo, n_sl).add_(corr_k)
        for j, (k, pos, idx_full, local, fac) in enumerate(sl.upml_h):
            if fused_slab:
                val, aux = y.cuda.fold(fh[k].reshape(S, -1), Bfd, r_avg[i][k].reshape(S, -1),
                                       sds[i]["h"][j], (fac["a1"], fac["b1"], fac["hist"],
                                                        fac["new"], fac["old"]), p, -1.0)
            else:
                rate = r_avg[i][k].reshape(S, -1).index_select(1, local)
                f_old = fh[k].reshape(S, -1).index_select(1, idx_full)
                val, aux = _fold_upml(f_old, Bfd.index_select(1, pos), rate, fac, p, -1.0)
            fh1[k].view(S, -1).index_copy_(1, idx_full, val)
            Bfd1.index_copy_(1, pos, aux)
    return fh1, psh1, Bfd1, H_new, Baux_new, Eext0, compact


def h_half_step(st: StepStatic3D, c: StepCoefs3D, fe, fh, psh, Bfd, E, H, Baux):
    """Yee and DG H half steps and the ghost planes; returns
    ``(fh, psh, Bfd, H, Baux, Eext, compact)`` (``Eext``: the projected trace of
    ``E^n`` used by the DG H update, needed by the interface penalty)."""
    y, d = st.yee, st.dg
    S = st.nshot
    fused = y is not None and y.cuda is not None and fe[0].is_cuda
    Eext, compact = None, False
    if d is not None and int(getattr(d, "lts", 1)) > 1 and st.couplings:
        fh, psh, Bfd, H, Baux, Eext, compact = lts_h_half_step(st, c, fe, fh, psh, Bfd, E, H,
                                                               Baux)
    else:
        if fused:
            fh, psh, Bfd = y.cuda.h_update(fe, fh, psh, Bfd)
        elif y is not None:
            rhs, psh = _yee_curl(y, fe, y.ph, psh, _fwd)
            fh_new = [torch.where(y.act[3 + k], fh[k] - y.chr * rhs[k], fh[k]) for k in range(3)]
            if y.upml is not None:
                fh_new, Bfd = _yee_upml(y.upml, "h", Bfd, rhs, fh, fh_new)
            fh = fh_new
        if d is not None:
            compact = d.cuda is not None and E.is_cuda
            Eext = interface_trace(st, fe, compact) if st.couplings else None
            H, Baux = dg_h_update(d, c, E, H, Baux, Eext)
    if d is not None and st.couplings:
        fh = list(fh)
        NY, NZ = y.shape[1], y.shape[2]
        rows = _cached(st, "ghosts", lambda: torch.as_tensor(
            [cp.ghost for cp in st.couplings], device=H.device))
        inner = _cached(st, "inners", lambda: torch.as_tensor(
            [cp.inner for cp in st.couplings], device=H.device))
        g = ghost_op(st)(H.reshape(S, -1)).view(S, len(st.couplings), 2, NY, NZ)
        # fh is a fresh output of this half step: the ghost planes are written in place
        fh[2].index_copy_(1, rows, 2.0 * g[:, :, 0] - fh[2].index_select(1, inner))
        fh[1].index_copy_(1, rows, 2.0 * g[:, :, 1] - fh[1].index_select(1, inner))
    return fh, psh, Bfd, H, Baux, Eext, compact


# --------------------------------------------------------------------------- E half step
def e_half_step(st: StepStatic3D, c: StepCoefs3D, fe, fh, pse, Dfd, E, H, Daux, Eext, compact):
    """Yee and DG E updates (sources not injected); returns ``(fe, pse, Dfd, E, Daux)``.
    With local time stepping the side-UPML nodes of the fine Yee planes and the PML
    elements of the band take ``p`` UPML sub-steps with the macro-step rates."""
    y, d = st.yee, st.dg
    S = st.nshot
    fused = y is not None and y.cuda is not None and fe[0].is_cuda
    p = int(getattr(d, "lts", 1)) if d is not None else 1
    lts = p > 1 and bool(st.couplings)
    pen = None
    if fused:
        pen = interface_penalty_rows(st, c, fe, E) if c.g_if is not None else None
        fe_new, pse, Dfd_new = y.cuda.e_update(fh, fe, pse, Dfd, c, pen)
    elif y is not None:
        rhs, pse = _yee_curl(y, fh, y.pe, pse, _bwd)
        if c.g_if is not None:
            rhs = _yee_interface_penalty(st, c, rhs, fe, E)
        fe_new = [torch.where(y.act[k], c.fd_ca * fe[k] + c.fd_cb * rhs[k], fe[k]) for k in range(3)]
        Dfd_new = Dfd
        if y.upml is not None:
            fe_new, Dfd_new = _yee_upml(y.upml, "e", Dfd, rhs, fe, fe_new,
                                        (c.fu_hist, c.fu_new, c.fu_old))
    else:
        fe_new, Dfd_new = fe, Dfd
    if lts and y is not None and y.upml is not None:
        fe_new = list(fe_new)
        if pen is None and c.g_if is not None:
            pen = interface_penalty_rows(st, c, fe, E)
        fused_slab = fused and hasattr(y.cuda, "slab_r")
        for i, sl in enumerate(slabs_of(st)):
            if not sl.upml_e:
                continue
            fh_s = [fh[k][:, sl.lo:sl.hi + 1] for k in range(3)]
            sd = y.cuda.slab_data(sl) if fused_slab else None
            rhs_s = y.cuda.slab_curl_bwd(sd, fh_s) if fused_slab else _slab_curl(y, fh_s, _bwd)
            if pen is not None:
                for comp in (1, 2):      # cached index: no host-to-device copy (graph capture)
                    rhs_s[comp] = rhs_s[comp].index_add(1, sl.row, pen[:, i, comp - 1][:, None])
            for j, (k, pos, idx_full, local, fac) in enumerate(sl.upml_e):
                hist, new, old = (t.index_select(0, pos) for t in (c.fu_hist_f, c.fu_new_f, c.fu_old_f))
                if fused_slab:
                    val, aux = y.cuda.fold(fe[k].reshape(S, -1), Dfd, rhs_s[k].reshape(S, -1),
                                           sd["e"][j], (fac["a1"], fac["b1"], hist, new, old), p, 1.0)
                else:
                    rate = rhs_s[k].reshape(S, -1).index_select(1, local)
                    f_old = fe[k].reshape(S, -1).index_select(1, idx_full)
                    val, aux = _fold_upml(f_old, Dfd.index_select(1, pos), rate,
                                          dict(fac, hist=hist, new=new, old=old), p, 1.0)
                fe_new[k].view(S, -1).index_copy_(1, idx_full, val)     # fresh outputs: in place
                Dfd_new.index_copy_(1, pos, aux)
    fe = fe_new
    Dfd = Dfd_new
    if d is not None:
        Xh = interface_x(st, c, Eext, compact) if c.g_if is not None else None
        T = dg_operator(d, (c.h0, c.h1, c.h2, c.h3), H, E, Xh)
        if d.cuda is not None and E.is_cuda and lts:
            E_new, Daux_new = d.cuda.lts_update(1, T, E, Daux, c.dg_ca.reshape(-1),
                                                c.dg_cb.reshape(-1), c.u_hist_f, c.u_new_f,
                                                c.u_old_f, p)
        elif d.cuda is not None and E.is_cuda:
            E_new, Daux_new = d.cuda.update_e(T, E, Daux, c.dg_ca, c.dg_cb, c.u_hist, c.u_new,
                                              c.u_old)
        else:
            curl = _dg_curl(T)
            E_new = c.dg_ca[None, None] * E + c.dg_cb[None, None] * curl
            Daux_new = Daux
            if d.pml_idx is not None:
                u = d.upml
                E_new, Daux_new = _upml(d.pml_idx, Daux, curl, E, E_new, u["a1"], u["b1"],
                                        c.u_hist, c.u_new, c.u_old)
        if lts and d.pml_idx is not None and not (d.cuda is not None and E.is_cuda):
            E_new, Daux_new = _dg_upml_fold(d, p, T, E, E_new, Daux, c.u_hist_f, c.u_new_f,
                                            c.u_old_f, 1.0, terms=True)
        if lts and getattr(d, "lts_split", False):
            E_new = lts_damp(d, c, E, E_new, "e")
        E, Daux = E_new, Daux_new
    return fe, pse, Dfd, E, Daux


def slabs_of(st: StepStatic3D):
    """:func:`lts_slabs` when the static has couplings, else an empty list."""
    return lts_slabs(st) if st.couplings else []


# --------------------------------------------------------------------------- step
def _fused_ok(st: StepStatic3D, c: StepCoefs3D, state) -> bool:
    """The fully fused global step (:mod:`.fused3d`) applies: CUDA kernels attached, no
    local time stepping and nothing to differentiate (autograd uses the PyTorch glue)."""
    from . import fused3d
    if not fused3d.any_available(st):
        return False
    if torch.is_grad_enabled() and (any(t.requires_grad for t in state)
                                    or any(t.requires_grad for t in c.tensors())):
        return False
    return True


def step(st: StepStatic3D, c: StepCoefs3D, state, src=None, amp=None):
    """One leapfrog step of the coupled 3D system (sources with samples ``amp``)."""
    if _fused_ok(st, c, state) and (amp is None or not amp.requires_grad
                                    or not torch.is_grad_enabled()):
        from . import fused3d
        out = fused3d.empty_state(st, state)
        with torch.no_grad():
            cc = c.replaced([t.contiguous() for t in c.tensors()])
            pre = None
            if fused3d.lts_available(st):
                key = tuple(t.data_ptr() for t in c.tensors())
                ent = st.__dict__.get("_lts_pre")
                if ent is None or ent[0] != key:
                    ent = st.__dict__["_lts_pre"] = (key, fused3d.prepare(st, cc))
                pre = ent[1]
            fused3d.step_any(st, cc, [t.contiguous() for t in state], out, src,
                             None if amp is None else amp.contiguous(), pre=pre)
        return tuple(out)
    return _step_reference(st, c, state, src, amp)


def _step_reference(st: StepStatic3D, c: StepCoefs3D, state, src=None, amp=None):
    """The step with PyTorch glue (differentiable by autograd)."""
    fe, fh = list(state[0:3]), list(state[3:6])
    pse, psh = list(state[6:12]), list(state[12:18])
    E, H, Daux, Baux = state[18], state[19], state[20], state[21]
    Dfd, Bfd = state[22], state[23]
    fh, psh, Bfd, H, Baux, Eext, compact = h_half_step(st, c, fe, fh, psh, Bfd, E, H, Baux)
    fe, pse, Dfd, E, Daux = e_half_step(st, c, fe, fh, pse, Dfd, E, H, Daux, Eext, compact)
    if src is not None:
        fe, E = inject(st, c, src, amp, fe, E)
    return tuple(fe) + tuple(fh) + tuple(pse) + tuple(psh) + (E, H, Daux, Baux, Dfd, Bfd)


def inject(st: StepStatic3D, c: StepCoefs3D, ops: PointOps3D, amp, fe, E):
    """Subtract ``cb w J`` (DeepGPR convention ``J = w dl delta``) from component ``ops.comp``."""
    S = st.nshot
    fe = list(fe)
    if ops.fd_shot.numel():
        f = fe[ops.comp]
        vals = (c.fd_cb.reshape(-1).index_select(0, ops.fd_node) * ops.fd_w
                * amp.index_select(0, ops.fd_item))
        # flat index_add (index_put with accumulate sorts and cannot be graph-captured)
        fe[ops.comp] = f.reshape(-1).index_add(0, ops.fd_shot * f[0].numel() + ops.fd_node,
                                               -vals).view(f.shape)
    if ops.dg_shot.numel():
        d = st.dg
        vals = (c.dg_cb.index_select(0, ops.dg_elem) * ops.dg_w
                * amp.index_select(0, ops.dg_item)[:, None])
        comp = E[:, ops.comp].reshape(S * d.K, d.Np).index_add(
            0, ops.dg_shot * d.K + ops.dg_elem, -vals).view(S, 1, d.K, d.Np)
        E = torch.cat([E[:, :ops.comp], comp, E[:, ops.comp + 1:]], dim=1)
    return fe, E


def sample(st: StepStatic3D, ops: PointOps3D, state):
    """Field component ``ops.comp`` at the receivers, ``(nshot, nrx)``."""
    S, n = st.nshot, ops.count
    if (st.yee is not None and st.yee.cuda is not None and torch.device(st.device).type == "cuda"
            and not (torch.is_grad_enabled() and (state[ops.comp].requires_grad
                                                  or state[18].requires_grad))):
        from .glue3d import point_glue
        g = point_glue(st, ops)
        if g is not None:
            out = torch.empty((S, n), dtype=st.dtype, device=state[ops.comp].device)
            g.sample(state[ops.comp].contiguous(), state[18].contiguous(), out)
            return out
    f, E = state[ops.comp], state[18]
    ref = f if f.numel() else E
    out = torch.zeros((S * n,), dtype=ref.dtype, device=ref.device)
    if ops.fd_shot.numel():
        vals = f.reshape(-1).index_select(0, ops.fd_shot * f[0].numel() + ops.fd_node) * ops.fd_w
        out = out.index_add(0, ops.fd_shot * n + ops.fd_item, vals)
    if ops.dg_shot.numel():
        d = st.dg
        e = E[:, ops.comp].reshape(S * d.K, d.Np).index_select(0, ops.dg_shot * d.K + ops.dg_elem)
        out = out.index_add(0, ops.dg_shot * n + ops.dg_item, (e * ops.dg_w).sum(-1))
    return out.view(S, n)


# --------------------------------------------------------------------------- energy
def energy(st: StepStatic3D, eps_fd, mu_fd, dg_eps, dg_mu, prev, now):
    """Leapfrog-invariant energy ``1/2 (eps E^n E^n + mu H^(n-1/2) H^(n+1/2))`` per
    shot (``prev``: state before the step, ``now``: after it). ``eps``/``mu`` are
    relative values on the Yee field grid and per DG element."""
    total = 0.0
    y, d = st.yee, st.dg
    if y is not None:
        cell = y.spacing[0] * y.spacing[1] * y.spacing[2]
        eps = VACUUM_PERMITTIVITY * eps_fd
        mu = VACUUM_PERMEABILITY * mu_fd
        for k in range(3):
            total = total + 0.5 * cell * (y.weights[k] * eps * prev[k] ** 2).sum(dim=(1, 2, 3))
            total = total + 0.5 * cell * (y.weights[3 + k] * mu * prev[3 + k] * now[3 + k]).sum(
                dim=(1, 2, 3))
    if d is not None:
        def quad(a, b):
            return (torch.matmul(a, d.mass) * b).sum(-1)       # (S, 3, K)
        jm = d.J[:, 0]
        e = VACUUM_PERMITTIVITY * dg_eps * jm
        m = VACUUM_PERMEABILITY * dg_mu * jm
        total = total + 0.5 * (e * quad(prev[18], prev[18])).sum(dim=(1, 2))
        total = total + 0.5 * (m * quad(prev[19], now[19])).sum(dim=(1, 2))
    return total


__all__ = [
    "Coupling3D", "DGStatic3D", "PointOps3D", "STATE_SIZE", "StepCoefs3D", "StepStatic3D",
    "TERMS", "YeeStatic3D", "build_coefficients", "cpml_axis_profiles", "dg_coefficients_3d",
    "dg_flux", "dg_h_update", "dg_operator", "dg_static_3d", "dg_terms", "e_half_step", "energy",
    "flux_coefficients", "h_half_step", "inject", "interface_penalty_rows", "lts_coefficients",
    "lts_h_half_step", "lts_slabs", "lts_weights", "sample", "step", "upml_coefficients",
    "yee_coefficients_3d", "yee_static_3d",
    "yee_upml_3d", "yee_upml_coefficients", "zero_state",
]

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

    NAMES = ("fd_ca", "fd_cb", "dg_ca", "dg_cb", "e0", "e1", "e2", "e3", "h0", "h1", "h2", "h3",
             "g_if", "u_hist", "u_new", "u_old", "fu_hist", "fu_new", "fu_old")

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


def dg_static_3d(geometry, face_tags, mu_r, dt, alpha, pml_sigma, dtype, device,
                 alpha_if: float = 0.0) -> DGStatic3D:
    """Static DG operators; ``pml_sigma``: per axis ``(K, Np)`` UPML conductivity
    profiles [S/m] (``None`` entries or ``None`` for no PML); ``alpha_if``: weight
    of the dissipative DG-Yee interface penalty."""
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
    pml_idx, upml = None, None
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
            q1 = 0.5 * dt * s_c1
            den = mu_p / dt + 0.5 * mu_p * s_c2
            upml = dict(s_c=t(s_c), s_c2=t(s_c2), a1=t((1 - q1) / (1 + q1)), b1=t(dt / (1 + q1)),
                        h_hist=t((mu_p / dt - 0.5 * mu_p * s_c2) / den),
                        h_new=t((1 / dt + 0.5 * s_c) / den), h_old=t((1 / dt - 0.5 * s_c) / den))
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
        ti(slotP.reshape(-1)), ti(ref.fmask.reshape(-1)), float(alpha_if), pml_idx, upml)


# --------------------------------------------------------------------------- coefficients
def yee_coefficients_3d(eps_r_pad, sigma_pad, dt):
    e_term = VACUUM_PERMITTIVITY * eps_r_pad / dt
    s_term = 0.5 * sigma_pad
    ea = e_term + s_term
    ca, cb = (e_term - s_term) / ea, 1.0 / ea
    pec = (sigma_pad > PEC_CONDUCTIVITY_THRESHOLD).detach()
    zero = torch.zeros_like(ca)
    return torch.where(pec, zero, ca), torch.where(pec, zero, cb)


def flux_coefficients(ds: DGStatic3D, eps_r, mu_r):
    """Face coefficients of the E- and H-fluxes from element materials.

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
    a = ds.alpha
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
    g = mf * (0.5 * ds.alpha_if) * Ym
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


def interface_trace(st: StepStatic3D, fe, compact: bool):
    """``E* = P fe|rows`` (projected Yee trace, y and z) at the interface face slots:
    compact ``(S, 2, nX)`` in coupling order for the fused kernels, otherwise
    dense ``(S, 2, K, 4 Nfp)`` (zero outside the interface)."""
    d, S = st.dg, st.nshot
    ys = [cp.to_dg[0](fe[1][:, cp.row].reshape(S, -1)) for cp in st.couplings]
    zs = [cp.to_dg[1](fe[2][:, cp.row].reshape(S, -1)) for cp in st.couplings]
    if compact:
        return torch.stack([torch.cat(ys, 1), torch.cat(zs, 1)], dim=1)
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


# --------------------------------------------------------------------------- step
def step(st: StepStatic3D, c: StepCoefs3D, state, src=None, amp=None):
    """One leapfrog step of the coupled 3D system (sources with samples ``amp``)."""
    fe, fh = list(state[0:3]), list(state[3:6])
    pse, psh = list(state[6:12]), list(state[12:18])
    E, H, Daux, Baux = state[18], state[19], state[20], state[21]
    Dfd, Bfd = state[22], state[23]
    y, d = st.yee, st.dg
    S = st.nshot
    fused = y is not None and y.cuda is not None and fe[0].is_cuda
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
        if d.cuda is not None and E.is_cuda:
            H, Baux = d.cuda.update_h(dg_operator(d, (c.e0, c.e1, c.e2, c.e3), E, H, Eext), H, Baux)
        else:
            curl = _dg_curl(dg_operator(d, (c.e0, c.e1, c.e2, c.e3), E, H, Eext))
            H_new = H - d.ch * curl
            if d.pml_idx is not None:
                u = d.upml
                H_new, Baux = _upml(d.pml_idx, Baux, -curl, H, H_new, u["a1"], u["b1"],
                                    u["h_hist"], u["h_new"], u["h_old"])
            H = H_new
        if st.couplings:
            hflat = H.reshape(S, 3, -1)
            NY, NZ = y.shape[1], y.shape[2]
            rows = _cached(st, "ghosts", lambda: torch.as_tensor(
                [cp.ghost for cp in st.couplings], device=H.device))
            hz = torch.stack([2.0 * cp.to_fd[0](hflat[:, 2].index_select(1, cp.nodes)).view(S, NY, NZ)
                              - fh[2][:, cp.inner] for cp in st.couplings], dim=1)
            hy = torch.stack([2.0 * cp.to_fd[1](hflat[:, 1].index_select(1, cp.nodes)).view(S, NY, NZ)
                              - fh[1][:, cp.inner] for cp in st.couplings], dim=1)
            fh[2] = fh[2].index_copy(1, rows, hz)
            fh[1] = fh[1].index_copy(1, rows, hy)
    if fused:
        pen = interface_penalty_rows(st, c, fe, E) if c.g_if is not None else None
        fe, pse, Dfd = y.cuda.e_update(fh, fe, pse, Dfd, c, pen)
    elif y is not None:
        rhs, pse = _yee_curl(y, fh, y.pe, pse, _bwd)
        if c.g_if is not None:
            rhs = _yee_interface_penalty(st, c, rhs, fe, E)
        fe_new = [torch.where(y.act[k], c.fd_ca * fe[k] + c.fd_cb * rhs[k], fe[k]) for k in range(3)]
        if y.upml is not None:
            fe_new, Dfd = _yee_upml(y.upml, "e", Dfd, rhs, fe, fe_new,
                                    (c.fu_hist, c.fu_new, c.fu_old))
        fe = fe_new
    if d is not None:
        Xh = interface_x(st, c, Eext, compact) if c.g_if is not None else None
        if d.cuda is not None and E.is_cuda:
            E, Daux = d.cuda.update_e(dg_operator(d, (c.h0, c.h1, c.h2, c.h3), H, E, Xh), E,
                                      Daux, c.dg_ca, c.dg_cb, c.u_hist, c.u_new, c.u_old)
        else:
            curl = _dg_curl(dg_operator(d, (c.h0, c.h1, c.h2, c.h3), H, E, Xh))
            E_new = c.dg_ca[None, None] * E + c.dg_cb[None, None] * curl
            if d.pml_idx is not None:
                u = d.upml
                E_new, Daux = _upml(d.pml_idx, Daux, curl, E, E_new, u["a1"], u["b1"],
                                    c.u_hist, c.u_new, c.u_old)
            E = E_new
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
    "dg_flux", "dg_operator", "dg_static_3d", "dg_terms", "energy", "flux_coefficients",
    "inject", "interface_penalty_rows", "sample", "step", "upml_coefficients",
    "yee_coefficients_3d", "yee_static_3d",
    "yee_upml_3d", "yee_upml_coefficients", "zero_state",
]

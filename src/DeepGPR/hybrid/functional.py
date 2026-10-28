"""Side-effect-free time step of the hybrid scheme.

One function, :func:`step`, advances the complete coupled state

``(Ez, Hx, Hy, psi_ex, psi_ey, psi_hx, psi_hy,        # Yee + Yee CPML
   E, Hx_dg, Hy_dg, q_ex, q_ey, q_hx, q_hy)             # DGTD + DG CPML``

by one leapfrog step, given

* :class:`StepStatic` - everything that does not depend on the materials
  (grids, masks, CPML recursions, DG operators, couplings), and
* :class:`StepCoefs` - the material-dependent coefficients (Yee ``ca``/``cb``,
  DG ``ca``/``cb`` and the face impedances of the upwind penalty).

It is used by :meth:`HybridTM2D.run` (forward modelling) and by
:mod:`DeepGPR.hybrid.adjoint`, whose backward pass applies the transpose of
exactly this step (reverse-mode differentiation of the step operator, or, on
the fused CUDA kernels of :mod:`.cuda2d`, the hand-written transpose of
:mod:`.vjp2d`), so the gradient is the discrete adjoint of the implemented
scheme. On CUDA the Yee and DG updates run as fused kernels with exact
transposes (same arithmetic); the step issues no host-device synchronisation,
so it can also be replayed as a CUDA Graph (:mod:`.graphs`).

The Yee part reproduces :class:`~DeepGPR.hybrid.yee.YeeTM2D` (and therefore
DeepGPR's native second-order update) on full-grid arrays: inactive nodes have
``ca = 1``, ``cb = 0`` and CPML coefficients vanish outside their faces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..config.constants import PEC_CONDUCTIVITY_THRESHOLD, VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY

STATE_SIZE = 14


@dataclass
class YeeStatic:
    dx: float
    dy: float
    shape: Tuple[int, int]
    act_e: torch.Tensor            # (NX, NY) bool, Ez nodes that are updated
    cHx: torch.Tensor              # Hx update coefficient (on Ez[j+1]-Ez[j])
    cHy: torch.Tensor              # Hy update coefficient (on Ez[i+1]-Ez[i])
    h_y: Tuple[torch.Tensor, ...]  # (A, B, E, F) y-face CPML for Hx
    h_x: Tuple[torch.Tensor, ...]  # (A, B, E, F) x-face CPML for Hy
    e_x: Tuple[torch.Tensor, ...]  # (A, B, E, F) x-face CPML for Ez (on dHy)
    e_y: Tuple[torch.Tensor, ...]  # (A, B, E, F) y-face CPML for Ez (on dHx)
    cuda: Optional[object] = None  # :class:`~DeepGPR.hybrid.cuda2d.Yee2Ops`


@dataclass
class DGStatic:
    K: int
    Np: int
    Nfp: int
    DrT: torch.Tensor
    DsT: torch.Tensor
    LIFTT: torch.Tensor
    rx: torch.Tensor
    ry: torch.Tensor
    sx: torch.Tensor
    sy: torch.Tensor
    nxf: torch.Tensor
    nyf: torch.Tensor
    fscale: torch.Tensor
    vmapM: torch.Tensor
    vmapP: torch.Tensor
    m_int: torch.Tensor
    m_pec: torch.Tensor
    m_abc: torch.Tensor
    m_if: torch.Tensor
    has_abc: bool
    alpha: float
    ch: torch.Tensor               # (K, 1) dt / mu (mu is not inverted)
    neighbour: torch.Tensor        # (K, 3) EToE
    pml_idx: Optional[torch.Tensor]
    pml_x: Optional[List[torch.Tensor]]
    pml_y: Optional[List[torch.Tensor]]
    cuda: Optional[object] = None  # :class:`~DeepGPR.hybrid.cuda2d.DG2Ops`


@dataclass
class CouplingStatic:
    row: int                       # Yee interface row
    ghost: int                     # Yee Hy row that receives the ghost value
    inner: int                     # Yee Hy row inside the Yee region next to it
    slots: torch.Tensor
    nodes: torch.Tensor
    to_dg: object                  # SparseOp E_b -> projected E* on DG face nodes
    to_fd: object                  # SparseOp Hy_DG -> H_b


@dataclass
class StepStatic:
    nshot: int
    yee: Optional[YeeStatic]
    dg: Optional[DGStatic]
    couplings: List[CouplingStatic] = field(default_factory=list)
    dtype: torch.dtype = torch.float64
    device: torch.device = torch.device("cpu")
    staircase: Optional[object] = None   # :class:`~DeepGPR.hybrid.staircase.StaircaseStatic`


@dataclass
class StepCoefs:
    fd_ca: Optional[torch.Tensor] = None
    fd_cb: Optional[torch.Tensor] = None
    dg_ca: Optional[torch.Tensor] = None
    dg_cb: Optional[torch.Tensor] = None
    Ym: Optional[torch.Tensor] = None
    Yp: Optional[torch.Tensor] = None
    Zm: Optional[torch.Tensor] = None
    Zp: Optional[torch.Tensor] = None

    NAMES = ("fd_ca", "fd_cb", "dg_ca", "dg_cb", "Ym", "Yp", "Zm", "Zp")

    def tensors(self) -> List[torch.Tensor]:
        return [getattr(self, n) for n in self.NAMES if getattr(self, n) is not None]

    def replaced(self, tensors: Sequence[torch.Tensor]) -> "StepCoefs":
        it = iter(tensors)
        return StepCoefs(**{n: (next(it) if getattr(self, n) is not None else None)
                            for n in self.NAMES})


# --------------------------------------------------------------------------- static
def yee_static(fd) -> YeeStatic:
    """Full-grid static arrays of a :class:`~DeepGPR.hybrid.yee.YeeTM2D`."""
    NX, NY = fd.NX, fd.NY
    z = lambda: torch.zeros((NX, NY), dtype=fd.dtype, device=fd.device)  # noqa: E731
    act_e = torch.zeros((NX, NY), dtype=torch.bool, device=fd.device)
    act_hx = torch.zeros_like(act_e)
    act_hy = torch.zeros_like(act_e)
    for a, b in fd.e_ranges:
        act_e[a:b, 1:NY - 1] = True
        act_hx[a:b, :NY - 1] = True
    for a, b in fd.hy_ranges:
        act_hy[a:b, 1:NY - 1] = True
    cHx = torch.where(act_hx, fd.ch_rhs / fd.dy, z())
    cHy = torch.where(act_hy, fd.ch_rhs / fd.dx, z())
    groups = {k: [z(), z(), z(), z()] for k in ("h_y", "h_x", "e_x", "e_y")}
    for entry in fd.cpml:
        ra, rb, re, rf = entry["coef"]
        kind = entry["kind"]
        if kind in ("e_xface", "h_xface"):
            a, b = entry["rows"]
            sl = (slice(a, b), slice(1, NY - 1))
            col = lambda v: v[:, None]  # noqa: E731
            spacing = fd.dx
            regions = [sl]
        else:
            a, b = entry["cols"]
            regions = [(slice(r0, r1), slice(a, b)) for r0, r1 in fd.e_ranges]
            col = lambda v: v[None, :]  # noqa: E731
            spacing = fd.dy
        key = {"e_xface": "e_x", "h_xface": "h_x", "e_yface": "e_y", "h_yface": "h_y"}[kind]
        A, B, E, Fm = groups[key]
        for rows, cols in regions:
            scale = fd.ch_rhs[rows, cols] if kind.startswith("h") else 1.0
            A[rows, cols] = scale * col(ra) / spacing
            B[rows, cols] = scale * col(rb)
            E[rows, cols] = col(re).expand(A[rows, cols].shape)
            Fm[rows, cols] = col(rf).expand(A[rows, cols].shape) / spacing
    return YeeStatic(fd.dx, fd.dy, (NX, NY), act_e, cHx, cHy, tuple(groups["h_y"]),
                     tuple(groups["h_x"]), tuple(groups["e_x"]), tuple(groups["e_y"]))


def dg_static(dg) -> DGStatic:
    """Static operators of a :class:`~DeepGPR.hybrid.dgtd.DGTM2D`."""
    neighbour = torch.as_tensor(dg.geometry.EToE, device=dg.device)
    return DGStatic(
        dg.K, dg.Np, dg.Nfp, dg.DrT, dg.DsT, dg.LIFTT, dg.rx, dg.ry, dg.sx, dg.sy,
        dg.nxf, dg.nyf, dg.fscale, dg.vmapM, dg.vmapP, dg.m_int, dg.m_pec, dg.m_abc, dg.m_if,
        dg.has_abc, dg.flux_alpha, dg.ch, neighbour, dg.pml_idx,
        getattr(dg, "pml_x", None), getattr(dg, "pml_y", None))


def coupling_static(couplings, fd) -> List[CouplingStatic]:
    out = []
    for name, (row, cp) in couplings.items():
        ghost = fd.ghost_top if name == "top" else fd.ghost_bottom
        inner = ghost - 1 if name == "top" else ghost + 1
        out.append(CouplingStatic(row, ghost, inner, cp.slots, cp.nodes, cp.to_dg, cp.to_fd))
    return out


# --------------------------------------------------------------------------- coefficients
def yee_coefficients(ys: YeeStatic, eps_r_pad, sigma_pad, dt):
    """``(ca, cb)`` of the Yee E update (``ca = 1``, ``cb = 0`` on inactive nodes)."""
    e_term = VACUUM_PERMITTIVITY * eps_r_pad / dt
    s_term = 0.5 * sigma_pad
    ea = e_term + s_term
    ca = (e_term - s_term) / ea
    cb = 1.0 / ea
    pec = (sigma_pad > PEC_CONDUCTIVITY_THRESHOLD).detach()
    zero, one = torch.zeros_like(ca), torch.ones_like(ca)
    ca = torch.where(ys.act_e, torch.where(pec, zero, ca), one)
    cb = torch.where(ys.act_e, torch.where(pec, zero, cb), zero)
    return ca, cb


def dg_coefficients(ds: DGStatic, eps_r, sigma, mu_r, dt):
    """``(ca, cb, Ym, Yp, Zm, Zp)`` of the DG E update and the upwind penalty."""
    eps = VACUUM_PERMITTIVITY * eps_r
    mu = VACUUM_PERMEABILITY * mu_r
    ea = eps / dt + 0.5 * sigma
    ca = ((eps / dt - 0.5 * sigma) / ea)[:, None]
    cb = (1.0 / ea)[:, None]
    Y = torch.sqrt(eps / mu)
    # (K, 3) -> (K, 3 Nfp), contiguous (the kernels read it flat); expand + reshape,
    # since repeat_interleave synchronises with the device
    rep = lambda a: a.unsqueeze(-1).expand(-1, -1, ds.Nfp).reshape(a.shape[0], -1).contiguous()  # noqa: E731
    Ym = rep(Y[:, None].expand(-1, 3))
    Yp = rep(Y[ds.neighbour])
    return ca, cb, Ym, Yp, 1.0 / Ym, 1.0 / Yp


def build_coefficients(st: StepStatic, fields, dt) -> StepCoefs:
    """Coefficients from material fields ``(eps_fd_pad, sig_fd_pad, dg_eps, dg_sig, dg_mu)``."""
    eps_fd, sig_fd, dg_eps, dg_sig, dg_mu = fields
    c = StepCoefs()
    if st.yee is not None:
        c.fd_ca, c.fd_cb = yee_coefficients(st.yee, eps_fd, sig_fd, dt)
    if st.dg is not None:
        c.dg_ca, c.dg_cb, c.Ym, c.Yp, c.Zm, c.Zp = dg_coefficients(st.dg, dg_eps, dg_sig,
                                                                   dg_mu, dt)
    return c


# --------------------------------------------------------------------------- state
def zero_state(st: StepStatic) -> Tuple[torch.Tensor, ...]:
    S = st.nshot
    kw = dict(dtype=st.dtype, device=st.device)
    empty = torch.zeros(0, **kw)
    if st.yee is not None:
        fd = [torch.zeros((S,) + st.yee.shape, **kw) for _ in range(7)]
    else:
        fd = [empty] * 7
    if st.dg is not None:
        d = st.dg
        dg = [torch.zeros((S, d.K, d.Np), **kw) for _ in range(3)]
        kp = 0 if d.pml_idx is None else d.pml_idx.numel()
        dg += [torch.zeros((S, kp, d.Np), **kw) for _ in range(4)]
    else:
        dg = [empty] * 7
    return tuple(fd + dg)


# --------------------------------------------------------------------------- DG operators
def _traces(d: DGStatic, f):
    flat = f.reshape(f.shape[0], -1)
    shape = (f.shape[0], d.K, 3 * d.Nfp)
    return flat[:, d.vmapM].view(shape), flat[:, d.vmapP].view(shape)


def _lift(d: DGStatic, flux):
    return torch.matmul(d.fscale * flux, d.LIFTT)


def dg_grad_e(d: DGStatic, c: StepCoefs, E, Hx, Hy, Eext):
    """``(D_x E, D_y E)``; ``Eext`` holds the projected Yee trace on interface faces."""
    Em, Ep = _traces(d, E)
    a = d.alpha
    Eext = Eext.view(Em.shape)
    dE = d.m_int * (0.5 * (Ep - Em)) + d.m_pec * (-Em) + d.m_if * (Eext - Em)
    if a > 0 or d.has_abc:
        Hxm, Hxp = _traces(d, Hx)
        Hym, Hyp = _traces(d, Hy)
        Htm = Hym * d.nxf - Hxm * d.nyf
        if a > 0:
            Htp = Hyp * d.nxf - Hxp * d.nyf
            up = (c.Yp * (Ep - Em) + (Htp - Htm)) / (c.Ym + c.Yp)
            dE = dE + d.m_int * a * (up - 0.5 * (Ep - Em))
        if d.has_abc:
            dE = dE + d.m_abc * (-0.5 * Em - 0.5 * Htm / c.Ym)
    dr = torch.matmul(E, d.DrT)
    ds = torch.matmul(E, d.DsT)
    dxE = d.rx * dr + d.sx * ds + _lift(d, d.nxf * dE)
    dyE = d.ry * dr + d.sy * ds + _lift(d, d.nyf * dE)
    return dxE, dyE


def dg_curl_h(d: DGStatic, c: StepCoefs, E, Hx, Hy):
    """``(D_x Hy, D_y Hx)``; upwind penalties use ``E^n``."""
    Hxm, Hxp = _traces(d, Hx)
    Hym, Hyp = _traces(d, Hy)
    a = d.alpha
    dHx = d.m_int * (0.5 * (Hxp - Hxm))
    dHy = d.m_int * (0.5 * (Hyp - Hym))
    P = None
    if a > 0:
        w = c.Zp / (c.Zm + c.Zp)
        dHx = dHx + d.m_int * a * (w - 0.5) * (Hxp - Hxm)
        dHy = dHy + d.m_int * a * (w - 0.5) * (Hyp - Hym)
        Em, Ep = _traces(d, E)
        P = d.m_int * a * (Ep - Em) / (c.Zm + c.Zp) - d.m_pec * a * Em / c.Zm
    if d.has_abc:
        dHx = dHx - d.m_abc * 0.5 * Hxm
        dHy = dHy - d.m_abc * 0.5 * Hym
        if P is None:
            Em, _ = _traces(d, E)
            P = torch.zeros_like(Em)
        P = P - d.m_abc * 0.5 * Em / c.Zm
    fx = d.nxf * dHy
    fy = d.nyf * dHx
    if P is not None:
        fx = fx + d.nxf * d.nxf * P
        fy = fy - d.nyf * d.nyf * P
    dxHy = d.rx * torch.matmul(Hy, d.DrT) + d.sx * torch.matmul(Hy, d.DsT) + _lift(d, fx)
    dyHx = d.ry * torch.matmul(Hx, d.DrT) + d.sy * torch.matmul(Hx, d.DsT) + _lift(d, fy)
    return dxHy, dyHx


# --------------------------------------------------------------------------- step
def _cached(st: StepStatic, key, make):
    """Small index tensors cached on the static (no host-device copy per step)."""
    cache = st.__dict__.setdefault("_index_cache", {})
    if key not in cache:
        cache[key] = make()
    return cache[key]


def interface_trace(st: StepStatic, Ez, compact: bool):
    """Projected Yee trace ``E*`` on the interface face slots: compact
    ``(S, nX)`` in coupling order (fused kernels) or dense ``(S, K * 3 Nfp)``."""
    d, S = st.dg, st.nshot
    rows = [cp.to_dg(Ez[:, cp.row, :]) for cp in st.couplings]
    if compact:
        return torch.cat(rows, 1)
    Eext = torch.zeros((S, d.K * 3 * d.Nfp), dtype=Ez.dtype, device=Ez.device)
    for cp, v in zip(st.couplings, rows):
        Eext = Eext.index_copy(1, cp.slots, v)
    return Eext


def step(st: StepStatic, c: StepCoefs, state, src=None, amp=None):
    """One leapfrog step of the coupled system (sources with samples ``amp``)."""
    (Ez, Hx, Hy, pex, pey, phx, phy, E, Hxd, Hyd, qex, qey, qhx, qhy) = state
    y, d = st.yee, st.dg
    S = st.nshot
    fy = y is not None and y.cuda is not None and Ez.is_cuda
    fdg = d is not None and d.cuda is not None and E.is_cuda
    if fy:
        Hx, Hy, phy, phx = y.cuda.h_update(Ez, Hx, Hy, phy, phx)
    elif y is not None:
        dEy = F.pad(Ez[:, :, 1:] - Ez[:, :, :-1], (0, 1))
        dEx = F.pad(Ez[:, 1:, :] - Ez[:, :-1, :], (0, 0, 0, 1))
        A, B, Em_, Fm = y.h_y
        Hx = Hx - y.cHx * dEy - (A * dEy + B * phy)
        phy = Em_ * phy - Fm * dEy
        A, B, Em_, Fm = y.h_x
        Hy = Hy + y.cHy * dEx + (A * dEx + B * phx)
        phx = Em_ * phx - Fm * dEx
    sc = st.staircase
    if fdg:
        if sc is not None:
            X = sc.trace(Ez, True)
        else:
            X = interface_trace(st, Ez, True) if st.couplings else None
        T = d.cuda.terms(0, E, Hxd, Hyd, X, c.Ym, c.Yp)
        Hxd, Hyd, qhx, qhy = d.cuda.update_h(T, Hxd, Hyd, qhx, qhy)
    elif d is not None:
        if sc is not None:
            Eext = sc.trace(Ez, False, d.K, 3 * d.Nfp)
        elif st.couplings:
            Eext = interface_trace(st, Ez, False)
        else:
            Eext = torch.zeros((S, d.K * 3 * d.Nfp), dtype=E.dtype, device=E.device)
        dxE, dyE = dg_grad_e(d, c, E, Hxd, Hyd, Eext)
        rhs_hx, rhs_hy = -dyE, dxE
        if d.pml_idx is not None:
            idx = d.pml_idx
            ra_x, rb_x, re_x, rf_x = d.pml_x
            ra_y, rb_y, re_y, rf_y = d.pml_y
            dx_p, dy_p = dxE[:, idx], dyE[:, idx]
            rhs_hx = rhs_hx.index_copy(1, idx, -(ra_y * dy_p + rb_y * qhx))
            rhs_hy = rhs_hy.index_copy(1, idx, ra_x * dx_p + rb_x * qhy)
            qhx = re_y * qhx - rf_y * dy_p
            qhy = re_x * qhy - rf_x * dx_p
        Hxd = Hxd + d.ch * rhs_hx
        Hyd = Hyd + d.ch * rhs_hy
    if sc is not None:
        Th, Tv = sc.traces(Hxd, Hyd)
        Hx, Hy = sc.ghosts(Hx, Hy, Th, Tv)
    elif d is not None and st.couplings:
        hy = Hyd.reshape(S, -1)
        rows = _cached(st, "ghosts", lambda: torch.as_tensor(
            [cp.ghost for cp in st.couplings], device=Hy.device))
        vals = torch.stack([2.0 * cp.to_fd(hy.index_select(1, cp.nodes)) - Hy[:, cp.inner, :]
                            for cp in st.couplings], dim=1)
        Hy = Hy.index_copy(1, rows, vals)
    Ez_old = Ez
    if fy:
        Ez, pex, pey = y.cuda.e_update(Hx, Hy, Ez, pex, pey, c.fd_ca, c.fd_cb)
    elif y is not None:
        dHyx = F.pad(Hy[:, 1:, :] - Hy[:, :-1, :], (0, 0, 1, 0))
        dHxy = F.pad(Hx[:, :, 1:] - Hx[:, :, :-1], (1, 0))
        Ax, Bx, Ex, Fx = y.e_x
        Ay, By, Ey, Fy = y.e_y
        rhs = (dHyx / y.dx - dHxy / y.dy + (Ax * dHyx + Bx * pex) - (Ay * dHxy + By * pey))
        pex = Ex * pex - Fx * dHyx
        pey = Ey * pey - Fy * dHxy
        Ez = c.fd_ca * Ez + c.fd_cb * rhs
    if sc is not None:
        Ez = sc.concave_update(Ez, Ez_old, Hx, Hy, c.fd_ca, c.fd_cb, Th, Tv)
    if fdg:
        T = d.cuda.terms(1, E, Hxd, Hyd, None, c.Zm, c.Zp)
        E, qex, qey = d.cuda.update_e(T, E, qex, qey, c.dg_ca, c.dg_cb)
    elif d is not None:
        dxHy, dyHx = dg_curl_h(d, c, E, Hxd, Hyd)
        rhs = dxHy - dyHx
        if d.pml_idx is not None:
            idx = d.pml_idx
            ra_x, rb_x, re_x, rf_x = d.pml_x
            ra_y, rb_y, re_y, rf_y = d.pml_y
            a_p, b_p = dxHy[:, idx], dyHx[:, idx]
            rhs = rhs.index_copy(1, idx, (ra_x * a_p + rb_x * qex) - (ra_y * b_p + rb_y * qey))
            qex = re_x * qex - rf_x * a_p
            qey = re_y * qey - rf_y * b_p
        E = c.dg_ca * E + c.dg_cb * rhs
    if src is not None:
        Ez, E = inject(st, c, src, amp, Ez, E)
    return (Ez, Hx, Hy, pex, pey, phx, phy, E, Hxd, Hyd, qex, qey, qhx, qhy)


@dataclass
class PointOps:
    """Batched point operators; source weights exclude ``cb`` (applied per step)."""

    fd_shot: torch.Tensor
    fd_node: torch.Tensor
    fd_item: torch.Tensor
    fd_w: torch.Tensor
    dg_shot: torch.Tensor
    dg_elem: torch.Tensor
    dg_item: torch.Tensor
    dg_w: torch.Tensor
    count: int


def inject(st: StepStatic, c: StepCoefs, ops: PointOps, amp, Ez, E):
    """Subtract ``cb w J`` (DeepGPR amplitude convention ``J = w delta``)."""
    S = st.nshot
    if ops.fd_shot.numel():
        shape = Ez.shape
        vals = (c.fd_cb.reshape(-1).index_select(0, ops.fd_node) * ops.fd_w
                * amp.index_select(0, ops.fd_item))
        # flat index_add (index_put with accumulate sorts and cannot be graph-captured)
        Ez = Ez.reshape(-1).index_add(0, ops.fd_shot * Ez[0].numel() + ops.fd_node,
                                      -vals).view(shape)
    if ops.dg_shot.numel():
        d = st.dg
        vals = c.dg_cb[ops.dg_elem] * ops.dg_w * amp[ops.dg_item][:, None]
        E = E.reshape(S * d.K, d.Np).index_add(0, ops.dg_shot * d.K + ops.dg_elem, -vals).view(
            S, d.K, d.Np)
    return Ez, E


def sample(st: StepStatic, ops: PointOps, state):
    """Ez at the receivers, ``(nshot, nrx)``."""
    S, n = st.nshot, ops.count
    Ez, E = state[0], state[7]
    out = torch.zeros((S * n,), dtype=Ez.dtype if Ez.numel() else E.dtype,
                      device=Ez.device if Ez.numel() else E.device)
    if ops.fd_shot.numel():
        vals = Ez.reshape(S, -1)[ops.fd_shot, ops.fd_node] * ops.fd_w
        out = out.index_add(0, ops.fd_shot * n + ops.fd_item, vals)
    if ops.dg_shot.numel():
        d = st.dg
        e = E.reshape(S * d.K, d.Np)[ops.dg_shot * d.K + ops.dg_elem]
        out = out.index_add(0, ops.dg_shot * n + ops.dg_item, (e * ops.dg_w).sum(-1))
    return out.view(S, n)


__all__ = [
    "CouplingStatic", "DGStatic", "PointOps", "STATE_SIZE", "StepCoefs", "StepStatic",
    "YeeStatic", "build_coefficients", "coupling_static", "dg_coefficients", "dg_static",
    "inject", "interface_trace", "sample", "step", "yee_coefficients", "yee_static",
    "zero_state",
]

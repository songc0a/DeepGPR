"""CUDA kernels of the 2D (TM) hybrid (``csrc/dg2d.cu``, ``csrc/yee2d.cu``).

They are compiled into the same library as the 3D kernels
(:func:`DeepGPR.hybrid.cuda3d.load_library`). :class:`DG2Ops` holds the static
data of a :class:`~DeepGPR.hybrid.functional.DGStatic` in kernel layouts, the
raw kernel calls (used by :mod:`.vjp2d`) and the differentiable fused
operators (curl terms with fluxes, updates with the DG CPML); :class:`Yee2Ops`
does the same for the Yee H and E updates. Every kernel has an exact,
atomic-free transpose (the cotangents of the material coefficients included).
"""

from __future__ import annotations

import ctypes

import numpy as np
import torch

from .cuda3d import _check, _cont, _grads, _ptrs, _stream, load_library


def _mirror_slots(vmapM, vmapP, neighbour, K, Nfp):
    """Slot of the neighbouring element holding the exterior node of every face
    slot (face nodes run in opposite directions; own slot on boundaries)."""
    NF = 3 * Nfp
    vm = vmapM.reshape(K, 3, Nfp)
    vp = vmapP.reshape(K, 3, Nfp)
    nb = neighbour
    slot = np.arange(K * NF).reshape(K, 3, Nfp)
    out = slot.copy()
    for f in range(3):
        k2 = nb[:, f]
        inner = k2 != np.arange(K)
        # face of k2 that points back to k
        f2 = np.argmax(nb[k2] == np.arange(K)[:, None], axis=1)
        for m in range(Nfp):
            cand = k2 * NF + f2 * Nfp + (Nfp - 1 - m)
            out[:, f, m] = np.where(inner, cand, slot[:, f, m])
    ok = vm.reshape(-1)[out.reshape(-1)] == vp.reshape(-1)
    if not np.all(ok):
        raise RuntimeError("2D mirror face slots not found.")
    return out.reshape(-1)


class DG2Ops:
    """Kernel data and operators of a 2D :class:`~DeepGPR.hybrid.functional.DGStatic`."""

    def __init__(self, lib, d, slots=()):
        if d.Np not in (3, 6, 10):
            raise ValueError("2D CUDA kernels support P1-P3 only.")
        self.lib = lib
        K, Np, Nfp = d.K, d.Np, d.Nfp
        self.K, self.Np, self.Nfp, self.NF = K, Np, Nfp, 3 * Nfp
        dev = d.LIFTT.device
        self.dtype = d.LIFTT.dtype
        self.code = {torch.float32: 0, torch.float64: 1}[self.dtype]
        c = lambda t: t.contiguous()  # noqa: E731
        i32 = lambda t: t.to(torch.int32).contiguous()  # noqa: E731
        face = lambda t: c(t.reshape(K, 3, Nfp)[..., 0])  # noqa: E731  per face (affine)
        self._keep = [c(d.rx.reshape(-1)), c(d.ry.reshape(-1)), c(d.sx.reshape(-1)),
                      c(d.sy.reshape(-1)), face(d.nxf), face(d.nyf), face(d.fscale),
                      face(d.m_int), face(d.m_pec), face(d.m_abc), face(d.m_if)]
        vmapM = d.vmapM.cpu().numpy()
        vmapP = d.vmapP.cpu().numpy()
        slotP = _mirror_slots(vmapM, vmapP, d.neighbour.cpu().numpy(), K, Nfp)
        fmask = vmapM[: self.NF]                      # element-local node of each slot (k = 0)
        self.vmapM, self.vmapP = i32(d.vmapM), i32(d.vmapP)
        self.slotP = torch.as_tensor(slotP, dtype=torch.int32, device=dev)
        self.fmask = torch.as_tensor(fmask, dtype=torch.int32, device=dev)
        self.nX = int(sum(int(t.numel()) for t in slots))
        self.xmap = None
        if self.nX:
            xmap = torch.full((K * self.NF,), -1, dtype=torch.int32, device=dev)
            xmap[torch.cat(list(slots))] = torch.arange(self.nX, dtype=torch.int32, device=dev)
            self.xmap = xmap
        self.Dr, self.Ds, self.LIFT = c(d.DrT.t()), c(d.DsT.t()), c(d.LIFTT.t())
        self._keep += [self.vmapM, self.vmapP, self.slotP, self.fmask, self.xmap,
                       self.Dr, self.Ds, self.LIFT]
        k = self._keep
        self.gp = _ptrs(k[0:11] + [self.vmapM, self.vmapP, self.slotP, self.fmask, self.xmap,
                                   self.Dr, self.Ds, self.LIFT])
        self.gd = (ctypes.c_double * 1)(float(d.alpha))
        # update data: ch, CPML map and recursion coefficients (rax rbx rex rfx ray rby rey rfy)
        self.ch = c(d.ch.reshape(-1))
        self.Kp = 0 if d.pml_idx is None else int(d.pml_idx.numel())
        self.pml = None
        pml = [None] * 8
        if self.Kp:
            m = torch.full((K,), -1, dtype=torch.int32, device=dev)
            m[d.pml_idx] = torch.arange(self.Kp, dtype=torch.int32, device=dev)
            self.pml = m
            pml = [c(t) for t in list(d.pml_x) + list(d.pml_y)]
        self._keep += [self.ch, self.pml] + pml
        self.up = _ptrs([self.ch, self.pml] + pml)
        self.gi = (ctypes.c_int * 3)(K, self.nX, self.Kp)
        self._work = None

    # ------------------------------------------------------------- raw kernels
    def terms_fwd(self, mode, E, Hx, Hy, X, cm, cp):
        """Curl terms ``(S, 2, K, Np)``: mode 0 ``(D_x E, D_y E)`` (flux with ``Ym``,
        ``Yp`` and the compact external trace ``X (S, nX)``), mode 1
        ``(D_x Hy, D_y Hx)`` (flux with ``Zm``, ``Zp``). Contiguous inputs."""
        S = E.shape[0]
        out = torch.empty((S, 2, self.K, self.Np), dtype=E.dtype, device=E.device)
        io = _ptrs([E, Hx, Hy, X, cm, cp, out])
        _check(self.lib.dg2d_terms_fwd(self.code, self.Np, mode, S, self.gi, self.gd, self.gp,
                                       io, _stream(E)), "2D terms forward")
        return out

    def terms_bwd(self, mode, E, Hx, Hy, gT, cm, cp, has_x, addE=None, addHx=None, addHy=None):
        """Transpose of :meth:`terms_fwd`: ``(gE, gHx, gHy, gX, gcm, gcp)`` with the
        coefficient cotangents per shot ``(S, K * 3 Nfp)``; the optional addends
        are summed into ``gE``/``gHx``/``gHy`` in the same pass."""
        S = E.shape[0]
        kw = dict(dtype=E.dtype, device=E.device)
        n = int(self.lib.dg2d_bwd_workspace(S, self.K, self.Np, self.Nfp))
        if self._work is None or self._work.numel() < n or self._work.dtype != E.dtype:
            self._work = torch.empty(n, **kw)
        gE, gHx, gHy = torch.empty_like(E), torch.empty_like(Hx), torch.empty_like(Hy)
        gX = torch.empty((S, self.nX), **kw) if has_x else None
        gcm = torch.empty((S, self.K * self.NF), **kw)
        gcp = torch.empty_like(gcm)
        io = _ptrs([E, Hx, Hy, gT, cm, cp, self._work, gX, gcm, gcp, gE, gHx, gHy, addE, addHx,
                    addHy])
        _check(self.lib.dg2d_terms_bwd(self.code, self.Np, mode, S, self.gi, self.gd, self.gp,
                                       io, _stream(E)), "2D terms backward")
        return gE, gHx, gHy, gX, gcm, gcp

    def update_fwd(self, mode, T, f0, f1, q0, q1, ca=None, cb=None):
        """mode 0: ``(Hx', Hy', qhx', qhy')`` from ``(Hx, Hy, qhx, qhy)``; mode 1:
        ``(E', qex', qey')`` from ``(E, qex, qey)`` with ``ca``/``cb`` ``(K,)``."""
        S = T.shape[0]
        n0 = torch.empty_like(f0)
        n1 = torch.empty_like(f1) if mode == 0 else None
        nq0, nq1 = torch.empty_like(q0), torch.empty_like(q1)
        io = _ptrs([T, f0, f1, q0, q1, ca, cb, n0, n1, nq0, nq1])
        _check(self.lib.dg2d_update_fwd(self.code, self.Np, mode, S, self.gi, self.up, io,
                                        _stream(T)), "2D DG update forward")
        return (n0, n1, nq0, nq1) if mode == 0 else (n0, nq0, nq1)

    def update_bwd(self, mode, T, f0, q0, q1, ca, cb, g0, g1, gq0, gq1):
        """Transpose of :meth:`update_fwd`: ``(gT, gf0, gf1, gq0, gq1, gca, gcb)``
        (mode 1: ``gf1`` is ``None``, ``gca``/``gcb`` per shot ``(S, K)``; ``T`` and
        ``f0`` are read in mode 1 only)."""
        S = g0.shape[0]
        kw = dict(dtype=g0.dtype, device=g0.device)
        gT = torch.empty((S, 2, self.K, self.Np), **kw)
        gf0 = torch.empty_like(g0)
        gf1 = torch.empty_like(g0) if mode == 0 else None
        gq0o, gq1o = torch.empty_like(q0), torch.empty_like(q1)
        gca = torch.empty((S, self.K), **kw) if mode == 1 else None
        gcb = torch.empty((S, self.K), **kw) if mode == 1 else None
        io = _ptrs([T, f0, q0, q1, ca, cb, g0, g1, gq0, gq1, gT, gf0, gf1, gq0o, gq1o, gca, gcb])
        _check(self.lib.dg2d_update_bwd(self.code, self.Np, mode, S, self.gi, self.up, io,
                                        _stream(g0)), "2D DG update backward")
        return gT, gf0, gf1, gq0o, gq1o, gca, gcb

    # ------------------------------------------------------------- autograd
    def terms(self, mode, E, Hx, Hy, X, cm, cp):
        return _Terms2.apply(self, mode, E, Hx, Hy, X, cm, cp)

    def update_h(self, T, Hx, Hy, qhx, qhy):
        return _UpdH2.apply(self, T, Hx, Hy, qhx, qhy)

    def update_e(self, T, E, qex, qey, ca, cb):
        return _UpdE2.apply(self, T, E, qex, qey, ca, cb)


class _Terms2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ops, mode, E, Hx, Hy, X, cm, cp):  # type: ignore[override]
        E, Hx, Hy, cm, cp = _cont((E, Hx, Hy, cm, cp))
        X = None if X is None else X.contiguous()
        out = ops.terms_fwd(mode, E, Hx, Hy, X, cm, cp)
        ctx.ops, ctx.mode, ctx.has_x = ops, mode, X is not None
        ctx.save_for_backward(E, Hx, Hy, cm, cp)
        return out

    @staticmethod
    def backward(ctx, gT):  # type: ignore[override]
        E, Hx, Hy, cm, cp = ctx.saved_tensors
        gE, gHx, gHy, gX, gcm, gcp = ctx.ops.terms_bwd(ctx.mode, E, Hx, Hy, gT.contiguous(), cm,
                                                       cp, ctx.has_x)
        shape = cm.shape
        return (None, None, gE, gHx, gHy, gX, gcm.sum(0).view(shape), gcp.sum(0).view(shape))


class _UpdH2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ops, T, Hx, Hy, qhx, qhy):  # type: ignore[override]
        T, Hx, Hy, qhx, qhy = _cont((T, Hx, Hy, qhx, qhy))
        out = ops.update_fwd(0, T, Hx, Hy, qhx, qhy)
        ctx.ops = ops
        ctx.save_for_backward(qhx, qhy)
        return out

    @staticmethod
    def backward(ctx, g0, g1, gq0, gq1):  # type: ignore[override]
        qhx, qhy = ctx.saved_tensors
        ref = g0 if g0 is not None else g1
        g0, g1 = _grads((g0, g1), (ref, ref))
        gq0, gq1 = _grads((gq0, gq1), (qhx, qhy))
        gT, gf0, gf1, gq0o, gq1o, _, _ = ctx.ops.update_bwd(0, None, None, qhx, qhy, None, None,
                                                            g0, g1, gq0, gq1)
        return None, gT, gf0, gf1, gq0o, gq1o


class _UpdE2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ops, T, E, qex, qey, ca, cb):  # type: ignore[override]
        T, E, qex, qey = _cont((T, E, qex, qey))
        ca1, cb1 = ca.reshape(-1).contiguous(), cb.reshape(-1).contiguous()
        out = ops.update_fwd(1, T, E, E, qex, qey, ca1, cb1)
        ctx.ops, ctx.shape = ops, ca.shape
        ctx.save_for_backward(T, E, qex, qey, ca1, cb1)
        return out

    @staticmethod
    def backward(ctx, gE, gq0, gq1):  # type: ignore[override]
        T, E, qex, qey, ca, cb = ctx.saved_tensors
        gE = torch.zeros_like(E) if gE is None else gE.contiguous()
        gq0, gq1 = _grads((gq0, gq1), (qex, qey))
        gT, gf0, _, gq0o, gq1o, gca, gcb = ctx.ops.update_bwd(1, T, E, qex, qey, ca, cb, gE,
                                                              None, gq0, gq1)
        return (None, gT, gf0, gq0o, gq1o, gca.sum(0).view(ctx.shape),
                gcb.sum(0).view(ctx.shape))


# --------------------------------------------------------------------------- Yee
class Yee2Ops:
    """Kernel data and operators of a 2D :class:`~DeepGPR.hybrid.functional.YeeStatic`."""

    def __init__(self, lib, y):
        self.lib = lib
        self.NX, self.NY = y.shape
        self.N = self.NX * self.NY
        self.dtype = y.cHx.dtype
        self.code = {torch.float32: 0, torch.float64: 1}[self.dtype]
        arrs = [y.cHx, y.cHy, *y.h_y, *y.h_x, *y.e_x, *y.e_y]
        self._keep = [a.contiguous() for a in arrs]
        self.cp = _ptrs(self._keep)
        self.gd = (ctypes.c_double * 2)(float(y.dx), float(y.dy))
        self._gi = {}

    def gi(self, S):
        if S not in self._gi:
            self._gi[S] = (ctypes.c_int * 3)(S, self.NX, self.NY)
        return self._gi[S]

    def _run(self, side, direction, S, items, ref):
        _check(self.lib.yee2d_run(self.code, side, direction, self.gi(S), self.gd, self.cp,
                                  _ptrs(items), _stream(ref)),
               f"2D Yee {'HE'[side]} {'forward' if direction == 0 else 'backward'}")

    # ------------------------------------------------------------- raw kernels
    def h_fwd(self, Ez, Hx, Hy, phy, phx):
        out = [torch.empty_like(t) for t in (Hx, Hy, phy, phx)]
        self._run(0, 0, Ez.shape[0], [Ez, Hx, Hy, phy, phx] + out, Ez)
        return out

    def h_bwd(self, gHx, gHy, gphy, gphx):
        """Transpose of :meth:`h_fwd`: ``(gEz, gHx, gHy, gphy, gphx)``."""
        out = [torch.empty_like(gHx) for _ in range(5)]
        self._run(0, 1, gHx.shape[0], [gHx, gHy, gphy, gphx] + out, gHx)
        return out

    def e_fwd(self, Hx, Hy, Ez, pex, pey, ca, cb):
        out = [torch.empty_like(t) for t in (Ez, pex, pey)]
        self._run(1, 0, Ez.shape[0], [Hx, Hy, Ez, pex, pey, ca, cb] + out, Ez)
        return out

    def e_bwd(self, Hx, Hy, Ez, pex, pey, ca, cb, gEz, gpex, gpey):
        """Transpose of :meth:`e_fwd`: ``(gHx, gHy, gEz, gpex, gpey, gca, gcb)`` with
        ``gca``/``gcb`` per shot ``(S, NX, NY)``."""
        out = [torch.empty_like(Ez) for _ in range(7)]
        self._run(1, 1, Ez.shape[0], [Hx, Hy, Ez, pex, pey, ca, cb, gEz, gpex, gpey] + out, Ez)
        return out

    # ------------------------------------------------------------- autograd
    def h_update(self, Ez, Hx, Hy, phy, phx):
        return _YeeH2.apply(self, Ez, Hx, Hy, phy, phx)

    def e_update(self, Hx, Hy, Ez, pex, pey, ca, cb):
        return _YeeE2.apply(self, Hx, Hy, Ez, pex, pey, ca, cb)


class _YeeH2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ops, Ez, Hx, Hy, phy, phx):  # type: ignore[override]
        out = ops.h_fwd(*_cont((Ez, Hx, Hy, phy, phx)))
        ctx.ops = ops
        return tuple(out)

    @staticmethod
    def backward(ctx, gHx, gHy, gphy, gphx):  # type: ignore[override]
        ref = next(g for g in (gHx, gHy, gphy, gphx) if g is not None)
        gs = _grads((gHx, gHy, gphy, gphx), (ref,) * 4)
        gEz, a, b, c, d = ctx.ops.h_bwd(*gs)
        return None, gEz, a, b, c, d


class _YeeE2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ops, Hx, Hy, Ez, pex, pey, ca, cb):  # type: ignore[override]
        Hx, Hy, Ez, pex, pey, ca, cb = _cont((Hx, Hy, Ez, pex, pey, ca, cb))
        out = ops.e_fwd(Hx, Hy, Ez, pex, pey, ca, cb)
        ctx.ops = ops
        ctx.save_for_backward(Hx, Hy, Ez, pex, pey, ca, cb)
        return tuple(out)

    @staticmethod
    def backward(ctx, gEz, gpex, gpey):  # type: ignore[override]
        Hx, Hy, Ez, pex, pey, ca, cb = ctx.saved_tensors
        gs = _grads((gEz, gpex, gpey), (Ez, pex, pey))
        gHx, gHy, gEz0, gpex0, gpey0, gca, gcb = ctx.ops.e_bwd(Hx, Hy, Ez, pex, pey, ca, cb, *gs)
        return None, gHx, gHy, gEz0, gpex0, gpey0, gca.sum(0), gcb.sum(0)


def attach(d, device, slots=()) -> bool:
    """Attach :class:`DG2Ops` to a 2D DG static (``d.cuda``) when possible."""
    if torch.device(device).type != "cuda":
        return False
    lib = load_library(device)
    if lib is None:
        return False
    d.cuda = DG2Ops(lib, d, slots)
    return True


def attach_yee(y, device) -> bool:
    """Attach :class:`Yee2Ops` to a 2D Yee static (``y.cuda``) when possible."""
    if torch.device(device).type != "cuda":
        return False
    lib = load_library(device)
    if lib is None:
        return False
    y.cuda = Yee2Ops(lib, y)
    return True


__all__ = ["DG2Ops", "Yee2Ops", "attach", "attach_yee"]

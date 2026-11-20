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

from .cuda3d import _check, _cont, _grads, _ptr, _ptrs, _stream, load_library


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

    def terms_bwd(self, mode, E, Hx, Hy, gT, cm, cp, has_x, addE=None, addHx=None, addHy=None,
                  out=None, acc_coef=None):
        """Transpose of :meth:`terms_fwd`: ``(gE, gHx, gHy, gX, gcm, gcp)`` with the
        coefficient cotangents per shot ``(S, K * 3 Nfp)``; the optional addends
        are summed into ``gE``/``gHx``/``gHy`` in the same pass. ``out = (gE, gHx, gHy)``:
        written there; ``acc_coef = (gcm, gcp)``: the coefficient cotangents are
        accumulated into these running sums (``None`` entries skipped)."""
        S = E.shape[0]
        kw = dict(dtype=E.dtype, device=E.device)
        n = int(self.lib.dg2d_bwd_workspace(S, self.K, self.Np, self.Nfp))
        if self._work is None or self._work.numel() < n or self._work.dtype != E.dtype:
            self._work = torch.empty(n, **kw)
        gE, gHx, gHy = ((torch.empty_like(E), torch.empty_like(Hx), torch.empty_like(Hy))
                        if out is None else out)
        gX = torch.empty((S, self.nX), **kw) if has_x else None
        if acc_coef is None:
            gcm = torch.empty((S, self.K * self.NF), **kw)
            gcp = torch.empty_like(gcm)
        else:
            gcm, gcp = acc_coef
            mode = mode | 4
        io = _ptrs([E, Hx, Hy, gT, cm, cp, self._work, gX, gcm, gcp, gE, gHx, gHy, addE, addHx,
                    addHy])
        _check(self.lib.dg2d_terms_bwd(self.code, self.Np, mode, S, self.gi, self.gd, self.gp,
                                       io, _stream(E)), "2D terms backward")
        return gE, gHx, gHy, gX, gcm, gcp

    def update_fwd(self, mode, T, f0, f1, q0, q1, ca=None, cb=None, out=None):
        """mode 0: ``(Hx', Hy', qhx', qhy')`` from ``(Hx, Hy, qhx, qhy)``; mode 1:
        ``(E', qex', qey')`` from ``(E, qex, qey)`` with ``ca``/``cb`` ``(K,)`` (written into
        ``out``, the same tuple, when given)."""
        S = T.shape[0]
        if out is None:
            n0 = torch.empty_like(f0)
            n1 = torch.empty_like(f1) if mode == 0 else None
            nq0, nq1 = torch.empty_like(q0), torch.empty_like(q1)
        elif mode == 0:
            n0, n1, nq0, nq1 = out
        else:
            (n0, nq0, nq1), n1 = out, None
        io = _ptrs([T, f0, f1, q0, q1, ca, cb, n0, n1, nq0, nq1])
        _check(self.lib.dg2d_update_fwd(self.code, self.Np, mode, S, self.gi, self.up, io,
                                        _stream(T)), "2D DG update forward")
        return (n0, n1, nq0, nq1) if mode == 0 else (n0, nq0, nq1)

    def update_bwd(self, mode, T, f0, q0, q1, ca, cb, g0, g1, gq0, gq1, out_q=None):
        """Transpose of :meth:`update_fwd`: ``(gT, gf0, gf1, gq0, gq1, gca, gcb)``
        (mode 1: ``gf1`` is ``None``, ``gca``/``gcb`` per shot ``(S, K)``; ``T`` and
        ``f0`` are read in mode 1 only); the auxiliary cotangents go to ``out_q``."""
        S = g0.shape[0]
        kw = dict(dtype=g0.dtype, device=g0.device)
        gT = torch.empty((S, 2, self.K, self.Np), **kw)
        gf0 = torch.empty_like(g0)
        gf1 = torch.empty_like(g0) if mode == 0 else None
        gq0o, gq1o = (torch.empty_like(q0), torch.empty_like(q1)) if out_q is None else out_q
        gca = torch.empty((S, self.K), **kw) if mode == 1 else None
        gcb = torch.empty((S, self.K), **kw) if mode == 1 else None
        io = _ptrs([T, f0, q0, q1, ca, cb, g0, g1, gq0, gq1, gT, gf0, gf1, gq0o, gq1o, gca, gcb])
        _check(self.lib.dg2d_update_bwd(self.code, self.Np, mode, S, self.gi, self.up, io,
                                        _stream(g0)), "2D DG update backward")
        return gT, gf0, gf1, gq0o, gq1o, gca, gcb

    # ------------------------------------------------------------- local time stepping
    def lts_z(self, Te, z, cb0, a: float):
        """``z + a cb0 (Te[0] - Te[1])`` (fused, differentiable; ``cb0`` ``(K,)``)."""
        return _LTSZ2.apply(self, Te, z, cb0, float(a))

    def lts_eta(self, T, ex, ey, tavg, beta: float, b: float, w: float):
        """``(beta ex - b ch T[1], beta ey + b ch T[0], tavg + w T)``; ``ex``/``ey``/``tavg``
        ``None`` on the first sub-step."""
        return _LTSEta2.apply(self, T, ex, ey, tavg, float(beta), float(b), float(w))

    # ------------------------------------------------------------- autograd
    def terms(self, mode, E, Hx, Hy, X, cm, cp):
        return _Terms2.apply(self, mode, E, Hx, Hy, X, cm, cp)

    def update_h(self, T, Hx, Hy, qhx, qhy):
        return _UpdH2.apply(self, T, Hx, Hy, qhx, qhy)

    def update_e(self, T, E, qex, qey, ca, cb):
        return _UpdE2.apply(self, T, E, qex, qey, ca, cb)


class _LTSZ2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ops, Te, z, cb0, a):  # type: ignore[override]
        Te, z = Te.contiguous(), z.contiguous()
        cb0 = cb0.reshape(-1).contiguous()
        out = torch.empty_like(z)
        _check(ops.lib.lts2d_z_fwd(ops.code, z.shape[0], ops.K, ops.Np, _ptr(Te), _ptr(z), _ptr(cb0),
                                   a, _ptr(out), _stream(z)), "2D LTS z update")
        ctx.ops, ctx.a = ops, a
        ctx.save_for_backward(Te, cb0)
        return out

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        ops = ctx.ops
        Te, cb0 = ctx.saved_tensors
        g = g.contiguous()
        gTe, gcb0 = torch.empty_like(Te), torch.empty_like(cb0)
        _check(ops.lib.lts2d_z_bwd(ops.code, g.shape[0], ops.K, ops.Np, _ptr(Te), _ptr(g), _ptr(cb0),
                                   ctx.a, _ptr(gTe), _ptr(gcb0), _stream(g)), "2D LTS z transpose")
        return None, gTe, g, gcb0.view(-1, 1), None


class _LTSEta2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ops, T, ex, ey, tavg, beta, b, w):  # type: ignore[override]
        T = T.contiguous()
        first = ex is None
        S = T.shape[0]
        shape = (S, ops.K, ops.Np)
        ex_out = torch.empty(shape, dtype=T.dtype, device=T.device)
        ey_out = torch.empty(shape, dtype=T.dtype, device=T.device)
        tavg_out = torch.empty_like(T)
        c = lambda t: None if t is None else t.contiguous()  # noqa: E731
        _check(ops.lib.lts2d_eta_fwd(ops.code, S, ops.K, ops.Np, _ptr(T), _ptr(c(ex)), _ptr(c(ey)),
                                     _ptr(c(tavg)), _ptr(ops.ch), beta, b, w, int(first), _ptr(ex_out),
                                     _ptr(ey_out), _ptr(tavg_out), _stream(T)), "2D LTS eta update")
        ctx.ops, ctx.first, ctx.coef = ops, first, (beta, b, w)
        ctx.shape, ctx.tshape, ctx.dtype_, ctx.device_ = shape, tuple(T.shape), T.dtype, T.device
        ctx.set_materialize_grads(False)
        return ex_out, ey_out, tavg_out

    @staticmethod
    def backward(ctx, gex, gey, gtavg):  # type: ignore[override]
        ops, (beta, b, w) = ctx.ops, ctx.coef
        S = ctx.shape[0]
        gT = torch.empty(ctx.tshape, dtype=ctx.dtype_, device=ctx.device_)
        c = lambda t: None if t is None else t.contiguous()  # noqa: E731
        mk = lambda: None if ctx.first else torch.empty(ctx.shape, dtype=ctx.dtype_, device=ctx.device_)  # noqa: E731
        gex_out, gey_out = mk(), mk()
        _check(ops.lib.lts2d_eta_bwd(ops.code, S, ops.K, ops.Np, _ptr(c(gex)), _ptr(c(gey)), _ptr(c(gtavg)),
                                     _ptr(ops.ch), beta, b, w, _ptr(gT), _ptr(gex_out), _ptr(gey_out),
                                     _stream(gT)), "2D LTS eta transpose")
        return None, gT, gex_out, gey_out, (None if ctx.first else c(gtavg)), None, None, None


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
        self.y = y
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
               f"2D Yee {'HE'[side]} {'forward' if (direction & 1) == 0 else 'backward'}")

    # ------------------------------------------------------------- raw kernels
    def h_fwd(self, Ez, Hx, Hy, phy, phx, out=None):
        out = [torch.empty_like(t) for t in (Hx, Hy, phy, phx)] if out is None else list(out)
        self._run(0, 0, Ez.shape[0], [Ez, Hx, Hy, phy, phx] + out, Ez)
        return out

    def h_bwd(self, gHx, gHy, gphy, gphx, out=None, gEz_acc=None):
        """Transpose of :meth:`h_fwd`: ``(gEz, gHx, gHy, gphy, gphx)``; ``out`` = the last
        four (written there); ``gEz_acc``: ``gEz`` accumulated into it."""
        rest = [torch.empty_like(gHx) for _ in range(4)] if out is None else list(out)
        gEz = torch.empty_like(gHx) if gEz_acc is None else gEz_acc
        self._run(0, 1 | (2 if gEz_acc is not None else 0), gHx.shape[0],
                  [gHx, gHy, gphy, gphx, gEz] + rest, gHx)
        return [gEz] + rest

    def e_fwd(self, Hx, Hy, Ez, pex, pey, ca, cb, out=None):
        out = [torch.empty_like(t) for t in (Ez, pex, pey)] if out is None else list(out)
        self._run(1, 0, Ez.shape[0], [Hx, Hy, Ez, pex, pey, ca, cb] + out, Ez)
        return out

    def e_bwd(self, Hx, Hy, Ez, pex, pey, ca, cb, gEz, gpex, gpey, out=None, src_acc=None,
              coef_acc=None, shot_sum=False):
        """Transpose of :meth:`e_fwd`: ``(gHx, gHy, gEz, gpex, gpey, gca, gcb)`` with
        ``gca``/``gcb`` per shot ``(S, NX, NY)`` (summed over the shots, ``(NX, NY)``, with
        ``shot_sum``). ``out = (gEz, gpex, gpey)``: written there; ``src_acc = (gHx, gHy)``:
        accumulated into; ``coef_acc = (gca, gcb)``: accumulated into (per shot, or summed
        over the shots in the kernel when of shape ``(NX, NY)``)."""
        new = lambda: torch.empty_like(Ez)  # noqa: E731
        gH = [new(), new()] if src_acc is None else list(src_acc)
        mid = [new(), new(), new()] if out is None else list(out)
        if coef_acc is None:
            gc = [torch.empty_like(Ez[0]), torch.empty_like(Ez[0])] if shot_sum else [new(), new()]
        else:
            gc = list(coef_acc)
            shot_sum = gc[0].numel() == self.N
        flags = (1 | (2 if src_acc is not None else 0) | (4 if coef_acc is not None else 0)
                 | (8 if shot_sum else 0))
        self._run(1, flags, Ez.shape[0],
                  [Hx, Hy, Ez, pex, pey, ca, cb, gEz, gpex, gpey] + gH + mid + gc, Ez)
        return gH + mid + gc

    # ------------------------------------------------------------- autograd
    # ------------------------------------------------------------- local time stepping
    def slab_data(self, sl):
        """Kernel data of a :class:`~DeepGPR.hybrid.functional.LTSSlab2D` (cached on it):
        ``(n, NY)`` uint8 masks of the fine Ez / Hx / Hy nodes and the static Yee factors
        ``cHx, cHy, Ay, Fy, Ax, Fx`` of the slab rows."""
        sd = getattr(sl, "cuda", None)
        if sd is None:
            y, lo, hi = self.y, sl.lo, sl.hi
            n = hi - lo + 1
            u8 = lambda m: m.expand(1, n, self.NY).reshape(n, self.NY).to(torch.uint8).contiguous()  # noqa: E731
            c = lambda t: t[lo:hi + 1].to(self.dtype).contiguous()  # noqa: E731
            sd = dict(n=n, masks=[u8(sl.fine_e), u8(sl.fine_hx), u8(sl.fine_hy)],
                      stat=[c(y.cHx), c(y.cHy), c(y.h_y[0]), c(y.h_y[3]), c(y.h_x[0]), c(y.h_x[3])])
            sl.cuda = sd
        return sd

    def slab_z(self, sd, ex, ey, z, cb0, a: float):
        """``where(fine_e, z + a cb0 curl_h(eta), z)`` on a slab."""
        return _SlabQ2.apply(self, sd, float(a), ex, ey, z, cb0)

    def slab_r(self, sd, z, eta, ravg, first: bool, last: bool, beta: float, b: float, w: float,
               r0=None):
        """``(r, eta_new, ravg_new, corr)`` from the Yee differences ``r = (dEy, dEx)`` of
        ``z``: ``r`` on the first sub-step only, ``eta_new`` (not on the last), ``ravg_new``
        and, with ``r0`` (last sub-step), the corrections ``(Hx, phy, Hy, phx)`` of the
        fine rows (:func:`DeepGPR.hybrid.functional.lts_h_half_step`)."""
        e = (None, None) if eta is None else tuple(eta)
        a = (None, None) if ravg is None else tuple(ravg)
        q = (None, None) if r0 is None else tuple(r0)
        out = _SlabR2.apply(self, sd, bool(first), bool(last), float(beta), float(b), float(w), z,
                            *e, *a, *q)
        return ((list(out[0:2]) if first else None), (None if last else list(out[2:4])),
                list(out[4:6]), (list(out[6:10]) if r0 is not None else None))

    def h_update(self, Ez, Hx, Hy, phy, phx):
        return _YeeH2.apply(self, Ez, Hx, Hy, phy, phx)

    def e_update(self, Hx, Hy, Ez, pex, pey, ca, cb):
        return _YeeE2.apply(self, Hx, Hy, Ez, pex, pey, ca, cb)


class _SlabQ2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ops, sd, a, ex, ey, z, cb0):  # type: ignore[override]
        ex, ey, z, cb0 = _cont((ex, ey, z, cb0))
        out = torch.empty_like(z)
        q = _ptrs([ex, ey, z, cb0, sd["masks"][0], out])
        _check(ops.lib.lts2d_slab_q(ops.code, 0, z.shape[0], sd["n"], ops.NY, ops.gd, q, a, _stream(z)),
               "LTS slab E rates (2D)")
        ctx.ops, ctx.sd, ctx.a = ops, sd, a
        ctx.save_for_backward(ex, ey, cb0)
        return out

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        ops, sd = ctx.ops, ctx.sd
        ex, ey, cb0 = ctx.saved_tensors
        g = g.contiguous()
        gex, gey = torch.empty_like(ex), torch.empty_like(ey)
        gcb0 = torch.empty_like(cb0) if ctx.needs_input_grad[6] else None
        q = _ptrs([ex, ey, g, cb0, sd["masks"][0], gex, gey, gcb0])
        _check(ops.lib.lts2d_slab_q(ops.code, 1, g.shape[0], sd["n"], ops.NY, ops.gd, q, ctx.a,
                                    _stream(g)), "LTS slab E rates transpose (2D)")
        return None, None, None, gex, gey, g, gcb0


class _SlabR2(torch.autograd.Function):
    """Ten outputs: ``(dEy, dEx)`` (first sub-step), ``(ex, ey)_new`` (not the last),
    ``(ravg_y, ravg_x)_new``, ``(corr_hx, corr_phy, corr_hy, corr_phx)`` (with ``r0``);
    unused ones empty."""

    @staticmethod
    def forward(ctx, ops, sd, first, last, beta, b, w, z, ex, ey, ay, ax, q0y, q0x):  # type: ignore[override]
        z = z.contiguous()
        eta = _cont((ex, ey)) if (ex is not None and not first and not last) else [None, None]
        ravg = _cont((ay, ax)) if not first else [None, None]
        cm = q0y is not None
        r0 = _cont((q0y, q0x)) if cm else [None, None]
        new = lambda: torch.empty_like(z)  # noqa: E731
        r_out = [new(), new()] if first else [None, None]
        eta_out = [new(), new()] if not last else [None, None]
        ravg_out = [new(), new()]
        corr = [new() for _ in range(4)] if cm else [None] * 4
        st = sd["stat"] + sd["masks"][1:]
        q = _ptrs([z] + list(eta) + list(ravg) + st + r_out + eta_out + ravg_out + list(r0) + corr)
        _check(ops.lib.lts2d_slab_r(ops.code, 0, z.shape[0], sd["n"], ops.NY, ops.gd, q, beta, b, w,
                                    int(first), _stream(z)), "LTS slab H rates (2D)")
        ctx.ops, ctx.sd, ctx.flags, ctx.coef = ops, sd, (first, last, cm), (beta, b, w)
        ctx.shape, ctx.dtype_, ctx.device_ = z.shape, z.dtype, z.device
        ctx.set_materialize_grads(False)
        empty = lambda: torch.empty(0, dtype=z.dtype, device=z.device)  # noqa: E731
        return tuple(t if t is not None else empty() for t in r_out + eta_out + ravg_out + corr)

    @staticmethod
    def backward(ctx, *gs):  # type: ignore[override]
        ops, sd = ctx.ops, ctx.sd
        first, last, cm = ctx.flags
        beta, b, w = ctx.coef
        c = lambda t: None if t is None else t.contiguous()  # noqa: E731
        gr = [c(t) for t in gs[0:2]] if first else [None, None]
        ge = [c(t) for t in gs[2:4]] if not last else [None, None]
        ga = [c(t) for t in gs[4:6]]
        gq = [c(t) for t in gs[6:10]] if cm else [None] * 4
        kw = dict(dtype=ctx.dtype_, device=ctx.device_)
        new = lambda: torch.empty(ctx.shape, **kw)  # noqa: E731
        gz = new()
        gein = [new(), new()] if (not first and not last) else [None, None]
        gain = [new(), new()] if (cm and not first) else [None, None]
        gr0 = [new(), new()] if cm else [None, None]
        st = sd["stat"] + sd["masks"][1:]
        q = _ptrs(gr + ge + ga + st + gq + [gz] + gein + gain + gr0)
        _check(ops.lib.lts2d_slab_r(ops.code, 1, ctx.shape[0], sd["n"], ops.NY, ops.gd, q, beta, b, w,
                                    int(first), _stream(gz)), "LTS slab H rates transpose (2D)")
        g_ravg = [None, None] if first else (gain if cm else ga)
        return (None,) * 7 + (gz,) + tuple(gein) + tuple(g_ravg) + tuple(gr0)


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
        gHx, gHy, gEz0, gpex0, gpey0, gca, gcb = ctx.ops.e_bwd(Hx, Hy, Ez, pex, pey, ca, cb, *gs,
                                                               shot_sum=True)
        return None, gHx, gHy, gEz0, gpex0, gpey0, gca.view(ca.shape), gcb.view(cb.shape)


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

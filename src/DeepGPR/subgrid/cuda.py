"""Fused CUDA coupling kernels of the Huygens subgridding (``csrc/hsg3d.cu``).

Compiled on first use with ``nvcc`` (like :mod:`DeepGPR.hybrid.cuda3d`) and
cached in ``$DEEPGPR_CACHE_DIR/subgrid``. :class:`TapsCuda` evaluates all
precursors of one field kind in one launch; :class:`CorrCuda` applies one
surface correction (all curl terms, two launches for E so that no launch has
two entries with the same target). Both are exposed as differentiable,
in-place operations whose backward passes are the transpose kernels.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import threading
from pathlib import Path
from typing import List

import numpy as np
import torch

from ..hybrid.cuda3d import _nvcc, _ptrs, _stream
from ..utils.logger import get_logger

_LOGGER = get_logger(__name__)
_SRC = Path(__file__).resolve().parent / "csrc" / "hsg3d.cu"
_LOCK = threading.Lock()
_LIBS: dict = {}
_P = ctypes.c_void_p
_I = ctypes.c_int


def _cache_dir() -> Path:
    root = os.environ.get("DEEPGPR_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "deepgpr"
    return base / "subgrid"


def load_library(device=None):
    """Compile (once) and load the coupling kernels; ``None`` if unavailable."""
    if os.environ.get("DEEPGPR_SUBGRID_DISABLE_CUDA") or not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    with _LOCK:
        if arch in _LIBS:
            return _LIBS[arch]
        lib = None
        try:
            src = _SRC.read_bytes()
            digest = hashlib.sha256(src + arch.encode()).hexdigest()[:16]
            path = _cache_dir() / f"libhsg3d_{arch}_{digest}.so"
            if not path.exists():
                nvcc = _nvcc()
                if nvcc is None:
                    raise RuntimeError("nvcc not found")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                subprocess.run([nvcc, "-O3", "-shared", "-Xcompiler", "-fPIC", f"-arch={arch}",
                                "-o", str(tmp), str(_SRC)], check=True, capture_output=True)
                os.replace(tmp, path)
            lib = ctypes.CDLL(str(path))
            lib.hsg3d_taps.restype = _I
            lib.hsg3d_taps.argtypes = [_I, _I, _P, _P, _P]
            lib.hsg3d_corr.restype = _I
            lib.hsg3d_corr.argtypes = [_I, _I, _P, _P, _P, _P]
        except Exception as exc:  # pragma: no cover - depends on the toolchain
            _LOGGER.warning("Subgrid CUDA kernels unavailable (%s); using PyTorch.", exc)
            lib = None
        _LIBS[arch] = lib
        return lib


def _check(rc: int, what: str):
    if rc != 0:
        raise RuntimeError(f"Subgrid CUDA kernel {what} failed (code {rc}).")


def _ll(vals):
    return (ctypes.c_longlong * len(vals))(*[int(v) for v in vals])


class TapsCuda:
    """All precursors of one kind: ``y[:, n] = sum_t w[n, t] x_{comp[n]}[:, idx[n, t]]``."""

    def __init__(self, lib, ops, comps, n_in: int, dtype, device):
        self.lib = lib
        T = max([op.T for op in ops] + [1])
        n = sum(op.n for op in ops)
        idx = torch.zeros((n, T), dtype=torch.long, device=device)
        w = torch.zeros((n, T), dtype=dtype, device=device)
        comp = torch.zeros(n, dtype=torch.int8, device=device)
        o = 0
        for op, c in zip(ops, comps):
            if op.n:
                idx[o:o + op.n, :op.T] = op.idx.view(op.n, op.T)
                w[o:o + op.n, :op.T] = op.w
                comp[o:o + op.n] = c
            o += op.n
        self.idx, self.w, self.comp = idx.contiguous(), w.contiguous(), comp
        self.n, self.T, self.N = n, T, int(n_in)
        self.code = 1 if dtype == torch.float64 else 0

    def fwd(self, x) -> torch.Tensor:
        S = x[0].shape[0]
        y = torch.empty((S, self.n), dtype=x[0].dtype, device=x[0].device)
        io = _ptrs([self.idx, self.w, self.comp] + list(x) + [y])
        _check(self.lib.hsg3d_taps(self.code, 0, _ll([S, self.n, self.T, self.N]), io,
                                   _stream(y)), "taps forward")
        return y

    def bwd(self, gy: torch.Tensor, gx) -> None:
        """``gx[c] += taps^T gy`` (atomics, in place)."""
        S = gy.shape[0]
        io = _ptrs([self.idx, self.w, self.comp] + list(gx) + [gy.contiguous()])
        _check(self.lib.hsg3d_taps(self.code, 1, _ll([S, self.n, self.T, self.N]), io,
                                   _stream(gy)), "taps backward")


class CorrCuda:
    """One surface correction (:class:`~DeepGPR.subgrid.hsg.Correction`) as fused launches."""

    def __init__(self, lib, corr, n_target: int, n_source: int, mode: int, dtype, device):
        self.lib = lib
        self.mode = mode
        self.code = 1 if dtype == torch.float64 else 0
        self.use_cb = corr.use_cb
        comp, idx, coef, scomp, sidx = [], [], [], [], []
        for t, (c, i, k, (a, b)) in enumerate(zip(corr.comp, corr.idx, corr.coef, corr.slices)):
            comp.append(torch.full((b - a,), c, dtype=torch.int8, device=device))
            idx.append(i)
            coef.append(k)
            if mode == 1:
                scomp.append(torch.full((b - a,), corr.src_comp[t], dtype=torch.int8, device=device))
                sidx.append(corr.src_idx[t])
        self.comp = torch.cat(comp)
        self.idx = torch.cat(idx).contiguous()
        self.coef = torch.cat(coef).contiguous()
        self.scomp = torch.cat(scomp) if mode == 1 else None
        self.sidx = torch.cat(sidx).contiguous() if mode == 1 else None
        self.m = int(self.idx.numel())
        self.N, self.Nsrc = int(n_target), int(n_source)
        # duplicate-free launch ranges: the terms are ordered (0, 2, 4, 1, 3, 5) and only
        # the two terms of one component share targets (E corrections at box edges)
        if corr.use_cb:
            mid = corr.slices[2][1]
            self.ranges = [(0, mid), (mid, self.m)]
        else:
            self.ranges = [(0, self.m)]

    def _io(self, a0, f, V0, V1, src, cb):
        sl = lambda t: None if t is None else t[a0:]  # noqa: E731
        offs = lambda t: None if t is None else t[:, a0:]  # noqa: E731
        return [sl(self.comp), sl(self.idx), sl(self.coef), cb if self.use_cb else None,
                sl(self.scomp), sl(self.sidx)] + list(f) + [offs(V0), offs(V1)] + (
            list(src) if src is not None else [None] * 3)

    def fwd(self, f, a=1.0, b=0.0, V0=None, V1=None, src=None, cb=None):
        """In place on ``f`` (3 contiguous fields)."""
        S = f[0].shape[0]
        ldv = 0 if V0 is None else V0.shape[1]
        for a0, a1 in self.ranges:
            io = _ptrs(self._io(a0, f, V0, V1, src, cb))
            gi = _ll([S, a1 - a0, self.mode, self.N, self.Nsrc, ldv])
            gd = (ctypes.c_double * 2)(a, b)
            _check(self.lib.hsg3d_corr(self.code, 0, gi, gd, io, _stream(f[0])), "corr forward")

    def bwd(self, gf, a=1.0, b=0.0, V0=None, V1=None, src=None, cb=None, gV0=None, gV1=None,
            gsrc=None, gcb=None):
        S = gf[0].shape[0]
        ldv = 0 if V0 is None else V0.shape[1]
        for a0, a1 in self.ranges:
            g0 = None if gV0 is None else gV0[:, a0:]
            g1 = None if gV1 is None else gV1[:, a0:]
            io = _ptrs(self._io(a0, gf, V0, V1, src, cb) + [g0, g1]
                       + (list(gsrc) if gsrc is not None else [None] * 3) + [gcb])
            gi = _ll([S, a1 - a0, self.mode, self.N, self.Nsrc, ldv])
            gd = (ctypes.c_double * 2)(a, b)
            _check(self.lib.hsg3d_corr(self.code, 1, gi, gd, io, _stream(gf[0])), "corr backward")


# --------------------------------------------------------------------------- autograd
class _Taps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op: TapsCuda, x0, x1, x2):  # type: ignore[override]
        ctx.op = op
        ctx.shapes = (x0.shape, x1.shape, x2.shape)
        return op.fwd([x0.contiguous(), x1.contiguous(), x2.contiguous()])

    @staticmethod
    def backward(ctx, gy):  # type: ignore[override]
        gx = [torch.zeros(s, dtype=gy.dtype, device=gy.device) for s in ctx.shapes]
        ctx.op.bwd(gy, gx)
        return (None, *gx)


class _CorrLerp(torch.autograd.Function):
    """``f += corr(a V0 + b V1)`` in place (IS corrections)."""

    @staticmethod
    def forward(ctx, op: CorrCuda, a, b, cb, V0, V1, f0, f1, f2):  # type: ignore[override]
        V0 = V0.contiguous()
        V1 = None if V1 is None else V1.contiguous()
        op.fwd([f0, f1, f2], a, b, V0, V1, None, cb)
        ctx.mark_dirty(f0, f1, f2)
        ctx.op, ctx.a, ctx.b = op, a, b
        ctx.has_v1 = V1 is not None
        ctx.has_cb = cb is not None
        ctx.save_for_backward(V0, V1 if V1 is not None else V0, cb if cb is not None else V0)
        return f0, f1, f2

    @staticmethod
    def backward(ctx, g0, g1, g2):  # type: ignore[override]
        V0, V1, cb = ctx.saved_tensors
        V1 = V1 if ctx.has_v1 else None
        cb = cb if ctx.has_cb else None
        g = [t.contiguous() for t in (g0, g1, g2)]
        gV0 = torch.zeros_like(V0) if ctx.needs_input_grad[4] else None
        gV1 = torch.zeros_like(V1) if (V1 is not None and ctx.needs_input_grad[5]) else None
        gcb = torch.zeros_like(cb) if (cb is not None and ctx.needs_input_grad[3]) else None
        ctx.op.bwd(g, ctx.a, ctx.b, V0, V1, None, cb, gV0, gV1, None, gcb)
        return None, None, None, gcb, gV0, gV1, g0, g1, g2


class _CorrGather(torch.autograd.Function):
    """``f += corr(src at the OS)`` in place (OS corrections)."""

    @staticmethod
    def forward(ctx, op: CorrCuda, cb, s0, s1, s2, f0, f1, f2):  # type: ignore[override]
        src = [s0.contiguous(), s1.contiguous(), s2.contiguous()]
        op.fwd([f0, f1, f2], 1.0, 0.0, None, None, src, cb)
        ctx.mark_dirty(f0, f1, f2)
        ctx.op = op
        ctx.has_cb = cb is not None
        ctx.save_for_backward(*src, cb if cb is not None else src[0])
        return f0, f1, f2

    @staticmethod
    def backward(ctx, g0, g1, g2):  # type: ignore[override]
        s0, s1, s2, cb = ctx.saved_tensors
        cb = cb if ctx.has_cb else None
        g = [t.contiguous() for t in (g0, g1, g2)]
        need_s = any(ctx.needs_input_grad[2:5])
        gsrc = [torch.zeros_like(t) for t in (s0, s1, s2)] if need_s else None
        gcb = torch.zeros_like(cb) if (cb is not None and ctx.needs_input_grad[1]) else None
        ctx.op.bwd(g, 1.0, 0.0, None, None, [s0, s1, s2], cb, None, None, gsrc, gcb)
        gs = gsrc if gsrc is not None else [None] * 3
        return (None, gcb, *gs, g0, g1, g2)


class SubgridCudaOps:
    """Fused coupling operators of one subgrid (attached as ``SubgridStatic.cuda``)."""

    def __init__(self, lib, sg, main_shape, dtype, device):
        n_main = int(np.prod(main_shape))
        n_fine = int(np.prod(sg.yee.shape))
        self.pe = TapsCuda(lib, sg.pe, sg.pe_comp, n_main, dtype, device)
        self.ph = TapsCuda(lib, sg.ph, sg.ph_comp, n_main, dtype, device)
        self.is_h = CorrCuda(lib, sg.is_h, n_fine, 0, 0, dtype, device)
        self.is_e = CorrCuda(lib, sg.is_e, n_fine, 0, 0, dtype, device)
        self.os_h = CorrCuda(lib, sg.os_h, n_main, n_fine, 1, dtype, device)
        self.os_e = CorrCuda(lib, sg.os_e, n_main, n_fine, 1, dtype, device)

    def precursors(self, kind: str, fields) -> torch.Tensor:
        return _Taps.apply(self.pe if kind == "e" else self.ph, *fields)

    def correct_is(self, kind: str, fields, V0, V1=None, a=1.0, b=0.0, cb=None) -> List[torch.Tensor]:
        op = self.is_h if kind == "h" else self.is_e
        return list(_CorrLerp.apply(op, float(a), float(b), cb, V0, V1, *fields))

    def correct_os(self, kind: str, fields, src, cb=None) -> List[torch.Tensor]:
        op = self.os_h if kind == "h" else self.os_e
        return list(_CorrGather.apply(op, cb, *src, *fields))


def attach(sg, main_shape, dtype, device) -> bool:
    """Attach :class:`SubgridCudaOps` to a subgrid static when possible."""
    if torch.device(device).type != "cuda":
        return False
    lib = load_library(device)
    if lib is None:
        return False
    sg.cuda = SubgridCudaOps(lib, sg, main_shape, dtype, device)
    return True


__all__ = ["CorrCuda", "SubgridCudaOps", "TapsCuda", "attach", "load_library"]

"""CUDA kernels of the 3D hybrid (``csrc/dgtd3d.cu``, ``csrc/yee3d.cu``), loaded with ``ctypes``.

The same library also holds the 2D kernels (``csrc/dg2d.cu``, ``csrc/yee2d.cu``),
wrapped in :mod:`.cuda2d`.

The library is compiled on first use with ``nvcc`` for the architecture of the
current GPU and cached in ``$DEEPGPR_CACHE_DIR/hybrid`` (default
``~/.cache/deepgpr/hybrid``), keyed by the source hash. When ``nvcc`` or a GPU
is missing, :func:`load_library` returns ``None`` and the solver uses the
PyTorch implementation of the same operator.

:class:`DGCudaOps` exposes the fused DG operator as a differentiable function:
its backward pass runs the transpose kernels (exact transpose of the forward
kernel, including the gradients of the face coefficients ``b0..b3``).
:class:`YeeCudaOps` does the same for the Yee H and E updates (differences,
CPML on slabs, side UPML, masks, interface-row penalty; gradients of ``ca``,
``cb`` and the UPML factors).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

import torch

from ..utils.logger import get_logger

_LOGGER = get_logger(__name__)
_CSRC = Path(__file__).resolve().parent / "csrc"
_SOURCES = (_CSRC / "dgtd3d.cu", _CSRC / "yee3d.cu", _CSRC / "dg2d.cu", _CSRC / "yee2d.cu")
_LOCK = threading.Lock()
_LIBS: dict = {}

_P = ctypes.c_void_p
_I = ctypes.c_int


def _cache_dir() -> Path:
    root = os.environ.get("DEEPGPR_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "deepgpr"
    return base / "hybrid"


def _nvcc() -> Optional[str]:
    path = shutil.which("nvcc")
    if path:
        return path
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env):
            cand = Path(os.environ[env]) / "bin" / "nvcc"
            if cand.exists():
                return str(cand)
    return None


def load_library(device=None):
    """Compile (once) and load the kernel library for ``device``; ``None`` if unavailable."""
    if os.environ.get("DEEPGPR_HYBRID_DISABLE_CUDA"):
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    with _LOCK:
        if arch in _LIBS:
            return _LIBS[arch]
        lib = None
        try:
            src = b"".join(p.read_bytes() for p in _SOURCES)
            digest = hashlib.sha256(src + arch.encode()).hexdigest()[:16]
            path = _cache_dir() / f"libdgtd3d_{arch}_{digest}.so"
            if not path.exists():
                nvcc = _nvcc()
                if nvcc is None:
                    raise RuntimeError("nvcc not found")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                cmd = [nvcc, "-O3", "-shared", "-Xcompiler", "-fPIC", f"-arch={arch}",
                       "-o", str(tmp)] + [str(p) for p in _SOURCES]
                subprocess.run(cmd, check=True, capture_output=True)
                os.replace(tmp, path)
            lib = ctypes.CDLL(str(path))
            lib.dgtd3d_terms_fwd.restype = _I
            lib.dgtd3d_terms_fwd.argtypes = [_I, _I, _I, _I] + [_P] * 17 + [_P, _I, _P]
            lib.dgtd3d_terms_bwd.restype = _I
            lib.dgtd3d_terms_bwd.argtypes = [_I, _I, _I, _I] + [_P] * 26 + [_P, _I, _P, _P, _P]
            lib.dgtd3d_bwd_workspace.restype = ctypes.c_longlong
            lib.dgtd3d_bwd_workspace.argtypes = [_I, _I, _I, _I]
            for name in ("dgtd3d_update_fwd", "dgtd3d_update_bwd"):
                fn = getattr(lib, name)
                fn.restype = _I
                fn.argtypes = [_I, _I, _I, _I, _I, _I, _P, _P]
            for name in ("yee3d_fwd", "yee3d_bwd"):
                fn = getattr(lib, name)
                fn.restype = _I
                fn.argtypes = [_I, _I, _P, _P, _P, _P, _P]
            # 2D kernels (csrc/dg2d.cu, csrc/yee2d.cu; wrappers in cuda2d.py)
            for name in ("dg2d_terms_fwd", "dg2d_terms_bwd"):
                fn = getattr(lib, name)
                fn.restype = _I
                fn.argtypes = [_I, _I, _I, _I, _P, _P, _P, _P, _P]
            lib.dg2d_bwd_workspace.restype = ctypes.c_longlong
            lib.dg2d_bwd_workspace.argtypes = [_I, _I, _I, _I]
            for name in ("dg2d_update_fwd", "dg2d_update_bwd"):
                fn = getattr(lib, name)
                fn.restype = _I
                fn.argtypes = [_I, _I, _I, _I, _P, _P, _P, _P]
            lib.yee2d_run.restype = _I
            lib.yee2d_run.argtypes = [_I, _I, _I, _P, _P, _P, _P, _P]
        except Exception as exc:  # pragma: no cover - depends on the toolchain
            _LOGGER.warning("3D DGTD CUDA kernels unavailable (%s); using PyTorch.", exc)
            lib = None
        _LIBS[arch] = lib
        return lib


def _ptr(t: Optional[torch.Tensor]):
    return None if t is None else ctypes.c_void_p(t.data_ptr())


def _ptrs(items):
    """Host array of device pointers (``None`` or empty -> NULL)."""
    arr = (ctypes.c_void_p * len(items))()
    for i, t in enumerate(items):
        arr[i] = None if (t is None or (torch.is_tensor(t) and t.numel() == 0)) else (
            t.data_ptr() if torch.is_tensor(t) else int(t))
    return arr


def _stream(t):
    return ctypes.c_void_p(torch.cuda.current_stream(t.device).cuda_stream)


def _check(rc: int, what: str):
    if rc != 0:
        raise RuntimeError(f"3D DGTD CUDA kernel {what} failed (code {rc}).")


def _cont(ts):
    return [t.contiguous() for t in ts]


class DGCudaOps:
    """Static operator data of a :class:`~DeepGPR.hybrid.functional3d.DGStatic3D` in
    the kernel layouts, the raw kernel calls (``terms_fwd``/``terms_bwd``,
    ``update_fwd``/``update_bwd``, used by :mod:`.vjp3d`) and their differentiable
    wrappers :meth:`terms`, :meth:`update_h`, :meth:`update_e`."""

    def __init__(self, lib, d, slots=()):
        if d.Np not in (4, 10):
            raise ValueError("CUDA kernels support P1 and P2 only.")
        self.lib = lib
        self.K, self.Np, self.Nfp = d.K, d.Np, d.Nfp
        c = lambda t: t.contiguous()  # noqa: E731
        i32 = lambda t: t.to(torch.int32).contiguous()  # noqa: E731
        self.R = c(d.R.reshape(3, 3, d.K))
        # affine elements: normals and face scale are constant per face
        self.n = c(d.n.reshape(3, d.K, 4, d.Nfp)[..., 0])
        self.fscale = c(d.fscale.reshape(d.K, 4, d.Nfp)[..., 0])
        self.vmapM, self.vmapP = i32(d.vmapM), i32(d.vmapP)
        self.slotP, self.fmask = i32(d.slotP), i32(d.fmask)
        self.D = tuple(c(DT.t()) for DT in d.DT)
        self.LIFT = c(d.LIFTT.t())
        self.dtype = d.LIFTT.dtype
        self.code = {torch.float32: 0, torch.float64: 1}[self.dtype]
        self._work = None
        # interface face slots (external input X / its cotangent, compact (S, 2, nX))
        self.nX = int(sum(int(t.numel()) for t in slots))
        self.xmap = None
        if self.nX:
            xmap = torch.full((d.K * 4 * d.Nfp,), -1, dtype=torch.int32, device=self.R.device)
            xmap[torch.cat(list(slots))] = torch.arange(self.nX, dtype=torch.int32,
                                                        device=self.R.device)
            self.xmap = xmap

    def setup_update(self, d):
        """Static data of the fused H/E updates (``ch``, UPML map and factors)."""
        self.ch = d.ch.reshape(-1).contiguous()
        self.pml = None
        self.Kp = 0
        self.u = {}
        if d.pml_idx is not None:
            pml = torch.full((d.K,), -1, dtype=torch.int32, device=self.ch.device)
            pml[d.pml_idx] = torch.arange(d.pml_idx.numel(), dtype=torch.int32,
                                          device=self.ch.device)
            self.pml = pml
            self.Kp = d.pml_idx.numel()
            self.u = {k: v.contiguous() for k, v in d.upml.items()}

    # ------------------------------------------------------------- raw kernels
    def terms_fwd(self, F, G, X, b0, b1, b2, b3):
        """The six curl terms ``(S, 6, K, Np)`` (contiguous inputs; ``X`` compact
        ``(S, 2, nX)`` on the interface slots or ``None``)."""
        S = F.shape[0]
        out = torch.empty((S, 6, self.K, self.Np), dtype=F.dtype, device=F.device)
        D = self.D
        _check(self.lib.dgtd3d_terms_fwd(
            self.code, self.Np, S, self.K, _ptr(F), _ptr(G), _ptr(X), _ptr(self.R), _ptr(self.n),
            _ptr(self.fscale), _ptr(self.vmapM), _ptr(self.vmapP), _ptr(b0), _ptr(b1), _ptr(b2),
            _ptr(b3), _ptr(D[0]), _ptr(D[1]), _ptr(D[2]), _ptr(self.LIFT), _ptr(out),
            _ptr(self.xmap), self.nX, _stream(F)), "forward")
        return out

    def terms_bwd(self, F, G, gT, b0, b1, b2, b3, has_x, addF=None, addG=None):
        """Exact transpose of :meth:`terms_fwd`: ``(gF, gG, gX, gb0, gb1, gb2, gb3)`` with
        ``gX`` compact ``(S, 2, nX)`` (or ``None``) and the face-coefficient cotangents
        per shot ``(S, [2,] 4, K)``. ``addF``/``addG`` (optional, shaped like ``F``/``G``)
        are added to ``gF``/``gG`` in the same pass."""
        S, K = F.shape[0], self.K
        kw = dict(dtype=F.dtype, device=F.device)
        nwork = int(self.lib.dgtd3d_bwd_workspace(S, K, self.Np, self.Nfp))
        if self._work is None or self._work.numel() < nwork or self._work.dtype != F.dtype \
                or self._work.device != F.device:
            self._work = torch.empty(nwork, **kw)
        work = self._work
        gF, gG = torch.empty_like(F), torch.empty_like(G)
        gX = torch.empty((S, 2, self.nX), **kw) if has_x else None
        gb0, gb1, gb3 = (torch.empty((S, 4, K), **kw) for _ in range(3))
        gb2 = torch.empty((S, 2, 4, K), **kw)
        D = self.D
        _check(self.lib.dgtd3d_terms_bwd(
            self.code, self.Np, S, K, _ptr(F), _ptr(G), _ptr(gT), _ptr(self.R), _ptr(self.n),
            _ptr(self.fscale), _ptr(self.vmapM), _ptr(self.vmapP), _ptr(self.slotP),
            _ptr(self.fmask), _ptr(b0), _ptr(b1), _ptr(b2), _ptr(b3), _ptr(D[0]), _ptr(D[1]),
            _ptr(D[2]), _ptr(self.LIFT), _ptr(work), _ptr(gF), _ptr(gG), _ptr(gX), _ptr(gb0),
            _ptr(gb1), _ptr(gb2), _ptr(gb3), _ptr(self.xmap), self.nX, _ptr(addF), _ptr(addG),
            _stream(F)), "backward")
        return gF, gG, gX, gb0, gb1, gb2, gb3

    def _factors(self, mode, hist, nw, old):
        u = self.u
        if mode == 1:
            return (u.get("a1"), u.get("b1"), hist, nw, old)
        return (u.get("a1"), u.get("b1"), u.get("h_hist"), u.get("h_new"), u.get("h_old"))

    def update_fwd(self, mode, T6, Fo, aux, ca=None, cb=None, hist=None, nw=None, old=None):
        """Fused update from the curl terms (``mode`` 0: H, 1: E, with ``ca``/``cb``
        flat ``(K,)`` and the UPML E factors); returns ``(F_new, aux_new)``."""
        S = Fo.shape[0]
        Fn = torch.empty_like(Fo)
        auxn = torch.empty_like(aux)
        q = _ptrs([T6, Fo, aux, Fn, auxn, self.ch if mode == 0 else None, ca, cb, self.pml,
                   *self._factors(mode, hist, nw, old)])
        _check(self.lib.dgtd3d_update_fwd(self.code, self.Np, mode, S, self.K, self.Kp, q,
                                          _stream(Fo)), "DG update forward")
        return Fn, auxn

    def update_bwd(self, mode, T6, Fo, aux, gFn, gauxn, ca=None, cb=None, hist=None, nw=None,
                   old=None):
        """Transpose of :meth:`update_fwd`: ``(gT6, gFo, gaux, gca, gcb, ghist, gnew, gold)``;
        the coefficient cotangents per shot ``(S, K)`` / ``(S, 3, Kp, Np)`` (E only).
        ``T6`` is read in the E mode only."""
        S = Fo.shape[0]
        kw = dict(dtype=Fo.dtype, device=Fo.device)
        gT6, gFo, gaux = torch.empty((S, 6, self.K, self.Np), **kw), torch.empty_like(Fo), \
            torch.zeros_like(aux)
        e = mode == 1
        gca = torch.empty((S, self.K), **kw) if e else None
        gcb = torch.empty((S, self.K), **kw) if e else None
        gh = torch.zeros_like(aux) if e and aux.numel() else None
        gn = torch.zeros_like(aux) if e and aux.numel() else None
        go = torch.zeros_like(aux) if e and aux.numel() else None
        q = _ptrs([T6, Fo, aux, gFn, gauxn, gT6, gFo, gaux, gca, gcb, gh, gn, go,
                   self.ch if mode == 0 else None, ca, cb, self.pml,
                   *self._factors(mode, hist, nw, old)])
        _check(self.lib.dgtd3d_update_bwd(self.code, self.Np, mode, S, self.K, self.Kp, q,
                                          _stream(Fo)), "DG update backward")
        return gT6, gFo, gaux, gca, gcb, gh, gn, go

    # ------------------------------------------------------------- autograd
    def terms(self, F, G, X, b0, b1, b2, b3):
        return _Terms.apply(self, F, G, X, b0, b1, b2, b3)

    def update_h(self, T6, H, aux):
        return _DGUpdate.apply(self, 0, T6, H, aux, None, None, None, None, None)

    def update_e(self, T6, E, aux, ca, cb, hist, nw, old):
        return _DGUpdate.apply(self, 1, T6, E, aux, ca, cb, hist, nw, old)


class _Terms(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ops: DGCudaOps, F, G, X, b0, b1, b2, b3):  # type: ignore[override]
        F, G = F.contiguous(), G.contiguous()
        X = None if X is None else X.contiguous()
        b0, b1, b2, b3 = (b.contiguous() for b in (b0, b1, b2, b3))
        out = ops.terms_fwd(F, G, X, b0, b1, b2, b3)
        ctx.ops, ctx.has_x = ops, X is not None
        ctx.save_for_backward(F, G, b0, b1, b2, b3)
        return out

    @staticmethod
    def backward(ctx, gT):  # type: ignore[override]
        F, G, b0, b1, b2, b3 = ctx.saved_tensors
        gF, gG, gX, gb0, gb1, gb2, gb3 = ctx.ops.terms_bwd(F, G, gT.contiguous(), b0, b1, b2,
                                                           b3, ctx.has_x)
        return (None, gF, gG, gX, gb0.sum(0).t(), gb1.sum(0).t(), gb2.sum(0).transpose(1, 2),
                gb3.sum(0).t())


class _DGUpdate(torch.autograd.Function):
    """Fused DG update from the curl terms ``T`` (``mode`` 0: H, 1: E), with UPML."""

    @staticmethod
    def forward(ctx, ops, mode, T6, Fo, aux, ca, cb, hist, nw, old):  # type: ignore[override]
        T6, Fo, aux = T6.contiguous(), Fo.contiguous(), aux.contiguous()
        if mode == 1:
            ca, cb = ca.reshape(-1).contiguous(), cb.reshape(-1).contiguous()
            hist = None if hist is None else hist.contiguous()
            nw = None if nw is None else nw.contiguous()
            old = None if old is None else old.contiguous()
        Fn, auxn = ops.update_fwd(mode, T6, Fo, aux, ca, cb, hist, nw, old)
        ctx.ops, ctx.mode = ops, mode
        ctx.save_for_backward(T6, Fo, aux, *(t if t is not None else torch.zeros(0) for t in
                                              (ca, cb, hist, nw, old)))
        return Fn, auxn

    @staticmethod
    def backward(ctx, gFn, gauxn):  # type: ignore[override]
        ops, mode = ctx.ops, ctx.mode
        T6, Fo, aux, ca, cb, hist, nw, old = ctx.saved_tensors
        gFn = torch.zeros_like(Fo) if gFn is None else gFn.contiguous()
        gauxn = torch.zeros_like(aux) if gauxn is None else gauxn.contiguous()
        e = mode == 1
        gT6, gFo, gaux, gca, gcb, gh, gn, go = ops.update_bwd(
            mode, T6, Fo, aux, gFn, gauxn, ca if e else None, cb if e else None,
            hist if e else None, nw if e else None, old if e else None)
        if mode == 0:
            return None, None, gT6, gFo, gaux, None, None, None, None, None
        red = lambda t, ref: None if (t is None or ref.numel() == 0) else t.sum(0)  # noqa: E731
        return (None, None, gT6, gFo, gaux, gca.sum(0).view(-1, 1), gcb.sum(0).view(-1, 1),
                red(gh, hist), red(gn, nw), red(go, old))


# --------------------------------------------------------------------------- Yee
class YeeCudaOps:
    """Kernel data of a :class:`~DeepGPR.hybrid.functional3d.YeeStatic3D` (one side
    each for the H and the E update), the raw kernel calls (``h_fwd``/``h_bwd``,
    ``e_fwd``/``e_bwd``, used by :mod:`.vjp3d`) and the differentiable fused updates."""

    def __init__(self, lib, y, nshot, rows=()):
        self.lib = lib
        self.y = y
        self.S = int(nshot)
        NX, NY, NZ = y.shape
        self.shape = (NX, NY, NZ)
        self.N = NX * NY * NZ
        dev = y.chr.device
        self.dtype = y.chr.dtype
        self.code = {torch.float32: 0, torch.float64: 1}[self.dtype]
        self.device = dev
        self._keep = []
        self.act = [y.act[k].to(torch.uint8).contiguous() for k in range(6)]
        self.side = {}
        for side, profiles, act, kind in (("h", y.ph, self.act[3:], "h"), ("e", y.pe, self.act[:3], "e")):
            sp, nslab = [], []
            for a in range(3):
                prof = profiles[a]
                if prof is None:
                    sp += [None] * 5
                    nslab.append(0)
                    continue
                A, B, E, Fm, idx = prof
                pos = torch.full((self.shape[a],), -1, dtype=torch.int32, device=dev)
                pos[idx] = torch.arange(idx.numel(), dtype=torch.int32, device=dev)
                tabs = [t.reshape(-1).contiguous() for t in (A, B, E, Fm)]
                self._keep += [pos] + tabs
                sp += [pos] + tabs
                nslab.append(idx.numel())
            upos, uoff, ntot = [None] * 3, [0, 0, 0], 0
            a1 = b1 = hist = nw = old = None
            if y.upml is not None:
                ents = y.upml[kind]
                off = y.upml["offsets_" + kind]
                for c, ent in enumerate(ents):
                    if ent["idx"].numel():
                        up = torch.full((self.N,), -1, dtype=torch.int32, device=dev)
                        up[ent["idx"]] = torch.arange(ent["idx"].numel(), dtype=torch.int32,
                                                      device=dev)
                        upos[c] = up
                        self._keep.append(up)
                    uoff[c] = off[c]
                ntot = off[-1]
                a1 = torch.cat([e["a1"] for e in ents]).contiguous()
                b1 = torch.cat([e["b1"] for e in ents]).contiguous()
                if kind == "h":
                    hist, nw, old = (torch.cat([e[k] for e in ents]).contiguous()
                                     for k in ("hist", "new", "old"))
                self._keep += [a1, b1] + ([hist, nw, old] if kind == "h" else [])
            ptrs = sp + list(act) + upos + [a1, b1, hist, nw, old]
            self.side[side] = dict(nslab=nslab, uoff=uoff, ntot=ntot, ptrs=ptrs)
        rowmap = torch.full((NX,), -1, dtype=torch.int32, device=dev)
        for r, row in enumerate(rows):
            rowmap[row] = r
        self.rowmap, self.nrows = rowmap, len(rows)
        self.gd = (ctypes.c_double * 3)(*y.spacing)
        self._gi, self._sp = {}, {}

    def gi(self, side, S):
        key = (side, S)
        if key not in self._gi:
            d = self.side[side]
            vals = [S, *self.shape, *d["nslab"], *d["uoff"], d["ntot"], self.nrows]
            self._gi[key] = (ctypes.c_int * len(vals))(*vals)
        return self._gi[key]

    def sp(self, side, hist=None, nw=None, old=None):
        """Static pointer table of a side (cached; the E side keys on the material
        UPML factors, whose storage changes between models)."""
        key = (side,) + tuple(0 if (t is None or t.numel() == 0) else t.data_ptr()
                              for t in (hist, nw, old))
        arr = self._sp.get(key)
        if arr is None:
            if len(self._sp) > 16:
                self._sp.clear()
            p = list(self.side[side]["ptrs"])
            if side == "e":
                p[23], p[24], p[25] = hist, nw, old
            arr = self._sp[key] = _ptrs(p)
        return arr

    # ------------------------------------------------------------- raw kernels
    def h_fwd(self, fe, fh, aux, psi):
        """H update (contiguous inputs): ``(fh_new[3], aux_new, psi_new[6])``."""
        S = fe[0].shape[0]
        out = [torch.empty_like(t) for t in fh]
        psi_out = [torch.empty_like(t) for t in psi]
        aux_out = torch.empty_like(aux)
        io = _ptrs(list(fe) + list(fh) + list(psi) + psi_out
                   + [aux, aux_out, None, None, self.y.chr, None, None] + out)
        _check(self.lib.yee3d_fwd(self.code, 0, self.gi("h", S), self.gd, self.sp("h"), io,
                                  _stream(fe[0])), "Yee H forward")
        return out, aux_out, psi_out

    def h_bwd(self, fe, fh, aux, psi, g, gaux, gpsi):
        """Transpose of :meth:`h_fwd`: ``(g_fe[3], g_fh[3], g_aux, g_psi[6])``."""
        S = fe[0].shape[0]
        gu = [torch.empty_like(t) for t in fh]
        gpsi_in = [torch.empty_like(t) for t in psi]
        gaux_in = torch.empty_like(aux)
        lam = torch.empty((6,) + fe[0].shape, dtype=fe[0].dtype, device=fe[0].device)
        gsrc = [torch.empty_like(t) for t in fe]
        io = _ptrs(list(fe) + list(fh) + list(psi) + [aux, None, None, self.y.chr, None, None]
                   + list(g) + list(gpsi) + [gaux] + gu + gpsi_in
                   + [gaux_in, lam, None, None, None, None, None, None] + gsrc)
        _check(self.lib.yee3d_bwd(self.code, 0, self.gi("h", S), self.gd, self.sp("h"), io,
                                  _stream(fe[0])), "Yee H backward")
        return gsrc, gu, gaux_in, gpsi_in

    def e_fwd(self, fh, fe, aux, ca, cb, hist, nw, old, pen, psi):
        """E update (contiguous inputs; empty ``hist``/``pen`` = none):
        ``(fe_new[3], aux_new, psi_new[6])``."""
        S = fe[0].shape[0]
        out = [torch.empty_like(t) for t in fe]
        psi_out = [torch.empty_like(t) for t in psi]
        aux_out = torch.empty_like(aux)
        io = _ptrs(list(fh) + list(fe) + list(psi) + psi_out
                   + [aux, aux_out, ca, cb, None, pen, self.rowmap if pen.numel() else None] + out)
        _check(self.lib.yee3d_fwd(self.code, 1, self.gi("e", S), self.gd,
                                  self.sp("e", hist, nw, old), io, _stream(fe[0])), "Yee E forward")
        return out, aux_out, psi_out

    def e_bwd(self, fh, fe, aux, ca, cb, hist, nw, old, pen, psi, g, gaux, gpsi):
        """Transpose of :meth:`e_fwd`: ``(g_fh[3], g_fe[3], g_aux, g_psi[6], g_ca, g_cb,
        g_hist, g_new, g_old, g_pen)``; ``g_ca``/``g_cb`` per shot ``(S, N)``, the UPML
        factor cotangents ``(S, n_upml)``, ``g_pen`` ``None`` without penalty."""
        S = fe[0].shape[0]
        dev, dt = fe[0].device, fe[0].dtype
        gu = [torch.empty_like(t) for t in fe]
        gpsi_in = [torch.empty_like(t) for t in psi]
        gaux_in = torch.empty_like(aux)
        lam = torch.empty((6,) + fe[0].shape, dtype=dt, device=dev)
        gsrc = [torch.empty_like(t) for t in fh]
        gca = torch.empty((S, self.N), dtype=dt, device=dev)
        gcb = torch.empty_like(gca)
        ntot = self.side["e"]["ntot"]
        gh = torch.zeros((S, ntot), dtype=dt, device=dev)
        gn, go = torch.zeros_like(gh), torch.zeros_like(gh)
        gpen = torch.zeros_like(pen) if pen.numel() else None
        io = _ptrs(list(fh) + list(fe) + list(psi)
                   + [aux, ca, cb, None, pen, self.rowmap if pen.numel() else None]
                   + list(g) + list(gpsi) + [gaux] + gu + gpsi_in
                   + [gaux_in, lam, gca, gcb, gh, gn, go, gpen] + gsrc)
        _check(self.lib.yee3d_bwd(self.code, 1, self.gi("e", S), self.gd,
                                  self.sp("e", hist, nw, old), io, _stream(fe[0])), "Yee E backward")
        return gsrc, gu, gaux_in, gpsi_in, gca, gcb, gh, gn, go, gpen

    # ------------------------------------------------------------- autograd
    def h_update(self, fe, fh, psh, Bfd):
        out = _YeeH.apply(self, *fe, *fh, Bfd, *psh)
        return list(out[0:3]), list(out[4:10]), out[3]

    def e_update(self, fh, fe, pse, Dfd, c, pen):
        z = torch.zeros(0, dtype=self.dtype, device=self.device)
        hist = c.fu_hist if c.fu_hist is not None else z
        nw = c.fu_new if c.fu_new is not None else z
        old = c.fu_old if c.fu_old is not None else z
        pen = z if pen is None else pen
        out = _YeeE.apply(self, *fh, *fe, Dfd, c.fd_ca, c.fd_cb, hist, nw, old, pen, *pse)
        return list(out[0:3]), list(out[4:10]), out[3]


def _grads(gs, refs):
    return [torch.zeros_like(t) if g is None else g.contiguous() for g, t in zip(gs, refs)]


class _YeeH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ops, fe0, fe1, fe2, fh0, fh1, fh2, aux, *psi):  # type: ignore[override]
        fe, fh, psi = _cont((fe0, fe1, fe2)), _cont((fh0, fh1, fh2)), _cont(psi)
        aux = aux.contiguous()
        out, aux_out, psi_out = ops.h_fwd(fe, fh, aux, psi)
        ctx.ops = ops
        ctx.save_for_backward(*fe, *fh, aux, *psi)
        return tuple(out) + (aux_out,) + tuple(psi_out)

    @staticmethod
    def backward(ctx, g0, g1, g2, gaux, *gpsi):  # type: ignore[override]
        sv = ctx.saved_tensors
        fe, fh, aux, psi = list(sv[0:3]), list(sv[3:6]), sv[6], list(sv[7:13])
        gsrc, gu, gaux_in, gpsi_in = ctx.ops.h_bwd(
            fe, fh, aux, psi, _grads((g0, g1, g2), fh), _grads((gaux,), (aux,))[0],
            _grads(gpsi, psi))
        return (None, *gsrc, *gu, gaux_in, *gpsi_in)


class _YeeE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ops, fh0, fh1, fh2, fe0, fe1, fe2, aux, ca, cb, hist, nw, old, pen,
                *psi):  # type: ignore[override]
        fh, fe, psi = _cont((fh0, fh1, fh2)), _cont((fe0, fe1, fe2)), _cont(psi)
        aux, ca, cb, hist, nw, old, pen = _cont((aux, ca, cb, hist, nw, old, pen))
        out, aux_out, psi_out = ops.e_fwd(fh, fe, aux, ca, cb, hist, nw, old, pen, psi)
        ctx.ops = ops
        ctx.save_for_backward(*fh, *fe, aux, ca, cb, hist, nw, old, pen, *psi)
        return tuple(out) + (aux_out,) + tuple(psi_out)

    @staticmethod
    def backward(ctx, g0, g1, g2, gaux, *gpsi):  # type: ignore[override]
        sv = ctx.saved_tensors
        fh, fe = list(sv[0:3]), list(sv[3:6])
        aux, ca, cb, hist, nw, old, pen = sv[6:13]
        psi = list(sv[13:19])
        gsrc, gu, gaux_in, gpsi_in, gca, gcb, gh, gn, go, gpen = ctx.ops.e_bwd(
            fh, fe, aux, ca, cb, hist, nw, old, pen, psi, _grads((g0, g1, g2), fe),
            _grads((gaux,), (aux,))[0], _grads(gpsi, psi))
        shape = ca.shape
        red = lambda t, ref: t.sum(0) if ref.numel() else None  # noqa: E731
        return (None, *gsrc, *gu, gaux_in, gca.sum(0).view(shape), gcb.sum(0).view(shape),
                red(gh, hist), red(gn, nw), red(go, old), gpen, *gpsi_in)


def attach_yee(y, nshot, device, rows=()) -> bool:
    """Attach :class:`YeeCudaOps` to a Yee static (``y.cuda``) when possible."""
    if torch.device(device).type != "cuda":
        return False
    lib = load_library(device)
    if lib is None:
        return False
    y.cuda = YeeCudaOps(lib, y, nshot, rows)
    return True


def attach(d, device, slots=()) -> bool:
    """Attach :class:`DGCudaOps` to a DG static (``d.cuda``) when possible
    (``slots``: interface face slots of the couplings, in coupling order)."""
    if torch.device(device).type != "cuda":
        return False
    lib = load_library(device)
    if lib is None:
        return False
    d.cuda = DGCudaOps(lib, d, slots)
    d.cuda.setup_update(d)
    return True


__all__ = ["DGCudaOps", "YeeCudaOps", "attach", "attach_yee", "load_library"]

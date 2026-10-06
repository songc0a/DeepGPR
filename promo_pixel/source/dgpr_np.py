"""NumPy driver for DeepGPR's native CPU kernel (deepgpr_cpu.so).

PyTorch is not installable in this sandbox, so the thin Python layer of
DeepGPR.compute (CPML extension, CPML coefficients, state allocation, argument
marshalling) is ported 1:1 to NumPy here. The forward FDTD solver and the exact
discrete adjoint are DeepGPR's own, unmodified native code.
"""
import ctypes
import importlib
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("DEEPGPR_SRC") or next(
    (p for p in (os.path.join(_HERE, "DeepGPR_src"), os.path.join(_HERE, "..", "..", "src", "DeepGPR"))
     if os.path.isdir(p)), os.path.join(_HERE, "DeepGPR_src"))

# Import the torch-free ABI tables without triggering DeepGPR/__init__ (torch).
for _name, _sub in (("DeepGPR", ""), ("DeepGPR.config", "config"), ("DeepGPR.native", "native")):
    _m = types.ModuleType(_name)
    _m.__path__ = [os.path.join(SRC, _sub)]
    sys.modules[_name] = _m
constants = importlib.import_module("DeepGPR.config.constants")
abi = importlib.import_module("DeepGPR.native.abi")

E0 = constants.VACUUM_PERMITTIVITY
M0 = constants.VACUUM_PERMEABILITY
C0 = constants.SPEED_OF_LIGHT
FACES = constants.CPML_FACE_NAMES

LIB = ctypes.CDLL(os.path.join(SRC, "lib", "deepgpr_cpu.so"))
for _fn in ("forward", "backward"):
    _sig, _ = abi.NATIVE_FUNCTIONS[_fn]
    getattr(LIB, _fn).argtypes = abi.ctypes_signature(_sig)
    getattr(LIB, _fn).restype = None
LIB.set_fdtd_order.argtypes = [ctypes.c_int]
LIB.deepgpr_last_error.restype = ctypes.c_char_p
assert LIB.deepgpr_abi_version() == constants.NATIVE_ABI_VERSION

_FP = ctypes.POINTER(ctypes.c_float)
_IP = ctypes.POINTER(ctypes.c_int)
_EMPTY = np.zeros(1, np.float32)


def _conv(kind, v):
    if kind == abi.FLOAT_PTR:
        return ctypes.cast(v.ctypes.data, _FP)
    if kind == abi.INT_PTR:
        return ctypes.cast(v.ctypes.data, _IP)
    if kind == abi.VOID_PTR:
        return ctypes.c_void_p(v.ctypes.data)
    if kind == abi.INT:
        return int(v)
    if kind == abi.FLOAT:
        return float(v)
    raise ValueError(kind)


def _invoke(name, order, args):
    sig, _ = abi.NATIVE_FUNCTIONS[name]
    exp = [n for n, _ in sig]
    missing = [n for n in exp if n not in args]
    extra = sorted(set(args) - set(exp))
    assert not missing and not extra, (missing, extra)
    LIB.deepgpr_clear_last_error()
    LIB.set_fdtd_order(order)
    getattr(LIB, name)(*[_conv(k, args[n]) for n, k in sig])
    err = LIB.deepgpr_last_error()
    if err:
        raise RuntimeError(err.decode())


def ricker(freq, nt, dt, peak_time):
    t = np.arange(nt, dtype=np.float64) * dt - peak_time
    p = (np.pi * freq * t) ** 2
    return ((1 - 2 * p) * np.exp(-p)).astype(np.float32)


def max_dt(dx, eps_min=1.0, ndim=2):
    return np.sqrt(eps_min) / (C0 * np.sqrt(ndim / dx ** 2))


def _pml_coeffs(eps_ext, mu_ext, dt, spacing, pml):
    """Port of solver.pml.build_pml_coeffs (quartic sigma, kappa=1, alpha=0)."""
    nx, ny, nz = eps_ext.shape
    out = []
    for face in range(6):
        p = pml[face]
        if p <= 0:
            out += [_EMPTY, _EMPTY]
            continue
        idx = (
            (0, slice(None), slice(None)),
            (nx - p, slice(None), slice(None)),
            (slice(None), 0, slice(None)),
            (slice(None), ny - p, slice(None)),
            (slice(None), slice(None), 0),
            (slice(None), slice(None), nz - p),
        )[face]
        er = np.float32(eps_ext[idx].mean())
        mr = np.float32(mu_ext[idx].mean())
        d = spacing[face // 2]
        m = constants.CPML_SCALING_PROFILES["quartic"]
        smax = (constants.CPML_SIGMA_MAX_FACTOR * (m + 1)) / (((M0 / E0) ** 0.5) * d * np.sqrt(er * mr))
        L = p + 1
        tmp = (np.linspace(0, (L - 1) + 0.5, 2 * L, dtype=np.float32) / (L - 1)) ** m
        es = (tmp[0:-1:2] * smax)[:-1]
        hs = (tmp[1::2] * smax)[:-1]
        coeffs = []
        for s in (es, hs):
            k = np.ones(p, np.float32)
            a = np.zeros(p, np.float32)
            R = np.zeros((4, 1, p), np.float32)
            t = (2 * E0 * k) + dt * (a * k + s)
            R[0, 0] = (2 * E0 + dt * a) / t
            R[1, 0] = (2 * E0 * k) / t
            R[2, 0] = ((2 * E0 * k) - dt * (a * k + s)) / t
            R[3, 0] = (2 * s * dt) / (k * t)
            coeffs.append(np.ascontiguousarray(R))
        out += coeffs
    return out


def _phi_shapes(face, pml, nx, ny, nz, nstep):
    p = pml[face]
    if p <= 0:
        return [None] * 4
    box = (
        (0, p, 0, ny, 0, nz), (nx - p, nx, 0, ny, 0, nz),
        (0, nx, 0, p, 0, nz), (0, nx, ny - p, ny, 0, nz),
        (0, nx, 0, ny, 0, p), (0, nx, 0, ny, nz - p, nz),
    )[face]
    a, b, c = box[1] - box[0], box[3] - box[2], box[5] - box[4]
    ax = face // 2
    if ax == 0:
        return [(nstep, a + 1, b, c + 1), (nstep, a + 1, b + 1, c), (nstep, a, b + 1, c), (nstep, a, b, c + 1)]
    if ax == 1:
        return [(nstep, a, b + 1, c + 1), (nstep, a + 1, b + 1, c), (nstep, a + 1, b, c), (nstep, a, b, c + 1)]
    return [(nstep, a, b + 1, c + 1), (nstep, a + 1, b, c + 1), (nstep, a + 1, b, c), (nstep, a, b + 1, c)]


class Sim:
    """One acquisition on one model: forward (optionally segmented) + adjoint."""

    def __init__(self, eps, sig, dx, dt, src_loc, rec_loc, pml=10, order=2, comp=2):
        eps = np.asarray(eps, np.float32)
        sig = np.asarray(sig, np.float32)
        if eps.ndim == 2:
            eps = eps[:, :, None]
            sig = sig[:, :, None]
        self.phys = eps.shape
        self.mode2d = eps.shape[2] == 1
        self.pml = [pml] * 4 + ([0, 0] if self.mode2d else [pml, pml])
        P = self.pml
        pad = ((P[0], P[1]), (P[2], P[3]), (P[4], P[5]))
        self.eps_ext = np.pad(eps, pad, mode="edge")
        self.sig_ext = np.pad(sig, pad, mode="edge")
        mu_ext = np.ones_like(self.eps_ext)
        halo = ((0, 1), (0, 1), (0, 1))
        self.eps_pad = np.ascontiguousarray(np.pad(self.eps_ext, halo))
        self.sig_pad = np.ascontiguousarray(np.pad(self.sig_ext, halo))
        self.mu_pad = np.ascontiguousarray(np.pad(mu_ext, halo))
        self.nx, self.ny, self.nz = self.eps_ext.shape
        self.dx = (dx, dx, dx)
        self.dt = float(dt)
        self.order = order
        self.comp = comp
        off = np.array(P[::2], np.int32)
        self.src = np.ascontiguousarray(np.asarray(src_loc, np.int32) + off)
        self.rec = np.ascontiguousarray(np.asarray(rec_loc, np.int32) + off)
        self.nstep, self.nsr = self.src.shape[:2]
        self.nrx = self.rec.shape[1]
        self.coeffs = _pml_coeffs(self.eps_ext, mu_ext, self.dt, self.dx, P)
        self.reset()

    def reset(self):
        fs = (self.nstep, self.nx + 1, self.ny + 1, self.nz + 1)
        self.fields = [np.zeros(fs, np.float32) for _ in range(6)]
        self.phi = []
        for face in range(6):
            for shp in _phi_shapes(face, self.pml, self.nx, self.ny, self.nz, self.nstep):
                self.phi.append(_EMPTY if shp is None else np.zeros(shp, np.float32))

    def _common(self, nt):
        a = dict(dt=self.dt, nt=nt, nshot=self.nstep, nreceiver=self.nrx,
                 dx=self.dx[0], dy=self.dx[1], dz=self.dx[2],
                 nx_fields=self.nx + 1, ny_fields=self.ny + 1, nz_fields=self.nz + 1,
                 receiver_component=self.comp, nsource=self.nsr, source_component=self.comp,
                 sampling_interval=1, fwi_mode=2 if self.mode2d else 3)
        for f, t in zip(FACES, self.pml):
            a[f"pml_{f}"] = t
        for i, f in enumerate(FACES):
            a[f"{f}_e_coeff"] = self.coeffs[2 * i]
            a[f"{f}_h_coeff"] = self.coeffs[2 * i + 1]
        upd = np.zeros((6, self.nx + 1, self.ny + 1, self.nz + 1), np.float32)
        for i, n in enumerate(("ce_hist", "ce_curl", "ce_rhs", "ch_hist", "ch_curl", "ch_rhs")):
            a[n] = upd[i]
        self._upd = upd
        a.update(eps_r_pad=self.eps_pad, sigma_pad=self.sig_pad, mu_r_pad=self.mu_pad)
        return a

    def forward(self, wavelet, history=False, grad=False):
        """Advance the states by len(wavelet) steps. Returns receiver data (nstep, nt, nrx)."""
        wav = np.asarray(wavelet, np.float32)
        nt = wav.shape[-1]
        src_wave = np.ascontiguousarray(np.broadcast_to(wav.reshape(-1, nt, 1), (self.nsr, nt, 1)))
        rec = np.zeros((self.nstep, nt, self.nrx), np.float32)
        a = self._common(nt)
        st = constants.WAVEFIELD_STORAGE_FLOAT32
        if history or grad:
            shape = (nt, self.nstep, self.nx, self.ny, self.nz)
            if not self.mode2d:
                shape = (3,) + shape
            self.E_saved = np.zeros(shape, np.float32)
            self.R_saved = np.zeros(shape, np.float32) if grad else _EMPTY
        else:
            st |= constants.WAVEFIELD_HISTORY_DISABLED
            self.E_saved = _EMPTY
            self.R_saved = _EMPTY
        a.update(E_saved=self.E_saved, R_saved=self.R_saved,
                 Ex=self.fields[0], Ey=self.fields[1], Ez=self.fields[2],
                 Hx=self.fields[3], Hy=self.fields[4], Hz=self.fields[5],
                 receiver_location=self.rec, receiver_data=rec,
                 source_location=self.src, source_waveform=src_wave,
                 storage_type=st, save_model_history=int(grad), use_async_offload=0)
        a.update(zip(abi.PHI_PARAMETER_NAMES, self.phi))
        _invoke("forward", self.order, a)
        self._nt = nt
        self._st = st
        return rec

    def backward(self, data_grad):
        """Exact discrete adjoint: returns (grad_eps, grad_sigma) on the physical model."""
        nt = self._nt
        a = self._common(nt)
        fs = (self.nstep, self.nx + 1, self.ny + 1, self.nz + 1)
        lam = [np.zeros(fs, np.float32) for _ in range(6)]
        lam += [_EMPTY if p.size == 1 and p is _EMPTY else np.zeros_like(p) for p in self.phi]
        g_eps = np.zeros((self.nx, self.ny, self.nz), np.float32)
        g_sig = np.zeros((self.nx, self.ny, self.nz), np.float32)
        dg = np.ascontiguousarray(data_grad, np.float32)
        names = ("lambda_ex", "lambda_ey", "lambda_ez", "lambda_hx", "lambda_hy", "lambda_hz") + abi.LAMBDA_PHI_PARAMETER_NAMES
        a.update(E_saved=self.E_saved, R_saved=self.R_saved, ndata_source=self.nrx,
                 receiver_location=self.rec, data_grad=dg, source_location=self.src,
                 grad_source=_EMPTY, source_requires_grad=0,
                 grad_eps_r=g_eps, grad_sigma=g_sig, eps_r_requires_grad=1, sigma_requires_grad=1,
                 storage_type=self._st, use_async_offload=0)
        a.update(zip(names, lam))
        _invoke("backward", self.order, a)
        P = self.pml
        sl = tuple(slice(P[2 * i], P[2 * i] + self.phys[i]) for i in range(3))
        return g_eps[sl].copy(), g_sig[sl].copy()

    def ez(self):
        """Current Ez on the physical model, shape (nstep, nx, ny, nz)."""
        P = self.pml
        sl = tuple(slice(P[2 * i], P[2 * i] + self.phys[i]) for i in range(3))
        return self.fields[2][(slice(None),) + sl]

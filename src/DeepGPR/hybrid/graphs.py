"""CUDA Graph replay of the hybrid time step (2D and 3D).

One step of :func:`~DeepGPR.hybrid.functional3d.step` (or of the 2D
:func:`~DeepGPR.hybrid.functional.step`) launches dozens to a hundred
kernels (fused Yee and DG kernels plus the small coupling, penalty, source and
receiver operations), so at moderate sizes the step is bound by the host
(Python, ``ctypes``, launch latency) rather than by the GPU. :class:`GraphedStep3D`
captures one step, the receiver sampling and the copy of the new state into
static input buffers in a CUDA Graph and replays it, which removes that
overhead. The arithmetic is unchanged (the same kernels with the same
arguments); replayed and eager runs agree to round-off (``tests/test_hybrid_3d.py``;
neither is bitwise reproducible, because ``index_add`` in the coupling and the
receiver sampling accumulates with atomics: run-to-run differences are 4e-16
in float64 and 2e-7 in float32 for both).

Replay pays off only while the step is bound by the host. The replayed step
copies its output into the static input buffers (one read and one write of the
state per step), whereas the eager step just rebinds them. Measured on an RTX
4090 (float32):
* test model (state < 1 MB): 0.90 ms eager against 0.24 ms replayed per step;
* B3 benchmark at r = 1 (state 75 MB): GPU-bound, 1.75 ms eager against
  1.87 ms replayed.

Graphs are therefore used only for states below ``DEEPGPR_HYBRID_GRAPH_MAX_MB``
(default 32 MB). ``DEEPGPR_HYBRID_DISABLE_GRAPHS=1`` turns them off.

The graph is tied to the static data, the coefficient tensors and the source /
receiver operators it was captured with; it is rebuilt for every propagation
(cheap compared with the time loop).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from ..utils.logger import get_logger
from . import functional as fn2
from . import functional3d as fn3

_LOGGER = get_logger(__name__)


def _module(st):
    return fn3 if isinstance(st, fn3.StepStatic3D) else fn2


def state_bytes(st) -> int:
    """Size of the state (computed from the shapes, nothing allocated)."""
    item = torch.tensor([], dtype=st.dtype).element_size()
    n = 0
    y, d = st.yee, st.dg
    if not isinstance(st, fn3.StepStatic3D):           # 2D: 7 Yee arrays, 3 DG fields + CPML
        if y is not None:
            n += 7 * st.nshot * y.shape[0] * y.shape[1]
        if d is not None:
            kp = 0 if d.pml_idx is None else d.pml_idx.numel()
            n += st.nshot * d.Np * (3 * d.K + 4 * kp)
        return n * item
    if y is not None:
        N = st.nshot * y.shape[0] * y.shape[1] * y.shape[2]
        n += 6 * N
        for prof in (y.pe, y.ph):
            for (_, a, _, _) in fn3.TERMS:
                if prof[a] is not None:
                    n += N // y.shape[a] * prof[a][4].numel()
        if y.upml is not None:
            n += st.nshot * (y.upml["offsets_e"][-1] + y.upml["offsets_h"][-1])
    if d is not None:
        n += 2 * st.nshot * 3 * d.K * d.Np
        if d.pml_idx is not None:
            n += 2 * st.nshot * 3 * d.pml_idx.numel() * d.Np
    return n * item


def enabled(st) -> bool:
    """Graphs on CUDA, not disabled, and a state small enough for replay to pay off."""
    if torch.device(st.device).type != "cuda" or not hasattr(torch.cuda, "CUDAGraph"):
        return False
    if os.environ.get("DEEPGPR_HYBRID_DISABLE_GRAPHS"):
        return False
    limit = float(os.environ.get("DEEPGPR_HYBRID_GRAPH_MAX_MB", "32")) * 2 ** 20
    return state_bytes(st) <= limit


class GraphedStep:
    """``rec = g(amp)`` advances :attr:`state` (static buffers) by one step.

    Args:
        st, c: Static data and coefficients (plain tensors).
        src: Source operator (``None``: no source).
        rec: Receiver operator (``None``: no sampling).
        nsrc: Number of source samples per step.
    """

    def __init__(self, st, c, src=None, rec=None, nsrc: int = 1, warmup: int = 2):
        F = _module(st)
        dev = torch.device(st.device)
        self.st, self.src, self.rec_ops = st, src, rec
        self.state = [t.clone() for t in F.zero_state(st)]
        self.amp = torch.zeros(max(1, nsrc), dtype=st.dtype, device=dev)
        amp = self.amp if src is not None else None
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.no_grad(), torch.cuda.stream(side):
            # warm-up (zero state, zero source: the state stays zero); builds the
            # cached index tensors and kernel argument tables outside the capture
            for _ in range(warmup):
                out = F.step(st, c, tuple(self.state), src, amp)
                if rec is not None:
                    F.sample(st, rec, out)
        torch.cuda.current_stream(dev).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            out = F.step(st, c, tuple(self.state), src, amp)
            self.rec = F.sample(st, rec, out) if rec is not None else None
            for a, b in zip(self.state, out):
                if a.numel():
                    a.copy_(b)
        for t in self.state:
            t.zero_()

    def set_state(self, state):
        for a, b in zip(self.state, state):
            if a.numel():
                a.copy_(b)

    def snapshot(self):
        return tuple(t.clone() for t in self.state)

    def __call__(self, amp: Optional[torch.Tensor] = None):
        if amp is not None:
            self.amp.copy_(amp)
        self.graph.replay()
        return self.rec


def make(st, c, src=None, rec=None, nsrc=1) -> Optional[GraphedStep]:
    """A :class:`GraphedStep`, or ``None`` when graphs are disabled or capture fails."""
    if not enabled(st):
        return None
    try:
        return GraphedStep(st, c, src, rec, nsrc)
    except Exception as exc:  # pragma: no cover - depends on the driver/toolkit
        _LOGGER.warning("CUDA Graph capture of the hybrid step failed (%s); using the eager "
                        "step. A failed capture can leave the CUDA RNG unusable in this "
                        "process; set DEEPGPR_HYBRID_DISABLE_GRAPHS=1 to skip it.", exc)
        return None


GraphedStep3D = GraphedStep   # former name

__all__ = ["GraphedStep", "GraphedStep3D", "enabled", "make", "state_bytes"]

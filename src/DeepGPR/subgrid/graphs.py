"""CUDA Graph replay of the subgridded time step (:func:`DeepGPR.subgrid.hsg.step`).

One step launches the main and ``2 r`` subgrid Yee updates plus the fused
coupling kernels, sources and receivers per subgrid; at GPR sizes the eager step
is bound by the host. :class:`GraphedSubgridStep` captures one step, the receiver
sampling and the copy of the new state into static input buffers (as
:class:`DeepGPR.hybrid.graphs.GraphedStep`) and replays it. The arithmetic is
that of the eager step. ``DEEPGPR_SUBGRID_DISABLE_GRAPHS=1`` turns replay off.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch

from ..utils.logger import get_logger
from . import hsg

_LOGGER = get_logger(__name__)


def enabled(st: hsg.SubgridStepStatic) -> bool:
    if torch.device(st.device).type != "cuda" or not hasattr(torch.cuda, "CUDAGraph"):
        return False
    return not os.environ.get("DEEPGPR_SUBGRID_DISABLE_GRAPHS")


class GraphedSubgridStep:
    """``(main, fine) = g(amp, amp_fine)`` advances :attr:`state` by one main step."""

    def __init__(self, st: hsg.SubgridStepStatic, c: hsg.SubgridCoefs, src, rec, nsrc: int,
                 warmup: int = 2, F=hsg):
        dev = torch.device(st.device)
        self.st = st
        self.state = [t.clone() for t in F.zero_state(st, rec)]
        self.amp = torch.zeros(max(1, nsrc), dtype=st.dtype, device=dev)
        self.amp_fine: Optional[List[Optional[torch.Tensor]]] = None
        if src is not None and src.has_sub:
            self.amp_fine = [torch.zeros((max(1, nsrc), sg.time_ratio), dtype=st.dtype, device=dev)
                             if src.sub[g].shot.numel() else None for g, sg in enumerate(st.subs)]
        amp = self.amp if src is not None else None
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.no_grad(), torch.cuda.stream(side):
            for _ in range(warmup):
                out = F.step(st, c, tuple(self.state), src, amp, self.amp_fine, rec)
                if rec is not None:
                    F.sample(st, rec, out)
        torch.cuda.current_stream(dev).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            out = F.step(st, c, tuple(self.state), src, amp, self.amp_fine, rec)
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

    def __call__(self, amp: Optional[torch.Tensor] = None, amp_fine=None):
        if amp is not None:
            self.amp.copy_(amp)
        if amp_fine is not None and self.amp_fine is not None:
            for buf, a in zip(self.amp_fine, amp_fine):
                if buf is not None and a is not None:
                    buf.copy_(a)
        self.graph.replay()
        return self.rec


def make(st, c, src=None, rec=None, nsrc: int = 1, F=hsg) -> Optional[GraphedSubgridStep]:
    """A :class:`GraphedSubgridStep` of the step of ``F`` (default :mod:`.hsg`), or
    ``None`` when disabled or capture fails."""
    if not enabled(st):
        return None
    try:
        return GraphedSubgridStep(st, c, src, rec, nsrc, F=F)
    except Exception as exc:  # pragma: no cover - depends on the driver/toolkit
        _LOGGER.warning("CUDA Graph capture of the subgrid step failed (%s); using the eager "
                        "step.", exc)
        return None


__all__ = ["GraphedSubgridStep", "enabled", "make"]

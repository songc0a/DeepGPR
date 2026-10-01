"""Backward-compatibility shim: use :mod:`DeepGPR.postprocessing.filters`."""

from __future__ import annotations

from .postprocessing.filters import apply_filter, design_fir_filter, hilbert_transform

__all__ = ["apply_filter", "design_fir_filter", "hilbert_transform"]

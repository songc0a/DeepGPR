"""Backward-compatibility shim for the pre-0.1 ``DeepGPR.compute2`` module.

:func:`compute` now lives in :mod:`DeepGPR.solver.compute`, the autograd
bridge in :mod:`DeepGPR.solver.autograd`, history storage in
:mod:`DeepGPR.solver.storage` and the memory report in
:mod:`DeepGPR.solver.memory`. The historical private helper names are kept as
aliases because the test-suite (and some user scripts) import them.

Note: monkey-patching names on *this* module no longer affects
:func:`compute`; patch :mod:`DeepGPR.solver.compute` /
:mod:`DeepGPR.solver.autograd` instead.
"""

from __future__ import annotations

from .config.constants import (
    INT8_BLOCK_DIM_MASK as _INT8_BLOCK_DIM_MASK,
    INT8_BLOCK_X_SHIFT as _INT8_BLOCK_X_SHIFT,
    INT8_BLOCK_Y_SHIFT as _INT8_BLOCK_Y_SHIFT,
    INT8_BLOCK_Z_SHIFT as _INT8_BLOCK_Z_SHIFT,
    INT8_REDUCTION_BACKENDS as _INT8_REDUCTION_BACKENDS,
    INT8_REDUCTION_SHIFT as _INT8_REDUCTION_SHIFT,
    WAVEFIELD_CONVERSION_BACKENDS as _WAVEFIELD_CONVERSION_BACKENDS,
    WAVEFIELD_CONVERSION_SHIFT as _WAVEFIELD_CONVERSION_SHIFT,
    WAVEFIELD_HISTORY_DISABLED as _WAVEFIELD_HISTORY_DISABLED,
    WAVEFIELD_STORAGE_INT8 as _WAVEFIELD_INT8,
)
from .native.bridge import _begin_native_call, _end_native_call
from .postprocessing.io import (
    normalize_forward_wavefield_directory as _normalize_forward_wavefield_directory,
)
from .postprocessing.io import save_forward_wavefield as _save_forward_wavefield
from .solver.autograd import DeepGPR
from .solver.modeling import compute
from .solver.memory import estimate_compute_memory as _estimate_compute_memory
from .solver.memory import format_compute_preview
from .solver.pml import pml_phi_element_count as _pml_phi_elements
from .solver.storage import WAVEFIELD_STORAGE_TYPES as _WAVEFIELD_STORAGE_TYPES
from .solver.storage import decompress_wavefield_history
from .solver.storage import encode_int8_storage_type as _encode_int8_storage_type
from .solver.storage import int8_history_layout as _int8_history_layout
from .solver.storage import normalize_compression_block_size as _normalize_compression_block_size
from .solver.storage import normalize_int8_reduction_backend as _normalize_int8_reduction_backend
from .solver.storage import normalize_wavefield_compression as _normalize_wavefield_compression
from .solver.storage import (
    normalize_wavefield_conversion_backend as _normalize_wavefield_conversion_backend,
)
from .solver.storage import normalize_wavefield_storage_dtype as _normalize_wavefield_storage_dtype
from .utils.formatting import format_memory_size as _format_memory_size
from .utils.tensor_checks import (
    check_nonzero_source_created_fields as _check_nonzero_source_created_fields,
)


def _print_compute_preview(**kwargs) -> None:
    """Print the pre-flight report (see :func:`DeepGPR.solver.memory.format_compute_preview`)."""
    print(format_compute_preview(**kwargs))


__all__ = ["DeepGPR", "compute", "decompress_wavefield_history"]

_COMPAT_PRIVATE = (
    _INT8_BLOCK_DIM_MASK,
    _INT8_BLOCK_X_SHIFT,
    _INT8_BLOCK_Y_SHIFT,
    _INT8_BLOCK_Z_SHIFT,
    _INT8_REDUCTION_BACKENDS,
    _INT8_REDUCTION_SHIFT,
    _WAVEFIELD_CONVERSION_BACKENDS,
    _WAVEFIELD_CONVERSION_SHIFT,
    _WAVEFIELD_HISTORY_DISABLED,
    _WAVEFIELD_INT8,
    _begin_native_call,
    _end_native_call,
    _normalize_forward_wavefield_directory,
    _save_forward_wavefield,
    _estimate_compute_memory,
    _pml_phi_elements,
    _WAVEFIELD_STORAGE_TYPES,
    _encode_int8_storage_type,
    _int8_history_layout,
    _normalize_compression_block_size,
    _normalize_int8_reduction_backend,
    _normalize_wavefield_compression,
    _normalize_wavefield_conversion_backend,
    _normalize_wavefield_storage_dtype,
    _format_memory_size,
    _check_nonzero_source_created_fields,
)

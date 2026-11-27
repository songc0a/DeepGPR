"""Tucker tensor tools for DeepGPR (two independent, optional modules).

Module A - model parameterization (:mod:`DeepGPR.tucker.model`)
    Differentiable Tucker representation of ``epsilon_r`` / ``sigma``:
    :func:`tucker_reconstruct` (also for factors produced by external
    networks), :class:`TuckerTensor`, :class:`TuckerMaterialModel`.

Module B - streaming wavefield sketch (:mod:`DeepGPR.tucker.sketch`,
:mod:`DeepGPR.tucker.wavefield`)
    One-pass randomized Tucker compression of the forward history that the
    discrete adjoint reads; enabled per call with
    ``DeepGPR.compute(..., wavefield_sketch=TuckerSketchConfig(...))``.
    :mod:`DeepGPR.tucker.blockwise`: block-wise, rank-adaptive Tucker history
    with adjoint shadow zones (``wavefield_sketch=BlockTuckerConfig(...)``).

The modules share only stateless helper functions; neither needs the other.
See ``docs/tucker_model_wavefield_compression.md``.
"""

from .model import (
    DenseMaterialField,
    MaterialMapping,
    TuckerMaterialField,
    TuckerMaterialModel,
    TuckerModelConfig,
    TuckerParameterCount,
    TuckerTensor,
    hosvd,
    randomized_hosvd,
    tucker_parameter_count,
    tucker_project,
    tucker_reconstruct,
)
from .sketch import (
    HISTORY_COMPONENTS,
    StreamingTuckerSketch,
    TuckerBlock,
    TuckerSketchConfig,
)
from .wavefield import ComponentErrorStats, HistorySizeReport, SketchedWavefieldHistory
from .blockwise import BlockFrameStats, BlockTuckerConfig, BlockTuckerHistory, surface_adjoint_keep_until
from ._tensor_ops import cosine_similarity, relative_error

__all__ = [
    "BlockFrameStats",
    "BlockTuckerConfig",
    "BlockTuckerHistory",
    "ComponentErrorStats",
    "DenseMaterialField",
    "HISTORY_COMPONENTS",
    "HistorySizeReport",
    "MaterialMapping",
    "SketchedWavefieldHistory",
    "StreamingTuckerSketch",
    "TuckerBlock",
    "TuckerMaterialField",
    "TuckerMaterialModel",
    "TuckerModelConfig",
    "TuckerParameterCount",
    "TuckerSketchConfig",
    "TuckerTensor",
    "cosine_similarity",
    "hosvd",
    "randomized_hosvd",
    "relative_error",
    "surface_adjoint_keep_until",
    "tucker_parameter_count",
    "tucker_project",
    "tucker_reconstruct",
]

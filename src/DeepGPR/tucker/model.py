"""Module A: differentiable Tucker low-rank parameterization of material models.

A 2D/3D material model is represented as

    M = G ×_1 U_1 ×_2 U_2 [×_3 U_3],   G: (r_1, ..., r_N),  U_n: (I_n, r_n)

and reconstructed on every forward pass, so gradients of any loss computed
from ``M`` (for example DeepGPR's FDTD adjoint gradients) flow back into the
core and the factors - and, when the factors are produced by an external
network, further into that network.

Public API
----------
:func:`tucker_reconstruct`
    Pure, fully differentiable ``(core, factors) -> tensor``. Accepts leaf
    parameters or tensors computed by external code (e.g. an implicit neural
    representation the caller builds outside DeepGPR).
:class:`TuckerTensor`
    ``nn.Module`` holding a trainable core and factors with an optional
    factor normalization that removes the Tucker scale gauge.
:class:`MaterialMapping`
    Differentiable maps from the unconstrained Tucker value to a physical
    property (identity, sigmoid bounds, softplus/exp lower bounds).
:class:`TuckerModelConfig`, :class:`TuckerMaterialField`,
:class:`DenseMaterialField`, :class:`TuckerMaterialModel`
    Independent ``epsilon_r`` and ``sigma`` representations (Tucker or dense)
    for dual-parameter inversion.
:func:`hosvd`, :func:`randomized_hosvd`, :func:`tucker_parameter_count`
    One-off initialisation and parameter accounting.

The randomized decomposition here is used only for one-off initialisation;
during inversion the core and the factors are the optimisation variables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ._tensor_ops import leading_left_singular_vectors, mode_dot, normalize_ranks, unfold

#: Accepted ``factor_normalization`` values.
FACTOR_NORMALIZATIONS = ("none", "rms", "orthonormal")

#: Accepted :class:`MaterialMapping` kinds.
MAPPING_KINDS = ("identity", "sigmoid", "softplus", "exp")

#: Accepted initialisation methods of :class:`TuckerModelConfig`.
INIT_METHODS = ("hosvd", "randomized", "random")

_NORM_FLOOR = 1.0e-30


# ---------------------------------------------------------------------------
# Pure reconstruction and decomposition
# ---------------------------------------------------------------------------


def _check_core_and_factors(core: Any, factors: Any) -> List[Optional[torch.Tensor]]:
    if not torch.is_tensor(core):
        raise TypeError("core must be a torch.Tensor.")
    if not core.is_floating_point():
        raise TypeError(f"core must be a floating-point tensor, got {core.dtype}.")
    if isinstance(factors, torch.Tensor) or not isinstance(factors, (list, tuple)):
        raise TypeError("factors must be a list or tuple with one matrix (or None) per mode.")
    if core.dim() != len(factors) or core.dim() < 1:
        raise ValueError(
            f"core has {core.dim()} modes but {len(factors)} factors were given; "
            "pass exactly one factor per core mode."
        )
    checked: List[Optional[torch.Tensor]] = []
    for mode, factor in enumerate(factors):
        if factor is None:
            checked.append(None)
            continue
        if not torch.is_tensor(factor) or factor.dim() != 2:
            raise ValueError(f"factor {mode} must be a 2D tensor (I_{mode}, r_{mode}).")
        if factor.shape[1] != core.shape[mode]:
            raise ValueError(
                f"factor {mode} has {factor.shape[1]} columns but the core mode-{mode} "
                f"size is {core.shape[mode]}."
            )
        if factor.dtype != core.dtype:
            raise TypeError(
                f"factor {mode} dtype {factor.dtype} differs from the core dtype {core.dtype}."
            )
        if factor.device != core.device:
            raise ValueError(
                f"factor {mode} is on {factor.device} but the core is on {core.device}."
            )
        checked.append(factor)
    return checked


def tucker_reconstruct(
    core: torch.Tensor, factors: Sequence[Optional[torch.Tensor]]
) -> torch.Tensor:
    """Return the full tensor ``core ×_1 U_1 ×_2 U_2 ... ×_N U_N``.

    The operation is a chain of differentiable ``tensordot`` calls: no
    ``detach``, ``.data`` or in-place writes, so autograd reaches ``core``,
    every factor and whatever produced them (e.g. an external network).

    Args:
        core: ``(r_1, ..., r_N)`` core tensor.
        factors: One ``(I_n, r_n)`` matrix per mode; ``None`` keeps mode ``n``
            unchanged (identity factor).

    Returns:
        ``(I_1, ..., I_N)`` tensor with the core's dtype and device.

    Raises:
        TypeError: Non-tensor inputs or mismatched dtypes.
        ValueError: Mismatched mode counts, ranks or devices.
    """
    checked = _check_core_and_factors(core, factors)
    # Apply the products that grow the tensor least first.
    order = sorted(
        (mode for mode, factor in enumerate(checked) if factor is not None),
        key=lambda mode: checked[mode].shape[0] / max(core.shape[mode], 1),  # type: ignore[union-attr]
    )
    result = core
    for mode in order:
        result = mode_dot(result, checked[mode], mode)  # type: ignore[arg-type]
    return result


def tucker_project(
    tensor: torch.Tensor, factors: Sequence[Optional[torch.Tensor]]
) -> torch.Tensor:
    """Return ``tensor ×_1 U_1^T ... ×_N U_N^T`` (the adjoint of reconstruction)."""
    if tensor.dim() != len(factors):
        raise ValueError("tensor must have one mode per factor.")
    result = tensor
    for mode, factor in enumerate(factors):
        if factor is not None:
            result = mode_dot(result, factor.transpose(0, 1), mode)
    return result


def hosvd(
    tensor: torch.Tensor, ranks: Union[int, Sequence[int]]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Truncated higher-order SVD (De Lathauwer et al., 2000).

    Args:
        tensor: ``(I_1, ..., I_N)`` floating-point tensor.
        ranks: Target ranks (each clipped to ``I_n``).

    Returns:
        ``(core, factors)`` with orthonormal factors.
    """
    ranks = normalize_ranks(ranks, tensor.dim())
    with torch.no_grad():
        factors = [
            leading_left_singular_vectors(unfold(tensor, mode), min(rank, tensor.shape[mode]))
            for mode, rank in enumerate(ranks)
        ]
        core = tucker_project(tensor, factors)
    return core, factors


def randomized_hosvd(
    tensor: torch.Tensor,
    ranks: Union[int, Sequence[int]],
    oversampling: int = 8,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Randomized HOSVD: Gaussian range finder per unfolding, then truncation.

    For each mode a ``(prod_{m != n} I_m) x (r_n + p)`` Gaussian test matrix
    is applied to the unfolding (Halko, Martinsson and Tropp, 2011), the
    sketch is orthonormalised, and the leading ``r_n`` left singular vectors
    of the projected unfolding are kept. Intended for one-off initialisation.

    Args:
        tensor: ``(I_1, ..., I_N)`` tensor.
        ranks: Target ranks.
        oversampling: Extra sketch columns ``p``.
        seed: Seed of the CPU generator drawing the test matrices.

    Returns:
        ``(core, factors)`` with orthonormal factors.
    """
    ranks = normalize_ranks(ranks, tensor.dim())
    if isinstance(oversampling, bool) or not isinstance(oversampling, int) or oversampling < 0:
        raise ValueError("oversampling must be a non-negative integer.")
    generator = torch.Generator()
    generator.manual_seed(0 if seed is None else int(seed))
    factors: List[torch.Tensor] = []
    with torch.no_grad():
        for mode, rank in enumerate(ranks):
            unfolded = unfold(tensor, mode)
            columns = min(rank + oversampling, unfolded.shape[0], unfolded.shape[1])
            test = torch.randn(
                (unfolded.shape[1], columns), generator=generator, dtype=tensor.dtype
            ).to(tensor.device)
            basis, _ = torch.linalg.qr(unfolded @ test)
            small = basis.transpose(0, 1) @ unfolded
            left = leading_left_singular_vectors(small, min(rank, small.shape[0]))
            factors.append(basis @ left)
        core = tucker_project(tensor, factors)
    return core, factors


class TuckerParameterCount(NamedTuple):
    """Parameter counts of a Tucker representation vs. a dense voxel grid.

    Attributes:
        full_grid: ``prod(I_n)``, the dense grid it replaces (the comparison basis).
        core: ``prod(r_n)``.
        factors: ``sum(I_n r_n)``.
        total: ``core + factors``.
        compression_ratio: ``full_grid / total`` (> 1 means fewer parameters).
    """

    full_grid: int
    core: int
    factors: int
    total: int
    compression_ratio: float


def tucker_parameter_count(
    shape: Sequence[int], ranks: Union[int, Sequence[int]]
) -> TuckerParameterCount:
    """Count the trainable values of a Tucker model of ``shape`` and ``ranks``.

    The comparison basis is a dense voxel parameterization of the same grid.
    When an external network generates the factors, *its* trainable
    parameters replace ``factors`` in a fair comparison; DeepGPR cannot count
    them.
    """
    shape = tuple(int(size) for size in shape)
    ranks = normalize_ranks(ranks, len(shape))
    full = math.prod(shape)
    core = math.prod(ranks)
    factors = sum(size * rank for size, rank in zip(shape, ranks))
    total = core + factors
    return TuckerParameterCount(full, core, factors, total, full / total)


# ---------------------------------------------------------------------------
# Factor normalization (scale gauge)
# ---------------------------------------------------------------------------


def _normalize_factor(raw: torch.Tensor, normalization: str) -> torch.Tensor:
    """Map a raw factor parameter to the effective factor."""
    if normalization == "none":
        return raw
    if normalization == "rms":
        norms = torch.linalg.norm(raw, dim=0, keepdim=True).clamp_min(_NORM_FLOOR)
        return raw * (math.sqrt(raw.shape[0]) / norms)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs


def _gauge_core(
    core: torch.Tensor, factors: Sequence[torch.Tensor], normalization: str
) -> torch.Tensor:
    """Core that, with ``_normalize_factor(factors)``, represents the same tensor."""
    result = core
    for mode, factor in enumerate(factors):
        if normalization == "none":
            continue
        if normalization == "rms":
            scale = torch.linalg.norm(factor, dim=0).clamp_min(_NORM_FLOOR) / math.sqrt(
                factor.shape[0]
            )
            shape = [1] * core.dim()
            shape[mode] = -1
            result = result * scale.reshape(shape)
        else:
            _, r = torch.linalg.qr(factor)
            signs = torch.sign(torch.diagonal(r))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            result = mode_dot(result, signs[:, None] * r, mode)
    return result


class TuckerTensor(nn.Module):
    """Trainable Tucker tensor ``G ×_1 U_1 ... ×_N U_N``.

    Scale gauge: ``(G ×_n A^{-1}, U_n A)`` represent the same tensor for any
    invertible ``A``, so with raw factors the optimizer can drift along this
    flat valley (factor norms growing while the core shrinks), and Adam's
    per-entry step sizes treat a core entry of ``O(||M||)`` and a factor entry
    of ``O(1/sqrt(I))`` very differently. ``factor_normalization`` fixes the
    gauge inside the forward pass:

    * ``"rms"`` (default): every factor column is rescaled to unit RMS,
      ``U_n = V_n sqrt(I_n) / ||V_n||_col``. The represented tensor is
      invariant to column scalings of ``V_n``, the core carries the physical
      scale (a constant model ``c`` has core ``c``), and factor entries are
      ``O(1)`` - one learning rate fits both.
    * ``"orthonormal"``: ``U_n = qr(V_n).Q`` (sign-fixed); removes scaling and
      shearing, leaving only rotations absorbed by the core.
    * ``"none"``: raw factors; use :meth:`orthogonalize_` periodically or
      :meth:`gauge_penalty` to control the drift.

    Args:
        core: Initial effective core ``(r_1, ..., r_N)``.
        factors: Initial effective factors ``(I_n, r_n)``.
        factor_normalization: ``"rms"``, ``"orthonormal"`` or ``"none"``.
        trainable: Register as ``nn.Parameter`` (``True``) or buffers.
    """

    def __init__(
        self,
        core: torch.Tensor,
        factors: Sequence[torch.Tensor],
        factor_normalization: str = "rms",
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if factor_normalization not in FACTOR_NORMALIZATIONS:
            raise ValueError(f"factor_normalization must be one of {FACTOR_NORMALIZATIONS}.")
        checked = _check_core_and_factors(core, list(factors))
        if any(factor is None for factor in checked):
            raise ValueError("TuckerTensor needs a factor matrix for every mode.")
        for mode, factor in enumerate(checked):
            assert factor is not None
            if factor_normalization == "orthonormal" and factor.shape[1] > factor.shape[0]:
                raise ValueError(
                    f"orthonormal factors need r_{mode} <= I_{mode} "
                    f"(got r={factor.shape[1]}, I={factor.shape[0]})."
                )
        self.factor_normalization = factor_normalization
        self.trainable = bool(trainable)
        # Parameters are stored contiguously: decompositions return strided
        # views (e.g. SVD column slices), and optimizers such as LBFGS call
        # ``grad.view(-1)``, which needs gradients in contiguous layout.
        contiguous = torch.contiguous_format
        with torch.no_grad():
            raw_factors = [factor.detach().clone(memory_format=contiguous) for factor in checked]  # type: ignore[union-attr]
            raw_core = _gauge_core(core.detach(), raw_factors, factor_normalization).clone(
                memory_format=contiguous
            )
        if self.trainable:
            self.core_param = nn.Parameter(raw_core)
            self.factor_params = nn.ParameterList([nn.Parameter(f) for f in raw_factors])
        else:
            self.register_buffer("core_param", raw_core)
            for mode, factor in enumerate(raw_factors):
                self.register_buffer(f"factor_buffer_{mode}", factor)

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        ranks: Union[int, Sequence[int]],
        method: str = "hosvd",
        oversampling: int = 8,
        seed: Optional[int] = None,
        factor_normalization: str = "rms",
        trainable: bool = True,
    ) -> "TuckerTensor":
        """Initialise from a full tensor by (randomized) HOSVD.

        Args:
            tensor: Full tensor to approximate.
            ranks: Tucker ranks.
            method: ``"hosvd"`` or ``"randomized"``.
            oversampling: Randomized sketch oversampling.
            seed: Randomized-method seed.
            factor_normalization: See the class docstring.
            trainable: See the class docstring.
        """
        if method == "hosvd":
            core, factors = hosvd(tensor, ranks)
        elif method == "randomized":
            core, factors = randomized_hosvd(tensor, ranks, oversampling, seed)
        else:
            raise ValueError("method must be 'hosvd' or 'randomized'.")
        return cls(core, factors, factor_normalization, trainable)

    @classmethod
    def random(
        cls,
        shape: Sequence[int],
        ranks: Union[int, Sequence[int]],
        scale: float = 1.0,
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: Any = None,
        factor_normalization: str = "rms",
        trainable: bool = True,
    ) -> "TuckerTensor":
        """Random initialisation: orthonormal random factors, Gaussian core.

        The represented tensor has RMS ``scale`` in expectation
        (``core ~ N(0, scale^2 prod(I_n / r_n))``).
        """
        shape = tuple(int(size) for size in shape)
        ranks = normalize_ranks(ranks, len(shape))
        generator = torch.Generator()
        generator.manual_seed(0 if seed is None else int(seed))
        factors = []
        for size, rank in zip(shape, ranks):
            gaussian = torch.randn((size, rank), generator=generator, dtype=torch.float64)
            q, _ = torch.linalg.qr(gaussian)
            factors.append(q.to(dtype=dtype, device=device))
        core_std = scale * math.sqrt(math.prod(shape) / math.prod(ranks))
        core = core_std * torch.randn(tuple(ranks), generator=generator, dtype=torch.float64)
        return cls(core.to(dtype=dtype, device=device), factors, factor_normalization, trainable)

    # -- accessors -----------------------------------------------------------

    @property
    def raw_factors(self) -> List[torch.Tensor]:
        """Raw factor parameters (or buffers) ``V_n``."""
        if self.trainable:
            return list(self.factor_params)
        return [getattr(self, f"factor_buffer_{mode}") for mode in range(self.core_param.dim())]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the represented tensor."""
        return tuple(int(factor.shape[0]) for factor in self.raw_factors)

    @property
    def ranks(self) -> Tuple[int, ...]:
        """Tucker ranks."""
        return tuple(int(size) for size in self.core_param.shape)

    def core(self) -> torch.Tensor:
        """Effective core (differentiable)."""
        return self.core_param

    def factors(self) -> List[torch.Tensor]:
        """Effective (normalized) factors (differentiable)."""
        return [_normalize_factor(raw, self.factor_normalization) for raw in self.raw_factors]

    def forward(self) -> torch.Tensor:  # type: ignore[override]
        """Reconstruct the full tensor."""
        return tucker_reconstruct(self.core(), self.factors())

    # -- gauge control -------------------------------------------------------

    @torch.no_grad()
    def orthogonalize_(self) -> "TuckerTensor":
        """Re-gauge in place to orthogonal, unit-RMS effective factors.

        Each factor becomes ``Q_n sqrt(I_n)`` (``factor = Q_n R_n``) and the
        core absorbs ``R_n / sqrt(I_n)``, so the represented tensor is
        unchanged (up to rounding) and :meth:`gauge_penalty` is zero.
        (With ``"orthonormal"`` normalization the effective factors stay
        ``Q_n``.) Optimizer moments refer to the old gauge; reset or recreate
        stateful optimizers such as Adam afterwards.
        """
        core = self.core()
        factors = self.factors()
        new_factors = []
        for mode, factor in enumerate(factors):
            q, r = torch.linalg.qr(factor)
            scale = math.sqrt(factor.shape[0])
            core = mode_dot(core, r / scale, mode)
            new_factors.append(q * scale)
        raw_core = _gauge_core(core, new_factors, self.factor_normalization)
        self.core_param.copy_(raw_core)
        for raw, factor in zip(self.raw_factors, new_factors):
            raw.copy_(factor)  # copy_ keeps the contiguous parameter layout
        return self

    def gauge_penalty(self) -> torch.Tensor:
        """``sum_n ||U_n^T U_n / I_n - I||_F^2`` on the effective factors.

        Add a small multiple to the loss to keep raw (``"none"``) factors near
        an orthogonal, unit-RMS gauge. Zero for exactly orthogonal unit-RMS
        columns.
        """
        penalty = self.core_param.new_zeros(())
        for factor in self.factors():
            gram = factor.transpose(0, 1) @ factor / factor.shape[0]
            identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            penalty = penalty + (gram - identity).square().sum()
        return penalty

    # -- export / accounting -------------------------------------------------

    def export(self) -> Dict[str, Any]:
        """Detached copies of the effective core and factors plus metadata."""
        with torch.no_grad():
            return {
                "core": self.core().detach().clone(),
                "factors": [factor.detach().clone() for factor in self.factors()],
                "factor_normalization": self.factor_normalization,
                "trainable": self.trainable,
            }

    @classmethod
    def from_export(cls, exported: Dict[str, Any]) -> "TuckerTensor":
        """Rebuild a :class:`TuckerTensor` from :meth:`export` output."""
        return cls(
            exported["core"],
            exported["factors"],
            exported.get("factor_normalization", "rms"),
            exported.get("trainable", True),
        )

    def parameter_count(self) -> TuckerParameterCount:
        """Trainable values compared with a dense grid of the same shape."""
        return tucker_parameter_count(self.shape, self.ranks)

    def memory_report(self) -> Dict[str, int]:
        """Measured bytes of the Tucker tensors and of the reconstructed grid.

        ``reconstructed_bytes`` is what every forward pass materialises: the
        FDTD solver needs the full grid, so Tucker reduces the *trainable*
        state (parameters, gradients, optimizer moments), not the model grid
        the solver reads.
        """
        tensors = [self.core_param, *self.raw_factors]
        parameter_bytes = sum(t.numel() * t.element_size() for t in tensors)
        gradient_bytes = sum(
            t.grad.numel() * t.grad.element_size()
            for t in tensors
            if getattr(t, "grad", None) is not None
        )
        element = self.core_param.element_size()
        return {
            "parameter_bytes": parameter_bytes,
            "gradient_bytes": gradient_bytes,
            "reconstructed_bytes": math.prod(self.shape) * element,
            "dense_parameter_bytes": math.prod(self.shape) * element,
        }


# ---------------------------------------------------------------------------
# Material mappings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialMapping:
    """Differentiable map from an unconstrained value ``t`` to a material value.

    * ``"identity"``: ``offset + scale * t`` (no bounds).
    * ``"sigmoid"``: ``lower + (upper - lower) * sigmoid(t)``; strict two-sided
      bounds, nonzero gradient everywhere.
    * ``"softplus"``: ``lower + scale * softplus(t)``; strict lower bound.
    * ``"exp"``: ``lower + scale * exp(t)``; strict lower bound, log scaling
      (useful for conductivity spanning decades).

    Hard clipping is intentionally not offered: it has zero gradient outside
    the bounds and can freeze cells permanently.

    Attributes:
        kind: One of :data:`MAPPING_KINDS`.
        lower, upper: Bounds (``upper`` only for ``"sigmoid"``).
        scale, offset: Affine parameters (see above).
    """

    kind: str = "identity"
    lower: Optional[float] = None
    upper: Optional[float] = None
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in MAPPING_KINDS:
            raise ValueError(f"mapping kind must be one of {MAPPING_KINDS}, got {self.kind!r}.")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("mapping scale must be a positive finite number.")
        if self.kind == "sigmoid":
            if self.lower is None or self.upper is None or not self.lower < self.upper:
                raise ValueError("sigmoid mapping needs bounds lower < upper.")
        elif self.kind == "identity":
            if self.lower is not None or self.upper is not None:
                raise ValueError(
                    "The identity mapping has no bounds; use 'sigmoid' (two-sided) or "
                    "'softplus'/'exp' (lower bound). Hard clipping would zero the gradient."
                )
        elif self.upper is not None:
            raise ValueError(f"'{self.kind}' mapping supports only a lower bound.")

    @property
    def _lower(self) -> float:
        return 0.0 if self.lower is None else float(self.lower)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        """Map an unconstrained tensor to material values (differentiable)."""
        if self.kind == "identity":
            return self.offset + self.scale * value
        if self.kind == "sigmoid":
            return self._lower + (float(self.upper) - self._lower) * torch.sigmoid(value)  # type: ignore[arg-type]
        if self.kind == "softplus":
            return self._lower + self.scale * nn.functional.softplus(value)
        return self._lower + self.scale * torch.exp(value)

    def inverse(self, material: torch.Tensor, margin: float = 1.0e-6) -> torch.Tensor:
        """Unconstrained value whose image is ``material``.

        Values on or outside the bounds are first moved ``margin`` (relative
        to the bound span or scale) inside.
        """
        material = material.detach()
        if self.kind == "identity":
            return (material - self.offset) / self.scale
        if self.kind == "sigmoid":
            span = float(self.upper) - self._lower  # type: ignore[arg-type]
            fraction = ((material - self._lower) / span).clamp(margin, 1.0 - margin)
            return torch.log(fraction) - torch.log1p(-fraction)
        scaled = ((material - self._lower) / self.scale).clamp_min(margin)
        if self.kind == "softplus":
            # softplus^{-1}(y) = y + log(1 - exp(-y)), stable for large y.
            return scaled + torch.log(-torch.expm1(-scaled))
        return torch.log(scaled)


def _mapping_from_config(config: "TuckerModelConfig") -> MaterialMapping:
    lower, upper = (None, None) if config.bounds is None else config.bounds
    if config.mapping == "identity":
        return MaterialMapping("identity", scale=config.scale, offset=config.offset)
    return MaterialMapping(config.mapping, lower=lower, upper=upper, scale=config.scale)


# ---------------------------------------------------------------------------
# Material fields and the dual-parameter model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuckerModelConfig:
    """Options of one material parameter (``epsilon_r`` or ``sigma``).

    Attributes:
        enabled: Tucker parameterization (``True``) or a dense voxel field.
        ranks: Tucker ranks (int or one per mode; clipped to the grid size).
        init: ``"hosvd"``, ``"randomized"`` (from the initial model) or
            ``"random"`` (ignores the initial model except for its shape).
        mapping: ``"identity"``, ``"sigmoid"``, ``"softplus"`` or ``"exp"``.
        bounds: ``(lower, upper)`` for ``"sigmoid"``; ``(lower, None)`` for
            ``"softplus"``/``"exp"``.
        scale, offset: Affine parameters of the mapping.
        factor_normalization: See :class:`TuckerTensor`.
        trainable: Whether the field is optimised (else a fixed buffer).
        oversampling: Sketch oversampling of ``init="randomized"``.
        seed: Seed of the randomized / random initialisation.
        random_scale: RMS of the unconstrained field for ``init="random"``.
    """

    enabled: bool = True
    ranks: Union[int, Sequence[int]] = 8
    init: str = "hosvd"
    mapping: str = "identity"
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    scale: float = 1.0
    offset: float = 0.0
    factor_normalization: str = "rms"
    trainable: bool = True
    oversampling: int = 8
    seed: Optional[int] = None
    random_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.init not in INIT_METHODS:
            raise ValueError(f"init must be one of {INIT_METHODS}, got {self.init!r}.")
        if self.factor_normalization not in FACTOR_NORMALIZATIONS:
            raise ValueError(f"factor_normalization must be one of {FACTOR_NORMALIZATIONS}.")
        if self.bounds is not None and len(self.bounds) != 2:
            raise ValueError("bounds must be a (lower, upper) pair.")
        _mapping_from_config(self)  # validates mapping/bounds


class TuckerMaterialField(nn.Module):
    """Material field ``m = mapping(TuckerTensor())``."""

    def __init__(self, tucker: TuckerTensor, mapping: MaterialMapping) -> None:
        super().__init__()
        self.tucker = tucker
        self.mapping = mapping

    @classmethod
    def from_model(
        cls, material: torch.Tensor, config: TuckerModelConfig
    ) -> "TuckerMaterialField":
        """Initialise from a full material model according to ``config``."""
        mapping = _mapping_from_config(config)
        ranks = [
            min(rank, size)
            for rank, size in zip(normalize_ranks(config.ranks, material.dim()), material.shape)
        ]
        if config.init == "random":
            tucker = TuckerTensor.random(
                material.shape,
                ranks,
                scale=config.random_scale,
                seed=config.seed,
                dtype=material.dtype,
                device=material.device,
                factor_normalization=config.factor_normalization,
                trainable=config.trainable,
            )
        else:
            tucker = TuckerTensor.from_tensor(
                mapping.inverse(material),
                ranks,
                method=config.init,
                oversampling=config.oversampling,
                seed=config.seed,
                factor_normalization=config.factor_normalization,
                trainable=config.trainable,
            )
        return cls(tucker, mapping)

    @classmethod
    def from_factors(
        cls,
        core: torch.Tensor,
        factors: Sequence[torch.Tensor],
        mapping: Optional[MaterialMapping] = None,
        factor_normalization: str = "rms",
        trainable: bool = True,
    ) -> "TuckerMaterialField":
        """Initialise from a user-supplied core and factors (unconstrained space)."""
        tucker = TuckerTensor(core, factors, factor_normalization, trainable)
        return cls(tucker, mapping or MaterialMapping())

    def forward(self) -> torch.Tensor:  # type: ignore[override]
        """Full material tensor (differentiable)."""
        return self.mapping(self.tucker())

    def parameter_count(self) -> TuckerParameterCount:
        """Tucker parameter accounting (see :func:`tucker_parameter_count`)."""
        return self.tucker.parameter_count()


class DenseMaterialField(nn.Module):
    """Dense voxel field ``m = mapping(V)`` (the non-Tucker option)."""

    def __init__(
        self, material: torch.Tensor, mapping: Optional[MaterialMapping] = None, trainable: bool = True
    ) -> None:
        super().__init__()
        self.mapping = mapping or MaterialMapping()
        value = self.mapping.inverse(material).clone(memory_format=torch.contiguous_format)
        self.trainable = bool(trainable)
        if self.trainable:
            self.value = nn.Parameter(value)
        else:
            self.register_buffer("value", value)

    def forward(self) -> torch.Tensor:  # type: ignore[override]
        """Full material tensor."""
        return self.mapping(self.value)

    def parameter_count(self) -> TuckerParameterCount:
        """Dense accounting (ratio 1)."""
        count = self.value.numel()
        return TuckerParameterCount(count, 0, 0, count, 1.0)


MaterialField = Union[TuckerMaterialField, DenseMaterialField]


def _make_field(material: torch.Tensor, config: Optional[TuckerModelConfig]) -> MaterialField:
    if config is None:
        config = TuckerModelConfig(enabled=False)
    if config.enabled:
        return TuckerMaterialField.from_model(material, config)
    return DenseMaterialField(material, _mapping_from_config(config), config.trainable)


class TuckerMaterialModel(nn.Module):
    """Independent ``epsilon_r`` and ``sigma`` fields for dual-parameter FWI.

    Each field is a :class:`TuckerMaterialField` or a
    :class:`DenseMaterialField`; nothing is shared between them (separate
    cores, factors, ranks, mappings and initialisations).

    ``forward()`` returns ``(epsilon_r, sigma)`` ready for
    :func:`DeepGPR.compute`; gradients of the FDTD loss flow back into the
    Tucker parameters through the solver's adjoint.
    """

    def __init__(self, epsilon: MaterialField, sigma: MaterialField) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma

    @classmethod
    def from_models(
        cls,
        epsilon_r: torch.Tensor,
        sigma: torch.Tensor,
        epsilon_config: Optional[TuckerModelConfig] = None,
        sigma_config: Optional[TuckerModelConfig] = None,
        dtype: Optional[torch.dtype] = None,
        device: Any = None,
    ) -> "TuckerMaterialModel":
        """Build both fields from initial full models.

        Args:
            epsilon_r, sigma: Initial models of identical shape.
            epsilon_config, sigma_config: Per-field options; ``None`` gives a
                trainable dense field with the identity mapping.
            dtype, device: Optional conversion of the initial models.
        """
        if epsilon_r.shape != sigma.shape:
            raise ValueError("epsilon_r and sigma must have the same shape.")
        if epsilon_r.dim() not in (2, 3):
            raise ValueError("Material models must be 2D or 3D.")
        epsilon_r = epsilon_r.detach().to(dtype=dtype or epsilon_r.dtype, device=device)
        sigma = sigma.detach().to(dtype=dtype or sigma.dtype, device=device)
        return cls(_make_field(epsilon_r, epsilon_config), _make_field(sigma, sigma_config))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """``(epsilon_r, sigma)`` full tensors."""
        return self.epsilon(), self.sigma()

    def parameter_groups(
        self, epsilon_lr: float, sigma_lr: float
    ) -> List[Dict[str, Any]]:
        """Optimizer parameter groups with separate learning rates."""
        groups = []
        for field, lr in ((self.epsilon, epsilon_lr), (self.sigma, sigma_lr)):
            parameters = [p for p in field.parameters() if p.requires_grad]
            if parameters:
                groups.append({"params": parameters, "lr": lr})
        return groups

    def parameter_report(self) -> Dict[str, Dict[str, Any]]:
        """Per-field parameter counts (dense grid is the comparison basis)."""
        return {
            name: dict(field.parameter_count()._asdict(), tucker=isinstance(field, TuckerMaterialField))
            for name, field in (("epsilon_r", self.epsilon), ("sigma", self.sigma))
        }


__all__ = [
    "DenseMaterialField",
    "FACTOR_NORMALIZATIONS",
    "INIT_METHODS",
    "MAPPING_KINDS",
    "MaterialMapping",
    "TuckerMaterialField",
    "TuckerMaterialModel",
    "TuckerModelConfig",
    "TuckerParameterCount",
    "TuckerTensor",
    "hosvd",
    "randomized_hosvd",
    "tucker_parameter_count",
    "tucker_project",
    "tucker_reconstruct",
]

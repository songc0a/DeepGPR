"""Module B (algorithm): one-pass streaming randomized Tucker sketch.

Implements the single-pass Tucker approximation of Sun, Guo, Luo, Tropp and
Udell, "Low-rank Tucker approximation of a tensor from streaming data",
SIAM J. Math. Data Sci. 2(4), 2020, with tensor-random-projection
(Khatri-Rao) factor maps. For a tensor ``X`` of shape ``(I_1, ..., I_M)``,
target ranks ``r_n``, sketch sizes ``k_n = r_n + p`` and core-sketch sizes
``s_n = 2 k_n + 1``:

Sketch (linear in X, one pass, the data are never revisited)
    factor sketches  ``Y_n = X_(n) Ω_n``,   ``Ω_n = ⊙_{m != n} A_n^(m)``,
                     ``A_n^(m) ~ N(0, 1)^{I_m x k_n}`` (never formed as a
                     ``prod_{m != n} I_m x k_n`` matrix)
    core sketch      ``Z = X ×_1 Φ_1 ... ×_M Φ_M``,  ``Φ_n ~ N(0, 1)^{s_n x I_n}``

Recover
    ``Q_n = qr(Y_n)``                    (``I_n x k_n``, orthonormal)
    ``C = Z ×_1 (Φ_1 Q_1)^† ... ×_M (Φ_M Q_M)^†``   (least squares per mode)
    ``X ≈ C ×_1 Q_1 ... ×_M Q_M``
    then a truncated HOSVD of the small ``k``-core reduces to the target ranks:
    ``C ≈ G ×_n V_n``, ``U_n = Q_n V_n``  (the stored representation).

Streaming: when mode 0 is the *stream mode* (time), slices ``X[t]`` arrive
one chunk at a time. Every sketch is linear, so a chunk of rows
``t in [a, b)`` contributes

    ``Y_n += X[a:b] contracted with A_n^(m) (m != n) and rows A_n^(0)[a:b]``
    ``Y_0[a:b] = X[a:b] contracted with A_0^(m) (m >= 1)``
    ``Z += X[a:b] ×_0 Φ_0[:, a:b] ×_{m>=1} Φ_m``

and the raw slices can be discarded immediately.

Modes with ``r_n >= I_n`` are *kept*: no factor is stored and ``Φ_n = I``.
If every mode is kept the representation is the raw tensor.

If ``X`` has multilinear rank ``<= k_n`` in every mode, the recovery is
exact (up to rounding) for Gaussian maps with probability one; the tests use
this to detect any unfolding, axis-order or core-recovery error.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from ._tensor_ops import (
    batched_mode_dot,
    khatri_rao_contract,
    matmul_precision,
    mode_dot,
    unfold,
)

RankSpec = Union[int, Sequence[int]]

#: History component names understood by :class:`TuckerSketchConfig.ranks`.
HISTORY_COMPONENTS = ("Ex", "Ey", "Ez", "Rx", "Ry", "Rz")


@dataclass(frozen=True)
class TuckerSketchConfig:
    """Options of the streaming Tucker compression of the FDTD history.

    Attributes:
        enabled: Master switch; ``False`` keeps the original solver path.
        ranks: Spatial target ranks: an int, one value per spatial axis
            ``(x, y[, z])``, or a mapping from a component name
            (``"Ex"``, ..., ``"Rz"``) or group (``"E"``, ``"R"``) to such a
            value (more specific keys win; a missing key falls back to
            ``"default"`` or 12). Singleton axes (2D) are never compressed.
        oversampling: Factor-sketch oversampling ``p`` (``k = r + p``).
        core_sketch_factor: Core-sketch size ``s = ceil(factor * k) + 1``
            (2.0 gives the recommended ``2k + 1``).
        seed: Base seed; every component draws independent maps from it.
        time_block: Frames per Tucker block. 1 compresses every saved frame
            on its own (``W_t ≈ G_t ×_1 U_x^t ×_2 U_y^t ×_3 U_z^t``); larger
            values add a streamed time mode whose spatial factors are shared
            by the block.
        time_rank: Time rank of a block (default ``min(time_block, 4)``).
        time_oversampling: Time-mode oversampling (default ``oversampling``).
        segment_steps: Time steps per native solver call (the full-resolution
            history buffer holds this many frames). Default 16; rounded up to
            a multiple of the gradient sampling interval.
        compute_dtype: Sketch/recovery dtype (float32 or float64).
        compress: Components to compress: ``"all"``, ``"E"``, ``"R"`` or a
            sequence of component names. Others are stored uncompressed.
        min_compression_ratio: Store a component uncompressed when its Tucker
            representation is not at least this many times smaller.
        measure_error: Measure the reconstruction error of every block that
            is completed inside one segment (extra cost; diagnostics).
        allow_tf32: Allow TF32 matmuls in the sketch on Ampere+ GPUs (faster,
            but about 1e-3 relative precision). Default ``False``.
        recovery_device: Where the small recovery factorizations run:
            ``"host"`` (CPU, float64; two device-host copies per segment) or
            ``"device"`` (the solver device, float64; no copies, but batched
            ``torch.linalg`` calls on tiny matrices are latency bound on GPUs).
        rhs: ``"stored"`` (default) compresses ``E^n`` and ``R^n`` separately.
            ``"from_e"`` compresses only ``E^n`` (plus the final frame) and
            lets the native kernel rebuild ``R^n = (E^(n+1) - ca E^n) / cb``
            from consecutive reconstructed frames: half the stored
            components, but the difference of two lossy frames can amplify
            the error. Requires ``model_gradient_sampling_interval=1``.
    """

    enabled: bool = True
    ranks: Union[RankSpec, Mapping[str, RankSpec]] = 12
    oversampling: int = 8
    core_sketch_factor: float = 2.0
    seed: int = 0
    time_block: int = 1
    time_rank: Optional[int] = None
    time_oversampling: Optional[int] = None
    segment_steps: Optional[int] = None
    compute_dtype: torch.dtype = torch.float32
    compress: Union[str, Sequence[str]] = "all"
    min_compression_ratio: float = 1.0
    measure_error: bool = False
    allow_tf32: bool = False
    recovery_device: str = "host"
    rhs: str = "stored"

    def __post_init__(self) -> None:
        for name in ("oversampling", "seed", "time_block"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.oversampling < 0:
            raise ValueError("oversampling must be >= 0.")
        if self.time_block < 1:
            raise ValueError("time_block must be >= 1.")
        for name in ("time_rank", "time_oversampling", "segment_steps"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"{name} must be an integer or None.")
        if self.time_rank is not None and not 1 <= self.time_rank <= self.time_block:
            raise ValueError("time_rank must be in [1, time_block].")
        if self.time_oversampling is not None and self.time_oversampling < 0:
            raise ValueError("time_oversampling must be >= 0.")
        if self.segment_steps is not None and self.segment_steps < 1:
            raise ValueError("segment_steps must be >= 1.")
        if not math.isfinite(self.core_sketch_factor) or self.core_sketch_factor < 1.0:
            raise ValueError("core_sketch_factor must be >= 1 (2.0 is recommended).")
        if self.compute_dtype not in (torch.float32, torch.float64):
            raise ValueError("compute_dtype must be torch.float32 or torch.float64.")
        if not math.isfinite(self.min_compression_ratio) or self.min_compression_ratio <= 0:
            raise ValueError("min_compression_ratio must be positive.")
        self.compressed_components(HISTORY_COMPONENTS)  # validates ``compress``
        if self.recovery_device not in ("host", "device"):
            raise ValueError("recovery_device must be 'host' or 'device'.")
        if self.rhs not in ("stored", "from_e"):
            raise ValueError("rhs must be 'stored' or 'from_e'.")
        if isinstance(self.ranks, Mapping):
            for key in self.ranks:
                if key not in (*HISTORY_COMPONENTS, "E", "R", "default"):
                    raise ValueError(
                        f"Unknown ranks key {key!r}; use a component name, 'E', 'R' or 'default'."
                    )

    @property
    def resolved_time_rank(self) -> int:
        """Time rank used for blocks (``time_rank`` or ``min(time_block, 4)``)."""
        return self.time_rank if self.time_rank is not None else min(self.time_block, 4)

    def ranks_for(self, component: str) -> RankSpec:
        """Spatial rank spec of one history component."""
        if not isinstance(self.ranks, Mapping):
            return self.ranks
        for key in (component, component[0], "default"):
            if key in self.ranks:
                return self.ranks[key]
        return 12

    def compressed_components(self, components: Sequence[str]) -> Tuple[str, ...]:
        """Subset of ``components`` that is Tucker-compressed."""
        spec = self.compress
        if isinstance(spec, str):
            if spec == "all":
                return tuple(components)
            if spec in ("E", "R"):
                return tuple(name for name in components if name[0] == spec)
            if spec in HISTORY_COMPONENTS:
                return tuple(name for name in components if name == spec)
            raise ValueError("compress must be 'all', 'E', 'R' or a sequence of component names.")
        names = tuple(spec)
        unknown = [name for name in names if name not in HISTORY_COMPONENTS]
        if unknown:
            raise ValueError(f"Unknown components in compress: {unknown}.")
        return tuple(name for name in components if name in names)


def component_seed(seed: int, name: str) -> int:
    """Deterministic, component-specific seed (independent maps per component)."""
    return (int(seed) * 1_000_003 + zlib.crc32(name.encode("ascii"))) % (2**63 - 1)


@dataclass
class SketchAccumulator:
    """Linear sketches of one batch of tensors (data never stored).

    Attributes:
        factor_sketches: ``{mode: (B, I_n, k_n)}`` for every compressed mode.
        core_sketch: ``(B, s_1, ..., s_M)`` (``s_n = I_n`` for kept modes).
        rows_filled: For a stream mode, how many leading rows were pushed.
    """

    factor_sketches: Dict[int, torch.Tensor]
    core_sketch: torch.Tensor
    rows_filled: int = 0

    def nbytes(self) -> int:
        """Bytes held by the sketches."""
        tensors = [*self.factor_sketches.values(), self.core_sketch]
        return sum(t.numel() * t.element_size() for t in tensors)


@dataclass
class TuckerBlock:
    """A batch of recovered Tucker approximations.

    Attributes:
        core: ``(B, r_1, ..., r_M)`` (kept modes have their full size).
        factors: ``{mode: (B, I_n, r_n)}``; kept modes are absent (identity).
        shape: Mode sizes ``(I_1, ..., I_M)``.
    """

    core: torch.Tensor
    factors: Dict[int, torch.Tensor] = field(default_factory=dict)
    shape: Tuple[int, ...] = ()

    allow_tf32: bool = False

    def reconstruct(self, rows: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Return ``(B, I_1, ..., I_M)``; ``rows=(a, b)`` restricts mode 0."""
        with matmul_precision(self.allow_tf32):
            return self._reconstruct(rows)

    def _reconstruct(self, rows: Optional[Tuple[int, int]]) -> torch.Tensor:
        core = self.core
        if rows is not None and 0 not in self.factors:
            core = core[:, rows[0] : rows[1]]
        factors = {}
        for mode, factor in self.factors.items():
            factors[mode] = factor[:, rows[0] : rows[1]] if (rows is not None and mode == 0) else factor
        # Cheapest growth first.
        order = sorted(factors, key=lambda mode: factors[mode].shape[1] / max(core.shape[mode + 1], 1))
        result = core
        for mode in order:
            result = batched_mode_dot(result, factors[mode], mode + 1)
        return result

    def nbytes(self) -> int:
        """Bytes of the core and the factors."""
        tensors = [self.core, *self.factors.values()]
        return sum(t.numel() * t.element_size() for t in tensors)


def _to_host(tensors: Sequence[torch.Tensor], target: Any = "cpu") -> List[torch.Tensor]:
    """Copy tensors to ``target`` float64 with one transfer (one synchronization)."""
    if not tensors:
        return []
    flat = torch.cat([t.reshape(-1) for t in tensors]).to(target, torch.float64)
    out, offset = [], 0
    for tensor in tensors:
        out.append(flat[offset : offset + tensor.numel()].view(tensor.shape))
        offset += tensor.numel()
    return out


def _to_device(tensors: Sequence[torch.Tensor], device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
    """Copy host tensors to ``device`` with one non-blocking transfer from pinned memory."""
    if not tensors:
        return []
    flat = torch.cat([t.reshape(-1) for t in tensors]).to(dtype)
    if device.type == "cuda" and flat.device.type == "cpu":
        flat = flat.pin_memory().to(device, non_blocking=True)
    else:
        flat = flat.to(device)
    out, offset = [], 0
    for tensor in tensors:
        out.append(flat[offset : offset + tensor.numel()].view(tensor.shape))
        offset += tensor.numel()
    return out


class StreamingTuckerSketch:
    """Random maps and the sketch / recover operators for one tensor shape.

    The object holds only the dimension-reduction maps (DRMs). Sketches are
    returned in :class:`SketchAccumulator` objects, so the same operator
    serves any number of independent tensors (batch entries).

    Args:
        shape: Mode sizes ``(I_1, ..., I_M)``.
        ranks: Target ranks (clipped to ``I_n``).
        oversampling: ``p`` per mode (an int or one value per mode).
        core_sketch_factor: ``s_n = min(I_n, ceil(factor * k_n) + 1)``.
        seed: Seed of the CPU generator (identical maps on every device).
        device: Device of the maps.
        dtype: Compute dtype.
        stream_mode: ``None`` or ``0`` (mode 0 arrives in chunks).
        allow_tf32: Allow TF32 matmuls (see :class:`TuckerSketchConfig`).
    """

    def __init__(
        self,
        shape: Sequence[int],
        ranks: Sequence[int],
        oversampling: Union[int, Sequence[int]] = 8,
        core_sketch_factor: float = 2.0,
        seed: int = 0,
        device: Any = None,
        dtype: torch.dtype = torch.float32,
        stream_mode: Optional[int] = None,
        allow_tf32: bool = False,
        recovery_device: str = "host",
    ) -> None:
        self.allow_tf32 = bool(allow_tf32)
        if recovery_device not in ("host", "device"):
            raise ValueError("recovery_device must be 'host' or 'device'.")
        self.recovery_device = recovery_device
        self.shape = tuple(int(size) for size in shape)
        if len(self.shape) < 1 or any(size < 1 for size in self.shape):
            raise ValueError(f"Invalid tensor shape {self.shape}.")
        if len(ranks) != len(self.shape):
            raise ValueError("ranks must have one entry per mode.")
        if stream_mode not in (None, 0):
            raise ValueError("stream_mode must be None or 0.")
        modes = len(self.shape)
        if isinstance(oversampling, int):
            oversampling = (oversampling,) * modes
        self.stream_mode = stream_mode
        self.dtype = dtype
        self.device = torch.device("cpu" if device is None else device)
        self.ranks = tuple(min(int(rank), size) for rank, size in zip(ranks, self.shape))
        self.kept = tuple(rank >= size for rank, size in zip(self.ranks, self.shape))
        self.sketch_sizes = tuple(
            size if kept else min(rank + int(extra), size)
            for size, rank, extra, kept in zip(self.shape, self.ranks, oversampling, self.kept)
        )
        self.core_sketch_sizes = tuple(
            size if kept else min(size, int(math.ceil(core_sketch_factor * k)) + 1)
            for size, k, kept in zip(self.shape, self.sketch_sizes, self.kept)
        )
        self.compressed_modes = tuple(mode for mode in range(modes) if not self.kept[mode])
        if len(self.compressed_modes) and modes < 2:
            raise ValueError("A one-mode tensor cannot be Tucker-compressed.")

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        self.factor_maps: Dict[int, Dict[int, torch.Tensor]] = {}
        self.core_maps: Dict[int, torch.Tensor] = {}
        for mode in self.compressed_modes:
            k = self.sketch_sizes[mode]
            self.factor_maps[mode] = {
                other: torch.randn((self.shape[other], k), generator=generator, dtype=dtype).to(
                    self.device
                )
                for other in range(modes)
                if other != mode
            }
            self.core_maps[mode] = torch.randn(
                (self.core_sketch_sizes[mode], self.shape[mode]), generator=generator, dtype=dtype
            ).to(self.device)
        # float64 copies for the small recovery factorizations.
        self.linalg_device = torch.device("cpu") if recovery_device == "host" else self.device
        self.core_maps_host = {
            mode: matrix.to(self.linalg_device, torch.float64) for mode, matrix in self.core_maps.items()
        }

    # -- bookkeeping ---------------------------------------------------------

    @property
    def is_raw(self) -> bool:
        """``True`` when every mode is kept (no compression)."""
        return not self.compressed_modes

    def map_nbytes(self) -> int:
        """Bytes of all random maps."""
        tensors = list(self.core_maps.values())
        for maps in self.factor_maps.values():
            tensors.extend(maps.values())
        return sum(t.numel() * t.element_size() for t in tensors)

    def stored_shape(self) -> Tuple[Tuple[int, ...], Dict[int, Tuple[int, int]]]:
        """Core shape and factor shapes of one recovered tensor (unbatched)."""
        core = tuple(size if kept else rank for size, rank, kept in zip(self.shape, self.ranks, self.kept))
        factors = {mode: (self.shape[mode], self.ranks[mode]) for mode in self.compressed_modes}
        return core, factors

    def stored_numel(self) -> int:
        """Values stored per recovered tensor (core + factors)."""
        core, factors = self.stored_shape()
        return math.prod(core) + sum(rows * cols for rows, cols in factors.values())

    def accumulator_numel(self) -> int:
        """Values held by one sketch accumulator."""
        factors = sum(self.shape[m] * self.sketch_sizes[m] for m in self.compressed_modes)
        return factors + math.prod(self.core_sketch_sizes)

    # -- sketching -----------------------------------------------------------

    def new_accumulator(self, batch: int) -> SketchAccumulator:
        """Zero sketches for ``batch`` independent tensors."""
        factor_sketches = {
            mode: torch.zeros(
                (batch, self.shape[mode], self.sketch_sizes[mode]), dtype=self.dtype, device=self.device
            )
            for mode in self.compressed_modes
        }
        core_sketch = torch.zeros((batch, *self.core_sketch_sizes), dtype=self.dtype, device=self.device)
        return SketchAccumulator(factor_sketches, core_sketch)

    def update(self, accumulator: SketchAccumulator, x: torch.Tensor, offset: int = 0) -> None:
        """Add the contribution of ``x`` to the sketches (one pass over ``x``).

        Args:
            accumulator: Sketches to update in place.
            x: ``(B, J_1, ..., J_M)``; ``J_n = I_n`` except the stream mode,
                whose ``J_0`` rows are rows ``offset ... offset + J_0 - 1``.
            offset: First stream-mode row of the chunk.
        """
        with matmul_precision(self.allow_tf32):
            self._update(accumulator, x, offset)

    def _update(self, accumulator: SketchAccumulator, x: torch.Tensor, offset: int) -> None:
        if x.dim() != len(self.shape) + 1:
            raise ValueError(
                f"Expected a ({len(self.shape) + 1})-D batch, got shape {tuple(x.shape)}."
            )
        rows = x.shape[1]
        expected = list(self.shape)
        if self.stream_mode == 0:
            if offset != accumulator.rows_filled:
                raise ValueError(
                    f"Stream rows must arrive in order: expected offset "
                    f"{accumulator.rows_filled}, got {offset}."
                )
            if offset + rows > self.shape[0]:
                raise ValueError("Stream chunk extends past the block length.")
            expected[0] = rows
        elif offset != 0:
            raise ValueError("offset is only valid for a stream mode.")
        if list(x.shape[1:]) != expected:
            raise ValueError(f"Chunk shape {tuple(x.shape[1:])} does not match {tuple(expected)}.")
        x = x.to(self.dtype)
        streamed = self.stream_mode == 0
        row_slice = slice(offset, offset + rows)

        def chunk_map(matrix: torch.Tensor, mode: int, transpose: bool) -> torch.Tensor:
            # Restrict maps of the stream mode to the rows of this chunk.
            if mode != 0 or not streamed:
                return matrix
            return matrix[:, row_slice] if transpose else matrix[row_slice]

        for mode in self.compressed_modes:
            maps = {other: chunk_map(m, other, False) for other, m in self.factor_maps[mode].items()}
            contribution = khatri_rao_contract(x, maps, keep=mode, batch=1)
            if streamed and mode == 0:
                accumulator.factor_sketches[0][:, row_slice] = contribution
            else:
                accumulator.factor_sketches[mode] += contribution

        # Core sketch: contract the modes with the largest reduction first.
        sketch = x
        order = sorted(
            range(len(self.shape)),
            key=lambda mode: (self.core_sketch_sizes[mode] / max(x.shape[mode + 1], 1)),
        )
        for mode in order:
            if self.kept[mode]:
                if streamed and mode == 0:
                    # Identity map: place the rows into their slots.
                    padded = sketch.new_zeros(
                        (*sketch.shape[:1], self.shape[0], *sketch.shape[2:])
                    )
                    padded[:, row_slice] = sketch
                    sketch = padded
                continue
            sketch = mode_dot(sketch, chunk_map(self.core_maps[mode], mode, True), mode + 1)
        accumulator.core_sketch += sketch
        if streamed:
            accumulator.rows_filled = offset + rows

    def sketch(self, x: torch.Tensor) -> SketchAccumulator:
        """Sketch a complete batch ``(B, I_1, ..., I_M)`` in one pass."""
        accumulator = self.new_accumulator(x.shape[0])
        self.update(accumulator, x)
        return accumulator

    # -- recovery ------------------------------------------------------------

    def recover(self, accumulator: SketchAccumulator, length: Optional[int] = None) -> TuckerBlock:
        """Recover the rank-truncated Tucker approximation from the sketches.

        Args:
            accumulator: Sketches of a complete batch.
            length: Actual stream-mode length (``< I_0`` for a partial last
                block); ``None`` means the full length.

        Returns:
            :class:`TuckerBlock` with the target ranks (fewer for a partial
            block shorter than the time rank).
        """
        return recover_many([(self, accumulator, length)])[0]

    def _checked_length(self, accumulator: SketchAccumulator, length: Optional[int]) -> Tuple[List[int], Optional[int]]:
        shape = list(self.shape)
        if self.stream_mode == 0:
            length = shape[0] if length is None else int(length)
            if not 1 <= length <= shape[0]:
                raise ValueError("length must be in [1, block length].")
            if accumulator.rows_filled < length:
                raise ValueError(
                    f"Only {accumulator.rows_filled} of {length} stream rows were pushed."
                )
            shape[0] = length
        elif length is not None:
            raise ValueError("length is only valid for a stream mode.")
        return shape, length


RecoveryRequest = Tuple[StreamingTuckerSketch, SketchAccumulator, Optional[int]]


def recover_many(requests: Sequence[RecoveryRequest]) -> List[TuckerBlock]:
    """Recover several sketches (any operators on one device) together.

    The small dense factorizations (QR of ``I_n x k`` sketches,
    pseudo-inverses of ``s_n x k`` matrices, ``k x k`` eigendecompositions)
    run on the host in float64; all tensor contractions stay on the device.
    Every request's host-bound matrices travel in one batched copy per
    direction, so the whole call synchronizes the device twice. (Batched
    ``torch.linalg`` calls on such tiny matrices are latency bound on a GPU
    and synchronize internally once or more per call.)

    Args:
        requests: ``(operator, accumulator, length)`` triples (see
            :meth:`StreamingTuckerSketch.recover`).

    Returns:
        One :class:`TuckerBlock` per request.
    """
    if not requests:
        return []
    device = requests[0][0].device
    linalg_device = requests[0][0].linalg_device
    if any(op.device != device or op.linalg_device != linalg_device for op, _, _ in requests):
        raise ValueError("recover_many needs operators with one device and one recovery device.")
    allow_tf32 = all(op.allow_tf32 for op, _, _ in requests)
    with matmul_precision(allow_tf32):
        items = []
        sketches: List[torch.Tensor] = []
        for operator, accumulator, length in requests:
            shape, length = operator._checked_length(accumulator, length)
            core = accumulator.core_sketch
            if operator.stream_mode == 0 and operator.kept[0]:
                core = core[:, :length]
            for mode in operator.compressed_modes:
                sketch = accumulator.factor_sketches[mode]
                if operator.stream_mode == 0 and mode == 0:
                    sketch = sketch[:, :length]
                sketches.append(sketch)
            items.append([operator, shape, length, core, {}])
        host_sketches = iter(_to_host(sketches, linalg_device))  # synchronization 1

        pseudo_inverses: List[torch.Tensor] = []
        for operator, _, length, _, bases in items:
            for mode in operator.compressed_modes:
                core_map = operator.core_maps_host[mode]
                if operator.stream_mode == 0 and mode == 0:
                    core_map = core_map[:, :length]
                q, _ = torch.linalg.qr(next(host_sketches))
                bases[mode] = q
                # (Φ_n Q_n)^† = R^{-1} P^T from the QR of Φ_n Q_n (full column rank).
                p, r = torch.linalg.qr(core_map @ q)
                pseudo_inverses.append(torch.linalg.solve_triangular(r, p.transpose(1, 2), upper=True))
        uploaded = iter(_to_device(pseudo_inverses, device, requests[0][0].dtype))

        # Core recovery on the device, then Grams for the HOSVD truncation.
        grams: List[torch.Tensor] = []
        for item in items:
            operator, core, bases = item[0], item[3], item[4]
            for mode in operator.compressed_modes:
                core = batched_mode_dot(core, next(uploaded).to(operator.dtype), mode + 1)
            truncate = [
                mode
                for mode in operator.compressed_modes
                if core.shape[mode + 1] > min(operator.ranks[mode], bases[mode].shape[2])
            ]
            for mode in truncate:
                unfolded = unfold(core, mode, batch=1)
                grams.append(unfolded @ unfolded.transpose(1, 2))
            item[3] = core
            item.append(truncate)
        host_grams = iter(_to_host(grams, linalg_device))  # synchronization 2

        projections: List[torch.Tensor] = []
        for operator, _, _, _, bases, truncate in items:
            for mode in truncate:
                rank = min(operator.ranks[mode], bases[mode].shape[2])
                _, vectors = torch.linalg.eigh(next(host_grams))
                vectors = vectors[..., -rank:].flip(-1)
                projections.append(vectors.transpose(1, 2))
                bases[mode] = bases[mode] @ vectors
        host_bases = [basis for item in items for basis in item[4].values()]
        uploaded = iter(_to_device([*projections, *host_bases], device, requests[0][0].dtype))
        projected = [next(uploaded) for _ in projections]
        blocks = []
        position = 0
        for operator, shape, _, core, bases, truncate in items:
            for mode in truncate:
                core = batched_mode_dot(core, projected[position].to(operator.dtype), mode + 1)
                position += 1
            device_bases = {mode: next(uploaded).to(operator.dtype) for mode in bases}
            blocks.append(TuckerBlock(core, device_bases, tuple(shape), operator.allow_tf32))
    return blocks


def resolve_spatial_ranks(spec: RankSpec, frame_shape: Sequence[int]) -> Tuple[int, ...]:
    """Expand a rank spec over the frame axes (singleton axes get rank 1).

    Args:
        spec: Int, or one value per frame axis (``len(frame_shape)``), or one
            value per non-singleton axis.
        frame_shape: Spatial frame shape, e.g. ``(hx, hy, hz)``.
    """
    sizes = tuple(int(size) for size in frame_shape)
    active = [axis for axis, size in enumerate(sizes) if size > 1]
    if isinstance(spec, bool):
        raise TypeError("ranks must be integers.")
    if isinstance(spec, int):
        values: List[int] = [spec if size > 1 else 1 for size in sizes]
    else:
        spec = tuple(spec)
        if len(spec) == len(sizes):
            values = [int(v) if size > 1 else 1 for v, size in zip(spec, sizes)]
        elif len(spec) == len(active):
            values = [1] * len(sizes)
            for axis, value in zip(active, spec):
                values[axis] = int(value)
        else:
            raise ValueError(
                f"ranks {spec!r} must have {len(sizes)} values (one per axis of the "
                f"{tuple(sizes)} frame) or {len(active)} (one per non-singleton axis)."
            )
    if any(isinstance(v, bool) or v < 1 for v in values):
        raise ValueError(f"ranks must be positive integers, got {values}.")
    return tuple(values)


__all__ = [
    "HISTORY_COMPONENTS",
    "RecoveryRequest",
    "SketchAccumulator",
    "StreamingTuckerSketch",
    "TuckerBlock",
    "TuckerSketchConfig",
    "component_seed",
    "recover_many",
    "resolve_spatial_ranks",
]

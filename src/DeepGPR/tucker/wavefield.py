"""Module B (storage): compressed FDTD history for the discrete adjoint.

:class:`SketchedWavefieldHistory` receives the saved frames of each history
component (``E^n`` and ``R^n`` of the gradient components) segment by
segment, compresses them with :class:`~DeepGPR.tucker.sketch.StreamingTuckerSketch`
and reconstructs any frame range on demand. Raw frames are only ever read
from the caller's segment buffer; nothing here keeps a full-resolution
history.

Storage is preallocated once per component, so ``nbytes()`` is the exact
size of the compressed history:

* Tucker component, ``time_block == 1``: per frame and shot a core
  ``(r_x, r_y[, r_z])`` and factors ``(I_n, r_n)``.
* Tucker component, ``time_block = T > 1``: per block of ``T`` frames and
  shot a core ``(r_t, r_x, r_y[, r_z])``, a time factor ``(T, r_t)`` and
  spatial factors shared by the block. A block may span several segments:
  its sketches stay open (linear streaming updates) until its last frame.
* Uncompressed component (not selected, all modes kept, or below
  ``min_compression_ratio``): raw frames of the history dtype.

Exact zeros: a Tucker reconstruction spreads small errors over the whole
frame. DeepGPR's material-gradient kernel relies on cells whose saved
``E^n`` and ``R^n`` are exactly zero (outer faces where a component is never
updated, PEC cells): there the adjoint value is discarded, because the
decoupled CPML-boundary cotangent can be arbitrarily large. Each compressed
component therefore records a *support mask* - cells that were nonzero in at
least one frame of any shot - and zeroes the reconstruction outside it. The
mask is stored as packed bits (``cells / 8`` bytes) and counted in the
storage size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ._tensor_ops import relative_error
from .sketch import (
    SketchAccumulator,
    StreamingTuckerSketch,
    TuckerBlock,
    TuckerSketchConfig,
    component_seed,
    recover_many,
    resolve_spatial_ranks,
)


_BIT_WEIGHTS = (1, 2, 4, 8, 16, 32, 64, 128)


def pack_bits(mask: torch.Tensor) -> torch.Tensor:
    """Pack a boolean tensor into ``uint8`` (8 cells per byte, little-endian bits)."""
    flat = mask.reshape(-1).to(torch.uint8)
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.cat([flat, flat.new_zeros(padding)])
    weights = torch.tensor(_BIT_WEIGHTS, dtype=torch.uint8, device=mask.device)
    return (flat.reshape(-1, 8) * weights).sum(dim=1, dtype=torch.uint8)


def unpack_bits(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Inverse of :func:`pack_bits`."""
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = (packed[:, None] >> shifts) & 1
    return bits.reshape(-1)[: math.prod(shape)].reshape(tuple(shape)).bool()


@dataclass
class ComponentErrorStats:
    """Reconstruction-error accumulators of one component.

    ``relative_l2`` is ``sqrt(sum ||W - Ŵ||^2 / sum ||W||^2)`` over the
    measured frames (NaN if none was measured); ``max_frame_relative`` the
    worst single frame (frames with ``||W|| = 0`` count as 0 if reconstructed
    exactly, else ``inf``). Blocks that span several segments cannot be
    measured (their raw frames are gone when the block completes); they are
    counted in ``unmeasured_frames``. Errors are measured before the support
    mask is applied (the support is only final after the last frame), so
    they bound the error of the frames the adjoint reads from above.
    """

    error_sq: float = 0.0
    reference_sq: float = 0.0
    max_frame_relative: float = 0.0
    frames: int = 0
    unmeasured_frames: int = 0

    @property
    def relative_l2(self) -> float:
        """Aggregate relative L2 error (0 for an all-zero exact history)."""
        if self.frames == 0:
            return float("nan")
        if self.reference_sq == 0.0:
            return 0.0 if self.error_sq == 0.0 else float("inf")
        return math.sqrt(self.error_sq / self.reference_sq)


@dataclass
class _PendingBlock:
    """Sketched block(s) of one component waiting for recovery."""

    store: "_ComponentStore"
    accumulator: SketchAccumulator
    length: Optional[int]
    first_block: int
    count: int
    reference: Optional[torch.Tensor] = None


class _ComponentStore:
    """Compressed (or raw) history of one component for all shots."""

    def __init__(
        self,
        name: str,
        frame_shape: Tuple[int, ...],
        shots: int,
        frames: int,
        config: TuckerSketchConfig,
        compress: bool,
        device: torch.device,
        history_dtype: torch.dtype,
    ) -> None:
        self.name = name
        self.frame_shape = frame_shape
        self.shots = shots
        self.frames = frames
        self.history_dtype = history_dtype
        self.device = device
        self.dtype = config.compute_dtype
        self.measure_error = config.measure_error
        self.errors = ComponentErrorStats()
        # Singleton axes (2D models) are not Tucker modes.
        self.active_axes = tuple(axis for axis, size in enumerate(frame_shape) if size > 1)
        spatial_shape = tuple(frame_shape[axis] for axis in self.active_axes)
        ranks = resolve_spatial_ranks(config.ranks_for(name), frame_shape)
        spatial_ranks = tuple(ranks[axis] for axis in self.active_axes)
        self.time_block = config.time_block if frames > 0 else 1
        self.time_block = min(self.time_block, max(frames, 1))
        self.operator: Optional[StreamingTuckerSketch] = None
        self.reason = "selected"
        tucker_modes = len(spatial_shape) + (1 if self.time_block > 1 else 0)
        if compress and tucker_modes >= 2:
            if self.time_block > 1:
                shape = (self.time_block, *spatial_shape)
                time_rank = min(config.resolved_time_rank, self.time_block)
                time_oversampling = (
                    config.oversampling if config.time_oversampling is None else config.time_oversampling
                )
                operator = StreamingTuckerSketch(
                    shape,
                    (time_rank, *spatial_ranks),
                    (time_oversampling,) + (config.oversampling,) * len(spatial_shape),
                    config.core_sketch_factor,
                    component_seed(config.seed, name),
                    device,
                    self.dtype,
                    stream_mode=0,
                    allow_tf32=config.allow_tf32,
                    recovery_device=config.recovery_device,
                )
            else:
                operator = StreamingTuckerSketch(
                    spatial_shape,
                    spatial_ranks,
                    config.oversampling,
                    config.core_sketch_factor,
                    component_seed(config.seed, name),
                    device,
                    self.dtype,
                    allow_tf32=config.allow_tf32,
                    recovery_device=config.recovery_device,
                )
            raw_bytes = (
                self.time_block
                * math.prod(spatial_shape)
                * torch.empty((), dtype=history_dtype).element_size()
            )
            tucker_bytes = operator.stored_numel() * torch.empty((), dtype=self.dtype).element_size()
            if operator.is_raw:
                self.reason = "ranks >= frame size"
            elif tucker_bytes * config.min_compression_ratio > raw_bytes:
                self.reason = "Tucker form not smaller than min_compression_ratio"
            else:
                self.operator = operator
        elif compress:
            self.reason = "fewer than two non-singleton modes"
        else:
            self.reason = "not selected"

        self.open_block: Optional[int] = None
        self.open_accumulator: Optional[SketchAccumulator] = None
        if self.operator is None:
            self.raw = torch.empty((frames, shots, *frame_shape), dtype=history_dtype, device=device)
            self.blocks = 0
            return
        self.raw = None
        self.support = torch.zeros(frame_shape, dtype=torch.bool, device=device)
        self.support_bits: Optional[torch.Tensor] = None
        self.blocks = -(-frames // self.time_block)
        core_shape, factor_shapes = self.operator.stored_shape()
        self.core = torch.zeros((self.blocks, shots, *core_shape), dtype=self.dtype, device=device)
        self.factors = {
            mode: torch.zeros((self.blocks, shots, *factor_shape), dtype=self.dtype, device=device)
            for mode, factor_shape in factor_shapes.items()
        }

    # -- accounting ----------------------------------------------------------

    @property
    def compressed(self) -> bool:
        """Whether this component is stored in Tucker form."""
        return self.operator is not None

    def storage_nbytes(self) -> int:
        """Bytes of the stored history of this component."""
        if self.raw is not None:
            return self.raw.numel() * self.raw.element_size()
        tensors = [self.core, *self.factors.values()]
        mask_bytes = (
            self.support_bits.numel()
            if self.support_bits is not None
            else -(-math.prod(self.frame_shape) // 8)
        )
        return sum(t.numel() * t.element_size() for t in tensors) + mask_bytes

    def raw_nbytes(self) -> int:
        """Bytes the uncompressed history of this component would take."""
        element = torch.empty((), dtype=self.history_dtype).element_size()
        return self.frames * self.shots * math.prod(self.frame_shape) * element

    def map_nbytes(self) -> int:
        """Bytes of the random maps."""
        return 0 if self.operator is None else self.operator.map_nbytes()

    def accumulator_nbytes(self) -> int:
        """Bytes of the currently open streaming sketch."""
        return 0 if self.open_accumulator is None else self.open_accumulator.nbytes()

    # -- helpers -------------------------------------------------------------

    def _block_length(self, block: int) -> int:
        return min(self.time_block, self.frames - block * self.time_block)

    def _to_tucker(self, frames: torch.Tensor) -> torch.Tensor:
        """``(F, S, *frame)`` -> ``(F, S, *active)`` (drop singleton axes)."""
        return frames.reshape(*frames.shape[:2], *(self.frame_shape[a] for a in self.active_axes))

    def _store(self, first_block: int, block: TuckerBlock, count: int) -> None:
        """Write ``count`` recovered blocks (batch = count * shots)."""
        core = block.core.reshape(count, self.shots, *block.core.shape[1:])
        target = self.core[first_block : first_block + count]
        slices = tuple(slice(0, size) for size in core.shape[2:])
        if tuple(core.shape[2:]) != tuple(target.shape[2:]):
            target.zero_()
        target[(slice(None), slice(None), *slices)] = core
        for mode, factor in block.factors.items():
            factor = factor.reshape(count, self.shots, *factor.shape[1:])
            destination = self.factors[mode][first_block : first_block + count]
            if tuple(factor.shape[2:]) != tuple(destination.shape[2:]):
                destination.zero_()
            destination[:, :, : factor.shape[2], : factor.shape[3]] = factor

    def _load(self, first_block: int, count: int) -> TuckerBlock:
        assert self.operator is not None
        core = self.core[first_block : first_block + count]
        factors = {
            mode: tensor[first_block : first_block + count].reshape(
                count * self.shots, *tensor.shape[2:]
            )
            for mode, tensor in self.factors.items()
        }
        return TuckerBlock(
            core.reshape(count * self.shots, *core.shape[2:]),
            factors,
            self.operator.shape,
            self.operator.allow_tf32,
        )

    def _measure(self, reference: torch.Tensor, estimate: torch.Tensor) -> None:
        """Accumulate errors; tensors are ``(batch, ...)`` frames or blocks."""
        reference = reference.to(torch.float64)
        estimate = estimate.to(torch.float64)
        difference = (estimate - reference).reshape(reference.shape[0], -1)
        reference_flat = reference.reshape(reference.shape[0], -1)
        error_sq = difference.square().sum(dim=1)
        reference_sq = reference_flat.square().sum(dim=1)
        self.errors.error_sq += float(error_sq.sum())
        self.errors.reference_sq += float(reference_sq.sum())
        ratio = torch.where(
            reference_sq > 0,
            (error_sq / reference_sq.clamp_min(1.0e-300)).sqrt(),
            torch.where(error_sq > 0, torch.full_like(error_sq, float("inf")), torch.zeros_like(error_sq)),
        )
        self.errors.max_frame_relative = max(self.errors.max_frame_relative, float(ratio.max()))
        self.errors.frames += reference.shape[0]

    # -- push / reconstruct --------------------------------------------------

    def push(self, frames: torch.Tensor, start: int) -> List[_PendingBlock]:
        """Consume frames ``start ... start + F - 1`` (``(F, S, *frame)``).

        Returns the blocks completed by these frames; the caller recovers
        them (:func:`recover_many`) and passes each to :meth:`complete`
        before the frames' buffer is reused.
        """
        count = frames.shape[0]
        pending: List[_PendingBlock] = []
        if self.raw is not None:
            self.raw[start : start + count].copy_(frames)
            if self.measure_error:  # an exact copy
                self.errors.reference_sq += float(frames.double().square().sum())
                self.errors.frames += count * self.shots
            return pending
        assert self.operator is not None
        # Running support (reductions only: no full-size temporaries).
        self.support |= (torch.amax(frames, dim=(0, 1)) != 0) | (torch.amin(frames, dim=(0, 1)) != 0)
        data = self._to_tucker(frames)
        block_size = self.time_block
        index = 0
        while index < count:
            global_frame = start + index
            block, offset = divmod(global_frame, block_size)
            remaining = count - index
            if self.open_block is None and offset == 0 and remaining >= block_size:
                # Whole blocks inside this chunk: sketch them as one batch.
                full = remaining // block_size
                # The last block may be shorter than time_block; whole-batch
                # processing needs equal lengths.
                while full and self._block_length(block + full - 1) != block_size:
                    full -= 1
                if full:
                    chunk = data[index : index + full * block_size]
                    if block_size == 1:
                        batch = chunk.reshape(full * self.shots, *chunk.shape[2:])
                    else:
                        batch = chunk.reshape(full, block_size, self.shots, *chunk.shape[2:])
                        batch = batch.transpose(1, 2).reshape(
                            full * self.shots, block_size, *chunk.shape[2:]
                        )
                    pending.append(
                        _PendingBlock(
                            self, self.operator.sketch(batch), None, block, full,
                            batch if self.measure_error else None,
                        )
                    )
                    index += full * block_size
                    continue
            # Streaming path: (part of) one block, sketches stay open.
            length = self._block_length(block)
            piece = min(length - offset, remaining)
            if self.open_block is None:
                if offset != 0:
                    raise RuntimeError(f"{self.name}: frames must be pushed in order.")
                self.open_block = block
                self.open_accumulator = self.operator.new_accumulator(self.shots)
            elif self.open_block != block:
                raise RuntimeError(f"{self.name}: block {self.open_block} was left incomplete.")
            chunk = data[index : index + piece].transpose(0, 1)
            assert self.open_accumulator is not None
            self.operator.update(self.open_accumulator, chunk, offset)
            if offset + piece == length:
                # The whole block is still in this chunk only if offset == 0.
                reference = chunk if self.measure_error and offset == 0 else None
                if self.measure_error and offset != 0:
                    self.errors.unmeasured_frames += length
                pending.append(
                    _PendingBlock(self, self.open_accumulator, length, block, 1, reference)
                )
                self.open_block = None
                self.open_accumulator = None
            index += piece
        if start + count == self.frames:
            self.support_bits = pack_bits(self.support)
            self.support = None  # type: ignore[assignment]
        return pending

    def complete(self, pending: _PendingBlock, recovered: TuckerBlock) -> None:
        """Store a recovered block (and measure its error if requested)."""
        self._store(pending.first_block, recovered, pending.count)
        if pending.reference is not None:
            self._measure(pending.reference, recovered.reconstruct())

    def reconstruct_into(self, out: torch.Tensor, start: int) -> None:
        """Write frames ``start ... start + F - 1`` into ``out`` ``(F, S, *frame)``."""
        count = out.shape[0]
        if self.raw is not None:
            out.copy_(self.raw[start : start + count])
            return
        if self.open_block is not None or self.support_bits is None:
            raise RuntimeError(f"{self.name}: the history is incomplete.")
        block_size = self.time_block
        target = self._to_tucker(out)
        if block_size == 1:
            values = self._load(start, count).reconstruct()
            target.copy_(values.reshape(count, self.shots, *values.shape[1:]))
        else:
            index = 0
            while index < count:
                block, offset = divmod(start + index, block_size)
                piece = min(self._block_length(block) - offset, count - index)
                values = self._load(block, 1).reconstruct(rows=(offset, offset + piece))
                target[index : index + piece].copy_(values.transpose(0, 1))
                index += piece
        out.masked_fill_(~unpack_bits(self.support_bits, self.frame_shape), 0)


@dataclass
class HistorySizeReport:
    """Measured byte counts of a sketched history.

    Attributes:
        raw_bytes: Uncompressed history of the stored components.
        stored_bytes: Compressed (and fallback raw) history actually kept.
        map_bytes: Random maps.
        open_sketch_bytes: Currently open streaming sketches.
        components: Per-component ``(stored_bytes, raw_bytes, compressed, reason)``.
    """

    raw_bytes: int
    stored_bytes: int
    map_bytes: int
    open_sketch_bytes: int
    components: Dict[str, Tuple[int, int, bool, str]] = field(default_factory=dict)

    @property
    def storage_ratio(self) -> float:
        """``raw_bytes / stored_bytes`` (history payload only)."""
        return self.raw_bytes / self.stored_bytes if self.stored_bytes else float("inf")


class SketchedWavefieldHistory:
    """Streaming Tucker-compressed history of several frame components.

    Args:
        components: Component names (e.g. ``("Ex", "Ey", "Ez", "Rx", "Ry", "Rz")``).
        frame_shape: Spatial shape of one frame per shot (e.g. ``(hx, hy, hz)``).
        shots: Number of shots (batch of independent frames per step).
        frames: Total number of saved frames.
        config: Compression options.
        device: Storage device.
        history_dtype: dtype of raw frames (float32 for DeepGPR).
    """

    def __init__(
        self,
        components: Sequence[str],
        frame_shape: Sequence[int],
        shots: int,
        frames: int,
        config: TuckerSketchConfig,
        device: torch.device,
        history_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.frame_shape = tuple(int(size) for size in frame_shape)
        self.shots = int(shots)
        self.frames = int(frames)
        selected = set(config.compressed_components(components))
        self.stores: Dict[str, _ComponentStore] = {
            name: _ComponentStore(
                name, self.frame_shape, self.shots, self.frames, config, name in selected,
                torch.device(device), history_dtype,
            )
            for name in components
        }
        self.frames_pushed = 0

    def push(self, frames: Dict[str, torch.Tensor], start: int) -> None:
        """Consume saved frames ``start ...`` of every component.

        Args:
            frames: ``{name: (F, S, *frame_shape)}`` views of the segment buffer.
            start: Global index of the first frame.
        """
        if start != self.frames_pushed:
            raise RuntimeError(f"Expected frame {self.frames_pushed}, got {start}.")
        counts = {tensor.shape[0] for tensor in frames.values()}
        if len(counts) != 1 or set(frames) != set(self.stores):
            raise ValueError("push needs the same number of frames for every component.")
        count = counts.pop()
        if start + count > self.frames:
            raise ValueError("More frames pushed than the history holds.")
        pending = [block for name, tensor in frames.items() for block in self.stores[name].push(tensor, start)]
        # All components' blocks are recovered together: two device-host
        # synchronizations per push instead of two per component.
        recovered = recover_many(
            [(block.store.operator, block.accumulator, block.length) for block in pending]  # type: ignore[misc]
        )
        for block, result in zip(pending, recovered):
            block.store.complete(block, result)
        self.frames_pushed += count

    def reconstruct_into(self, out: Dict[str, torch.Tensor], start: int) -> None:
        """Write frames ``start ...`` of every component into ``out`` views."""
        if self.frames_pushed != self.frames:
            raise RuntimeError(
                f"Only {self.frames_pushed} of {self.frames} frames were pushed."
            )
        for name, tensor in out.items():
            self.stores[name].reconstruct_into(tensor, start)

    def reconstruct(self, name: str, start: int = 0, count: Optional[int] = None) -> torch.Tensor:
        """Return frames of one component as a new ``(F, S, *frame)`` tensor."""
        count = self.frames - start if count is None else count
        store = self.stores[name]
        out = torch.empty(
            (count, self.shots, *self.frame_shape), dtype=store.history_dtype, device=store.device
        )
        store.reconstruct_into(out, start)
        return out

    def size_report(self) -> HistorySizeReport:
        """Measured storage of the history (see :class:`HistorySizeReport`)."""
        components = {
            name: (store.storage_nbytes(), store.raw_nbytes(), store.compressed, store.reason)
            for name, store in self.stores.items()
        }
        return HistorySizeReport(
            raw_bytes=sum(store.raw_nbytes() for store in self.stores.values()),
            stored_bytes=sum(store.storage_nbytes() for store in self.stores.values()),
            map_bytes=sum(store.map_nbytes() for store in self.stores.values()),
            open_sketch_bytes=sum(store.accumulator_nbytes() for store in self.stores.values()),
            components=components,
        )

    def error_report(self) -> Dict[str, ComponentErrorStats]:
        """Per-component reconstruction errors (``measure_error=True`` only)."""
        return {name: store.errors for name, store in self.stores.items()}


def history_relative_error(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """``||W - Ŵ|| / ||W||`` with the zero-reference rule of :func:`relative_error`."""
    return relative_error(estimate, reference)


__all__ = [
    "ComponentErrorStats",
    "pack_bits",
    "unpack_bits",
    "HistorySizeReport",
    "SketchedWavefieldHistory",
    "history_relative_error",
]

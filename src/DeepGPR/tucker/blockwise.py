"""Block-wise Tucker history with adaptive ranks and adjoint shadow zones.

A Tucker approximation of a whole frame fails once the wavefield has spread over the domain: many curved
wavefronts crossing the grid are not of low multilinear rank, so a fixed global rank leaves late frames (deep
reflections) almost unrecovered. Over a few wavelengths the field is close to a few plane waves, which is.
:class:`BlockTuckerHistory` therefore splits every frame of every history component into blocks of at most
32 cells per axis and compresses each block with a sequentially truncated HOSVD whose ranks are chosen per
block and per mode from a relative error tolerance: the discarded mode-``n`` Gram eigenvalues of block ``b``
sum to at most ``tolerance**2 * ||X_b||**2 / 3``, so ``||X_b - X̂_b|| <= tolerance * ||X_b||``. Block sides
<= 32 keep all mode eigenproblems of a frame inside cuSOLVER's batched Jacobi solver (one call per mode).

Blocks are not stored at all (reconstructed as zeros) when

* the frame is later than the block's ``keep_until`` frame. With the adjoint run backward from the end of the
  record, the adjoint field at a cell is still exactly zero after ``T - tau_r(x)`` (``tau_r``: first-arrival
  time from the receivers), so forward values there never enter the material gradient - the "shadow zones" of
  Boehm et al. (2016, Geophysics 81(6), R385). Cells whose parameters are not inverted (e.g. fixed air) can be
  marked ``-1`` (never needed). :func:`surface_adjoint_keep_until` gives a rigorous bound for receivers on one
  horizontal row;
* the block norm is below ``floor`` times the largest norm among the needed blocks of that frame (cells the
  forward front has not reached).

Only the time-local, per-block structure differs from :class:`~DeepGPR.tucker.wavefield.SketchedWavefieldHistory`;
the solver interface (``push``, ``reconstruct_into``, ``size_report``, ``error_report``) is the same, so
``DeepGPR.compute(..., wavefield_sketch=BlockTuckerConfig(...))`` uses it in place of the global sketch.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ._tensor_ops import matmul_precision
from .sketch import TuckerSketchConfig
from .wavefield import ComponentErrorStats, HistorySizeReport

C0 = 299_792_458.0
MAX_BLOCK = 32  # cuSOLVER's batched Jacobi eigensolver (syevjBatched) handles matrices up to 32 x 32
PAYLOAD_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class BlockTuckerConfig(TuckerSketchConfig):
    """Options of the block-wise Tucker history (``ranks``/``oversampling`` of the parent are unused).

    Attributes:
        block: Block size per frame axis (each <= 32).
        origin: Cell index per axis where the block grid starts (cells before it form a partial first block),
            e.g. the first ground row so that no block straddles the surface.
        tolerance: Relative error bound per block (ST-HOSVD truncation).
        floor: Blocks with ``||X_b|| < floor * max ||X_b'||`` (needed blocks of the same frame) are dropped.
        max_rank: Optional cap of every mode rank.
        keep_until: Tensor broadcastable to the frame shape (each axis 1 or full): last frame index whose
            value the adjoint needs at that cell; ``-1`` = never. ``None`` keeps every block. With
            ``rhs="from_e"`` one more frame is kept (``R^n`` is rebuilt from ``E^(n+1)``).
        storage_device: Where the compressed history lives (``"cpu"`` or a CUDA device).
        chunk_blocks: Blocks factorized together (bounds the temporary memory).
        tight_regions: Boxes ``(lo0, hi0, lo1, hi1, lo2, hi2, tolerance)`` (frame cells, half-open) whose
            overlapping blocks use the smaller of ``tolerance`` and the region's (``0`` = keep every nonzero
            singular value). Meant for the blocks around a source: they hold the near-singular direct wave,
            so a tolerance relative to their norm leaves large absolute errors where the gradient is strong.
        host_limit_bytes: With ``storage_device="cpu"`` and ``spill_dir``: payloads beyond this many bytes of
            host memory go to an unlinked file in ``spill_dir`` (read back on demand; the file disappears with
            the history). ``None`` keeps everything in host memory.
        spill_dir: Folder of the spill file (a local disk).
        payload_dtype: Stored type of factors and cores: ``"float32"``, ``"float16"`` or ``"bfloat16"``. In half
            precision each block's core is divided by its largest magnitude (one float32 scale per block) and the
            factors are stored as they are (orthonormal columns, entries in [-1, 1]).
        device_chunk_bytes: With a CUDA ``storage_device``: the records (payload, block ids, ranks, scales) are
            packed into device buffers of this many bytes, allocated as needed, instead of one allocation each.
            Each frame's few-MB payload would otherwise be carved out of a cached segment of the compression
            temporaries and pin it, so the reserved memory grows by about one temporary segment per frame
            (70.9 M cells, tolerance 0.1: 23 GiB reserved at 13 GiB allocated). Use at least
            ``max_split_size_mb`` of ``PYTORCH_CUDA_ALLOC_CONF`` so that every buffer is a segment of its own.
    """

    block: Tuple[int, int, int] = (32, 32, 32)
    origin: Tuple[int, int, int] = (0, 0, 0)
    tolerance: float = 0.01
    floor: float = 1.0e-4
    max_rank: Optional[int] = None
    keep_until: Optional[torch.Tensor] = field(default=None, compare=False, repr=False)
    storage_device: str = "cpu"
    chunk_blocks: int = 8192
    tight_regions: Tuple[Tuple[float, ...], ...] = ()
    host_limit_bytes: Optional[int] = None
    spill_dir: Optional[str] = None
    payload_dtype: str = "float32"
    device_chunk_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.time_block != 1:
            raise ValueError("BlockTuckerConfig compresses every frame on its own (time_block=1).")
        if len(self.block) != 3 or any(int(b) < 1 or int(b) > MAX_BLOCK for b in self.block):
            raise ValueError(f"block needs three sides in [1, {MAX_BLOCK}].")
        if len(self.origin) != 3 or any(int(o) < 0 for o in self.origin):
            raise ValueError("origin needs three non-negative cell indices.")
        if not 0.0 < self.tolerance < 1.0:
            raise ValueError("tolerance must be in (0, 1).")
        if not self.floor >= 0.0:
            raise ValueError("floor must be >= 0.")
        if self.host_limit_bytes is not None and self.host_limit_bytes < 0:
            raise ValueError("host_limit_bytes must be >= 0.")
        if self.max_rank is not None and self.max_rank < 1:
            raise ValueError("max_rank must be >= 1.")
        if self.payload_dtype not in PAYLOAD_DTYPES:
            raise ValueError(f"payload_dtype must be one of {sorted(PAYLOAD_DTYPES)}.")
        if self.device_chunk_bytes is not None and self.device_chunk_bytes < 1:
            raise ValueError("device_chunk_bytes must be >= 1.")
        for region in self.tight_regions:
            if len(region) != 7 or not 0.0 <= float(region[6]) < 1.0:
                raise ValueError("tight_regions entries are (lo0, hi0, lo1, hi1, lo2, hi2, tolerance in [0, 1)).")


@dataclass
class _Record:
    """Compressed blocks of one frame of one component and shot."""

    ids: torch.Tensor  # (na,) int32 block indices
    ranks: torch.Tensor  # (na, 3) int16
    payload: Optional[torch.Tensor]  # per chunk all U0, all U1, all U2, all cores (row-major, rank-trimmed)
    chunks: Tuple[int, ...] = ()  # blocks per chunk, in order (empty: chunks of config.chunk_blocks)
    spilled: Tuple[int, int] = (-1, 0)  # (byte offset in the spill file, element count) when payload is None
    scales: Optional[torch.Tensor] = None  # (na,) float32 core scales (half-precision payloads)
    itemsize: int = 4  # bytes per payload element

    def nbytes(self) -> int:
        payload = (self.payload.numel() if self.payload is not None else self.spilled[1]) * self.itemsize
        extra = [t for t in (self.ids, self.ranks, self.scales) if t is not None]
        return payload + sum(t.numel() * t.element_size() for t in extra)


@dataclass
class BlockFrameStats:
    """Per-frame block counts of one component (summed over shots)."""

    blocks: int = 0  # blocks with nonzero values
    stored: int = 0
    shadow: int = 0  # nonzero blocks skipped by keep_until
    floor: int = 0  # needed nonzero blocks below the floor
    payload_bytes: int = 0
    mean_rank: float = 0.0  # mean of (r0 r1 r2)^(1/3) over stored blocks


def surface_adjoint_keep_until(
    eps_r: torch.Tensor,
    receiver_row: int,
    dx: float,
    dt: float,
    record_steps: int,
    pml: int,
    margin: float = 2.0e-9,
    fixed_above: bool = True,
) -> torch.Tensor:
    """``keep_until`` (``(1, ny + 2 pml, 1)``, extended frame) for receivers on the row ``receiver_row``.

    Any path from the receiver row to row ``j`` crosses every row in between, so the receiver-to-cell
    traveltime is at least ``tau(j) = sum_k dx * min_{x,z} sqrt(eps_r[:, k, :]) / c`` over the rows strictly
    between them - a lower bound valid for any lateral receiver position and any model (no ray tracing). The
    adjoint source lasts until the end of the record ``T = record_steps * dt``; read backward, the adjoint field
    is zero at row ``j`` after ``T - tau(j)``. Frames up to ``(T - tau(j) + margin) / dt`` are kept. Rows above
    the receiver row get ``-1`` when ``fixed_above`` (their parameters are not inverted, e.g. air), the bottom
    CPML rows continue the bound with the last row's slowness (materials are replicated into the CPML).

    Args:
        eps_r: ``(nx, ny, nz)`` relative permittivity of the current model (no CPML).
        receiver_row: Row (second axis, no CPML) of the receivers.
        dx, dt: Cell size (m) and time step (s).
        record_steps: Number of time steps of the record (the history has ``record_steps + 1`` E frames).
        pml: CPML thickness in cells.
        margin: Safety margin (s) added to the cutoff time.
        fixed_above: Mark the rows above the receiver row (and the top CPML) as never needed.
    """
    slowness = torch.sqrt(eps_r.detach().double().amin(dim=(0, 2))) / C0  # min slowness per row (s/m)
    ny = slowness.numel()
    rows = torch.arange(-pml, ny + pml)
    clipped = rows.clamp(0, ny - 1)
    s = slowness[clipped]
    cum = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(s * dx, 0)])  # cum[i] = sum s[:i] * dx
    r = receiver_row + pml  # index in the extended rows
    tau = torch.zeros(rows.numel(), dtype=torch.float64)
    below = torch.arange(rows.numel()) > r
    idx = torch.arange(rows.numel())
    tau[below] = (cum[idx[below]] - cum[r + 1]).clamp_min(0.0)  # rows strictly between r and j
    above = idx < r
    tau[above] = (cum[r] - cum[idx[above] + 1]).clamp_min(0.0)
    last = torch.floor((record_steps * dt - tau + margin) / dt).clamp(-1, record_steps).to(torch.int32)
    if fixed_above:
        last[above] = -1
    return last.reshape(1, -1, 1)


class BlockTuckerHistory:
    """Block-wise, rank-adaptive Tucker history (drop-in for ``SketchedWavefieldHistory``).

    Args:
        components: Component names.
        frame_shape: ``(hx, hy, hz)`` cells of one frame.
        shots: Shots per frame.
        frames: Frames that will be pushed.
        config: :class:`BlockTuckerConfig`.
        device: Device of the pushed frames (compression runs there).
        history_dtype: dtype of the raw frames (float32).
    """

    def __init__(
        self,
        components: Sequence[str],
        frame_shape: Sequence[int],
        shots: int,
        frames: int,
        config: BlockTuckerConfig,
        device: torch.device,
        history_dtype: torch.dtype = torch.float32,
    ) -> None:
        if len(frame_shape) != 3:
            raise ValueError("BlockTuckerHistory needs 3D frames (hx, hy, hz).")
        self.config = config
        self.components = tuple(components)
        self.frame_shape = tuple(int(s) for s in frame_shape)
        self.shots = int(shots)
        self.frames = int(frames)
        self.device = torch.device(device)
        self.history_dtype = history_dtype
        self.storage_device = torch.device(config.storage_device)
        self.block = tuple(min(int(b), s) for b, s in zip(config.block, self.frame_shape))
        self.pad_before = tuple((b - int(o) % b) % b for b, o in zip(self.block, config.origin))
        self.padded = tuple(-(-(p + s) // b) * b for p, s, b in zip(self.pad_before, self.frame_shape, self.block))
        self.nblocks = tuple(p // b for p, b in zip(self.padded, self.block))
        self.crop = tuple(slice(p, p + s) for p, s in zip(self.pad_before, self.frame_shape))
        self.last_needed = self._block_keep_until(config.keep_until)
        self.block_tolerance = self._block_tolerance(config)
        if config.rhs == "from_e":
            # R^n = (E^(n+1) - ca E^n) / cb is rebuilt from the next frame: keep one frame past the cutoff.
            self.last_needed = torch.where(self.last_needed >= 0, self.last_needed + 1, self.last_needed)
        self.records: Dict[str, List[Optional[_Record]]] = {n: [None] * (self.frames * self.shots) for n in self.components}
        self.frame_stats: Dict[str, List[BlockFrameStats]] = {n: [BlockFrameStats() for _ in range(self.frames)] for n in self.components}
        self.errors: Dict[str, ComponentErrorStats] = {n: ComponentErrorStats() for n in self.components}
        self._buffer = torch.zeros(self.padded, device=self.device, dtype=torch.float32)
        self.frames_pushed = 0
        self.host_bytes = 0  # payload bytes held in host memory
        self.spilled_bytes = 0  # payload bytes in the spill file
        self._spill_fd: Optional[int] = None
        self.payload_dtype = PAYLOAD_DTYPES[config.payload_dtype]
        self.itemsize = torch.empty(0, dtype=self.payload_dtype).element_size()
        self._chunks: List[torch.Tensor] = []  # device_chunk_bytes: uint8 buffers the records are packed into
        self._chunk_used = 0  # bytes used in the last buffer

    def __del__(self) -> None:
        if getattr(self, "_spill_fd", None) is not None:
            os.close(self._spill_fd)
            self._spill_fd = None

    @property
    def device_chunk_reserved_bytes(self) -> int:
        """Bytes of the device buffers of ``device_chunk_bytes`` (used + unused tail of the last one)."""
        return sum(c.numel() for c in self._chunks)

    def _to_storage(self, t: torch.Tensor) -> torch.Tensor:
        """``t`` on the storage device; packed into a device buffer with ``device_chunk_bytes``."""
        size = self.config.device_chunk_bytes
        if size is None or self.storage_device.type != "cuda" or t.numel() == 0:
            return t.to(self.storage_device)
        t = t.contiguous()
        nbytes = t.numel() * t.element_size()
        offset = -(-self._chunk_used // 256) * 256  # aligned for every dtype view
        if not self._chunks or offset + nbytes > self._chunks[-1].numel():
            self._chunks.append(torch.empty(max(int(size), nbytes), dtype=torch.uint8, device=self.storage_device))
            offset = 0
        view = self._chunks[-1][offset : offset + nbytes].view(t.dtype).view(t.shape)
        view.copy_(t)
        self._chunk_used = offset + nbytes
        return view

    def _store_payload(self, payload: torch.Tensor) -> Tuple[Optional[torch.Tensor], Tuple[int, int]]:
        """Payload to the storage device, or to the spill file once ``host_limit_bytes`` is reached."""
        cfg = self.config
        nbytes = payload.numel() * payload.element_size()
        if (cfg.host_limit_bytes is None or cfg.spill_dir is None or self.storage_device.type != "cpu"
                or self.host_bytes + nbytes <= cfg.host_limit_bytes):
            self.host_bytes += nbytes
            return self._to_storage(payload), (-1, 0)
        if self._spill_fd is None:
            fd, path = tempfile.mkstemp(prefix="block_tucker_", suffix=".spill", dir=cfg.spill_dir)
            os.unlink(path)  # freed when the descriptor is closed
            self._spill_fd = fd
        data = payload.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
        offset = self.spilled_bytes
        view = memoryview(data)
        while view:
            view = view[os.pwrite(self._spill_fd, view, offset + len(data) - len(view)):]
        self.spilled_bytes += nbytes
        return None, (offset, payload.numel())

    def _load_payload(self, record: _Record) -> torch.Tensor:
        if record.payload is not None:
            return record.payload.to(self.device)
        offset, count = record.spilled
        data = bytearray(count * record.itemsize)
        view, done = memoryview(data), 0
        while done < len(data):
            chunk = os.preadv(self._spill_fd, [view[done:]], offset + done)
            if chunk <= 0:
                raise IOError("short read from the block Tucker spill file")
            done += chunk
        return torch.frombuffer(data, dtype=self.payload_dtype).to(self.device)

    # -- geometry --------------------------------------------------------------------------------------------

    def _block_keep_until(self, keep: Optional[torch.Tensor]) -> torch.Tensor:
        """Per-block maximum of ``keep_until`` (``-1`` for cells outside the frame)."""
        total = math.prod(self.nblocks)
        if keep is None:
            return torch.full((total,), self.frames, dtype=torch.int64, device=self.device)
        k = torch.as_tensor(keep).detach().to("cpu", torch.int32)
        if k.dim() != 3 or any(a not in (1, s) for a, s in zip(k.shape, self.frame_shape)):
            raise ValueError(f"keep_until of shape {tuple(k.shape)} does not broadcast to {self.frame_shape}.")
        for axis in range(3):
            if k.shape[axis] == 1:
                continue
            shape = list(k.shape)
            before, after = self.pad_before[axis], self.padded[axis] - self.pad_before[axis] - self.frame_shape[axis]
            parts = []
            if before:
                parts.append(torch.full(shape[:axis] + [before] + shape[axis + 1 :], -1, dtype=torch.int32))
            parts.append(k)
            if after:
                parts.append(torch.full(shape[:axis] + [after] + shape[axis + 1 :], -1, dtype=torch.int32))
            k = torch.cat(parts, dim=axis).unflatten(axis, (self.nblocks[axis], self.block[axis])).amax(dim=axis + 1)
        return k.expand(self.nblocks).reshape(-1).to(self.device, torch.int64)

    def _block_tolerance(self, config: BlockTuckerConfig) -> torch.Tensor:
        """Per-block tolerance: ``config.tolerance``, lowered in the blocks overlapping ``tight_regions``."""
        tol = torch.full(self.nblocks, float(config.tolerance), dtype=torch.float64)
        for region in config.tight_regions:
            index = []
            for axis in range(3):
                lo = max(int(region[2 * axis]), 0)
                hi = min(int(region[2 * axis + 1]), self.frame_shape[axis])
                if hi <= lo:
                    break
                b = self.block[axis]
                index.append(slice((lo + self.pad_before[axis]) // b, (hi - 1 + self.pad_before[axis]) // b + 1))
            else:
                tol[tuple(index)] = tol[tuple(index)].clamp_max(float(region[6]))
        return tol.reshape(-1).to(self.device)

    def _view(self, padded: torch.Tensor) -> torch.Tensor:
        """``(nbx, nby, nbz, bx, by, bz)`` view of a padded frame."""
        (n0, n1, n2), (b0, b1, b2) = self.nblocks, self.block
        return padded.view(n0, b0, n1, b1, n2, b2).permute(0, 2, 4, 1, 3, 5)

    def _unravel(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, n1, n2 = self.nblocks
        div = lambda a, b: torch.div(a, b, rounding_mode="floor")  # noqa: E731
        return div(ids, n1 * n2), div(ids, n2) % n1, ids % n2

    # -- compression -----------------------------------------------------------------------------------------

    def _st_hosvd(self, x: torch.Tensor, tolerance: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Rank-adaptive ST-HOSVD of ``x`` ``(na, b0, b1, b2)`` with per-block ``tolerance`` ``(na,)``: ranks
        ``(na, 3)``, zero-padded factors and core."""
        na = x.shape[0]
        energy = torch.linalg.vector_norm(x, dim=(1, 2, 3)).double().square()
        budget = tolerance.square() * energy / 3.0
        y = x
        ranks, factors = [], []
        for mode in range(3):
            moved = y.movedim(1 + mode, 1)
            rest = moved.shape[2:]
            m = moved.reshape(na, moved.shape[1], -1)
            gram = m @ m.transpose(1, 2)
            w, v = torch.linalg.eigh(gram)  # ascending; batched Jacobi for sides <= 32
            w = w.double().clamp_min(0.0)
            tail = torch.cumsum(w, dim=1)  # tail[k] = sum of the k + 1 smallest eigenvalues
            r = moved.shape[1] - (tail <= budget[:, None]).sum(dim=1)
            r = r.clamp_min(1)
            if self.config.max_rank is not None:
                r = r.clamp_max(self.config.max_rank)
            top = int(r.max())
            u = v.flip(-1)[:, :, :top]
            u = u * (torch.arange(top, device=u.device)[None, :] < r[:, None]).to(u.dtype)[:, None, :]
            y = (u.transpose(1, 2) @ m).reshape(na, top, *rest).movedim(1, 1 + mode)
            ranks.append(r)
            factors.append(u)
        return torch.stack(ranks, dim=1), factors, y

    @staticmethod
    def _masks(ranks: torch.Tensor, sizes: Sequence[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Boolean masks selecting the rank-trimmed factor and core entries (row-major per block)."""
        tops = [int(ranks[:, i].max()) for i in range(3)]
        cols = [torch.arange(t, device=ranks.device)[None, :] < ranks[:, i : i + 1] for i, t in enumerate(tops)]
        factor_masks = [c[:, None, :].expand(ranks.shape[0], s, t) for c, s, t in zip(cols, sizes, tops)]
        core_mask = cols[0][:, :, None, None] & cols[1][:, None, :, None] & cols[2][:, None, None, :]
        return factor_masks, core_mask

    def _compress(self, frame: torch.Tensor, n: int, stats: BlockFrameStats) -> Optional[_Record]:
        buf = self._buffer
        buf.zero_()
        buf[self.crop].copy_(frame)
        view = self._view(buf)
        norms = torch.linalg.vector_norm(view, dim=(3, 4, 5)).reshape(-1)
        nonzero = norms > 0
        needed = self.last_needed >= n
        peak = torch.where(needed, norms, torch.zeros_like(norms)).max()
        active = needed & nonzero & (norms >= self.config.floor * peak)
        stats.blocks += int(nonzero.sum())
        stats.shadow += int((nonzero & ~needed).sum())
        stats.floor += int((nonzero & needed & ~active).sum())
        ids = active.nonzero().squeeze(1)
        if ids.numel() == 0:
            return None
        # Blocks with a lowered tolerance (tight_regions) go into chunks of their own: a chunk is padded to its
        # largest rank, so one near-full-rank block would otherwise inflate the work of all the others.
        tight = self.block_tolerance[ids] < self.config.tolerance
        groups = [ids[tight], ids[~tight]] if bool(tight.any()) else [ids]
        ids = torch.cat(groups)
        sizes = []
        for g in groups:
            sizes += [min(self.config.chunk_blocks, g.numel() - k) for k in range(0, g.numel(), self.config.chunk_blocks)]
        ix, iy, iz = self._unravel(ids)
        half = self.payload_dtype != torch.float32
        all_ranks, payloads, all_scales = [], [], []
        with matmul_precision(self.config.allow_tf32):
            start = 0
            for size in sizes:
                sl = slice(start, start + size)
                start += size
                x = view[ix[sl], iy[sl], iz[sl]]
                ranks, factors, core = self._st_hosvd(x, self.block_tolerance[ids[sl]])
                factor_masks, core_mask = self._masks(ranks, self.block)
                if half:  # core entries outside the ranks are zero (masked factors)
                    scale = core.abs().amax(dim=(1, 2, 3)).clamp_min(torch.finfo(torch.float32).tiny)
                    core = core / scale[:, None, None, None]
                    all_scales.append(scale)
                payloads += [f[mask] for f, mask in zip(factors, factor_masks)] + [core[core_mask]]
                all_ranks.append(ranks)
        ranks = torch.cat(all_ranks)
        # Payload layout per chunk: U0s, U1s, U2s, cores; chunks are concatenated in order.
        payload, spilled = self._store_payload(torch.cat(payloads).to(self.payload_dtype))
        record = _Record(
            self._to_storage(ids.to(torch.int32)),
            self._to_storage(ranks.to(torch.int16)),
            payload,
            tuple(sizes),
            spilled,
            self._to_storage(torch.cat(all_scales).float()) if half else None,
            self.itemsize,
        )
        stats.stored += ids.numel()
        stats.payload_bytes += record.nbytes()
        stats.mean_rank += float(ranks.double().prod(dim=1).pow(1.0 / 3.0).sum())
        return record

    def _decompress(self, record: Optional[_Record], out: torch.Tensor) -> None:
        """Write the frame of ``record`` into ``out`` ``(hx, hy, hz)``."""
        if record is None:
            out.zero_()
            return
        buf = self._buffer
        buf.zero_()
        view = self._view(buf)
        ids = record.ids.to(self.device, torch.int64)
        ranks_all = record.ranks.to(self.device, torch.int64)
        payload = self._load_payload(record).float()
        scales = record.scales.to(self.device) if record.scales is not None else None
        offset = 0
        sizes = record.chunks or tuple(
            min(self.config.chunk_blocks, ids.numel() - k) for k in range(0, ids.numel(), self.config.chunk_blocks))
        with matmul_precision(self.config.allow_tf32):
            start = 0
            for size in sizes:
                sl = slice(start, start + size)
                start += size
                ranks = ranks_all[sl]
                na = ranks.shape[0]
                factor_masks, core_mask = self._masks(ranks, self.block)
                factors = []
                for mask in factor_masks:
                    f = torch.zeros(mask.shape, device=self.device, dtype=torch.float32)
                    count = int(mask.sum())
                    f[mask] = payload[offset : offset + count]
                    offset += count
                    factors.append(f)
                core = torch.zeros(core_mask.shape, device=self.device, dtype=torch.float32)
                count = int(core_mask.sum())
                core[core_mask] = payload[offset : offset + count]
                offset += count
                if scales is not None:
                    core *= scales[sl][:, None, None, None]
                t = torch.einsum("nabc,nia->nibc", core, factors[0])
                t = torch.einsum("nibc,njb->nijc", t, factors[1])
                t = torch.einsum("nijc,nkc->nijk", t, factors[2])
                ix, iy, iz = self._unravel(ids[sl])
                view[ix, iy, iz] = t
                del t, core, factors
        out.copy_(buf[self.crop])

    # -- solver interface ------------------------------------------------------------------------------------

    def push(self, frames: Dict[str, torch.Tensor], start: int) -> None:
        """Compress frames ``start ...`` (``{name: (F, S, hx, hy, hz)}``)."""
        if start != self.frames_pushed:
            raise RuntimeError(f"Expected frame {self.frames_pushed}, got {start}.")
        counts = {t.shape[0] for t in frames.values()}
        if len(counts) != 1 or set(frames) != set(self.components):
            raise ValueError("push needs the same number of frames for every component.")
        count = counts.pop()
        if start + count > self.frames:
            raise ValueError("More frames pushed than the history holds.")
        for j in range(count):
            n = start + j
            for name, tensor in frames.items():
                stats = self.frame_stats[name][n]
                for s in range(self.shots):
                    record = self._compress(tensor[j, s], n, stats)
                    self.records[name][n * self.shots + s] = record
                    if self.config.measure_error:
                        estimate = torch.empty(self.frame_shape, device=self.device, dtype=torch.float32)
                        self._decompress(record, estimate)
                        err = self.errors[name]
                        err.error_sq += float(torch.linalg.vector_norm(estimate - tensor[j, s])) ** 2
                        ref = float(torch.linalg.vector_norm(tensor[j, s])) ** 2
                        err.reference_sq += ref
                        err.frames += 1
                if stats.stored:
                    stats.mean_rank /= stats.stored
        self.frames_pushed += count

    def reconstruct_into(self, out: Dict[str, torch.Tensor], start: int) -> None:
        """Write frames ``start ...`` of every component into ``out`` views ``(F, S, hx, hy, hz)``."""
        if self.frames_pushed != self.frames:
            raise RuntimeError(f"Only {self.frames_pushed} of {self.frames} frames were pushed.")
        for name, tensor in out.items():
            for j in range(tensor.shape[0]):
                for s in range(self.shots):
                    self._decompress(self.records[name][(start + j) * self.shots + s], tensor[j, s])

    def reconstruct(self, name: str, start: int = 0, count: Optional[int] = None) -> torch.Tensor:
        """Frames of one component as a new ``(F, S, hx, hy, hz)`` tensor."""
        count = self.frames - start if count is None else count
        out = torch.empty((count, self.shots, *self.frame_shape), device=self.device, dtype=torch.float32)
        self.reconstruct_into({name: out}, start)
        return out

    def size_report(self) -> HistorySizeReport:
        element = torch.empty((), dtype=self.history_dtype).element_size()
        raw = self.frames * self.shots * math.prod(self.frame_shape) * element
        components = {}
        for name in self.components:
            stored = sum(r.nbytes() for r in self.records[name] if r is not None)
            components[name] = (stored, raw, True, f"block Tucker {self.block}, tol {self.config.tolerance:g}")
        return HistorySizeReport(
            raw_bytes=raw * len(self.components),
            stored_bytes=sum(c[0] for c in components.values()),
            map_bytes=0,
            open_sketch_bytes=0,
            components=components,
        )

    def error_report(self) -> Dict[str, ComponentErrorStats]:
        return dict(self.errors)


__all__ = ["BlockFrameStats", "BlockTuckerConfig", "BlockTuckerHistory", "surface_adjoint_keep_until"]

"""Segmented native solver whose adjoint reads a Tucker-sketched history.

:class:`DeepGPRSketched` is the ``wavefield_sketch`` counterpart of
:class:`DeepGPR.solver.autograd.DeepGPR`. The FDTD update itself is
unchanged: the native forward advances the full-resolution E/H/CPML state
``segment_steps`` steps at a time, exactly as checkpointed ``compute`` calls
do. After each segment the saved ``E^n`` / ``R^n`` frames (the only forward
quantities the material-gradient kernel reads) are streamed into a
:class:`~DeepGPR.tucker.wavefield.SketchedWavefieldHistory` and the segment
buffer is reused. The backward pass runs the native adjoint segment by
segment in reverse, each time decompressing that segment's frames into the
same buffer. Adjoint states, receiver cotangents and source gradients are
carried across segments exactly; only the material-gradient kernel sees
approximated history values.

Memory: the full-resolution history never exists. Besides the compressed
history, one segment buffer of ``2 * components * frames * shots * cells``
float32 values and the random maps are allocated.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..native.bridge import invoke, native_stream_scope, require_capability
from ..native.loader import get_deepgpr_lib
from ..tucker.sketch import TuckerSketchConfig
from ..tucker.wavefield import ComponentErrorStats, HistorySizeReport, SketchedWavefieldHistory
from ..utils.logger import get_logger
from ..utils.tensor_checks import check_tensors_for_nan_inf
from .autograd import (
    STATE_GRADIENT_NAMES,
    STATE_NAMES,
    STATE_TENSOR_COUNT,
    TM2D_INACTIVE_STATE_INDICES,
    SolverConfig,
    SolverTensors,
    _allocate_update_coefficients,
    _check_forward_capabilities,
    _own_state_gradient,
    build_backward_arguments,
    build_forward_arguments,
)
from .storage import encode_storage_type, saved_history_shape, saved_time_steps

_LOGGER = get_logger(__name__)

#: Default native steps per segment.
DEFAULT_SEGMENT_STEPS = 16

#: Index of the first state tensor in ``DeepGPRSketched.apply`` arguments.
_FIRST_STATE_ARGUMENT = 7


@dataclass
class WavefieldSketchStats:
    """Measurements of one ``compute(..., wavefield_sketch=...)`` call.

    Pass an instance as ``wavefield_sketch_stats``; the solver fills it.

    Attributes:
        keep_history: Keep a reference to the compressed history after
            backward (for diagnostics; holds its memory until released).
        synchronize: Synchronize CUDA around each phase so the timings are
            exact (adds a little overhead).
        components: History components (``("Ez", "Rz")`` or six names).
        segments, segment_steps: Native segments and their length.
        frames: Saved history frames.
        segment_buffer_bytes: Full-resolution segment buffer (E and R).
        size: Measured compressed history size (after the forward pass).
        forward_native_seconds, compress_seconds: Forward phase timings.
        backward_native_seconds, decompress_seconds: Backward phase timings.
        errors: Per-component reconstruction errors (``measure_error``).
        history: The compressed history if ``keep_history`` (else ``None``).
    """

    keep_history: bool = False
    synchronize: bool = False
    components: Tuple[str, ...] = ()
    segments: int = 0
    segment_steps: int = 0
    frames: int = 0
    segment_buffer_bytes: int = 0
    size: Optional[HistorySizeReport] = None
    forward_native_seconds: float = 0.0
    compress_seconds: float = 0.0
    backward_native_seconds: float = 0.0
    decompress_seconds: float = 0.0
    errors: Dict[str, ComponentErrorStats] = field(default_factory=dict)
    history: Optional[SketchedWavefieldHistory] = None


class _Timer:
    """Accumulate wall time of a phase, optionally CUDA-synchronized."""

    def __init__(self, device: torch.device, synchronize: bool) -> None:
        self.sync = synchronize and device.type == "cuda"
        self.device = device

    def __call__(self) -> float:
        if self.sync:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()


def history_components(mode: int, rhs_from_e: bool = False) -> Tuple[str, ...]:
    """History components read by the material-gradient kernel."""
    electric = ("Ex", "Ey", "Ez") if mode == 3 else ("Ez",)
    if rhs_from_e:
        return electric
    return electric + tuple("R" + name[1] for name in electric)


def resolve_segment_steps(requested: Optional[int], nt: int, sampling_interval: int) -> int:
    """Segment length: a multiple of the sampling interval, at most ``nt`` rounded up."""
    steps = DEFAULT_SEGMENT_STEPS if requested is None else int(requested)
    steps = -(-steps // sampling_interval) * sampling_interval
    whole = -(-nt // sampling_interval) * sampling_interval
    return max(sampling_interval, min(steps, whole))


def _segments(nt: int, steps: int) -> List[Tuple[int, int]]:
    return [(start, min(start + steps, nt)) for start in range(0, nt, steps)]


def _prefix_view(buffer: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Contiguous view of the first ``prod(shape)`` elements of ``buffer``."""
    return buffer.view(-1)[: math.prod(shape)].view(shape)


def _component_views(
    e_view: torch.Tensor, r_view: Optional[torch.Tensor], mode: int
) -> Dict[str, torch.Tensor]:
    """``{name: (F, S, hx, hy, hz)}`` views into the native history layout.

    ``r_view=None`` returns only the E components (``rhs="from_e"``).
    """
    if mode == 3:
        names = ("x", "y", "z")
        views = {f"E{axis}": e_view[index] for index, axis in enumerate(names)}
        if r_view is not None:
            views.update({f"R{axis}": r_view[index] for index, axis in enumerate(names)})
        return views
    views = {"Ez": e_view}
    if r_view is not None:
        views["Rz"] = r_view
    return views


class DeepGPRSketched(torch.autograd.Function):
    """Autograd function: segmented native solve with a Tucker-sketched history.

    Call signature::

        DeepGPRSketched.apply(config, tensors, sketch, stats, eps_r, sigma,
                              source_amplitudes, *states)

    Returns the 30 advanced states, an empty ``E_saved`` placeholder and the
    receiver data, like :class:`DeepGPR.solver.autograd.DeepGPR`.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        config: SolverConfig,
        tensors: SolverTensors,
        sketch: TuckerSketchConfig,
        stats: Optional[WavefieldSketchStats],
        eps_r: torch.Tensor,
        sigma: torch.Tensor,
        source_amplitudes: torch.Tensor,
        *states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Run the native forward in segments and sketch the history."""
        if len(states) != STATE_TENSOR_COUNT:
            raise ValueError(f"Expected {STATE_TENSOR_COUNT} state tensors, got {len(states)}.")
        if config.storage.compression != "none" or config.storage.dtype != torch.float32:
            raise ValueError("The Tucker-sketched history needs float32 native frames.")
        if config.reconstruct_rhs or config.use_async_offload:
            raise ValueError(
                "The Tucker-sketched history stores E and R; reconstruct_rhs and "
                "use_async_offload are not supported."
            )
        ctx.set_materialize_grads(False)
        device = config.device
        source_amplitudes = source_amplitudes.contiguous()
        source_location = tensors.source_location.to(device=device, dtype=torch.int32).contiguous()
        receiver_location = tensors.receiver_location.to(device=device, dtype=torch.int32).contiguous()
        c_lib = get_deepgpr_lib(device)
        eps_r_requires_grad = bool(eps_r.requires_grad)
        sigma_requires_grad = bool(sigma.requires_grad)
        needs_model_gradient = eps_r_requires_grad or sigma_requires_grad
        _check_forward_capabilities(c_lib, config, needs_model_gradient)
        if needs_model_gradient and config.history_region == "physical" and bool(config.pmlthick.any()):
            raise ValueError(
                "wavefield_history_region='physical' omits PML material sensitivities. "
                "Use 'extended' for material gradients with PML."
            )

        interval = config.sampling_interval
        rhs_from_e = sketch.rhs == "from_e"
        if rhs_from_e and needs_model_gradient:
            if interval != 1:
                raise ValueError(
                    "TuckerSketchConfig(rhs='from_e') requires model_gradient_sampling_interval=1."
                )
            require_capability(
                c_lib,
                "deepgpr_supports_rhs_reconstruction",
                "The loaded native library cannot rebuild R^n from the E history; "
                "use TuckerSketchConfig(rhs='stored') or rebuild the libraries.",
            )
        if needs_model_gradient:
            steps = resolve_segment_steps(sketch.segment_steps, config.nt, interval)
        else:
            # No history is needed: one call, history disabled.
            steps = max(config.nt, 1)
        segments = _segments(config.nt, steps)
        storage_type = encode_storage_type(
            config.storage,
            needs_model_gradient,
            rhs_from_e=rhs_from_e and needs_model_gradient,
            physical_history=config.history_region == "physical",
            tm2d=config.tm2d_fast_path,
        )
        components = history_components(config.mode, rhs_from_e)
        frames_total = saved_time_steps(config.nt, interval)
        # E-only history: one more E frame (E^nt) closes the last R^n.
        stored_frames = frames_total + 1 if rhs_from_e else frames_total
        timer = _Timer(device, bool(stats is not None and stats.synchronize))

        history: Optional[SketchedWavefieldHistory] = None
        e_buffer = r_buffer = torch.empty(0, device=device, dtype=torch.float32)
        segment_frames = saved_time_steps(steps, interval)
        if needs_model_gradient:
            history = SketchedWavefieldHistory(
                components,
                config.history_cells,
                config.nstep,
                stored_frames,
                sketch,
                device,
                torch.float32,
            )
            shape = saved_history_shape(config.mode, segment_frames, config.nstep, *config.history_cells)
            e_buffer = torch.empty(shape, device=device, dtype=torch.float32)
            # R history, or (rhs="from_e") the one-frame E^(n+1) buffer.
            r_frames = 1 if rhs_from_e else segment_frames
            r_buffer = torch.empty(
                saved_history_shape(config.mode, r_frames, config.nstep, *config.history_cells),
                device=device,
                dtype=torch.float32,
            )

        update_coefficients = _allocate_update_coefficients(config)
        receiver_amplitudes = torch.empty(
            (config.nstep, config.nt, config.nrx), device=device, dtype=config.dtype
        )
        receiver_buffer = torch.empty(
            (config.nstep, steps, config.nrx), device=device, dtype=config.dtype
        )
        materials = (tensors.eps_r_pad, tensors.sigma_pad, tensors.mu_r_pad)
        native_seconds = compress_seconds = 0.0
        for start, stop in segments:
            count = stop - start
            frames = saved_time_steps(count, interval)
            segment_config = dataclasses.replace(config, nt=count)
            history_shape = saved_history_shape(
                config.mode, frames, config.nstep, *config.history_cells
            )
            e_view = _prefix_view(e_buffer, history_shape) if history is not None else e_buffer
            if history is None:
                r_view = r_buffer
            elif rhs_from_e:
                r_view = r_buffer  # receives E after the segment's last step
            else:
                r_view = _prefix_view(r_buffer, history_shape)
            receiver_view = (
                receiver_amplitudes
                if len(segments) == 1
                else _prefix_view(receiver_buffer, (config.nstep, count, config.nrx))
            )
            arguments = build_forward_arguments(
                segment_config,
                materials=materials,
                pml_coefficients=tensors.pml_coefficients,
                states=states,
                update_coefficients=update_coefficients,
                e_saved=e_view,
                r_saved=r_view,
                receiver_data=receiver_view,
                source_location=source_location,
                receiver_location=receiver_location,
                source_waveform=source_amplitudes[:, start:stop].contiguous(),
                storage_type=storage_type,
                save_model_history=needs_model_gradient,
                use_async_offload=False,
            )
            began = timer()
            with native_stream_scope(device):
                invoke(c_lib, "forward", config.fdtd_order, arguments)
            if receiver_view is not receiver_amplitudes:
                receiver_amplitudes[:, start:stop].copy_(receiver_view)
            native_done = timer()
            native_seconds += native_done - began
            if history is not None:
                history.push(
                    _component_views(e_view, None if rhs_from_e else r_view, config.mode),
                    start // interval,
                )
                if rhs_from_e and stop == config.nt:
                    history.push(_component_views(r_view, None, config.mode), frames_total)
                compress_seconds += timer() - native_done

        if config.debug:
            check_tensors_for_nan_inf(d="forward", **dict(zip(STATE_NAMES, states)))

        if stats is not None:
            stats.components = components if history is not None else ()
            stats.segments = len(segments)
            stats.segment_steps = steps
            stats.frames = frames_total if history is not None else 0
            stats.segment_buffer_bytes = (e_buffer.numel() + r_buffer.numel()) * e_buffer.element_size()
            stats.forward_native_seconds = native_seconds
            stats.compress_seconds = compress_seconds
            stats.backward_native_seconds = 0.0
            stats.decompress_seconds = 0.0
            if history is not None:
                stats.size = history.size_report()
                stats.errors = history.error_report()
            stats.history = history if stats.keep_history else None

        # Saved even without material gradients: initial-state cotangents
        # (checkpointed segments) also need the native adjoint.
        ctx.save_for_backward(
            tensors.mu_r_pad,
            source_location,
            receiver_location,
            *tensors.pml_coefficients,
            tensors.eps_r_pad,
            tensors.sigma_pad,
        )
        ctx.config = config
        ctx.stats = stats
        ctx.history = history
        ctx.segments = segments
        ctx.steps = steps
        ctx.storage_type = storage_type
        ctx.rhs_from_e = rhs_from_e
        ctx.eps_r_requires_grad = eps_r_requires_grad
        ctx.sigma_requires_grad = sigma_requires_grad
        ctx.source_requires_grad = bool(source_amplitudes.requires_grad)
        ctx.source_shape = tuple(source_amplitudes.shape)
        ctx.state_shapes = tuple(state.shape for state in states)
        ctx.receiver_shape = (config.nstep, config.nt, config.nrx)
        ctx.released = False
        e_saved = torch.empty(0, device=device, dtype=torch.float32)
        ctx.mark_non_differentiable(e_saved)
        return (*states, e_saved, receiver_amplitudes)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Optional[torch.Tensor]):  # type: ignore[override]
        """Native adjoint segment by segment, decompressing each segment's history."""
        config: SolverConfig = ctx.config
        if ctx.released:
            raise RuntimeError(
                "The sketched forward history was already released by a previous "
                "backward pass. Re-run DeepGPR.compute before calling backward again."
            )
        device, dtype = config.device, config.dtype
        needs_model_gradient = ctx.eps_r_requires_grad or ctx.sigma_requires_grad
        history: Optional[SketchedWavefieldHistory] = ctx.history
        stats: Optional[WavefieldSketchStats] = ctx.stats
        timer = _Timer(device, bool(stats is not None and stats.synchronize))

        saved = tuple(tensor.contiguous() for tensor in ctx.saved_tensors)
        mu_r_pad, source_location, receiver_location = saved[:3]
        coefficients = saved[3:15]
        eps_r_pad, sigma_pad = saved[15:17]
        materials = (eps_r_pad, sigma_pad, mu_r_pad)

        state_cotangents = grad_outputs[:STATE_TENSOR_COUNT]
        data_grad = grad_outputs[STATE_TENSOR_COUNT + 1]
        lambdas = [
            _own_state_gradient(gradient, shape, device, dtype)
            for gradient, shape in zip(state_cotangents, ctx.state_shapes)
        ]
        data_grad = (
            torch.zeros(ctx.receiver_shape, device=device, dtype=dtype)
            if data_grad is None
            else data_grad.contiguous()
        )
        nx, ny, nz = config.nx, config.ny, config.nz
        grad_eps_r = (
            torch.zeros((nx, ny, nz), device=device, dtype=dtype)
            if ctx.eps_r_requires_grad
            else torch.empty(0, device=device, dtype=dtype)
        )
        grad_sigma = (
            torch.zeros((nx, ny, nz), device=device, dtype=dtype)
            if ctx.sigma_requires_grad
            else torch.empty(0, device=device, dtype=dtype)
        )
        grad_source = (
            torch.zeros(ctx.source_shape, device=device, dtype=dtype)
            if ctx.source_requires_grad
            else torch.empty(0, device=device, dtype=dtype)
        )
        steps = ctx.steps
        interval = config.sampling_interval
        e_buffer = r_buffer = torch.empty(0, device=device, dtype=torch.float32)
        rhs_from_e = ctx.rhs_from_e
        if needs_model_gradient:
            frames_per_segment = saved_time_steps(steps, interval)
            shape = saved_history_shape(config.mode, frames_per_segment, config.nstep, *config.history_cells)
            e_buffer = torch.empty(shape, device=device, dtype=torch.float32)
            r_buffer = torch.empty(
                saved_history_shape(
                    config.mode, 1 if rhs_from_e else frames_per_segment, config.nstep, *config.history_cells
                ),
                device=device,
                dtype=torch.float32,
            )
        source_buffer = (
            torch.empty((ctx.source_shape[0], steps, *ctx.source_shape[2:]), device=device, dtype=dtype)
            if ctx.source_requires_grad
            else grad_source
        )
        update_coefficients = _allocate_update_coefficients(config)
        c_lib = get_deepgpr_lib(device)
        native_seconds = decompress_seconds = 0.0
        single = len(ctx.segments) == 1
        for start, stop in reversed(ctx.segments):
            count = stop - start
            frames = saved_time_steps(count, interval)
            segment_config = dataclasses.replace(config, nt=count)
            began = timer()
            if history is not None:
                history_shape = saved_history_shape(
                    config.mode, frames, config.nstep, *config.history_cells
                )
                e_view = _prefix_view(e_buffer, history_shape)
                if rhs_from_e:
                    r_view = r_buffer
                    history.reconstruct_into(_component_views(e_view, None, config.mode), start)
                    history.reconstruct_into(_component_views(r_view, None, config.mode), stop)
                else:
                    r_view = _prefix_view(r_buffer, history_shape)
                    history.reconstruct_into(
                        _component_views(e_view, r_view, config.mode), start // interval
                    )
            else:
                e_view, r_view = e_buffer, r_buffer
            decoded = timer()
            decompress_seconds += decoded - began
            if ctx.source_requires_grad and not single:
                source_view = _prefix_view(source_buffer, (ctx.source_shape[0], count, *ctx.source_shape[2:]))
                source_view.zero_()
            else:
                source_view = grad_source
            arguments = build_backward_arguments(
                segment_config,
                materials=materials,
                pml_coefficients=coefficients,
                lambdas=tuple(lambdas),
                update_coefficients=update_coefficients,
                e_saved=e_view,
                r_saved=r_view,
                data_grad=data_grad if single else data_grad[:, start:stop].contiguous(),
                source_location=source_location,
                receiver_location=receiver_location,
                grad_source=source_view,
                grad_eps_r=grad_eps_r,
                grad_sigma=grad_sigma,
                source_requires_grad=ctx.source_requires_grad,
                eps_r_requires_grad=ctx.eps_r_requires_grad,
                sigma_requires_grad=ctx.sigma_requires_grad,
                storage_type=ctx.storage_type,
                use_async_offload=False,
            )
            with native_stream_scope(device):
                invoke(c_lib, "backward", config.fdtd_order, arguments)
            if ctx.source_requires_grad and not single:
                grad_source[:, start:stop].copy_(source_view)
            native_seconds += timer() - decoded

        if config.tm2d_fast_path:
            for index in TM2D_INACTIVE_STATE_INDICES:
                lambdas[index].zero_()
        state_needs_grad = ctx.needs_input_grad[
            _FIRST_STATE_ARGUMENT : _FIRST_STATE_ARGUMENT + STATE_TENSOR_COUNT
        ]
        if config.debug:
            to_check = {
                name: gradient
                for name, gradient, required in zip(STATE_GRADIENT_NAMES, lambdas, state_needs_grad)
                if required
            }
            if ctx.eps_r_requires_grad:
                to_check["grad_eps_r"] = grad_eps_r
            if ctx.sigma_requires_grad:
                to_check["grad_sigma"] = grad_sigma
            if ctx.source_requires_grad:
                to_check["grad_source"] = grad_source
            check_tensors_for_nan_inf(d="backward", **to_check)
        if stats is not None:
            stats.backward_native_seconds = native_seconds
            stats.decompress_seconds = decompress_seconds

        ctx.history = None
        ctx.released = True
        return (
            None,
            None,
            None,
            None,
            grad_eps_r if ctx.eps_r_requires_grad else None,
            grad_sigma if ctx.sigma_requires_grad else None,
            grad_source if ctx.source_requires_grad else None,
            *(gradient if required else None for gradient, required in zip(lambdas, state_needs_grad)),
        )


__all__ = [
    "DEFAULT_SEGMENT_STEPS",
    "DeepGPRSketched",
    "WavefieldSketchStats",
    "history_components",
    "resolve_segment_steps",
]

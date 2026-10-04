"""PyTorch autograd bridge to the native forward and adjoint FDTD solvers.

:class:`DeepGPR` runs the native forward solver in ``forward`` and the exact
discrete adjoint in ``backward``. The electromagnetic state tensors (E, H and
the 24 CPML auxiliary arrays) are both inputs and outputs: the native solver
advances them in place, which is what makes segment-wise checkpointing work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..config.constants import CPML_FACE_NAMES, TM2D_INACTIVE_PHI_INDICES
from ..native.abi import LAMBDA_PHI_PARAMETER_NAMES, PHI_PARAMETER_NAMES
from ..native.bridge import invoke, native_stream_scope, require_capability
from ..native.loader import get_deepgpr_lib
from ..utils.logger import get_logger
from ..utils.tensor_checks import check_nonzero_source_created_fields, check_tensors_for_nan_inf
from .storage import (
    WavefieldStorageConfig,
    decompress_wavefield_history,
    encode_storage_type,
    history_spatial_shape,
    int8_history_layout,
    saved_history_shape,
    saved_time_steps,
)

_LOGGER = get_logger(__name__)

#: Number of state tensors (Ex, Ey, Ez, Hx, Hy, Hz + 24 CPML arrays).
STATE_TENSOR_COUNT = 30

#: Display names of the 30 state tensors, used in debug diagnostics.
STATE_NAMES: Tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") + tuple(
    f"{face}{field}Phi{index}"
    for face in CPML_FACE_NAMES
    for field in ("E", "H")
    for index in (1, 2)
)

#: Names of the incoming state gradients, used in debug diagnostics.
STATE_GRADIENT_NAMES: Tuple[str, ...] = (
    "lambda_ex",
    "lambda_ey",
    "lambda_ez",
    "lambda_hx",
    "lambda_hy",
    "lambda_hz",
) + LAMBDA_PHI_PARAMETER_NAMES

_UPDATE_COEFFICIENT_NAMES = ("ce_hist", "ce_curl", "ce_rhs", "ch_hist", "ch_curl", "ch_rhs")

#: State indices (Ex, Ey, Hz and their CPML auxiliaries) inactive on the 2D TM path.
TM2D_INACTIVE_STATE_INDICES: Tuple[int, ...] = (0, 1, 5) + tuple(
    6 + index for index in TM2D_INACTIVE_PHI_INDICES
)


@dataclass(frozen=True)
class SolverConfig:
    """Scalar configuration of one native solve (not differentiated).

    Attributes:
        device: Execution device.
        dtype: Native float dtype (``torch.float32``).
        grid_spacing: ``(dx, dy, dz)`` [m].
        dt: Time step [s].
        nx, ny, nz: Extended grid dimensions (including CPML).
        nt: Time steps.
        nstep, nsr, nrx: Shots, sources per shot, receivers per shot.
        pmlthick: ``int32`` tensor of six face thicknesses.
        source_direction, receiver_component: Field components 0/1/2.
        sampling_interval: History sampling interval.
        storage: Resolved history-storage options.
        save_wavefield_history: Whether histories are written.
        use_async_offload: Requested CUDA pinned-memory offload.
        fdtd_order: 2, 4 or 8.
        mode: Gradient mode 2 or 3.
        debug: Whether to run the expensive NaN/Inf checks.
        reconstruct_rhs: Store only E (sampling interval 1) and rebuild R^n
            from consecutive E frames in the adjoint; requested by
            ``wavefield_rhs_history`` and applied only when a material
            gradient uses the history.
        history_region: ``"extended"`` (whole grid) or ``"physical"`` (saved
            frames cover only the physical model cells).
        tm2d_fast_path: Use the CUDA 2D Ez-TM fast path (decided by
            :func:`DeepGPR.solver.modeling.compute` from the geometry, the
            source component and the initial states).
    """

    device: torch.device
    dtype: torch.dtype
    grid_spacing: Tuple[float, float, float]
    dt: float
    nx: int
    ny: int
    nz: int
    nt: int
    nstep: int
    nsr: int
    nrx: int
    pmlthick: torch.Tensor
    source_direction: int
    receiver_component: int
    sampling_interval: int
    storage: WavefieldStorageConfig
    save_wavefield_history: bool
    use_async_offload: bool
    fdtd_order: int
    mode: int
    debug: bool
    reconstruct_rhs: bool = False
    history_region: str = "extended"
    tm2d_fast_path: bool = False

    @property
    def pml(self) -> List[int]:
        """Face thicknesses as Python ints ``[x0, xm, y0, ym, z0, zm]``."""
        return [int(self.pmlthick[i]) for i in range(6)]

    @property
    def history_cells(self) -> Tuple[int, int, int]:
        """Spatial cells ``(hx, hy, hz)`` of one saved history frame."""
        block = self.storage.block_size if self.storage.compression == "int8" else None
        return history_spatial_shape(
            self.nx, self.ny, self.nz, self.pml, self.history_region, block
        )


@dataclass(frozen=True)
class SolverTensors:
    """Non-differentiated tensors consumed by the native solver.

    Attributes:
        mu_r_pad, eps_r_pad, sigma_pad: Materials with CPML and Yee halo.
        source_location, receiver_location: ``int32`` extended-grid indices.
        pml_descriptors: Six face descriptors (see :mod:`.pml`).
        pml_coefficients: Twelve coefficient tensors ordered
            ``x0_e, x0_h, xm_e, xm_h, ..., zm_e, zm_h``.
    """

    mu_r_pad: torch.Tensor
    eps_r_pad: torch.Tensor
    sigma_pad: torch.Tensor
    source_location: torch.Tensor
    receiver_location: torch.Tensor
    pml_descriptors: Tuple[torch.Tensor, ...]
    pml_coefficients: Tuple[torch.Tensor, ...]


def _coefficient_arguments(coefficients: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
    """Map the 12 CPML coefficient tensors to their native parameter names."""
    arguments: Dict[str, torch.Tensor] = {}
    for face_index, face in enumerate(CPML_FACE_NAMES):
        arguments[f"{face}_e_coeff"] = coefficients[2 * face_index]
        arguments[f"{face}_h_coeff"] = coefficients[2 * face_index + 1]
    return arguments


def _common_arguments(config: SolverConfig) -> Dict[str, Any]:
    """Scalar arguments shared by the native forward and backward calls."""
    dx, dy, dz = config.grid_spacing
    arguments: Dict[str, Any] = {
        "dt": config.dt,
        "nt": config.nt,
        "nshot": config.nstep,
        "nreceiver": config.nrx,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "nx_fields": config.nx + 1,
        "ny_fields": config.ny + 1,
        "nz_fields": config.nz + 1,
        "receiver_component": config.receiver_component,
        "nsource": config.nsr,
        "source_component": config.source_direction,
        "sampling_interval": config.sampling_interval,
        "fwi_mode": config.mode,
    }
    for face, thickness in zip(CPML_FACE_NAMES, config.pml):
        arguments[f"pml_{face}"] = thickness
    return arguments


def _allocate_update_coefficients(config: SolverConfig) -> Dict[str, torch.Tensor]:
    """Scratch space the native code fills with the FDTD update coefficients."""
    update_coeffs = torch.zeros(
        (6, config.nx + 1, config.ny + 1, config.nz + 1),
        device=config.device,
        dtype=config.dtype,
    )
    return dict(zip(_UPDATE_COEFFICIENT_NAMES, update_coeffs.unbind(0)))


def build_forward_arguments(
    config: SolverConfig,
    *,
    materials: Tuple[Any, Any, Any],
    pml_coefficients: Tuple[Any, ...],
    states: Tuple[Any, ...],
    update_coefficients: Dict[str, Any],
    e_saved: Any,
    r_saved: Any,
    receiver_data: Any,
    source_location: Any,
    receiver_location: Any,
    source_waveform: Any,
    storage_type: int,
    save_model_history: bool,
    use_async_offload: bool,
) -> Dict[str, Any]:
    """Return the ``{parameter: value}`` mapping for the native ``forward``.

    Pure function (no tensor operations), so it can be unit-tested with any
    array type that exposes a memory address.

    Args:
        config: Solve configuration.
        materials: ``(eps_r_pad, sigma_pad, mu_r_pad)``.
        pml_coefficients: 12 CPML coefficient arrays (``x0_e, x0_h, ...``).
        states: 30 state arrays (Ex, Ey, Ez, Hx, Hy, Hz, 24 CPML arrays).
        update_coefficients: Six scratch arrays keyed ``ce_hist`` ... ``ch_rhs``.
        e_saved, r_saved: History buffers (``r_saved`` holds the final E
            frame when ``storage_type`` carries the E-only flag).
        receiver_data: ``(nstep, nt, nrx)`` output buffer.
        source_location, receiver_location: ``int32`` extended-grid indices.
        source_waveform: ``(nsr, nt, 1)`` waveforms.
        storage_type: Encoded native storage type.
        save_model_history: Whether the R history is written.
        use_async_offload: Effective CUDA offload flag.
    """
    if len(states) != STATE_TENSOR_COUNT:
        raise ValueError(f"Expected {STATE_TENSOR_COUNT} state tensors, got {len(states)}.")
    arguments: Dict[str, Any] = _common_arguments(config)
    arguments.update(update_coefficients)
    arguments.update(_coefficient_arguments(pml_coefficients))
    arguments.update(
        {
            "eps_r_pad": materials[0],
            "sigma_pad": materials[1],
            "mu_r_pad": materials[2],
            "E_saved": e_saved,
            "R_saved": r_saved,
            "Ex": states[0],
            "Ey": states[1],
            "Ez": states[2],
            "Hx": states[3],
            "Hy": states[4],
            "Hz": states[5],
            "receiver_location": receiver_location,
            "receiver_data": receiver_data,
            "source_location": source_location,
            "source_waveform": source_waveform,
            "storage_type": storage_type,
            "save_model_history": int(save_model_history),
            "use_async_offload": int(use_async_offload),
        }
    )
    arguments.update(zip(PHI_PARAMETER_NAMES, states[6:]))
    return arguments


def build_backward_arguments(
    config: SolverConfig,
    *,
    materials: Tuple[Any, Any, Any],
    pml_coefficients: Tuple[Any, ...],
    lambdas: Tuple[Any, ...],
    update_coefficients: Dict[str, Any],
    e_saved: Any,
    r_saved: Any,
    data_grad: Any,
    source_location: Any,
    receiver_location: Any,
    grad_source: Any,
    grad_eps_r: Any,
    grad_sigma: Any,
    source_requires_grad: bool,
    eps_r_requires_grad: bool,
    sigma_requires_grad: bool,
    storage_type: int,
    use_async_offload: bool,
) -> Dict[str, Any]:
    """Return the ``{parameter: value}`` mapping for the native ``backward``.

    Args:
        config: Solve configuration.
        materials: ``(eps_r_pad, sigma_pad, mu_r_pad)``.
        pml_coefficients: 12 CPML coefficient arrays.
        lambdas: 30 adjoint state arrays (advanced in place).
        update_coefficients: Six scratch arrays.
        e_saved, r_saved: Forward histories.
        data_grad: Receiver-data cotangent ``(nstep, nt, nrx)``.
        source_location, receiver_location: ``int32`` extended-grid indices.
        grad_source, grad_eps_r, grad_sigma: Gradient output buffers.
        source_requires_grad, eps_r_requires_grad, sigma_requires_grad: Flags.
        storage_type: Encoded native storage type.
        use_async_offload: Effective CUDA offload flag.
    """
    if len(lambdas) != STATE_TENSOR_COUNT:
        raise ValueError(f"Expected {STATE_TENSOR_COUNT} adjoint states, got {len(lambdas)}.")
    arguments: Dict[str, Any] = _common_arguments(config)
    arguments.update(update_coefficients)
    arguments.update(_coefficient_arguments(pml_coefficients))
    arguments.update(
        {
            "eps_r_pad": materials[0],
            "sigma_pad": materials[1],
            "mu_r_pad": materials[2],
            "E_saved": e_saved,
            "R_saved": r_saved,
            "ndata_source": config.nrx,
            "receiver_location": receiver_location,
            "data_grad": data_grad,
            "source_location": source_location,
            "grad_source": grad_source,
            "source_requires_grad": int(source_requires_grad),
            "grad_eps_r": grad_eps_r,
            "grad_sigma": grad_sigma,
            "eps_r_requires_grad": int(eps_r_requires_grad),
            "sigma_requires_grad": int(sigma_requires_grad),
            "storage_type": storage_type,
            "use_async_offload": int(use_async_offload),
        }
    )
    arguments.update(zip(STATE_GRADIENT_NAMES, lambdas))
    return arguments


def _check_forward_capabilities(c_lib: Any, config: SolverConfig) -> None:
    """Reject native libraries that lack a feature this call needs."""
    if bool(config.pmlthick.any()):
        require_capability(
            c_lib,
            "deepgpr_supports_external_pml",
            "The loaded library lacks external-PML material gradients. "
            "Rebuild the CPU/CUDA shared libraries from the current sources.",
        )
    storage = config.storage
    if config.tm2d_fast_path:
        require_capability(
            c_lib,
            "deepgpr_supports_tm2d_fast_path",
            "The loaded native library has no 2D TM fast path. Rebuild deepgpr.cu.",
        )
    if config.history_region == "physical":
        require_capability(
            c_lib,
            "deepgpr_supports_physical_history",
            "The loaded native library cannot restrict histories to the physical "
            "model. Rebuild the CPU/CUDA shared libraries from the current sources.",
        )
    if config.reconstruct_rhs:
        require_capability(
            c_lib,
            "deepgpr_supports_rhs_reconstruction",
            "The loaded native library cannot rebuild R^n from the E history. "
            "Rebuild the CPU/CUDA shared libraries from the current sources.",
        )
    if config.save_wavefield_history and storage.conversion_backend != "legacy":
        require_capability(
            c_lib,
            "deepgpr_supports_conversion_backends",
            "The loaded CUDA library does not contain selectable native "
            "FP16/BF16 conversion backends. Rebuild deepgpr.cu.",
        )
    if config.save_wavefield_history and storage.compression == "int8":
        require_capability(
            c_lib,
            "deepgpr_supports_int8_wavefield",
            "The loaded CUDA library does not contain the fused block-INT8 "
            "wavefield implementation. Rebuild deepgpr.cu from the current source.",
        )
        if storage.int8_reduction_backend != "current":
            require_capability(
                c_lib,
                "deepgpr_supports_int8_reduction_backends",
                "The loaded CUDA library does not contain selectable INT8 "
                "reduction backends. Rebuild deepgpr.cu.",
            )


def _allocate_final_frame(config: SolverConfig) -> torch.Tensor:
    """Device buffer for the final E^nt frame of an E-only history.

    It closes the last ``E^(n+1) - E^n`` pair. The frame is kept inside the
    autograd context (never the returned state tensors, which callers may
    advance in place) and stays on the device even with async offload.
    """
    storage = config.storage
    shape = saved_history_shape(config.mode, 1, config.nstep, *config.history_cells)
    if storage.compression == "int8":
        assert storage.block_size is not None
        packed_bytes = int8_history_layout(shape, storage.block_size)["packed_bytes"]
        return torch.empty(packed_bytes, device=config.device, dtype=torch.int8)
    return torch.empty(shape, device=config.device, dtype=storage.dtype)


def _allocate_histories(
    config: SolverConfig,
    needs_model_gradient: bool,
    use_async_offload: bool,
    reconstruct_rhs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """Allocate ``E_saved`` and ``R_saved`` for the configured storage mode.

    ``R_saved`` is only materialised when a material gradient is requested.
    With ``reconstruct_rhs`` it is the one-frame final-E buffer instead of an
    R history.
    """
    device = config.device
    storage = config.storage
    nt_saved = saved_time_steps(config.nt, config.sampling_interval)
    shape = saved_history_shape(config.mode, nt_saved, config.nstep, *config.history_cells)

    if not config.save_wavefield_history:
        empty = torch.empty(0, device=device, dtype=storage.dtype)
        return empty, torch.empty(0, device=device, dtype=storage.dtype), shape
    if reconstruct_rhs and needs_model_gradient:
        final_frame = _allocate_final_frame(config)
    else:
        final_frame = None
    if storage.compression == "int8":
        assert storage.block_size is not None
        packed_bytes = int8_history_layout(shape, storage.block_size)["packed_bytes"]
        e_saved = torch.empty(packed_bytes, device=device, dtype=torch.int8)
        if final_frame is not None:
            return e_saved, final_frame, shape
        r_saved = (
            torch.empty(packed_bytes, device=device, dtype=torch.int8)
            if needs_model_gradient
            else torch.empty(0, device=device, dtype=torch.int8)
        )
        return e_saved, r_saved, shape
    if final_frame is not None:
        if use_async_offload:
            e_saved = torch.empty(shape, device="cpu", dtype=storage.dtype, pin_memory=True)
        else:
            e_saved = torch.empty(shape, device=device, dtype=storage.dtype)
        return e_saved, final_frame, shape
    if use_async_offload:
        e_saved = torch.empty(shape, device="cpu", dtype=storage.dtype, pin_memory=True)
        r_saved = (
            torch.empty(shape, device="cpu", dtype=storage.dtype, pin_memory=True)
            if needs_model_gradient
            else torch.empty(0, device="cpu", dtype=storage.dtype)
        )
        return e_saved, r_saved, shape
    e_saved = torch.empty(shape, device=device, dtype=storage.dtype)
    r_saved = (
        torch.empty(shape, device=device, dtype=storage.dtype)
        if needs_model_gradient
        else torch.empty(0, device=device, dtype=storage.dtype)
    )
    return e_saved, r_saved, shape


def _own_state_gradient(
    gradient: Optional[torch.Tensor], shape: torch.Size, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return a contiguous gradient buffer that the native adjoint may overwrite.

    The native backward advances the incoming state cotangents in place. They
    are therefore always copied (or freshly zero-allocated when autograd
    passes ``None``) so caller-owned tensors - ``grad_outputs`` passed to
    ``backward``, retained gradients, hook arguments - are never modified.
    """
    if gradient is None:
        return torch.zeros(shape, device=device, dtype=dtype)
    return gradient.clone(memory_format=torch.contiguous_format)


class DeepGPR(torch.autograd.Function):
    """Autograd function wrapping the native DeepGPR solvers.

    Call signature (via :meth:`apply`)::

        DeepGPR.apply(config, tensors, eps_r, sigma, source_amplitudes, *states)

    where ``states`` are the 30 state tensors (Ex, Ey, Ez, Hx, Hy, Hz and the
    24 CPML arrays). Returns the 30 advanced states followed by ``E_saved``
    and the receiver data ``(nstep, nt, nrx)``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        config: SolverConfig,
        tensors: SolverTensors,
        eps_r: torch.Tensor,
        sigma: torch.Tensor,
        source_amplitudes: torch.Tensor,
        *states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Run the native forward solver.

        Args:
            ctx: Autograd context.
            config: Scalar solve configuration.
            tensors: Materials, geometry and CPML coefficients.
            eps_r, sigma: Extended materials (only their ``requires_grad`` is
                read here; gradients flow back to them).
            source_amplitudes: ``(nsr, nt, 1)`` float32 waveforms.
            *states: The 30 state tensors, advanced in place.
        """
        if len(states) != STATE_TENSOR_COUNT:
            raise ValueError(f"Expected {STATE_TENSOR_COUNT} state tensors, got {len(states)}.")
        ctx.set_materialize_grads(False)
        device = config.device
        source_amplitudes = source_amplitudes.contiguous()
        source_location = tensors.source_location.to(device=device, dtype=torch.int32).contiguous()
        receiver_location = tensors.receiver_location.to(
            device=device, dtype=torch.int32
        ).contiguous()

        c_lib = get_deepgpr_lib(device)
        _check_forward_capabilities(c_lib, config)

        eps_r_requires_grad = bool(eps_r.requires_grad)
        sigma_requires_grad = bool(sigma.requires_grad)
        needs_model_gradient = eps_r_requires_grad or sigma_requires_grad
        use_async_offload = bool(config.use_async_offload and device.type == "cuda")
        reconstruct_rhs = bool(
            config.reconstruct_rhs and config.save_wavefield_history and needs_model_gradient
        )
        storage_type = encode_storage_type(
            config.storage,
            config.save_wavefield_history,
            rhs_from_e=reconstruct_rhs,
            physical_history=config.history_region == "physical",
            tm2d=config.tm2d_fast_path,
        )

        if config.save_wavefield_history:
            ctx.save_for_backward(
                tensors.mu_r_pad,
                source_location,
                receiver_location,
                *tensors.pml_coefficients,
                tensors.eps_r_pad,
                tensors.sigma_pad,
            )
        else:
            ctx.save_for_backward()

        ctx.config = config
        ctx.eps_r_requires_grad = eps_r_requires_grad
        ctx.sigma_requires_grad = sigma_requires_grad
        ctx.source_requires_grad = bool(source_amplitudes.requires_grad)
        ctx.source_shape = tuple(source_amplitudes.shape)
        ctx.storage_type = storage_type
        ctx.reconstruct_rhs = reconstruct_rhs
        ctx.use_async_offload = use_async_offload
        ctx.state_shapes = tuple(state.shape for state in states)
        ctx.receiver_shape = (config.nstep, config.nt, config.nrx)

        e_saved, r_saved, history_shape = _allocate_histories(
            config, needs_model_gradient, use_async_offload, reconstruct_rhs
        )
        receiver_amplitudes = torch.empty(
            (config.nstep, config.nt, config.nrx), device=device, dtype=config.dtype
        )

        arguments = build_forward_arguments(
            config,
            materials=(tensors.eps_r_pad, tensors.sigma_pad, tensors.mu_r_pad),
            pml_coefficients=tensors.pml_coefficients,
            states=states,
            update_coefficients=_allocate_update_coefficients(config),
            e_saved=e_saved,
            r_saved=r_saved,
            receiver_data=receiver_amplitudes,
            source_location=source_location,
            receiver_location=receiver_location,
            source_waveform=source_amplitudes,
            storage_type=storage_type,
            save_model_history=config.save_wavefield_history and needs_model_gradient,
            use_async_offload=use_async_offload,
        )

        with native_stream_scope(device):
            invoke(c_lib, "forward", config.fdtd_order, arguments)

        if config.debug:
            int8 = config.storage.compression == "int8"
            block = config.storage.block_size or (1, 1, 1)
            diagnostic_e = (
                decompress_wavefield_history(e_saved, history_shape, block)
                if int8 and config.save_wavefield_history
                else e_saved
            )
            r_shape = (
                saved_history_shape(config.mode, 1, config.nstep, *config.history_cells)
                if reconstruct_rhs
                else history_shape
            )
            diagnostic_r = (
                decompress_wavefield_history(r_saved, r_shape, block)
                if int8 and r_saved.numel()
                else r_saved
            )
            check_tensors_for_nan_inf(
                d="forward",
                E_saved=diagnostic_e,
                R_saved=diagnostic_r,
                **dict(zip(STATE_NAMES, states)),
            )
            check_nonzero_source_created_fields(c_lib, source_amplitudes, *states[:6])

        ctx.E_saved = e_saved
        # With reconstruct_rhs the second buffer is the internal final E frame.
        ctx.R_saved = None if reconstruct_rhs else r_saved
        ctx.E_final = r_saved if reconstruct_rhs else None
        ctx.mark_non_differentiable(e_saved)
        return (*states, e_saved, receiver_amplitudes)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Optional[torch.Tensor]):  # type: ignore[override]
        """Run the native adjoint solver and return the input gradients.

        Args:
            ctx: Autograd context saved by :meth:`forward`.
            *grad_outputs: 30 state cotangents, the (ignored) ``E_saved``
                cotangent and the receiver-data cotangent. ``None`` entries
                are treated as zeros.
        """
        config: SolverConfig = ctx.config
        if not config.save_wavefield_history:
            raise RuntimeError(
                "Forward wavefield history is disabled; adjoint/model-gradient "
                "backward is unavailable."
            )
        if ctx.E_saved is None:
            raise RuntimeError(
                "DeepGPR's saved forward wavefield was already released by a previous "
                "backward pass. Re-run DeepGPR.compute before calling backward again "
                "(retain_graph=True cannot reuse the native adjoint)."
            )

        device, dtype = config.device, config.dtype
        state_cotangents = grad_outputs[:STATE_TENSOR_COUNT]
        data_grad = grad_outputs[STATE_TENSOR_COUNT + 1]

        saved = tuple(tensor.contiguous() for tensor in ctx.saved_tensors)
        mu_r_pad, source_location, receiver_location = saved[:3]
        coefficients = saved[3:15]
        eps_r_pad, sigma_pad = saved[15:17]

        e_saved = ctx.E_saved.contiguous()
        r_saved = (ctx.E_final if ctx.reconstruct_rhs else ctx.R_saved).contiguous()
        lambdas = [
            _own_state_gradient(gradient, shape, device, dtype)
            for gradient, shape in zip(state_cotangents, ctx.state_shapes)
        ]
        data_grad = (
            torch.zeros(ctx.receiver_shape, device=device, dtype=dtype)
            if data_grad is None
            else data_grad.contiguous()
        )

        if ctx.eps_r_requires_grad:
            grad_eps_r = torch.zeros((config.nx, config.ny, config.nz), device=device, dtype=dtype)
        else:
            grad_eps_r = torch.empty(0, device=device, dtype=dtype)
        if ctx.sigma_requires_grad:
            grad_sigma = torch.zeros((config.nx, config.ny, config.nz), device=device, dtype=dtype)
        else:
            grad_sigma = torch.empty(0, device=device, dtype=dtype)
        if ctx.source_requires_grad:
            grad_source = torch.zeros(ctx.source_shape, device=device, dtype=dtype)
        else:
            grad_source = torch.empty(0, device=device, dtype=dtype)

        arguments = build_backward_arguments(
            config,
            materials=(eps_r_pad, sigma_pad, mu_r_pad),
            pml_coefficients=coefficients,
            lambdas=tuple(lambdas),
            update_coefficients=_allocate_update_coefficients(config),
            e_saved=e_saved,
            r_saved=r_saved,
            data_grad=data_grad,
            source_location=source_location,
            receiver_location=receiver_location,
            grad_source=grad_source,
            grad_eps_r=grad_eps_r,
            grad_sigma=grad_sigma,
            source_requires_grad=ctx.source_requires_grad,
            eps_r_requires_grad=ctx.eps_r_requires_grad,
            sigma_requires_grad=ctx.sigma_requires_grad,
            storage_type=ctx.storage_type,
            use_async_offload=ctx.use_async_offload,
        )

        c_lib = get_deepgpr_lib(device)
        with native_stream_scope(device):
            invoke(c_lib, "backward", config.fdtd_order, arguments)
        if config.tm2d_fast_path:
            # The 2D TM path never couples Ex, Ey, Hz (and their CPML
            # auxiliaries) back into Ez, Hx, Hy: their initial-state
            # cotangents are zero by construction.
            for index in TM2D_INACTIVE_STATE_INDICES:
                lambdas[index].zero_()

        # needs_input_grad: (config, tensors, eps_r, sigma, source, *states)
        state_needs_grad = ctx.needs_input_grad[5 : 5 + STATE_TENSOR_COUNT]
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

        # Release the (potentially very large) histories as early as possible.
        ctx.E_saved = None
        ctx.R_saved = None
        ctx.E_final = None

        return (
            None,
            None,
            grad_eps_r if ctx.eps_r_requires_grad else None,
            grad_sigma if ctx.sigma_requires_grad else None,
            grad_source if ctx.source_requires_grad else None,
            *(
                gradient if required else None
                for gradient, required in zip(lambdas, state_needs_grad)
            ),
        )


__all__ = [
    "DeepGPR",
    "STATE_GRADIENT_NAMES",
    "STATE_NAMES",
    "STATE_TENSOR_COUNT",
    "TM2D_INACTIVE_STATE_INDICES",
    "SolverConfig",
    "SolverTensors",
    "build_backward_arguments",
    "build_forward_arguments",
]

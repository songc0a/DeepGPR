"""High-level entry point: :func:`compute` (FDTD forward modelling with autograd)."""

from __future__ import annotations

import math
import warnings
from datetime import datetime
from typing import Any, Optional, Tuple

import torch

from ..config.constants import (
    FIELD_COMPONENTS,
    INT8_SPECIALISED_REDUCTION_VOLUME,
    SUPPORTED_FDTD_ORDERS,
    SUPPORTED_GRADIENT_MODES,
)
from ..config.defaults import (
    DEFAULT_CONVERSION_BACKEND,
    DEFAULT_FDTD_ORDER,
    DEFAULT_GRADIENT_MODE,
    DEFAULT_GRADIENT_SAMPLING_INTERVAL,
    DEFAULT_INT8_REDUCTION_BACKEND,
    DEFAULT_PML_THICKNESS,
    DEFAULT_RECEIVER_COMPONENT,
    DEFAULT_SOURCE_COMPONENT,
    DEFAULT_WAVEFIELD_COMPRESSION,
    DEFAULT_WAVEFIELD_RHS_HISTORY,
)
from ..postprocessing.io import normalize_forward_wavefield_directory, save_forward_wavefield
from ..preprocessing.grid import normalize_grid_spacing
from ..preprocessing.model_setup import PreparedModel, _shift_locations, initialization
from ..native.loader import get_deepgpr_lib, library_supports
from ..utils.logger import get_logger
from ..utils.validators import require_bool
from .autograd import DeepGPR, SolverConfig, SolverTensors
from .fields import create_or_separate
from .memory import format_compute_preview
from .pml import build_pml_coeffs, build_pml_phi
from .sampling import AUTO_SAMPLING_INTERVAL, recommended_sampling_interval, source_max_frequency
from .storage import (
    WavefieldStorageConfig,
    normalize_compression_block_size,
    normalize_int8_reduction_backend,
    normalize_wavefield_compression,
    normalize_wavefield_conversion_backend,
    normalize_wavefield_rhs_history,
    normalize_wavefield_storage_dtype,
    resolve_rhs_reconstruction,
    saved_history_shape,
    saved_time_steps,
)

_LOGGER = get_logger(__name__)


def _resolve_alias(name: str, value: Any, alias_name: str, alias_value: Any) -> Any:
    """Merge a keyword with its deprecated alias, rejecting double specification."""
    if value is not None and alias_value is not None:
        raise TypeError(f"Specify only one of {name} and its deprecated alias {alias_name}.")
    return alias_value if value is None else value


def _resolve_storage_options(
    device: torch.device,
    storage_dtype: Any,
    compression: Any,
    conversion_backend: Any,
    int8_reduction_backend: Any,
    compression_block_size: Any,
    compression_rate: Any,
    save_wavefield_history: bool,
    use_async_offload: bool,
    has_output_directory: bool,
) -> Tuple[torch.dtype, str, str, str]:
    """Validate the history-storage options (everything but the block size).

    Returns:
        ``(storage_dtype, compression, conversion_backend, int8_reduction_backend)``
        with ``"auto"`` conversion already resolved.
    """
    storage_dtype = normalize_wavefield_storage_dtype(storage_dtype)
    compression = normalize_wavefield_compression(compression)
    conversion_backend = normalize_wavefield_conversion_backend(conversion_backend)
    int8_reduction_backend = normalize_int8_reduction_backend(int8_reduction_backend)
    if conversion_backend == "auto":
        conversion_backend = (
            "native_scalar"
            if device.type == "cuda" and compression == "none" and storage_dtype == torch.float16
            else "legacy"
        )
    if compression == "zfp":
        raise NotImplementedError(
            "wavefield_compression='zfp' requires a block-level CUDA decoder fused "
            "with DeepGPR's material-gradient kernel. No such optional backend is "
            "compiled in this source tree; the GPU-native fused 'int8' mode remains "
            "available without third-party dependencies."
        )
    if compression_rate is not None:
        raise ValueError(
            "wavefield_compression_rate is only meaningful for the optional ZFP backend."
        )
    if not save_wavefield_history and use_async_offload:
        raise ValueError(
            "use_async_offload=True is incompatible with "
            "save_wavefield_history=False because there is no history to offload."
        )
    if not save_wavefield_history and has_output_directory:
        raise ValueError("save_forward_wavefield_path requires save_wavefield_history=True.")
    if compression == "int8":
        if device.type != "cuda":
            raise ValueError("wavefield_compression='int8' is CUDA-only and GPU-resident.")
        if storage_dtype != torch.float32:
            raise ValueError(
                "wavefield_storage_dtype must remain float32 when "
                "wavefield_compression='int8'; the compressed history format is "
                "selected independently by wavefield_compression."
            )
        if use_async_offload:
            raise ValueError(
                "use_async_offload=True is incompatible with GPU-resident INT8 "
                "wavefield compression."
            )
    elif compression_block_size is not None:
        raise ValueError(
            "wavefield_compression_block_size is only valid when wavefield_compression='int8'."
        )
    if compression != "int8" and int8_reduction_backend not in ("auto", "current"):
        raise ValueError(
            "int8_reduction_backend is only selectable when wavefield_compression='int8'."
        )
    if conversion_backend != "legacy":
        if device.type != "cuda":
            raise ValueError("Native wavefield conversion backends are CUDA-only.")
        if compression != "none" or storage_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "wavefield_conversion_backend is only selectable for uncompressed "
                "CUDA float16/bfloat16 histories."
            )
    return storage_dtype, compression, conversion_backend, int8_reduction_backend


def _resolve_int8_block(
    compression: str, block_size: Any, spatial_mode: int, reduction_backend: str
) -> Tuple[Optional[Tuple[int, int, int]], str]:
    """Resolve the INT8 block and the ``"auto"`` reduction backend."""
    resolved_block = (
        normalize_compression_block_size(block_size, spatial_mode)
        if compression == "int8"
        else None
    )
    if reduction_backend == "auto":
        reduction_backend = (
            "cub_block"
            if compression == "int8"
            and math.prod(resolved_block) == INT8_SPECIALISED_REDUCTION_VOLUME  # type: ignore[arg-type]
            else "current"
        )
    if (
        compression == "int8"
        and reduction_backend != "current"
        and math.prod(resolved_block) != INT8_SPECIALISED_REDUCTION_VOLUME  # type: ignore[arg-type]
    ):
        raise ValueError(
            "The cub_block and warp_shuffle INT8 reduction experiments require "
            "a compression block containing exactly 64 voxels."
        )
    return resolved_block, reduction_backend


def _resolve_sampling_interval(
    requested: Any, automatic: bool, source_amplitudes: torch.Tensor, dt: Any, report: bool
) -> int:
    """Resolve ``"auto"`` and report intervals relative to the empirical bound.

    An explicit interval above ``floor(1 / (4 f_max dt))`` raises a
    :class:`RuntimeWarning` that states the bound; intervals inside it (and the
    automatic choice) are only logged at INFO level.

    Args:
        requested: Positive ``int`` or ``"auto"`` (already validated).
        automatic: Whether ``requested`` is ``"auto"``.
        source_amplitudes: ``(nsr, nt, 1)`` source waveforms.
        dt: Time step [s].
        report: Whether a material gradient will use the history.
    """
    if not automatic and (requested == 1 or not report):
        return int(requested)
    max_frequency = source_max_frequency(source_amplitudes, dt)
    bound = recommended_sampling_interval(source_amplitudes, dt)
    if automatic:
        if report:
            _LOGGER.info(
                "model_gradient_sampling_interval='auto' selected %d "
                "(source f_max=%.4e Hz, dt=%.4e s).",
                bound,
                max_frequency,
                float(dt),
            )
        return bound
    if requested > bound:
        warnings.warn(
            f"model_gradient_sampling_interval={requested} exceeds the recommended bound "
            f"{bound} = floor(1 / (4 f_max dt)) for the source spectrum (f_max="
            f"{max_frequency:.4e} Hz, dt={float(dt):.4e} s). The temporal-sampling gradient "
            "approximation degrades quickly above this bound; use "
            "model_gradient_sampling_interval='auto' or an interval <= the bound.",
            RuntimeWarning,
            stacklevel=3,
        )
    else:
        _LOGGER.info(
            "model_gradient_sampling_interval=%d uses the weighted temporal-sampling "
            "approximation (recommended bound %d for source f_max=%.4e Hz).",
            requested,
            bound,
            max_frequency,
        )
    return int(requested)


def compute(
    device,
    dx=None,
    dt=None,
    source_amplitudes=None,
    source_location=None,
    receiver_location=None,
    er=None,
    se=None,
    mr=None,
    E=None,
    H=None,
    PML=None,
    pmlthick=DEFAULT_PML_THICKNESS,
    source_direction=DEFAULT_SOURCE_COMPONENT,
    reciever_direction=DEFAULT_RECEIVER_COMPONENT,
    model_gradient_sampling_interval=DEFAULT_GRADIENT_SAMPLING_INTERVAL,
    wavefield_storage_dtype=torch.float32,
    use_async_offload=False,
    fdtd_order=DEFAULT_FDTD_ORDER,
    mode=DEFAULT_GRADIENT_MODE,
    debug=False,
    print_parameters=False,
    save_forward_wavefield_path=None,
    *,
    eps_r=None,
    sigma=None,
    mu_r=None,
    receiver_component=None,
    save_wavefield_history=True,
    wavefield_compression=DEFAULT_WAVEFIELD_COMPRESSION,
    wavefield_compression_block_size=None,
    wavefield_conversion_backend=DEFAULT_CONVERSION_BACKEND,
    int8_reduction_backend=DEFAULT_INT8_REDUCTION_BACKEND,
    wavefield_compression_rate=None,
    wavefield_rhs_history=DEFAULT_WAVEFIELD_RHS_HISTORY,
):
    """Run 2D/3D FDTD forward modelling of Maxwell's equations with autograd.

    The call is differentiable with respect to ``eps_r``, ``sigma``,
    ``source_amplitudes`` and the initial states ``E``, ``H`` and ``PML``.
    Gradients are computed by the native exact discrete adjoint.

    Args:
        device: ``"cpu"``, ``"cuda"``, ``"cuda:1"`` or a ``torch.device``.
        dx: Scalar grid spacing or ``(dx, dy, dz)`` [m].
        dt: Time step [s]; must satisfy the CFL limit.
        source_amplitudes: ``(nwaveforms, nt, 1)`` (or ``(nwaveforms, nt)``);
            ``nwaveforms`` is 1 (shared) or the number of sources per shot.
        source_location: ``(nstep, nsr, 3)`` integer coordinates in the
            *physical* model frame.
        receiver_location: ``(nstep, nrx, 3)`` integer coordinates.
        er: Deprecated alias of ``eps_r``.
        se: Deprecated alias of ``sigma``.
        mr: Deprecated alias of ``mu_r``.
        E: Optional initial ``(Ex, Ey, Ez)`` on the full solver grid.
        H: Optional initial ``(Hx, Hy, Hz)``.
        PML: Optional 24 CPML auxiliary tensors. ``E``/``H``/``PML`` returned
            by ``compute`` or :func:`checkpoint_initial_field` must be passed
            back unchanged; never crop or re-pad them. The solver advances them
            **in place**; clone them if you still need the old values.
        pmlthick: External CPML thickness: an int, four values
            ``[x0, xm, y0, ym]`` or six values ``[x0, xm, y0, ym, z0, zm]``.
            z faces must be zero in 2D.
        source_direction: Source polarisation (0: Ex, 1: Ey, 2: Ez).
        reciever_direction: Deprecated alias of ``receiver_component``.
        model_gradient_sampling_interval: Store every n-th forward step for
            the gradient; values > 1 give an approximate gradient. ``"auto"``
            selects ``floor(1 / (4 f_max dt))`` from the source spectrum (see
            :func:`DeepGPR.recommended_sampling_interval`); explicit values
            above that bound raise a :class:`RuntimeWarning`.
        wavefield_storage_dtype: History dtype: float32, float16, bfloat16
            (or ``"fp32"``, ``"fp16"``, ``"bf16"``, ...).
        use_async_offload: CUDA only: offload histories to pinned host memory.
        fdtd_order: Spatial order 2, 4 or 8.
        mode: Gradient mode: 2 (Ez only, 2D TM) or 3 (Ex, Ey, Ez).
        debug: Run NaN/Inf validation after the native calls.
        print_parameters: Print a configuration and memory report first.
        save_forward_wavefield_path: Directory where ``E_saved`` is written as
            ``forward_wavefield_HH-MM.pt``; ``None`` disables file output.
        eps_r: Relative permittivity of the physical model (air + target),
            ``(nx, ny)`` or ``(nx, ny, nz)``. The CPML is added internally;
            material gradients have exactly the input shape.
        sigma: Electrical conductivity [S/m], same shape as ``eps_r``.
        mu_r: Relative permeability or ``None`` (ones). Not differentiable.
        receiver_component: Recorded component (0: Ex, 1: Ey, 2: Ez).
        save_wavefield_history: ``False`` skips the history (forward only,
            much less memory); backward then raises.
        wavefield_compression: ``"none"`` or CUDA block ``"int8"``.
        wavefield_compression_block_size: INT8 block; default ``(8, 8)`` in
            2D and ``(4, 4, 4)`` in 3D.
        wavefield_conversion_backend: CUDA FP16/BF16 conversion: ``"auto"``,
            ``"legacy"``, ``"native_scalar"`` or ``"native_vec2"``.
        int8_reduction_backend: CUDA INT8 reduction: ``"auto"``,
            ``"current"``, ``"cub_block"`` or ``"warp_shuffle"``.
        wavefield_compression_rate: Reserved for an optional ZFP backend.
        wavefield_rhs_history: ``"auto"`` (default) stores only the E history
            for float32 storage with sampling interval 1 and rebuilds
            ``R^n = (E^(n+1) - ca E^n) / cb`` in the adjoint (half the history
            memory; gradients unchanged up to rounding). Low-precision and INT8
            histories keep the E+R pair. ``"stored"`` always keeps E+R;
            ``"reconstructed"`` requests the E-only history for any storage
            (sampling interval 1).

    Returns:
        ``(E_saved, (Ex, Ey, Ez), (Hx, Hy, Hz), PML, receiver_amplitudes)``
        where ``receiver_amplitudes`` has shape ``(nstep, nt, nrx)``.

    Raises:
        TypeError, ValueError, NotImplementedError: For invalid arguments.
        DeepGPR.utils.exceptions.NativeLibraryError: For missing/stale native
            libraries or capabilities.
    """
    eps_r = _resolve_alias("eps_r", eps_r, "er", er)
    sigma = _resolve_alias("sigma", sigma, "se", se)
    mu_r = _resolve_alias("mu_r", mu_r, "mr", mr)
    receiver_component = reciever_direction if receiver_component is None else receiver_component

    device = torch.device(device)
    if fdtd_order not in SUPPORTED_FDTD_ORDERS:
        raise ValueError("fdtd_order must be one of 2, 4, or 8.")
    if mode not in SUPPORTED_GRADIENT_MODES:
        raise ValueError("mode must be 2 or 3.")
    require_bool("print_parameters", print_parameters)
    require_bool("save_wavefield_history", save_wavefield_history)
    output_directory = normalize_forward_wavefield_directory(save_forward_wavefield_path)
    run_time = datetime.now() if output_directory is not None else None
    automatic_sampling = (
        isinstance(model_gradient_sampling_interval, str)
        and model_gradient_sampling_interval.lower() == AUTO_SAMPLING_INTERVAL
    )
    if not automatic_sampling and (
        isinstance(model_gradient_sampling_interval, bool)
        or not isinstance(model_gradient_sampling_interval, int)
        or model_gradient_sampling_interval < 1
    ):
        raise ValueError("model_gradient_sampling_interval must be a positive integer or 'auto'.")
    (
        wavefield_storage_dtype,
        wavefield_compression,
        wavefield_conversion_backend,
        int8_reduction_backend,
    ) = _resolve_storage_options(
        device,
        wavefield_storage_dtype,
        wavefield_compression,
        wavefield_conversion_backend,
        int8_reduction_backend,
        wavefield_compression_block_size,
        wavefield_compression_rate,
        save_wavefield_history,
        use_async_offload,
        output_directory is not None,
    )
    wavefield_rhs_history = normalize_wavefield_rhs_history(wavefield_rhs_history)
    if source_direction not in FIELD_COMPONENTS or receiver_component not in FIELD_COMPONENTS:
        raise ValueError("source_direction and receiver_component must be 0, 1, or 2.")
    if getattr(mu_r, "requires_grad", False):
        raise NotImplementedError(
            "DeepGPR does not currently return relative-permeability gradients."
        )

    grid_spacing = normalize_grid_spacing(dx)
    mu_r_supplied = mu_r is not None
    prepared = PreparedModel(
        *initialization(
            device,
            eps_r,
            sigma,
            mu_r,
            source_amplitudes,
            source_location,
            receiver_location,
            grid_spacing,
            dt,
            pmlthick,
            fdtd_order,
        )
    )
    eps_r, sigma, mu_r_pad = prepared.er, prepared.se, prepared.mr
    nx, ny, nz, nt = prepared.nx, prepared.ny, prepared.nz, prepared.nt
    nstep, nsr, nrx = prepared.nstep, prepared.nsr, prepared.nrx
    pmlthick = prepared.pmlthick
    source_amplitudes = prepared.source_amplitudes
    time_step = float(dt)

    source_location = _shift_locations(source_location, pmlthick, device)
    receiver_location = _shift_locations(receiver_location, pmlthick, device)

    compression_block_size, int8_reduction_backend = _resolve_int8_block(
        wavefield_compression,
        wavefield_compression_block_size,
        prepared.mode,
        int8_reduction_backend,
    )

    needs_model_gradient = eps_r.requires_grad or sigma.requires_grad
    model_gradient_sampling_interval = _resolve_sampling_interval(
        model_gradient_sampling_interval,
        automatic_sampling,
        source_amplitudes,
        dt,
        save_wavefield_history and needs_model_gradient,
    )
    stores_model_history = save_wavefield_history and needs_model_gradient
    lossless_storage = (
        wavefield_compression == "none" and wavefield_storage_dtype == torch.float32
    )
    probe_library = (
        stores_model_history
        and model_gradient_sampling_interval == 1
        and (
            wavefield_rhs_history == "reconstructed"
            or (wavefield_rhs_history == "auto" and lossless_storage)
        )
    )
    reconstruct_rhs = resolve_rhs_reconstruction(
        wavefield_rhs_history,
        sampling_interval=model_gradient_sampling_interval,
        stores_model_history=stores_model_history,
        library_supported=probe_library
        and library_supports(get_deepgpr_lib(device), "deepgpr_supports_rhs_reconstruction"),
        lossless_storage=lossless_storage,
    )
    if (
        save_wavefield_history
        and needs_model_gradient
        and mode == 2
        and (prepared.mode != 2 or source_direction != 2 or receiver_component != 2)
    ):
        raise ValueError(
            "mode=2 is an exact model-gradient mode only for 2D Ez-TM modeling. "
            "Use mode=3 for 3D or other electric-field components."
        )

    if print_parameters:
        print(
            format_compute_preview(
                device=device,
                grid_spacing=grid_spacing,
                dt=time_step,
                nx=nx,
                ny=ny,
                nz=nz,
                nt=nt,
                nstep=nstep,
                nsr=nsr,
                nrx=nrx,
                source_amplitudes=source_amplitudes,
                source_location=source_location,
                receiver_location=receiver_location,
                er=eps_r,
                se=sigma,
                mr=mu_r_pad,
                mr_supplied=mu_r_supplied,
                pmlthick=pmlthick,
                source_direction=source_direction,
                receiver_component=receiver_component,
                model_gradient_sampling_interval=model_gradient_sampling_interval,
                wavefield_storage_dtype=wavefield_storage_dtype,
                wavefield_conversion_backend=wavefield_conversion_backend,
                int8_reduction_backend=int8_reduction_backend,
                wavefield_compression=wavefield_compression,
                wavefield_compression_block_size=compression_block_size,
                save_wavefield_history=save_wavefield_history,
                reconstruct_rhs=reconstruct_rhs,
                use_async_offload=use_async_offload,
                fdtd_order=fdtd_order,
                mode=mode,
                debug=debug,
                save_forward_wavefield_path=output_directory,
                E=E,
                H=H,
                PML=PML,
            )
        )

    electric = create_or_separate(E, nx, ny, nz, nstep, device, prepared.dtype)
    magnetic = create_or_separate(H, nx, ny, nz, nstep, device, prepared.dtype)
    # ``dt`` is forwarded exactly as the caller supplied it (not ``float(dt)``)
    # so the CPML coefficient arithmetic is unchanged for tensor/NumPy inputs.
    pml_arrays = build_pml_coeffs(
        eps_r, mu_r_pad, dt, grid_spacing, nx, ny, nz, pmlthick, device, prepared.dtype
    )
    descriptors, coefficients = pml_arrays[:6], pml_arrays[6:]
    phi = build_pml_phi(*descriptors, nstep, PML, device)

    config = SolverConfig(
        device=device,
        dtype=prepared.dtype,
        grid_spacing=grid_spacing,
        dt=time_step,
        nx=nx,
        ny=ny,
        nz=nz,
        nt=nt,
        nstep=nstep,
        nsr=nsr,
        nrx=nrx,
        pmlthick=pmlthick,
        source_direction=source_direction,
        receiver_component=receiver_component,
        sampling_interval=model_gradient_sampling_interval,
        storage=WavefieldStorageConfig(
            dtype=wavefield_storage_dtype,
            compression=wavefield_compression,
            block_size=compression_block_size,
            conversion_backend=wavefield_conversion_backend,
            int8_reduction_backend=int8_reduction_backend,
        ),
        save_wavefield_history=save_wavefield_history,
        use_async_offload=bool(use_async_offload),
        fdtd_order=fdtd_order,
        mode=mode,
        debug=bool(debug),
        reconstruct_rhs=reconstruct_rhs,
    )
    tensors = SolverTensors(
        mu_r_pad=mu_r_pad,
        eps_r_pad=prepared.ere,
        sigma_pad=prepared.see,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_descriptors=tuple(descriptors),
        pml_coefficients=tuple(coefficients),
    )
    _LOGGER.debug(
        "compute: device=%s order=%d mode=%d grid=(%d, %d, %d) nt=%d shots=%d "
        "history=%s storage=%s compression=%s",
        device,
        fdtd_order,
        mode,
        nx,
        ny,
        nz,
        nt,
        nstep,
        save_wavefield_history,
        wavefield_storage_dtype,
        wavefield_compression,
    )

    outputs = DeepGPR.apply(
        config, tensors, eps_r, sigma, source_amplitudes, *electric, *magnetic, *phi
    )
    states, e_saved, receiver_amplitudes = outputs[:30], outputs[30], outputs[31]

    if output_directory is not None:
        metadata = None
        if save_wavefield_history and wavefield_compression == "int8":
            nt_saved = saved_time_steps(nt, model_gradient_sampling_interval)
            metadata = {
                "compression": "int8",
                "block_size": compression_block_size,
                "uncompressed_shape": saved_history_shape(mode, nt_saved, nstep, nx, ny, nz),
            }
        save_forward_wavefield(e_saved, output_directory, run_time, metadata=metadata)

    return (
        e_saved,
        tuple(states[0:3]),
        tuple(states[3:6]),
        tuple(states[6:30]),
        receiver_amplitudes,
    )


__all__ = ["compute"]

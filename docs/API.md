<!-- Moved from README.md in 0.1.0; kept up to date with the [Unreleased] changes in CHANGELOG.md. -->

# `DeepGPR.compute` API reference

`compute` is a core function for 3D/2D Finite-Difference Time-Domain (FDTD) forward modeling, primarily designed for Ground Penetrating Radar (GPR) and electromagnetic wave propagation. It fully supports backpropagation (e.g., for Full Waveform Inversion, FWI) utilizing PyTorch's `autograd` engine.

## Function signature

```python
def compute(device, dx=None, dt=None,
            source_amplitudes=None,
            source_location=None,
            receiver_location=None,
            er=None, se=None, mr=None,
            E=None, H=None, PML=None,
            pmlthick=10, source_direction=2, reciever_direction=2,
            model_gradient_sampling_interval=1,
            wavefield_storage_dtype=torch.float32,
            use_async_offload=False,
            fdtd_order=2,
            mode=2,
            debug=False,
            print_parameters=False,
            save_forward_wavefield_path=None,
            *, eps_r=None, sigma=None, mu_r=None, receiver_component=None,
            save_wavefield_history=True,
            wavefield_compression="none",
            wavefield_compression_block_size=None,
            wavefield_conversion_backend="auto",
            int8_reduction_backend="auto",
            wavefield_compression_rate=None,
            wavefield_rhs_history="auto",
            wavefield_history_region="extended"):
```

`eps_r`, `sigma`, `mu_r` and `receiver_component` are the canonical names;
`er`, `se`, `mr` and `reciever_direction` remain accepted as deprecated aliases.
The implementation lives in `DeepGPR/solver/modeling.py`; the defaults are
defined in `DeepGPR/config/defaults.py`.

Set `print_parameters=True` on an ordinary call to print the normalized
configuration and memory estimate immediately before the native solver starts:

```python
result = DeepGPR.compute(
    device="cuda:0",
    # Other model, source, and acquisition arguments...
    print_parameters=True,
)
```

## Input parameters
### 1. Basic Physics & Grid Parameters

| Parameter | Data Type | Description |
| :--- | :--- | :--- |
| **`device`** | `torch.device` / `str` | PyTorch computation device, e.g., `'cuda:0'` or `'cpu'`. CUDA loads `deepgpr.so/.dll`; CPU loads `deepgpr_cpu.so/.dll/.dylib`. |
| **`dx`** | `float` / 3-value `list`, `tuple`, or `Tensor` | Grid spacing in meters. A scalar uses an isotropic grid ($dx = dy = dz$). Three values specify independent $(dx, dy, dz)$ spacings for the finite differences, CFL condition, CPML coefficients, and source scaling. |
| **`dt`** | `float` | Time step size. It is checked against a material-aware CFL limit that includes the selected 2/4/8-order stencil. Typically in seconds (s). |
| **`fdtd_order`** | `int` | Spatial finite-difference order used by the FDTD field updates. Supported values are `2`, `4`, and `8`; default is `2` for compatibility with earlier versions. |
| **`mode`** | `int` | FWI gradient mode. `2` keeps the previous Ez-only model-gradient calculation. `3` uses Ex, Ey, and Ez electric-field contributions for relative permittivity and conductivity gradients. |
| **`debug`** | `bool` | Runs expensive NaN/Inf and zero-field validation checks when `True`. Backward validation covers material, source, and initial-state gradients actually requested by autograd. The default `False` keeps these checks disabled for faster production runs. |
| **`print_parameters`** | `bool` | Prints a preflight summary before native FDTD execution. The summary includes all simulation options (including the resolved R-history mode and history region) and a tensor-payload memory estimate for model/state tensors, CPML, saved `E_saved`/`R_saved` wavefields (or the E-only history and its final frame), receiver buffers, gradients, low-precision snapshots, and CUDA offload buffers. CPU and CUDA estimates are reported separately with a 20% capacity margin. The same estimate is available as `DeepGPR.estimate_compute_memory(...)` (keyword arguments include `reconstruct_rhs` and `history_region`). |
| **`save_forward_wavefield_path`** | `str` / path-like / `None` | Directory used to save `E_saved` after a successful forward run. The default `None` performs no file I/O. Files use the local 24-hour start time, for example `forward_wavefield_14-35.pt`; a numeric suffix prevents overwriting when multiple runs start in the same minute. |
### 2. Medium Model Parameters

Supply **only the physical model, including any air layer and the target region**. Do not include PML in `eps_r`, `sigma`, or `mu_r`. For 2D simulations use `(nx, ny)` or `(nx, ny, 1)`.

Every `compute` call replicates the current material values at each boundary outward by `pmlthick`. For `[px0, px1, py0, py1, pz0, pz1]`, the internal grid is `Nx = nx + px0 + px1`, `Ny = ny + py0 + py1`, `Nz = nz + pz0 + pz1`. In 2D, `Nz = 1`. Sources and receivers use zero-based **input model coordinates**; the solver adds `[px0, py0, pz0]` internally without modifying your tensors. PML thickness can exceed the physical model size.

After `loss.backward()`, `eps_r.grad` and `sigma.grad` have exactly their respective input shapes, including air cells. Update these physical tensors with your external FWI optimizer; the next call regenerates PML from the updated model. If air must remain fixed, apply your physical air mask in the optimizer.

**Migration:** remove old manually padded PML cells and subtract the old low-face padding from acquisition indices. Keep actual air layers. Existing states generated on a different computational grid must be regenerated.

| Parameter | Data Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| **`eps_r`** (`er`) | `Tensor` (float) | `(nx, ny, nz)` or `(nx, ny)` | Relative permittivity ($\epsilon_r$). Values must be $\ge 1$. `er` is the deprecated alias. |
| **`sigma`** (`se`) | `Tensor` (float) | `(nx, ny, nz)` or `(nx, ny)` | Electrical conductivity ($\sigma$). Values must be non-negative. `se` is the deprecated alias. |
| **`mu_r`** (`mr`) | `Tensor` (float) | `(nx, ny, nz)` or `(nx, ny)` | Relative permeability ($\mu_r$). Optional; defaults to 1 for the entire space. `mr` is the deprecated alias. |

> **Dimension Key**: `nx`, `ny`, and `nz` represent the number of grid cells along the X, Y, and Z axes, respectively.

### 3. Source & Receiver Setup

This section defines the geometric observation system (coordinates) and the excitation waveforms.

| Parameter | Data Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| **`source_amplitudes`** | `Tensor` (float) | `(num_waveforms, nt, 1)` | Source excitation waveforms. `nt` is the total number of time steps.<br>- If `num_waveforms == 1`: All sources share this single waveform.<br>- If `num_waveforms == nsr`: Each source uses its corresponding waveform. |
| **`source_location`** | `Tensor` (int) | `(nstep, nsr, 3)` | Grid coordinate indices of the sources in the unextended physical model.<br>The last dimension corresponds to `[x_idx, y_idx, z_idx]`. |
| **`receiver_location`** | `Tensor` (int) | `(nstep, nrx, 3)` | Grid coordinate indices of the receivers in the unextended physical model.<br>The last dimension corresponds to `[x_idx, y_idx, z_idx]`. |
| **`source_direction`** | `int` | Scalar | Polarization direction/component of the source excitation.<br>`0` = X, `1` = Y, `2` = Z (e.g., exciting $E_z$). |
| **`receiver_component`** (`reciever_direction`) | `int` | Scalar | The component recorded by the receivers.<br>`0` = $E_x$, `1` = $E_y$, `2` = $E_z$. The misspelled name remains as a deprecated alias. |

> **Core Shape Definitions**:
> *   `nstep`: Number of shots/batches (independent simulation tasks running in parallel).
> *   `nsr`: Number of **sources** per single simulation.
> *   `nrx`: Number of **receivers** per single simulation.
> *   `nt`: Total number of time steps to simulate.

### 4. Boundary Conditions & Optimization

| Parameter | Data Type | Format | Description |
| :--- | :--- | :--- | :--- |
| **`pmlthick`** | `int` / `list` / `Tensor`| Scalar or list of 4/6 | External PML thickness in grid cells, added by edge replication.<br>- Integer `p`: All active boundaries have thickness `p` (no Z padding in 2D).<br>- `[x0, xm, y0, ym]`: X/Y faces with no Z PML.<br>- `[x0, xm, y0, ym, z0, zm]`: Six independent faces; Z values must be zero in 2D.<br>- Zero disables that face. |
| **`model_gradient_sampling_interval`**| `int` / `"auto"` | Scalar | Wavefield sampling interval during forward propagation (Default: 1).<br>A larger integer reduces VRAM use for `E_saved` and `R_saved`, but uses an explicitly approximate model gradient. The last incomplete sampling block is weighted by its actual length.<br>`"auto"` selects `floor(1 / (4 f_max dt))` from the source spectrum; an explicit value above that bound raises a `RuntimeWarning` stating the bound. See [4.4](#44-temporal-sampling-interval). |
| **`save_wavefield_history`** | `bool` | Scalar | Independently controls allocation and native writes of the E/R histories used by adjoint model-gradient backward (Default: `True`). `False` still executes the complete FDTD, CPML, source-injection, and receiver-recording path, returns an empty `E_saved`, and raises a clear error if history-dependent backward is attempted. |
| **`wavefield_storage_dtype`** | `torch.dtype` / `str` | `float32`, `float16`, or `bfloat16` | Storage format for saved `E_saved` and `R_saved` model-gradient wavefields. FDTD propagation remains float32. `float16` and `bfloat16` halve saved-wavefield memory at the cost of gradient accuracy; `bfloat16` has the safer dynamic range. String aliases such as `"fp16"` and `"bf16"` are accepted. |
| **`wavefield_conversion_backend`** | `str` | `"auto"`, `"legacy"`, `"native_scalar"`, or `"native_vec2"` | CUDA FP16/BF16 history conversion. The audited default `"auto"` uses NVIDIA scalar intrinsics for CUDA FP16 and the legacy path otherwise. Explicit values retain the correctness/performance A/B paths; vec2 is not the default because it was slower on RTX 4090. |
| **`wavefield_compression`** | `str` | `"none"`, `"int8"`, or `"zfp"` | `"int8"` enables CUDA-native per-block symmetric INT8 histories with FP32 scales and inline decode in the fused material-gradient kernel. `"none"` is the unchanged default. `"zfp"` is reserved for an optional fused CUDA decoder and is rejected by the current dependency-free build instead of silently materializing a decoded global-memory history. |
| **`wavefield_compression_block_size`** | sequence / `None` | 2D or 3D spatial block | INT8 block shape; defaults to `(8, 8)` in 2D and `(4, 4, 4)` in 3D. The block volume must be a power of two no larger than 256. Partial boundary blocks are supported. |
| **`int8_reduction_backend`** | `str` | `"auto"`, `"current"`, `"cub_block"`, or `"warp_shuffle"` | Tile maximum reduction. The audited default `"auto"` selects NVIDIA CUB `BlockReduce` for 64-voxel tiles and preserves the prior shared-memory tree for other valid tile sizes. Explicit values expose the retained A/B implementations. |
| **`wavefield_compression_rate`** | scalar / `None` | Optional ZFP setting | Reserved for an optional ZFP backend. It is rejected unless that backend is selected and available. |
| **`wavefield_history_region`** | `str` | `"extended"` or `"physical"` | Cells covered by the saved history (see [4.6](#46-physical-region-history)). `"extended"` (default) stores the whole grid including the external PML. `"physical"` stores only the physical model cells; material gradients are unchanged because CPML cells never receive one, and history memory falls in proportion to the cell count. |
| **`wavefield_rhs_history`** | `str` | `"auto"`, `"stored"`, or `"reconstructed"` | Storage of the right-hand-side history `R^n` used by the material gradient (see [4.5](#45-e-only-history-at-sampling-interval-1)). `"auto"` (default) stores only `E^n` for uncompressed `float32` histories with `model_gradient_sampling_interval=1` and rebuilds `R^n` in the adjoint, halving history memory with unchanged gradients; other storage modes keep `E^n` and `R^n`. `"stored"` always keeps both (the earlier history layout); `"reconstructed"` requests the E-only history for any storage mode and requires interval 1. |
| **`use_async_offload`** | `bool` | Scalar | CUDA-only VRAM optimization flag (Default: `False`).<br>If `True`, `E_saved` and `R_saved` are asynchronously offloaded to page-locked host memory (`pin_memory` CPU RAM). This reduces GPU VRAM consumption at the cost of PCIe transfers. On CPU this option is ignored. |

### 4.1 FWI Gradient Mode

`mode` only changes how the model gradients are accumulated during backpropagation:

- `mode=2` (default): Saves Ez in `E_saved` and computes relative permittivity/conductivity gradients from Ez only.
- `mode=3`: Saves Ex, Ey, and Ez in `E_saved` and computes relative permittivity/conductivity gradients from all three electric-field components. This is intended for complete 3D Maxwell FWI. The receiver component is not changed by this option.

When `eps_r` or `sigma` requires gradients, `mode=2` is restricted to 2D Ez-TM modeling; use `mode=3` for 3D gradients. Source-waveform gradients are supported. Gradients with respect to `mu_r` are not currently implemented and are rejected explicitly.

### 4.2 Discrete Adjoint Gradient

The backward solver applies the exact reverse-mode transpose of each executed operation in reverse order: receiver sampling, source injection, electric CPML, electric update, magnetic CPML, and magnetic update. The derivative transpose is applied to the material-weighted field cotangent, so heterogeneous update coefficients and anisotropic grid spacing are handled by the executed discrete operator. Every electric and magnetic CPML auxiliary state has a separate cotangent recurrence on all six faces.

CPML is treated as a fixed numerical boundary during backward. Boundary coefficient averages are detached, and gradients are cropped to the physical model; replicated PML sensitivities are not summed into model edges. All physical cells remain eligible for material gradients, including the first cell at a low face. PML materials and coefficients are rebuilt on the next forward call. Thus a finite-difference check must keep boundary values fixed (or freeze the extended PML and its coefficients explicitly); perturbing model edges and regenerating PML also changes a numerical boundary that this FWI gradient intentionally holds fixed.

The material-gradient formulation in DeepGPR was informed in part by the differentiable FDTD implementation in [TIDE](https://github.com/Vcholerae1/tide-GPR), particularly its treatment of the discrete Maxwell electric-field update in gradient computation. We gratefully acknowledge the TIDE project and its authors for this work.

Use `model_gradient_sampling_interval=1` and `wavefield_storage_dtype=torch.float32` for a directional derivative check in the physical model region. Temporal subsampling and lower-precision storage deliberately approximate the gradient. Run [the gradient-check notebook](../tests/03_gradient_2d.ipynb) after rebuilding the native ABI 6 libraries with `deepgpr_supports_external_pml`. See `tests/test_external_pml.py` for physical edge and checkpoint regression checks.

### 4.3 GPU-native block INT8 history

Only the sampled forward state used by the material adjoint is compressed. The
live Ex/Ey/Ez/Hx/Hy/Hz and CPML states remain float32. The executed discrete
update requires `E^n` and `R^n`, where `E^(n+1) = ca E^n + cb R^n`; consequently
`mode=2` stores compressed Ez/Rz and `mode=3` stores compressed Ex/Ey/Ez and all
three corresponding RHS components. With `wavefield_rhs_history="reconstructed"`
only the E components are compressed (plus one packed final frame) and the
fused kernel decodes `E^n` and `E^(n+1)` to rebuild `R^n`. Magnetic histories
are not stored.

The packed tensor contains a contiguous signed-INT8 payload followed by a
four-byte-aligned contiguous FP32 scale array. Backward maps one CUDA block to
one compression tile, loads each E/R scale once into shared memory, decodes each
value in a register, and immediately accumulates the epsilon/conductivity
gradient. It does not allocate or write a reconstructed global-memory history.
`use_async_offload=True`, CPU execution, and a non-float32
`wavefield_storage_dtype` are explicitly incompatible with `"int8"`.

The current shared-memory maximum reduction remains available as
`int8_reduction_backend="current"`. The RTX 4090 audit selected CUB
`BlockReduce` for the default 64-voxel tiles; `"auto"` falls back to the current
tree for other supported power-of-two tile volumes.

`E_saved` is an opaque one-dimensional `torch.int8` packed tensor in this mode.
For diagnostics only, reconstruct it with
`DeepGPR.decompress_wavefield_history(E_saved, original_shape, block_size)`.
This helper materializes FP32 and is never called by autograd backward.

### 4.4 Temporal sampling interval

`model_gradient_sampling_interval = S > 1` stores every `S`-th forward step and
weights it by the length of its sampling block. This is an **approximation** of
the exact discrete gradient (`S = 1`). Its error stays small while the history
is sampled at least four times per period of the highest frequency in the
forward wavefield, giving the empirical bound

`S <= floor(1 / (4 f_max dt))`,

where `f_max` is the highest frequency at which the amplitude spectrum of the
source waveform still exceeds `SOURCE_SPECTRUM_CUTOFF_FRACTION = 3e-3` of its
peak (about `3 f_peak` for a Ricker wavelet). Several waveforms use the most
conservative (smallest) value.

- `model_gradient_sampling_interval="auto"` uses this bound (logged at INFO
  level). The default remains `1`.
- An explicit `S > 1` inside the bound is logged at INFO level; an explicit `S`
  above the bound raises a `RuntimeWarning` that states the bound.
- Segmented or checkpointed runs should evaluate
  `DeepGPR.recommended_sampling_interval(full_source_waveform, dt)` once on the
  complete waveform and pass that integer to every segment: a short segment of
  a waveform has a different spectrum. `DeepGPR.source_max_frequency` returns
  the underlying `f_max`.

Measured relative L2 error of the ε gradient against `S = 1` (FP32):

| configuration | loss | S=5 | S=8 (auto) | S=10 | S=16 | S=20 |
|---|---|---:|---:|---:|---:|---:|
| `examples/2.2DFWI.ipynb` (200 MHz Ricker, dt=5e-11; CPU) | MSE | 9e-6 | 1.4e-5 | 2.8e-5 | 4.9e-4 | 5.8e-3 |
| same | L1 | 8e-4 | 2.2e-3 | 3.4e-3 | 4.6e-2 | 0.11 |
| 90x60 two-layer model, surface antennas, nt=700 (CPU) | MSE | 3.7e-5 | 2.4e-4 | 1.8e-4 | 3.0e-2 | 3.4e-2 |
| same | L1 | 1.7e-3 | 3.5e-3 | 6.5e-3 | 0.18 | 0.16 |

The error grows quickly once `S` exceeds the bound. It is concentrated within a
few cells of the sources and receivers, where the wavefield contains the most
high-frequency energy: in the two-layer case above, cells farther than six
cells from every antenna carry a relative error of 4e-7 (MSE) and 3e-5 (L1) at
`S = 8`; in a small 3D `mode=3` case at `S = 8` the L1 gradient error is 29 %
over the whole model but 1.8e-6 beyond six cells from the antennas. The bound looks only at the **forward source spectrum**. The adjoint
source of a non-smooth misfit such as L1 (`sign(residual)`) is much broader
band, which is why the L1 rows are one to two orders of magnitude worse at the
same `S`. Use `S = 1` with `float32` storage for gradient checks.

### 4.5 E-only history at sampling interval 1

The executed electric update is `E^(n+1) = ca E^n + cb R^n`, so with
`model_gradient_sampling_interval=1` the stored `R^n` is redundant: the adjoint
can rebuild `R^n = (E^(n+1) - ca E^n) / cb` from two consecutive saved frames
with the same arithmetic the forward solver used to store it. With
`wavefield_rhs_history="auto"` and uncompressed `float32` storage DeepGPR
therefore stores only `E^n`:

- History memory halves (one E history plus one extra frame instead of E and
  R). `float32` gradients are unchanged: bitwise on CPU; on CUDA the remaining
  differences come from the atomic adjoint scatter.
- The last step needs `E^nt`, the field at the end of the forward run. It is
  kept in an internal one-frame buffer of the autograd context (on the device,
  also with `use_async_offload=True`), never taken from the returned `(Ex, Ey,
  Ez)` states, which callers may advance in place.
- `E_saved` keeps its documented meaning, shape and dtype.
- With `use_async_offload=True` the backward pass reads one E frame per step
  instead of an E and an R frame.

For `float16`, `bfloat16` and INT8 histories `"auto"` keeps the E+R pair. The
permittivity integrand is `-eps0 cb / dt (E^(n+1) - E^n)`; rebuilding it from
two independently rounded E frames has an error proportional to `|E|` instead
of `|E^(n+1) - E^n|`, which was up to 3.5x larger in some measured cases (mostly
comparable; see `tests/profiling_results/history_adjoint_gpu_report.md`).
`wavefield_rhs_history="reconstructed"` enables the E-only history for these
modes as well when memory matters more. `"stored"` restores the earlier E+R
layout in every mode. Native libraries without the
`deepgpr_supports_rhs_reconstruction` capability (for example older prebuilt
Windows/macOS binaries) fall back to E+R under `"auto"` and reject
`"reconstructed"`.

### 4.6 Physical-region history

Material gradients are never accumulated in CPML cells, so their history is
only needed by diagnostics. `wavefield_history_region="physical"` restricts
`E_saved` (and the internal R history or final E frame) to the physical model
cells `[px0, Nx - px1) x [py0, Ny - py1) x [pz0, Nz - pz1)` of the extended grid:

- `E_saved` has the physical spatial shape `(nx, ny, nz)` instead of
  `(Nx, Ny, Nz)`; its values are exactly the corresponding slice of the
  extended history.
- float32, float16 and bfloat16 gradients are identical to `"extended"` (bitwise
  on CPU; on CUDA up to the atomic adjoint scatter).
- INT8: the block INT8 scale of a tile depends on every value in that tile.
  The physical box is therefore widened to whole tiles of the full-grid tiling
  (origin rounded down, end rounded up to a tile boundary of the extended
  grid), so every tile, scale and decoded value of the physical cells equals
  the extended history and the gradients are unchanged. The packed
  uncompressed shape is that widened box; `save_forward_wavefield_path` records
  it as `uncompressed_shape` together with `history_origin`.
- Files saved with `save_forward_wavefield_path` contain a dictionary with
  `wavefield`, `history_region="physical"`, `history_origin`, `pmlthick` and
  `uncompressed_shape` (plus the INT8 entries). `"extended"` histories keep the
  previous format (a bare tensor, or the INT8 dictionary).
- Requires native libraries with `deepgpr_supports_physical_history`; older
  libraries are rejected for `"physical"`.

### 4.7 CUDA 2D TM fast path

In 2D (`nz = 1`) the grid still stores all six field components with a halo
layer (`(nstep, Nx+1, Ny+1, 2)`). Ex, Ey and Hz form a subsystem that is driven
only by itself (through the CPML) and by an Ex/Ey source; it reaches Ez, Hx and
Hy only through z-derivatives of Ex and Ey. With an Ez source and zero
Ex/Ey/Hz they therefore stay exactly zero.

The CUDA backend uses this automatically: when the model is 2D,
`source_direction=2`, and Ex, Ey, Hz and the CPML auxiliaries that couple them
(`x0/xm/y0/ym` E-phi1 and H-phi2) are zero, the forward and adjoint kernels
launch only for the `k = 0` layer and only update/transpose Ez, Hx and Hy;
the CPML corrections skip the z faces. States created by DeepGPR (`E`, `H`, `PML` left
as `None`) qualify without a check; caller-supplied states are checked once
(one device synchronisation). Every other call — a non-Ez source, or a
non-zero Ex/Ey/Hz state or coupled CPML auxiliary — silently uses the general
kernels, so its results are unchanged.

- Receiver data and the final Ez/Hx/Hy states are bitwise identical to the
  general kernels (the zero z-derivative terms are kept in the arithmetic).
- On the fast path the gradients with respect to the initial Ex, Ey, Hz and
  the coupled CPML auxiliaries are returned as zero: on that path they cannot
  influence any output except their own (identically zero) values. Material
  and source gradients are unaffected.
- The CPU backend always uses its general loops.
- CUDA kernels index at most `2^31 - 1` field cells per shot (cells of all
  shots together may exceed that); larger grids raise `NativeLibraryError`.

### 5. Field Variable States (Checkpoints / Initial Fields)

For starting a forward simulation from scratch ($t=0$), these three parameters should be passed as `None` (the system will automatically initialize zero-tensors).

| Parameter | Data Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| **`E`** | `tuple` / `None` | 3 Tensors | Initial state of the electric field components `(Ex, Ey, Ez)`. Each tensor shape is `(nstep, Nx+1, Ny+1, Nz+1)`, including external PML and the Yee field halo. |
| **`H`** | `tuple` / `None` | 3 Tensors | Initial state of the magnetic field components `(Hx, Hy, Hz)`. Shapes identical to `E`. |
| **`PML`** | `tuple` / `None` | 24 Tensors | Auxiliary state variables ($\Phi$ fields) for the PML boundary updates. |

`checkpoint_initial_field` accepts the same unextended physical model and PML settings and allocates matching full-grid states. Pass returned `E`, `H`, and all 24 `PML` tensors back unchanged in shape. The solver only extends materials; it never pads checkpoint states again. Continuation requires identical model geometry, PML settings, and shot ordering, with consistent materials, grid spacing, and time step for that trajectory.

The native solver advances input states in place. When a state is retained for PyTorch checkpoint recomputation or reused in another branch, clone every tensor before passing it to `compute`:

```python
# Inside a checkpointed segment; boundary contains E, H, then all 24 PML states.
work = tuple(t.clone() for t in boundary)
result = DeepGPR.compute(
    device=device, dx=dx, dt=dt,
    eps_r=eps_r, sigma=sigma,  # physical model on every segment
    source_amplitudes=segment_source,
    source_location=source_location, receiver_location=receiver_location,
    pmlthick=pmlthick, E=work[:3], H=work[3:6], PML=work[6:],
)
return (*result[1], *result[2], *result[3], result[-1])
```

Keep material values fixed throughout one segmented trajectory and its backward pass. After the FWI optimizer step, start the next forward simulation from zero or from an initial state appropriate to that new model. See [the checkpoint example](../tests/checkpoint_example.ipynb).

---

## Return values

The function returns a tuple of 5 elements. These are used to extract synthetic data, initiate the gradient flow for backpropagation, or serve as initial parameters (`E`, `H`, `PML`) for subsequent time-stepped calculations.

```python
return E_saved, (Ex, Ey, Ez), (Hx, Hy, Hz), (x0EPhi1...zmHPhi2), receiver_amplitudes
```

1.  **`E_saved`**: The pre-update electric field history `E^n` saved for gradient calculation and diagnostics. Depending on `wavefield_rhs_history`, an internal `R_saved` tensor stores the corresponding discrete right-hand side `R^n`, or an internal one-frame buffer stores the final `E^nt` from which the adjoint rebuilds `R^n` (§4.5).
    *   With `save_wavefield_history=False`, `E_saved` is a zero-length tensor and neither E nor R history storage/compression kernels are launched.
    *   **Shape when `mode=2`**: `(nt_saved, nstep, Nx, Ny, Nz)`, storing Ez only.
    *   **Shape when `mode=3`**: `(3, nt_saved, nstep, Nx, Ny, Nz)`, storing components in `[Ex, Ey, Ez]` order.
    *   `Nx`, `Ny`, `Nz` include external PML. Histories, saved files, and full E/H/PML states keep that grid; only material gradients are cropped. To plot a physical history, slice spatial axes with `[px0:px0+nx, py0:py0+ny, pz0:pz0+nz]`. With `wavefield_history_region="physical"` the spatial axes are already `(nx, ny, nz)` (§4.6).
    *   `nt_saved` depends on `nt` and `model_gradient_sampling_interval`.
    *   Dtype is selected by `wavefield_storage_dtype` when compression is disabled. With `wavefield_compression="int8"`, this is an opaque packed one-dimensional `torch.int8` tensor containing values and FP32 scales.
    *   Set `save_forward_wavefield_path="/path/to/output"` to save a CPU-loadable `.pt` file. Uncompressed extended histories save the tensor directly. INT8 mode saves a dictionary containing `wavefield`, `compression`, `block_size`, and `uncompressed_shape` so diagnostics can reconstruct it safely; `wavefield_history_region="physical"` always saves a dictionary that also records `history_region`, `history_origin` and `pmlthick` (§4.6).
2.  **`(Ex, Ey, Ez)`**: The 3D electric field state at the final time step.
3.  **`(Hx, Hy, Hz)`**: The 3D magnetic field state at the final time step.
4.  **`(PML_Tuple)`**: A tuple of 24 Tensors recording the final time step state of the PML auxiliary $\Phi$ variables.
5.  **`receiver_amplitudes`**: **The core output.** The waveform signals recorded by the receivers over the entire simulation time.
    *   **Shape**: `(nstep, nt, nrx)`
    *   **Meaning**: `[Shot Index, Time Step, Receiver Index]`. This output is sliced to the component specified by `receiver_component` (or its deprecated alias `reciever_direction`).

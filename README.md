# DeepGPR

**Official Website / Documentation:** [https://songc0a.github.io/DeepGPR/](https://songc0a.github.io/DeepGPR/)

DeepGPR provides a wave propagation module for PyTorch, designed for applications such as Ground Penetrating Radar (GPR) imaging and inversion. Its core concepts are derived from Deepwave. You can use it to perform both forward modeling and backpropagation—thereby enabling the simulation of wave propagation to generate synthetic data—as well as for Full Waveform Inversion (FWI). Furthermore, you can integrate this wave propagation functionality into a larger operational pipeline—incorporating various wavelets, loss functions, and other components—to achieve end-to-end forward and reverse propagation, powered by automatic differentiation and our high-performance operators.


## Features

Supports 2D and 3D forward modeling of Maxwell's equations—via the Finite-Difference Time-Domain (FDTD) method—for both single and multiple excitation scenarios.

Gradients of the output receiver data can be computed with respect to model parameters (relative permittivity, conductivity), the initial wavefield, and source amplitudes.

Utilizes CPML, allowing the width of the PML layer to be configured independently for each boundary.

The compute backend can run on CUDA GPUs or on CPU. The CPU backend is implemented in C and is selected automatically when `device='cpu'`.

The FDTD spatial finite-difference order can be selected with `fdtd_order=2`, `4`, or `8` (default: `2`).

The FWI gradient mode can be selected with `mode=2` or `mode=3`. `mode=2` keeps the previous Ez-only gradient behavior, while `mode=3` uses Ex, Ey, and Ez forward/adjoint electric-field contributions for relative permittivity and conductivity gradients.

Supports techniques such as checkpointing, DDP, and the utilization of CPU memory to minimize GPU memory consumption, thereby enabling the execution of large-scale models.


## System Requirements

- **OS**: Linux, Windows, and macOS for CPU execution; Linux and Windows for CUDA execution
- **Environment**: Python 3.8+, CUDA Toolkit for CUDA execution
- **Libraries**: `torch`, `numpy`, `scipy`, `matplotlib`
- **Hardware**: NVIDIA GPU with sufficient VRAM for CUDA execution; CPU execution works without a GPU.


## Start

Before CUDA use, you must ensure that you have an NVIDIA graphics card and have installed a CUDA-enabled version of PyTorch. For CPU use, install a CPU build of PyTorch and include a compiled `deepgpr_cpu` shared library in `src/DeepGPR/lib`.

DeepGPR can then be installed using

```bash
  pip install DeepGPR
```



## A Small Forward Modeling Test

```python
import torch
import DeepGPR
import matplotlib.pyplot as plt

# Set up the parameters and models
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dx=0.02  # Or [dx, dy, dz], for example [0.02, 0.015, 0.01]
dt=3e-11
nt=2000
er = torch.ones(100, 100,1) * 2  
er[50:,:]=5
se = torch.zeros_like(er)  
er.requires_grad_()
source_location=torch.tensor([[[10,10,0]]],device=device,dtype=torch.int)
receiver_location=torch.tensor([[[10,90,0]]],device=device,dtype=torch.int)
freq=2e8
peak_time = 1 / freq
source_amplitudes = torch.zeros((1,nt,1),device=device)
source_amplitudes[0,:,0]=DeepGPR.wavelet.ricker(
    freq, nt, dt, peak_time, device=device
)


#forward modeling
r = DeepGPR.compute(
    device=device, dx=dx, dt=dt, 
    source_amplitudes=source_amplitudes,
    source_location=source_location, 
    receiver_location=receiver_location, 
    er=er, se=se,
    fdtd_order=2
)

(r[-1]**2).sum().backward()

_, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(r[-1].detach().flatten().cpu().numpy())
ax[0].set_title("Receiver data")
ax[1].imshow(er.grad.detach())
ax[1].set_title("Gradient")
plt.show()
```

![result](./Fig/example.png)

There are more examples in the ./examples.

## Source Wavelets

Wavelets are available from the `DeepGPR.wavelet` module. Every function
returns a one-dimensional PyTorch tensor and accepts optional `dtype` and
`device` arguments.

```python
ricker = DeepGPR.wavelet.ricker(freq, nt, dt, peak_time, device=device)
gaussian = DeepGPR.wavelet.gaussian(freq, nt, dt, peak_time, device=device)
derivative = DeepGPR.wavelet.gaussian_derivative(
    freq, nt, dt, peak_time, device=device
)
morlet = DeepGPR.wavelet.morlet(
    freq, nt, dt, peak_time, cycles=3.0, device=device
)
burst = DeepGPR.wavelet.sine_burst(
    freq, nt, dt, peak_time, cycles=3.0, device=device
)
```

`DeepGPR.ricker(...)` remains available as a backward-compatible alias.



The following figures present representative 2D and 3D full-waveform inversion (FWI) examples. For each case, the true model, initial model, and inverted result are shown to evaluate the reconstruction performance of the proposed method.

### 2D FWI Result

The 2D example illustrates the inversion performance on a two-dimensional subsurface model. The comparison between the true model, initial model, and inverted result shows that the proposed method can effectively recover the main structural features from the initial model.

| True Model | Initial Model | Inverted Result |
|---|---|---|
| ![2D true model](./Fig/2dfwitrue.png) | ![2D initial model](./Fig/2dfwiinit.png) | ![2D inverted result](./Fig/2dfwipred.png) |

### 3D FWI Result

The 3D example demonstrates the applicability of the proposed method to three-dimensional full-waveform inversion. For visualization, the figures below show the central slice of the 3D model, including the true model, the initial model, and the inverted result. The comparison indicates that the proposed method can reconstruct the dominant subsurface structures in the 3D case and improve the model consistency relative to the initial model.

| Model Type | Central Slice of 3D Model |
|---|---|
| True Model | ![3D true model central slice](./Fig/3dfwitrue.png) |
| Initial Model | ![3D initial model central slice](./Fig/3dfwiinit.png) |
| Inverted Result | ![3D inverted result central slice](./Fig/3dfwipred.png) |

# `compute` Interface Documentation

`compute` is a core function for 3D/2D Finite-Difference Time-Domain (FDTD) forward modeling, primarily designed for Ground Penetrating Radar (GPR) and electromagnetic wave propagation. It fully supports backpropagation (e.g., for Full Waveform Inversion, FWI) utilizing PyTorch's `autograd` engine.

## 📝 Function Signature

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
            save_forward_wavefield_path=None):
```

Set `print_parameters=True` on an ordinary call to print the normalized
configuration and memory estimate immediately before the native solver starts:

```python
result = DeepGPR.compute(
    device="cuda:0",
    # Other model, source, and acquisition arguments...
    print_parameters=True,
)
```

## 📥 Input Parameters
### 1. Basic Physics & Grid Parameters

| Parameter | Data Type | Description |
| :--- | :--- | :--- |
| **`device`** | `torch.device` / `str` | PyTorch computation device, e.g., `'cuda:0'` or `'cpu'`. CUDA loads `deepgpr.so/.dll`; CPU loads `deepgpr_cpu.so/.dll/.dylib`. |
| **`dx`** | `float` / 3-value `list`, `tuple`, or `Tensor` | Grid spacing in meters. A scalar uses an isotropic grid ($dx = dy = dz$). Three values specify independent $(dx, dy, dz)$ spacings for the finite differences, CFL condition, CPML coefficients, and source scaling. |
| **`dt`** | `float` | Time step size. It is checked against a material-aware CFL limit that includes the selected 2/4/8-order stencil. Typically in seconds (s). |
| **`fdtd_order`** | `int` | Spatial finite-difference order used by the FDTD field updates. Supported values are `2`, `4`, and `8`; default is `2` for compatibility with earlier versions. |
| **`mode`** | `int` | FWI gradient mode. `2` keeps the previous Ez-only model-gradient calculation. `3` uses Ex, Ey, and Ez electric-field contributions for relative permittivity and conductivity gradients. |
| **`debug`** | `bool` | Runs expensive NaN/Inf and zero-field validation checks when `True`. Backward validation covers material, source, and initial-state gradients actually requested by autograd. The default `False` keeps these checks disabled for faster production runs. |
| **`print_parameters`** | `bool` | Prints a preflight summary before native FDTD execution. The summary includes all simulation options and a tensor-payload memory estimate for model/state tensors, CPML, saved `E_saved`/`R_saved` wavefields, receiver buffers, gradients, low-precision snapshots, and CUDA offload buffers. CPU and CUDA estimates are reported separately with a 20% capacity margin. |
| **`save_forward_wavefield_path`** | `str` / path-like / `None` | Directory used to save `E_saved` after a successful forward run. The default `None` performs no file I/O. Files use the local 24-hour start time, for example `forward_wavefield_14-35.pt`; a numeric suffix prevents overwriting when multiple runs start in the same minute. |
### 2. Medium Model Parameters

This section defines the electromagnetic properties of the simulation space. For 2D simulations, set `nz=1`.

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
| **`source_location`** | `Tensor` (int) | `(nstep, nsr, 3)` | Grid coordinate indices of the sources.<br>The last dimension corresponds to `[x_idx, y_idx, z_idx]`. |
| **`receiver_location`** | `Tensor` (int) | `(nstep, nrx, 3)` | Grid coordinate indices of the receivers.<br>The last dimension corresponds to `[x_idx, y_idx, z_idx]`. |
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
| **`pmlthick`** | `int` / `list` / `Tensor`| Scalar or list of 6 | Thickness (in grid layers) of the PML (Perfectly Matched Layer) absorbing boundaries.<br>- Integer `p`: All six boundaries have thickness `p` (Z-boundaries are ignored in 2D).<br>- List `[x0, xm, y0, ym, z0, zm]`: Specific thicknesses for the 6 boundaries. |
| **`model_gradient_sampling_interval`**| `int` | Scalar | Wavefield sampling interval during forward propagation (Default: 1).<br>A larger integer reduces VRAM use for `E_saved` and `R_saved`, but uses an explicitly approximate model gradient. The last incomplete sampling block is weighted by its actual length. |
| **`wavefield_storage_dtype`** | `torch.dtype` / `str` | `float32`, `float16`, or `bfloat16` | Storage format for saved `E_saved` and `R_saved` model-gradient wavefields. FDTD propagation remains float32. `float16` and `bfloat16` halve saved-wavefield memory at the cost of gradient accuracy; `bfloat16` has the safer dynamic range. String aliases such as `"fp16"` and `"bf16"` are accepted. |
| **`use_async_offload`** | `bool` | Scalar | CUDA-only VRAM optimization flag (Default: `False`).<br>If `True`, `E_saved` and `R_saved` are asynchronously offloaded to page-locked host memory (`pin_memory` CPU RAM). This reduces GPU VRAM consumption at the cost of PCIe transfers. On CPU this option is ignored. |

### 4.1 FWI Gradient Mode

`mode` only changes how the model gradients are accumulated during backpropagation:

- `mode=2` (default): Saves Ez in `E_saved` and computes relative permittivity/conductivity gradients from Ez only.
- `mode=3`: Saves Ex, Ey, and Ez in `E_saved` and computes relative permittivity/conductivity gradients from all three electric-field components. This is intended for complete 3D Maxwell FWI. The receiver component is not changed by this option.

When `eps_r` or `sigma` requires gradients, `mode=2` is restricted to 2D Ez-TM modeling; use `mode=3` for 3D gradients. Source-waveform gradients are supported. Gradients with respect to `mu_r` are not currently implemented and are rejected explicitly.

### 4.2 Discrete Adjoint Gradient

The backward solver applies the exact reverse-mode transpose of each executed operation in reverse order: receiver sampling, source injection, electric CPML, electric update, magnetic CPML, and magnetic update. The derivative transpose is applied to the material-weighted field cotangent, so heterogeneous update coefficients and anisotropic grid spacing are handled by the executed discrete operator. Every electric and magnetic CPML auxiliary state has a separate cotangent recurrence on all six faces.

CPML is treated as a fixed numerical boundary. Its boundary material averages are explicitly detached, and CPML cells are excluded from the returned material gradients. This separation must be retained when optimizing a model.

The material-gradient formulation in DeepGPR was informed in part by the differentiable FDTD implementation in [TIDE](https://github.com/Vcholerae1/tide-GPR), particularly its treatment of the discrete Maxwell electric-field update in gradient computation. We gratefully acknowledge the TIDE project and its authors for this work.

Use `model_gradient_sampling_interval=1` and `wavefield_storage_dtype=torch.float32` for a directional derivative check in the physical model region. Temporal subsampling and lower-precision storage deliberately approximate the gradient. Run [the gradient-check notebook](examples/4.GradientCheck.ipynb) after rebuilding the native ABI 6 libraries.

## CPU Backend Build

The CPU backend is a plain C shared library and is built with OpenMP by default. Build it into `src/DeepGPR/lib` before running with `device='cpu'`. You can control CPU thread count with `OMP_NUM_THREADS`.

```bash
# Linux
cc -std=c99 -O3 -fopenmp -fPIC -shared -o src/DeepGPR/lib/deepgpr_cpu.so src/DeepGPR/lib/deepgpr_cpu.c

# macOS
brew install libomp
LIBOMP_PREFIX="$(brew --prefix libomp)"
LIBOMP_RUNTIME_NAME="/opt/llvm-openmp/lib/libomp.dylib"
cc -std=c99 -O3 -Xpreprocessor -fopenmp -DDEEPGPR_USE_OPENMP -I"$LIBOMP_PREFIX/include" -L"$LIBOMP_PREFIX/lib" -fPIC -shared -o src/DeepGPR/lib/deepgpr_cpu.dylib src/DeepGPR/lib/deepgpr_cpu.c -lomp
cp "$LIBOMP_PREFIX/lib/libomp.dylib" src/DeepGPR/lib/libomp.dylib
install_name_tool -id "$LIBOMP_RUNTIME_NAME" src/DeepGPR/lib/libomp.dylib
LIBOMP_DEP="$(otool -L src/DeepGPR/lib/deepgpr_cpu.dylib | awk '/libomp\.dylib/ {print $1; exit}')"
install_name_tool -change "$LIBOMP_DEP" "$LIBOMP_RUNTIME_NAME" src/DeepGPR/lib/deepgpr_cpu.dylib
codesign --force --sign - src/DeepGPR/lib/libomp.dylib
codesign --force --sign - src/DeepGPR/lib/deepgpr_cpu.dylib
```

On Windows, build `src\DeepGPR\lib\deepgpr_cpu.dll` with MSVC:

```powershell
cl /LD /O2 /openmp /Fe:src\DeepGPR\lib\deepgpr_cpu.dll src\DeepGPR\lib\deepgpr_cpu.c
```

### 5. Field Variable States (Checkpoints / Initial Fields)

For starting a forward simulation from scratch ($t=0$), these three parameters should be passed as `None` (the system will automatically initialize zero-tensors).

| Parameter | Data Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| **`E`** | `tuple` / `None` | 3 Tensors | Initial state of the electric field components `(Ex, Ey, Ez)`. Each tensor shape is `(nstep, nx+1, ny+1, nz+1)`. |
| **`H`** | `tuple` / `None` | 3 Tensors | Initial state of the magnetic field components `(Hx, Hy, Hz)`. Shapes identical to `E`. |
| **`PML`** | `tuple` / `None` | 24 Tensors | Auxiliary state variables ($\Phi$ fields) for the PML boundary updates. |

---

## 📤 Return Values

The function returns a tuple of 5 elements. These are used to extract synthetic data, initiate the gradient flow for backpropagation, or serve as initial parameters (`E`, `H`, `PML`) for subsequent time-stepped calculations.

```python
return E_saved, (Ex, Ey, Ez), (Hx, Hy, Hz), (x0EPhi1...zmHPhi2), receiver_amplitudes
```

1.  **`E_saved`**: The pre-update electric field history `E^n` saved for gradient calculation and diagnostics. An internal `R_saved` tensor stores the corresponding discrete right-hand side `R^n`.
    *   **Shape when `mode=2`**: `(nt_saved, nstep, nx, ny, nz)`, storing Ez only.
    *   **Shape when `mode=3`**: `(3, nt_saved, nstep, nx, ny, nz)`, storing components in `[Ex, Ey, Ez]` order.
    *   `nt_saved` depends on `nt` and `model_gradient_sampling_interval`.
    *   Dtype is selected by `wavefield_storage_dtype`.
    *   Set `save_forward_wavefield_path="/path/to/output"` to save this tensor as a CPU-loadable `.pt` file. Load it with `torch.load(path, map_location="cpu")`.
2.  **`(Ex, Ey, Ez)`**: The 3D electric field state at the final time step. 
3.  **`(Hx, Hy, Hz)`**: The 3D magnetic field state at the final time step.
4.  **`(PML_Tuple)`**: A tuple of 24 Tensors recording the final time step state of the PML auxiliary $\Phi$ variables.
5.  **`receiver_amplitudes`**: **The core output.** The waveform signals recorded by the receivers over the entire simulation time.
    *   **Shape**: `(nstep, nt, nrx)`
    *   **Meaning**: `[Shot Index, Time Step, Receiver Index]`. This output is sliced to the component specified by `receiver_component` (or its deprecated alias `reciever_direction`).


## Tests and Verification

Fast unit tests and the complete numerical verification notebooks are kept in
the single [`tests`](tests) directory:

```bash
python -m unittest discover -s tests -p "test_*.py"
python tests/run_notebook.py tests/00_local_backend_and_contracts.ipynb
```

Run notebooks `00` through `09` in numeric order, followed by
`99_verification_summary.ipynb`. See [`tests/README.md`](tests/README.md) for
the complete matrix and CUDA verification options.

# Cite information
If you find our codes useful, please kindly cite this article. Thanks.

@article{liu2026fast,

  title={Fast ground penetrating radar dual-parameter full waveform inversion method accelerated by hybrid compilation of CUDA kernel function and PyTorch},

  author={Liu, Lei and Song, Chao and He, Liangsheng and Wang, Silin and Feng, Xuan and Liu, Cai},
  journal={Computers \& Geosciences},

  pages={106101},

  year={2026},

  publisher={Elsevier}

}

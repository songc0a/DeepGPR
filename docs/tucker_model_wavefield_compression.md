# Tucker model parameterization and streaming Tucker sketching of the adjoint history

DeepGPR provides two independent, optional Tucker modules:

| | Module A: model parameterization | Module B: wavefield-history sketch |
|---|---|---|
| What is low-rank | the material model `eps_r`, `sigma` | the forward history `E^n`, `R^n` read by the adjoint |
| Code | `DeepGPR.tucker.model` | `DeepGPR.tucker.sketch`, `DeepGPR.tucker.wavefield`, `DeepGPR.solver.sketched` |
| Switch | build the model with `TuckerMaterialModel` / `tucker_reconstruct`, or not | `DeepGPR.compute(..., wavefield_sketch=TuckerSketchConfig(...))` |
| Affects | the optimisation variables | memory of the gradient computation; the gradient becomes approximate |
| Default | off (plain tensors) | off (`wavefield_sketch=None`) |

The modules share only stateless helpers (`DeepGPR/tucker/_tensor_ops.py`: mode
products, unfoldings, error metrics). They never share cores, factors, random
maps or optimizer state, and either works without the other.

| Model Tucker | History sketch | Mode |
|---|---|---|
| off | off | original DeepGPR (bitwise identical results, Test C) |
| on | off | low-rank model parameterization, exact discrete adjoint |
| off | on | voxel model, approximate gradient from the compressed history |
| on | on | both |

Implicit neural representations (SIREN, Fourier features, MLPs) are **not**
part of DeepGPR. DeepGPR provides the differentiable reconstruction that an
external network can feed (Section 2.3).

Contents: 1 Mathematics of Module A, 2 Module A API, 3 Mathematics of Module B,
4 Module B integration and data flow, 5 Module B API, 6 Checkpointing,
7 Error sources, 8 Tests, 9 Benchmarks, 10 Limitations and open items.

---

## 1. Module A: Tucker parameterization of the material model

A 3D model `M ∈ R^{Nx×Ny×Nz}` is represented as

```
M = G ×_1 U_x ×_2 U_y ×_3 U_z,   G ∈ R^{rx×ry×rz},  U_x ∈ R^{Nx×rx}, U_y ∈ R^{Ny×ry}, U_z ∈ R^{Nz×rz}
```

(`×_n` is the mode-n product), and a 2D model as `M = U_x G U_y^T`. Any ranks
`1 <= r_n <= N_n` are allowed and need not be equal.

**Reconstruction.** `tucker_reconstruct(core, factors)` applies the mode
products as a chain of `torch.tensordot` calls, ordered by increasing growth
`N_n / r_n` (cost `O(prod(N) r_last)` for the last product). There is no
`detach`, `.data` or in-place write, so autograd differentiates the result with
respect to the core, every factor, and whatever computed them. The gradients
are the adjoint mode products

```
dL/dG   = dL/dM ×_1 U_x^T ×_2 U_y^T ×_3 U_z^T
dL/dU_x = (dL/dM)_(1) (G ×_2 U_y ×_3 U_z)_(1)^T     (and similarly for U_y, U_z)
```

which Test A checks against autograd (1e-13) and Test C checks through the
FDTD adjoint (the core gradient equals `tucker_project(dL/dM, factors)` to
float32 rounding).

**Gradient path through the solver.** `DeepGPR.compute` casts the material to
float32, extends it into the CPML (`_ExtendModel`, whose transpose sums the
replicated PML sensitivities back onto the model edges) and passes it to the
native solver through `DeepGPR.solver.autograd.DeepGPR`, a
`torch.autograd.Function` whose backward runs the native exact discrete
adjoint and returns `dL/d eps_r` and `dL/d sigma` on the extended grid. Every
step before the native call is an ordinary differentiable PyTorch operation, so
a non-leaf `eps_r` (a Tucker reconstruction or a network output) receives its
gradient by the chain rule. No change to the native code or to the autograd
Function was needed for Module A. A float64 Tucker model is accepted; the
solver runs in float32 and the gradient is cast back to float64 by autograd.

**Scale gauge.** For any invertible `A_n`, `(G ×_n A_n^{-1}, U_n A_n)`
represents the same tensor. With raw factors the optimizer can drift along
this flat valley, and Adam's per-entry steps treat core entries of size
`O(||M||)` and factor entries of size `O(1/sqrt(N_n))` very differently.
`TuckerTensor(factor_normalization=...)` fixes the gauge inside the forward
pass:

* `"rms"` (default): `U_n = V_n sqrt(N_n) / ||V_n||_col` (weight
  normalization of each column). Column scalings of the raw `V_n` do not change
  `M`; the core carries the physical scale (a constant model `c` has core
  `c`); factor entries are `O(1)`.
* `"orthonormal"`: `U_n = qr(V_n).Q` with a sign fix.
* `"none"`: raw factors, with `orthogonalize_()` (re-gauge in place to
  orthogonal unit-RMS factors, model unchanged) and `gauge_penalty()`
  (`sum_n ||U_n^T U_n / N_n - I||^2`) as tools.

Measured on a rank-(4,4,4) fit of a 60×60×50 tensor (constant background plus
a rank-3 anomaly, 500 Adam iterations, relative loss reduction): `"rms"`
reaches 1e-10..1e-11 for every learning rate in 0.003..0.03; raw factors are
slower at 0.003 (2e-4..2e-3) and stall at 0.1 (0.25); QR-orthonormal factors
converge slowly with Adam (0.42 at lr 0.03). The default is therefore `"rms"`.
Test A keeps a reduced version of this check.

**Learning rate.** Adam moves every trainable entry by about `lr` per step,
and each factor entry influences a whole fibre of the model, so the model-space
step of a Tucker parameterization is much larger than that of a voxel model at
the same `lr`. In the reduced FWI examples of Test C, voxel Adam converged with
`lr = 0.02..0.05`, the Tucker models diverged at `lr >= 0.005` (2D) and
converged with `lr = 0.001..0.002`. L-BFGS with a Wolfe line search (used in
the benchmarks) needs no learning rate.

**Bounds.** `MaterialMapping` maps the unconstrained Tucker value `T` to a
material value differentiably: `"identity"` (`offset + scale T`), `"sigmoid"`
(`lower + (upper - lower) sigmoid(T)`), `"softplus"` (`lower + scale
softplus(T)`), `"exp"` (`lower + scale exp(T)`). Hard clipping is not offered
(zero gradient). Bounds matter in practice: a low-rank approximation of a model
with an air layer (`eps_r = 1`) undershoots below 1, which `compute` rejects;
`mapping="softplus", bounds=(1.0, None)` avoids it (Test C).

**Initialisation.** `TuckerModelConfig(init=...)`: `"hosvd"` (truncated HOSVD
of `mapping.inverse(initial model)`), `"randomized"` (Gaussian range finder per
unfolding, Halko-Martinsson-Tropp, then truncation), `"random"` (orthonormal
random factors, Gaussian core). `TuckerMaterialField.from_factors(core,
factors)` starts from user tensors. The decomposition is used once, for
initialisation; during the inversion the core and factors are the optimisation
variables (no per-iteration re-decomposition or sketching of the model).

**Parameter accounting.** `tucker_parameter_count(shape, ranks)` returns the
dense-grid count (the comparison basis), the core count `prod(r)`, the factor
count `sum(N_n r_n)`, the total and the ratio. For 120×120×100: ranks 8 →
3 264 values (441×), 12 → 5 808 (248×), 16 → 9 536 (151×), 24 → 21 120 (68×).
`TuckerTensor.memory_report()` gives measured bytes of parameters and
gradients and of the reconstructed grid. The reconstructed full model is still
materialised on every forward pass (the FDTD solver needs it), so Tucker
reduces the trainable state (parameters, gradients, optimizer moments) and
regularises the model; it does not reduce the solver's memory. If an external
network produces the factors, its own trainable parameters replace the factor
count in a fair comparison; DeepGPR cannot count them.

## 2. Module A API

```python
import DeepGPR
from DeepGPR.tucker import (TuckerModelConfig, TuckerMaterialModel, TuckerTensor,
                            tucker_reconstruct, tucker_parameter_count)

model = TuckerMaterialModel.from_models(
    eps_init, sigma_init,
    TuckerModelConfig(ranks=(12, 12, 12), mapping="softplus", bounds=(1.0, None)),
    TuckerModelConfig(ranks=(6, 8, 8), mapping="softplus", bounds=(0.0, None), scale=1e-3),
)
optimizer = torch.optim.Adam(model.parameter_groups(epsilon_lr=1e-3, sigma_lr=1e-3))
eps_r, sigma = model()                     # full tensors, differentiable
data = DeepGPR.compute(..., eps_r=eps_r, sigma=sigma, mode=3)[-1]
loss = ((data - observed) ** 2).sum()
loss.backward()                            # gradients reach the Tucker cores/factors
optimizer.step()
```

| Object | Purpose |
|---|---|
| `tucker_reconstruct(core, factors)` | pure differentiable reconstruction; `None` factor = identity |
| `tucker_project(tensor, factors)` | adjoint `tensor ×_n U_n^T` |
| `hosvd`, `randomized_hosvd` | one-off decompositions for initialisation |
| `TuckerTensor(core, factors, factor_normalization="rms", trainable=True)` | trainable container; `from_tensor`, `random`, `export`/`from_export`, `state_dict`, `.to()`, `orthogonalize_`, `gauge_penalty`, `parameter_count`, `memory_report` |
| `MaterialMapping(kind, lower, upper, scale, offset)` | differentiable bound map with `inverse` |
| `TuckerModelConfig` | per-field options: `enabled`, `ranks`, `init`, `mapping`, `bounds`, `scale`, `offset`, `factor_normalization`, `trainable`, `oversampling`, `seed`, `random_scale` |
| `TuckerMaterialField` / `DenseMaterialField` | one field, Tucker or voxel, with a mapping |
| `TuckerMaterialModel.from_models(eps, sigma, eps_config, sigma_config)` | independent fields; `forward() -> (eps_r, sigma)`, `parameter_groups`, `parameter_report` |

`enabled=False` gives a dense voxel field (optionally with the same mapping);
`trainable=False` a fixed one. Using plain tensors for `eps_r`/`sigma`
bypasses all of this.

### 2.3 External networks (INR/SIREN interface)

DeepGPR does not implement or depend on any network. An external model only
has to produce the core and/or factor matrices as tensors; reconstruction,
FDTD and the adjoint are differentiable end to end:

```python
Ux = net_x(x_coords)              # (Nx, rx)   any external torch code
Uy = net_y(y_coords)              # (Ny, ry)
Uz = net_z(z_coords)              # (Nz, rz)
eps_r = 1.0 + torch.nn.functional.softplus(
    DeepGPR.tucker_reconstruct(core_eps, (Ux, Uy, Uz)))
data = DeepGPR.compute(..., eps_r=eps_r, sigma=sigma, mode=3)[-1]
loss_fn(data, observed).backward()  # reaches core_eps and the parameters of net_x/y/z
```

Requirements: factors `(N_n, r_n)` with `r_n` equal to the core's mode sizes,
same dtype and device as the core (a clear `TypeError`/`ValueError` otherwise).
Keep the material inside the solver's admissible range (`eps_r >= 1`,
`sigma >= 0`) with a smooth map. `examples/tucker/external_factors.py` runs
this pattern with plain tensors standing in for network outputs, and Test A /
Test C verify that the gradients reach the upstream tensors (finite
differences in Test A, through the FDTD adjoint in Test C).

---

## 3. Module B: streaming randomized Tucker sketch

### 3.1 What the adjoint needs

The native forward (`deepgpr.cu`, `forward`) saves, at every saved step `n`,
the pre-update field `E^n` and the update right-hand side
`R^n = (E^(n+1) - ca E^n) / cb` (component-wise, after source injection). The
material-gradient kernel `accumulate_material_gradients_gpu` is the only
consumer:

```
g_eps   += λ_E^(n+1) · (dca/deps E^n + dcb/deps R^n) · w
g_sigma += λ_E^(n+1) · (dca/dsig E^n + dcb/dsig R^n) · w
```

H, the 24 CPML auxiliary arrays and the adjoint propagation itself do not use
the forward history (the adjoint recursion is linear in `λ` and depends only
on the materials). The history therefore consists of `E^n` and `R^n` of the
gradient components: `Ez` in mode 2 (2D TM), `Ex, Ey, Ez` in mode 3. In
DeepGPR's array layout every component lives on the same `(nx, ny, nz)` cell
array of the extended grid (Yee staggering is implicit in the stencils), so
all frames of a run have one shape; the sketch nevertheless supports any
shape per component and independent ranks (`ranks={"Ex": ..., "R": ...}`).

By default DeepGPR stores only `E^n` (plus `E^nt`) for float32 storage and
rebuilds `R^n` exactly (`wavefield_rhs_history="auto"`); low-precision and INT8
histories keep `E` and `R`. The sketch keeps both (`rhs="stored"`, default)
or only `E` (`rhs="from_e"`, Section 7).

### 3.2 Algorithm (Sun, Guo, Luo, Tropp and Udell, 2020)

Source: Y. Sun, Y. Guo, C. Luo, J. A. Tropp and M. Udell, *Low-rank Tucker
approximation of a tensor from streaming data*, SIAM J. Math. Data Sci. 2(4),
2020 (arXiv:1904.10951), with the tensor random projection (Khatri-Rao
structured maps) of the same authors; the matrix analogue is Tropp, Yurtsever,
Udell and Cevher, SIAM J. Sci. Comput. 41(4), 2019.

For a tensor `X ∈ R^{I_1×...×I_M}`, target ranks `r_n`, oversampling `p`:

```
k_n = min(r_n + p, I_n)              factor-sketch size
s_n = min(I_n, ceil(2.0 k_n) + 1)    core-sketch size (core_sketch_factor = 2.0)

Sketch (linear, one pass):
  Y_n = X_(n) Ω_n,   Ω_n = A_n^(1) ⊙ ... ⊙ A_n^(n-1) ⊙ A_n^(n+1) ⊙ ... ⊙ A_n^(M),  A_n^(m) ~ N(0,1)^{I_m×k_n}
  Z   = X ×_1 Φ_1 ×_2 ... ×_M Φ_M,     Φ_n ~ N(0,1)^{s_n×I_n}

Recover:
  Q_n = qr(Y_n)                                   (I_n × k_n, orthonormal)
  C   = Z ×_1 (Φ_1 Q_1)^† ×_2 ... ×_M (Φ_M Q_M)^†   (k_1 × ... × k_M)
  X̂   = C ×_1 Q_1 ... ×_M Q_M
  truncate: C ≈ G ×_n V_n (HOSVD of the small k-core, V_n: k_n × r_n),  U_n = Q_n V_n
  stored: G (r_1 × ... × r_M) and U_n (I_n × r_n)
```

* `Y_n` is computed as `khatri_rao_contract`: one dense contraction of the
  largest other mode, then column-wise ("diagonal in j") contractions of the
  remaining modes. The `prod_{m≠n} I_m × k_n` matrix `Ω_n` is never formed, nor
  is any unfolding of `X` copied (intermediate size `<= numel(X) k_n / I_max`).
* `(Φ_n Q_n)^†` is applied as `R^{-1} P^T` from a QR of the `s_n × k_n` matrix
  (full column rank with probability one); no normal equations.
* If `X` has multilinear rank `<= k_n` in every mode, the recovery is exact
  for Gaussian maps (with probability one): `range(Y_n) = range(X_(n))` and
  `Z ×_n (Φ_n Q_n)^†` inverts the core sketch exactly. Test B uses this to
  catch unfolding, axis-order and core-recovery errors (error 1e-11 in
  float64; a permuted rank vector gives > 1e-2).
* For general tensors Sun et al. bound the expected error by a constant times
  the mode-wise tail energies `sum_n τ_{ρ}^2(X_(n))` (the constant depends on
  `k_n`, `s_n`); `s = 2k + 1` is their recommendation. The truncation adds at
  most the HOSVD truncation error of the small core. Test B measures the
  one-pass error within 4× of the HOSVD error of the same rank on smooth
  fields, and on Gaussian noise (flat spectrum) it is larger than HOSVD (the
  known weakness of one-pass sketches).
* Modes with `r_n >= I_n` are *kept* (identity factor, `Φ_n = I`); singleton
  axes (2D frames) are dropped; a component whose Tucker form is not smaller
  than its raw frames (`min_compression_ratio`) or has fewer than two
  non-singleton modes is stored raw. This is the uncompressed fallback.
* The maps are drawn from a CPU `torch.Generator` seeded with
  `seed * 1000003 + crc32(component)`, so they are identical on CPU and GPU,
  reproducible, and independent between components.
* Small dense factorizations (QR of `I_n × k_n` sketches, pseudo-inverses of
  `s_n × k_n` matrices, `k_n × k_n` eigendecompositions) run on the host in
  float64 (`recovery_device="host"`, default); all contractions run on the
  device. `recover_many` recovers every block completed in a segment, for all
  components, with one batched device-to-host copy per direction, i.e. two
  synchronizations per segment. `recovery_device="device"` keeps everything on
  the GPU; batched `torch.linalg` calls on such tiny matrices are latency-bound
  and synchronize internally (measured 4× slower on the 2D example, Section 9).
* TF32 is disabled inside the sketch (`allow_tf32=False`): PyTorch 1.11 enables
  TF32 matmuls by default on Ampere+ GPUs (10-bit mantissa), which raised the
  float32 sketch error by ~25× in the 2D example (1.1e-4 instead of 4.9e-6 at
  rank 16).

### 3.3 Streaming in time

`time_block = 1` (default) compresses every saved frame on its own:
`W_t ≈ G_t ×_1 U_x^t ×_2 U_y^t ×_3 U_z^t`, i.e. the sketch above with `M = 3`
(or 2). The frame is touched once while it sits in the segment buffer.

`time_block = T > 1` treats `T` consecutive frames as a 4-mode tensor
`(T, I_x, I_y, I_z)` with mode 0 (time) as the *stream mode*: frames arrive in
chunks `[a, b)` and only update linear sketches,

```
Y_n   += X[a:b] contracted with A_n^(m) (m ≠ n, spatial) and rows A_n^(0)[a:b]
Y_0[a:b] = X[a:b] contracted with A_0^(m) (all spatial m)
Z     += X[a:b] ×_0 Φ_0[:, a:b] ×_x Φ_x ×_y Φ_y ×_z Φ_z
```

The block is recovered when its last frame arrives, possibly several native
segments later; its raw frames are never held together. The stored form is a
core `(r_t, r_x, r_y, r_z)`, a time factor `(T, r_t)` and spatial factors
shared by the block. A partial last block is recovered with its true length.
Chunked streaming equals the one-shot sketch to 1e-13 (Test B).

---

## 4. Module B integration and data flow

`compute(..., wavefield_sketch=cfg)` with `cfg.enabled` and
`save_wavefield_history=True` routes the call to
`DeepGPR.solver.sketched.DeepGPRSketched` instead of the original autograd
Function. The FDTD update formulas, CPML, receivers and sources are the
unchanged native code:

```
forward                                         backward
-------                                         --------
for each segment [t0, t1) (segment_steps):      for each segment in reverse:
  native forward(nt = t1 - t0) on the full-       reconstruct the segment's E^n, R^n
    resolution E/H/CPML state (in place)            from the sketches into the buffer
    -> E^n, R^n of the segment into one           native backward(nt = t1 - t0):
       reusable float32 buffer                      λ (30 states) carried exactly,
  stream the frames into the Tucker sketch          data cotangent slice, source
  (buffer reused for the next segment)              gradient slice, eps/sigma
                                                    gradients accumulated (+=)
```

* The state that continues the time stepping (E, H, 24 CPML arrays) is never
  approximated; only the history read by the material-gradient kernel is.
  Receiver data, the loss, source gradients and initial-state gradients are
  therefore exact (Test C: bitwise identical data; source and state gradients
  equal to 1e-6).
* Segments chain exactly: with ranks above the frame size (raw fallback) the
  segmented path reproduces the original gradients bit for bit in 2D (CUDA
  and CPU) and to the 3D CUDA atomic-accumulation floor (Test C), for
  sampling intervals 1 and 3. This verifies the time-index alignment between
  stored frames and the adjoint.
* Segments are multiples of `model_gradient_sampling_interval`, so the
  weighted temporal sampling is the same as in one call.
* The CUDA 2D Ez-TM fast path, physical/extended history regions and both
  native backends (CUDA, CPU/OpenMP) work unchanged.
* `E_saved` is returned empty. `WavefieldSketchStats(keep_history=True)` keeps
  the compressed history for inspection (`history.reconstruct(name, ...)`).

**Exact zeros.** The native kernel discards the adjoint value where the saved
`E^n` and `R^n` are both exactly zero, because the cotangent of decoupled
outer-boundary cells (components that are never updated there) can be
arbitrarily large. A Tucker reconstruction spreads small errors into those
cells; without care this destroyed the 3D gradient (cosine 0.01, concentrated
at the corner cell that receives the replicated CPML sensitivities). Every
compressed component therefore records a support mask (cells nonzero in at
least one frame of any shot, accumulated with reductions during the forward
pass, stored as packed bits) and zeroes the reconstruction outside it. With the
mask the 3D gradient matches (cosine 0.9999996 at rank 6).

**Memory allocation.**

| Allocation | Size | Lifetime |
|---|---|---|
| compressed history | per frame/block: core + factors (+ mask bits per component) | forward → end of backward |
| segment buffer | `(2 or 1+1/F) × components × F × shots × cells × 4 B` (`F` frames per segment) | forward, backward |
| random maps | `sum_n (sum_{m≠n} I_m k_n + s_n I_n)` per component (KiB) | until backward |
| open streaming sketch (`time_block > 1`) | `sum_n I_n k_n + prod_n s_n` per component and shot | while a block is open |
| transient sketch/recovery tensors | `~ F × shots × cells × k / I_max` | one segment |

All of it lives on the solver device; nothing is offloaded. `segment_steps`
trades the buffer size against the number of native calls (Section 9).

**Explicit rejections** (`ValueError`): `wavefield_compression="int8"`,
`wavefield_storage_dtype` other than float32, `use_async_offload=True`,
`save_forward_wavefield_path`, `wavefield_rhs_history="reconstructed"`, and
`rhs="from_e"` with a sampling interval above 1. `TypeError` for a non-config
`wavefield_sketch`.

## 5. Module B API

```python
from DeepGPR.tucker import TuckerSketchConfig
from DeepGPR.solver.sketched import WavefieldSketchStats

cfg = TuckerSketchConfig(ranks=(16, 16, 16), oversampling=8, seed=42)
stats = WavefieldSketchStats()
out = DeepGPR.compute(..., eps_r=eps, sigma=sigma, mode=3,
                      wavefield_sketch=cfg, wavefield_sketch_stats=stats)
loss = ((out[-1] - observed) ** 2).sum()
loss.backward()
print(stats.size.stored_bytes, stats.size.raw_bytes, stats.segment_buffer_bytes)
```

`TuckerSketchConfig` fields: `enabled`, `ranks` (int, per axis, or a mapping by
component `"Ex"`..`"Rz"` / group `"E"`, `"R"` / `"default"`), `oversampling`,
`core_sketch_factor`, `seed`, `time_block`, `time_rank`, `time_oversampling`,
`segment_steps` (default 16), `compute_dtype` (float32/float64), `compress`
(`"all"`, `"E"`, `"R"` or names; others stored raw), `min_compression_ratio`,
`measure_error`, `allow_tf32`, `rhs` (`"stored"` / `"from_e"`).

`WavefieldSketchStats` fields: `components`, `segments`, `segment_steps`,
`frames`, `segment_buffer_bytes`, `size` (`HistorySizeReport`: `raw_bytes` of
E+R, `stored_bytes`, `map_bytes`, per-component sizes and fallback reasons,
`storage_ratio`), phase timings (`forward_native_seconds`, `compress_seconds`,
`backward_native_seconds`, `decompress_seconds`; exact with
`synchronize=True`), `errors` (per-component `relative_l2`,
`max_frame_relative`, `frames`, `unmeasured_frames`), `history` (with
`keep_history=True`).

The algorithmic layer is usable on its own:
`StreamingTuckerSketch(shape, ranks, oversampling, seed=..., stream_mode=0 or None)`
with `new_accumulator`, `update`, `sketch`, `recover` → `TuckerBlock`
(`reconstruct(rows=...)`), and `SketchedWavefieldHistory(components,
frame_shape, shots, frames, config, device)` with `push`, `reconstruct_into`,
`reconstruct`, `size_report`, `error_report`.

## 6. Relation to checkpointing

DeepGPR's checkpointing is the user-level pattern of
`tests/checkpoint_example.ipynb`: the time axis is split into segments, each a
`compute` call with explicit E/H/CPML states under
`torch.utils.checkpoint`; backward recomputes one segment and keeps only that
segment's history. Module B does not replace it and builds no separate
checkpoint system:

* The segmented native driver of Module B is the same mechanism (state passed
  between native calls in place), without recomputation: it keeps a
  compressed history instead.
* **Checkpoint + sketch** works by passing `wavefield_sketch` to the
  `compute` calls inside the checkpointed segments. The boundary states that
  restart the recomputation are full-precision and are never compressed; only
  the history inside each recomputed segment is sketched. The first
  (no-grad) pass stores no history at all. Test C: with lossless ranks the
  combination equals plain checkpointing bit for bit; with rank 16 it equals
  the sketch alone to 1e-5 (frames, maps and therefore approximations are the
  same).
* Lossy compression of the checkpoint states themselves is not offered: an
  approximated restart state would change the recomputed forward wavefield and
  its history without control.

## 7. Error sources

1. **Truncation of the history.** The material gradient is linear in the
   history, so the gradient error is `sum_n λ^(n+1) · (a δE^n + b δR^n)`. A
   per-frame relative error `e_W` gives a gradient error of the same order;
   it decreases with rank (Test C: 2D eps gradient 4.7e-2 / 9.1e-3 / 1.8e-3
   at ranks 6 / 10 / 16; 3D 1.6e-2 / 8.9e-4 / 2.7e-4 at ranks 4 / 6 / 10).
2. **Cancellation in the sigma gradient.** Its lossless floor is 50× higher
   than that of eps (3D: 4.7e-6 vs 9e-8, from float32 summation order alone),
   so the same history error yields a larger relative sigma error (typically
   2-8×).
3. **`rhs="from_e"`**: `R^n` rebuilt from two reconstructed frames. Because
   consecutive frames are sketched with the same maps and evolve smoothly,
   their errors are correlated and the difference is not amplified in the
   measured cases: eps/sigma gradient errors equal those of `rhs="stored"`
   within a factor 2 (sometimes smaller) at half the storage (Section 9). This
   differs from FP16 histories (independent rounding per frame), for which
   DeepGPR keeps `R` by default. `"stored"` remains the default.
4. **Time blocks** share spatial factors over `T` frames; at equal spatial rank
   the error grows (≈ 1.5-3× in the tests) while storage drops 3-7×.
5. **Precision.** Float32 sketches with TF32 disabled reach ~1e-6 relative
   reconstruction error when the rank is sufficient; float64 (`compute_dtype`)
   reaches 1e-11 on exactly low-rank data.
6. **Randomness.** Errors depend on the seed through the Gaussian maps; the
   expected-error bound holds per seed with high probability; results are
   reproducible for a fixed seed.

The approximate gradient is not the exact discrete gradient. All accuracy
claims here are measured against the uncompressed DeepGPR gradient.

## 8. Tests

| File | Content |
|---|---|
| `tests/test_tucker_model.py` (Test A, 27) | 2D/3D reconstruction vs independent einsum, many rank shapes, axis-order detection, dtypes, CPU/GPU, gradcheck, analytic gradients, HOSVD/randomized init exactness, parameter counts, gauge invariance, orthogonalize, export/load/state_dict, LBFGS-compatible parameters, optimizer convergence, mappings, independent dual model, external-factor gradients (finite differences) |
| `tests/test_tucker_sketch.py` (Test B, 23) | Khatri-Rao contraction vs explicit matrix, exact recovery of random low-rank tensors, wrong-rank detection, ranks/oversampling, error vs HOSVD, oversampling, high-rank data, zero tensors, kept modes, seeds, CUDA = CPU, TF32 guard, chunked stream = one shot, partial blocks and order checks, per-frame and time-block histories across segments, exact byte accounting and no retention of raw frames, support mask, fallbacks and component selection, 2D/1D frames, error statistics, config validation |
| `tests/test_tucker_fdtd_integration.py` (Test C, 13) | reduced 2D FWI example and 3D OverThrust example: disabled sketch = original path (bitwise), lossless sketch = original gradient (bitwise / atomic floor) for sampling 1 and 3 and both `rhs`, gradient accuracy and cosines for eps and sigma, error decreasing with rank, Tucker gradients through the exact and the sketched adjoint (chain rule), external factor tensors through FDTD, the four on/off modes reducing the loss, checkpoint + sketch, exact source/state gradients, no-grad forward, double backward, rejected combinations, measured peak memory below the default history |
| `tests/tucker_cases.py` | the reduced cases |

Run: `python -m unittest tests.test_tucker_model tests.test_tucker_sketch tests.test_tucker_fdtd_integration`.

## 9. Benchmarks

See Section 9 results below (`examples/tucker/benchmark_tucker.py`).

## 10. Limitations and open items

See the end of this document.

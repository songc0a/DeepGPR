# Huygens subgridding for 3D Yee-FDTD (forward, adjoint, adaptive refinement, DGTD surface)

`DeepGPR.subgrid` adds local grid refinement to DeepGPR's 3D Yee scheme:
refined boxes (*subgrids*) with spacing `dx / r` and time step `dt / r` (`r`
odd), coupled to the main grid through Huygens surfaces. It is built on the
hybrid package's 3D Yee operators and fused CUDA kernels
(`DeepGPR.hybrid.functional3d`, `DeepGPR.hybrid.cuda3d`); neither those
modules nor the native DeepGPR solver were changed.

| Step | Content | Module / example | Status |
|---|---|---|---|
| 1 | 3D CUDA subgridding (forward) | `hsg.py`, `cuda.py`, `csrc/hsg3d.cu`, `graphs.py`, `solver.py` | implemented, verified (Section 2) |
| 2 | Adaptive grid sizes (a priori refinement plan) | `adapt.py`, `examples/subgrid/adaptive_targets.py` | implemented, verified (Section 3) |
| 3 | Exact discrete adjoint, dual-parameter FWI | `adjoint.py`, `examples/subgrid/fwi3d_subgrid.py` | gradient exact (finite differences, CUDA = autograd); modelling floor 5.6-12x lower; voxel images not better than the main grid alone; known-shape inversion removes the eps bias (Section 4) |
| 4 | Adjoint-error-driven dynamic AMR-FWI | `indicators.py`, `amr.py`, `examples/subgrid/amr_fwi3d.py` | indicator verified (effectivity 0.93-1.01); dynamic regridding works; no image gain over the baselines (Section 5) |
| 5 | DGTD rugged surface + subgridded FDTD + FWI | `hybrid.py`, `examples/subgrid/{dgtd_subgrid_accuracy,fwi3d_dgtd_subgrid}.py` | implemented, step (1e-15) and gradient verified; target response 42 % -> 30 % error; known-shape FWI eps_A 13.8 -> 15.7 (true 16); voxel images equal (Section 6) |

All numbers below were measured on one RTX 4090 (24 GB, shared with other
processes), PyTorch with the fused CUDA kernels unless stated otherwise.

## 1. Algorithm (`hsg.py`)

The scheme is Berenger's Huygens subgridding (IEEE Trans. Antennas Propag.
54(12), 3797-3804, 2006, doi:10.1109/TAP.2006.886519) in the arrangement of
gprMax's current `gprMax/subgrids` code (read for this work; gprMax is GPLv3,
DeepGPR MIT, so no gprMax code was copied, the operators were derived anew):

* A subgrid is a complete Yee grid (spacing `dx/r`, `dt/r`) with its own CPML
  (`pml_cells`, default 6), covering the working region (inner surface, IS),
  `is_os_sep` main cells (default 3) up to the outer surface (OS), `r//2 + 2`
  cells and the CPML.
* IS (subgrid): total field inside, scattered field outside. Main-grid fields,
  interpolated to the subgrid (*precursors*), are injected as a TF/SF incident
  field. OS (main grid): incident field only inside, total outside; the
  subgrid's scattered field, sampled at the coinciding fine nodes (odd `r`),
  is injected.
* One rule serves both surfaces: for a Yee difference whose two nodes lie on
  different sides, the node across the surface is converted to the field type
  of the updated node (`d += s inc` on the IS, `d -= s scat` on the OS,
  `s = -1/+1` on the lower/upper face). Edges and corners are corrected once
  per crossing difference (gprMax's face-by-face rule).
* Precursors: tangential E on the IS planes and tangential H half a fine
  cell outside them, by tensor-product linear interpolation of the main
  lattice of the same component. The main nodes just outside the box are used,
  so edge nodes are interpolated, not clamped as in gprMax. Optional normal
  smoothing `(1/4, 1/2, 1/4)` (gprMax's `filter`, default on). Linear
  interpolation in time between the bracketing main levels.
* Time stepping per main step (`h = (r-1)/2`): main H; subgrid `h x (H, E) + H`
  (`hsg_2`) and the OS H correction; main E and sources; subgrid `h x (E, H) + E`
  (`hsg_1`) and the OS E correction.
* Sources inside a subgrid use the band-limited (FFT) resampling of the
  main-rate wavelet at `dt/r`, with the main grid's half-step delay, so a
  source carries the same current wherever it is placed. Receivers inside a
  subgrid record main time `n+1` during step `n+1`, and the driver shifts them.

The *switched* Huygens subgridding (Hartley, Giannopoulos and Davidson, IEEE
Trans. Antennas Propag. 70(8), 6872-6882, 2022, doi:10.1109/TAP.2022.3161371),
whose stabilisation needs no subgrid PML, is **not** implemented. Here the
stabilisation is the subgrid CPML and the filter, as in gprMax's code.

Models live on a model grid of spacing `h`. The main grid (`dx = coarsen * h`)
takes averages over the `coarsen^3` block of each node. A subgrid of ratio `r`
(a divisor of `coarsen`) takes block averages over `(coarsen/r)^3` cells, or the
model itself when `r = coarsen`. Both maps are linear and differentiable.

## 2. Step 1: 3D CUDA subgridding (forward)

CUDA: the main and subgrid Yee updates use the existing fused kernels
(`hybrid/csrc/yee3d.cu`, unchanged). `subgrid/csrc/hsg3d.cu` adds two kernels:

* a tap gather (all precursors of one kind in one launch);
* a fused correction scatter (all curl terms of one surface; time interpolation
  folded in; two launches for E so that no launch has duplicate targets, which
  keeps the forward pass free of atomics).

Each kernel has an exact transpose. They are exposed as differentiable,
in-place autograd functions. `graphs.py` replays one step as a CUDA Graph.

Verification (`tests/test_subgrid_3d.py`, 11 tests):

| Check | Result |
|---|---|
| precursors of linear fields | exact (1e-10) |
| fused CUDA step vs PyTorch step, f64 | 1.4e-15 |
| CUDA Graph vs eager | identical |
| spurious reflection, homogeneous medium, 400 MHz | 0.3 % of the direct wave (-50 dB) |
| antennas inside the subgrid, eps 20 sphere, vs uniform 1 cm grid | 0.2-0.3 % (coarse 3 cm: 29-38 %) |
| scattered field of a 3 cm eps-20 sphere, antennas outside, 200 MHz | 5 % (coarse: 50-57 %) |
| same at 400 MHz | 33-45 % (coarse: 83-92 %); the residual is coarse-grid dispersion along the ~0.55 m path outside the box |
| two subgrids, 200 MHz | 2.5-5 % (coarse: 35-53 %); float32 within 1.2e-3 of float64 |
| 30 000 steps, filter on and off | no growth (receivers decay to 2e-6 of the peak; the remaining constant field is the dipole's static field) |

Cost (GPR-scale model 1.8 x 2.4 x 2.4 m, main 3 cm, 0.85 M nodes, subgrid
0.6 m cube, 0.91 M fine nodes, float32):

| | ms / main step |
|---|---|
| main grid only | 0.20 |
| subgrid, coupling as PyTorch ops | 1.65 (host bound: ~170 small kernels) |
| subgrid, fused coupling kernels | 0.88 (GPU busy 0.53 ms; Yee 0.43 ms at ~770 GB/s) |
| fused + CUDA Graph (2 sources) | 0.87 (eager 1.10); in float64 the replay's state copy makes it slower (2.12 vs 1.87) |

In `adaptive_targets.py` (Section 3) the subgrid run takes 0.28 s against
5.1 s for the uniform 1 cm grid (18x), with a peak of 0.22 GB against 2.39 GB.

## 3. Step 2: adaptive grid sizes (`adapt.py`)

`plan_refinement(eps, sigma, h, f_max, ppw=10, ...)` chooses:

1. the main spacing: the largest odd `coarsen` with `coarsen * h <= lambda/ppw`
   on all but `refine_fraction` (2 %) of the volume;
2. a required ratio per main node, from two criteria:
   * dispersion: `lambda_min / ppw` in the node's block;
   * representation: the block-averaged material, interpolated back, differs
     from the model by more than 15 %, i.e. a structure smaller than a main
     cell;
3. boxes. Connected flags that reach the domain sides (a ground surface, for
   example) cannot be refined by a Huygens subgrid and are reported. The rest
   is clipped to the region where a box fits (reported as `clipped`) and
   clustered by Berger-Rigoutsos (IEEE Trans. Syst. Man Cybern. 21(5),
   1278-1286, 1991, doi:10.1109/21.120081), padded, and merged while the outer
   surfaces come closer than two main cells. Antennas that would sit between
   an inner and an outer surface enlarge the box.

Example `examples/subgrid/adaptive_targets.py`: model grid 1 cm, 1.5 x 2.1 x
2.1 m, soil eps 6, a 4 cm sphere and a 5 cm pipe segment. Planning takes
0.5 s and gives two ratio-3 boxes on the 3 cm main grid. The air-soil surface
(10 082 flagged nodes) is reported as not refinable.

Scattered-field error vs the uniform 1 cm grid:

| Case | subgrid | main grid only |
|---|---|---|
| air + dielectric targets, 160 MHz | 10.1 % | 13.8 % |
| buried dielectric, 160 / 100 MHz | 10.2 / 3.8 % | 15.6 / 10.4 % |
| buried PEC, 250 / 160 / 100 MHz | 26.7 / 14.5 / 3.9 % | 154 / 164 / 96 % |
| air + PEC targets, 160 MHz | 13.3 % | 137 % |

At 100 MHz the subgrid error drops to 4 %. The residual at higher
frequencies is therefore the main grid's dispersion outside the boxes, not the
subgridding. Tests: `tests/test_subgrid_adapt.py` (6, including the AMR regrid of Section 5).

## 4. Step 3: exact discrete adjoint and dual-parameter FWI (`adjoint.py`)

`SubgridPropagator(solver, src, rec)` returns `data = prop(eps, sigma, w)`.
Its backward pass is the checkpointed reverse sweep (`sqrt(nt)` checkpoints)
of exactly `hsg.step`: one reverse-mode VJP per step through the fused kernels'
hand-written transposes. Gradients reach the model-grid `eps_r`, `sigma` and
the wavelet, including the subgrid-rate resampling.

Verification (`tests/test_subgrid_adjoint.py`; sources and receivers on the
main grid and inside the subgrid, random heterogeneous model, float64):

| Check | Result |
|---|---|
| CUDA (hand-written transposes) vs CPU (autograd of the PyTorch step) | eps 4.5e-16, sigma 3.6e-15, wavelet 2.3e-15 |
| finite differences: eps in the box / eps everywhere / sigma / wavelet | 2.6e-11 / 8.2e-11 / 1.7e-9 / 3e-12 |
| checkpoint interval 1 vs 40 | < 1e-12 |

Cost (Section 2 model, float32, 1 shot, 400 steps):

| | forward ms/step | gradient ms/step | peak memory |
|---|---|---|---|
| subgrid | 0.81 | 4.09 | 2.76 GB |
| main grid only | 0.23 | 1.00 | 1.52 GB |

A hand-written VJP of the whole step (as `hybrid/vjp3d.py`) is not
implemented; the gradient costs about five forward steps.

### FWI (`examples/subgrid/fwi3d_subgrid.py`)

Setup:

* 0.9 m cube, soil eps 4 / 2 mS/m, anomaly A (eps and sigma) and anomaly B
  (sigma only);
* 9 shots, receiver grid on a plane, L-BFGS on eps and sigma;
* observed data: the anomaly response of a uniform 1 cm grid added to each
  solver's own background prediction. Raw 1 cm data are dominated by the
  3 cm grid's direct-wave dispersion: 7e-3 of misfit against a 4e-5 anomaly
  signal, which no inversion can remove, since it arises outside the box;
* receivers within 10 cm of their source muted.

*Floor* = the misfit of the true model on each solver (its modelling error).

| Preset, start | solver | floor | final misfit | eps err (6 cm smooth) | int dEps A | int dSig A / B | s / gradient | peak |
|---|---|---|---|---|---|---|---|---|
| small (4 cm, 250 MHz), background, 25 it | coarse | 0.074 | 0.0012 | 0.946 (0.816) | 0.18 | 0.06 / 0.07 | 0.49 | 3.9 GB |
| | subgrid | 0.011 | 0.0012 | 0.946 (0.830) | 0.18 | 0.07 / 0.08 | 3.57 | 10.9 GB |
| resolved (6 cm, 400 MHz, TV), background, 30 it | coarse | 0.151 | 0.0213 | 0.766 (0.509) | 0.64 | 0.20 / 0.11 | 0.43 | 3.7 GB |
| | subgrid | 0.027 | 0.0213 | 0.806 (0.573) | 0.58 | 0.18 / 0.11 | 2.79 | 10.3 GB |
| resolved, smoothed true start, 30 it | coarse | 0.151 | 0.0199 | 0.693 (0.273) | 0.79 | 0.44 / 0.47 | 0.46 | 3.7 GB |
| | subgrid | 0.027 | 0.0187 | 0.715 (0.296) | 0.79 | 0.46 / 0.48 | 2.98 | 10.3 GB |
| strong (eps 25, 200 mS/m), background, 30 it | coarse | 0.882 | 0.053 | 0.916 (0.737) | 0.23 | -0.04 / 0.03 | 0.43 | 3.7 GB |
| | subgrid | 0.075 | 0.051 | 0.919 (0.749) | 0.25 | -0.10 / 0.03 | 2.77 | 10.3 GB |

Parametric inversion: shapes known, the three values unknown
(`--param objects`, 15 iterations).

| Preset | solver | eps A (true 12) | sigma A (true 20 mS/m) | sigma B (true 30 mS/m) | misfit |
|---|---|---|---|---|---|
| resolved | coarse | 10.67 | 17.0 | 28.1 | 0.093 |
| | subgrid | 12.13 | 8.8 | 30.4 | 0.023 |
| strong (true 25 / 50 / 200) | coarse | 9.87 | 0 (bound) | 280 | 0.38 |
| | subgrid | 9.99 | 0 (bound) | 298 | 0.33 |

Values are bounded (eps >= 1, sigma >= 0) as in the voxel inversions.

What these runs show:

* The subgrid lowers the modelling floor of the anomaly response 5.6-12 times.
* In voxel FWI both solvers fit the data equally well, well below the coarse
  floor, so the coarse solver absorbs its modelling error in the model. The
  images are equally good: they are limited by the data's resolution and the
  regularisation, not by the forward accuracy. This holds from the
  background and from the smoothed true model.
* With known shapes, the subgrid removes the coarse bias of eps_A (-17 % of
  the contrast to +1.6 %) and of sigma_B. sigma_A is poorly constrained next
  to eps_A, and the subgrid run reaches a misfit below its own floor by moving
  sigma_A.
* The same holds below the DG surface (Section 6), more clearly.
* The strong-contrast case does not converge from the background with either
  solver.

## 5. Step 4: adjoint-error-driven dynamic AMR-FWI (`indicators.py`, `amr.py`)

The indicator is a dual-weighted residual (Becker and Rannacher, Acta
Numerica 10, 1-102, 2001, doi:10.1017/S0962492901000010; for parameter
identification Becker and Vexler, Numer. Math. 96, 435-459, 2004,
doi:10.1007/s00211-003-0482-9), applied to the discrete Yee scheme of the main
grid:

    dJ ~ sum_n <g_E^n, rho_E^n> + <g_H^n, rho_H^n>

* `g` are the exact discrete cotangents of zero-valued perturbations added to
  the main E and H updates, taken from the reverse sweep.
* `rho` is the computed solution's residual in a more accurate scheme: the
  fourth- minus second-order staggered curl, plus the leapfrog's third time
  difference `/24`.
* Per main node the signed contributions are summed over time and taken in
  absolute value. The main CPML and the outer-surface boxes are excluded.
* Marking uses Doerfler's rule (SIAM J. Numer. Anal. 33(3), 1106-1124, 1996,
  doi:10.1137/0733054).

**Effectivity** (`tests/test_subgrid_indicators.py`): the summed indicator
against the measured change `<g, d(1 cm) - d(3 cm)> * 9/8` (the 1 cm grid
keeps 1/9 of the error) is 1.006 at 200 MHz, 0.976 at 300 MHz and 0.932 at
400 MHz.

**What it marks.** For the total misfit at the background model it marks
mainly the antenna region, where the grid-dependent near field dominates the
modelling error. For FWI of the anomaly response, `amr.indicator(...,
eps0=, sig0=)` evaluates the differential indicator `eta(m) - eta(m0)` with
the same adjoint source (one extra gradient pass). It vanishes at `m = m0` and
concentrates where refinement would change the predicted anomaly response.

**AMR-FWI** (`amr_fwi3d.py`, resolved preset):

* model updates and refinement allowed in the whole subsurface below the
  antennas;
* three cycles of 10 L-BFGS iterations;
* regridding (`amr.regrid`) between cycles: Doerfler marking with `theta`
  0.5, plus the a priori criteria on the current model, Berger-Rigoutsos;
  refinement persists across cycles.

After the coarse cycle, the indicator places one box over the anomaly zone,
`(0.36-0.63) x (0.24-0.63) x (0.24-0.63)` m, and keeps it in the next cycle.
(A first version re-padded persistent boxes at every regrid and grew them;
fixed, and covered by `tests/test_subgrid_adapt.py::RegridTest`.)

| Schedule (same update region and restarts) | final misfit | eps err (6 cm) | time |
|---|---|---|---|
| AMR (cycle 0 coarse, then the indicator's box) | 0.0509 | 0.950 (0.933) | 162 s |
| main grid only | 0.0511 | 0.951 (0.937) | 43 s |
| fixed box placed with prior knowledge | 0.0514 | 0.957 (0.963) | 212 s |

The large update region slows all three equally; as in Section 4, the image
is not limited by the forward accuracy here. The indicator and the regridding
work, but they bring no image gain in this test.

## 6. Step 5: DGTD rugged surface + subgridded FDTD + FWI (`hybrid.py`)

`HybridSubgridMaxwell3D(eps, sigma, h, coarsen=3, surface=..., dg_band=...,
subgrids=[...])` wraps `HybridMaxwell3D`: a body-fitted DGTD band around the
surface, with the Yee grid above and below it. The hybrid sees the block
averages of the model at spacing `coarsen * h`. The subgrids are attached to
the Yee grid below the band and away from the side UPML.

The step runs:

1. the hybrid's H half: Yee H, DG H, ghost rows;
2. `hsg_2` and the OS H correction;
3. the hybrid's E half: Yee E with the interface penalty, DG E, sources;
4. `hsg_1` and the OS E correction.

It reuses the hybrid's operators block by block. The same
`SubgridPropagator` gives the exact adjoint through all three
discretisations. Sources and receivers act on the hybrid part.

Verification (`tests/test_subgrid_hybrid.py`):

* without subgrids, data equal `HybridMaxwell3D` to 4.6e-16 (to < 1e-12 in the test);
* finite differences (float64): eps in the subgrid 2.1e-7 (J barely depends on it), eps in
  the DG band 5.9e-10, eps everywhere 2.4e-9, sigma 1.1e-7.

Accuracy (`examples/subgrid/dgtd_subgrid_accuracy.py`). Model: rugged sine
surface, soil eps 6, a 3 cm eps-20 sphere 0.66 m deep, 250 MHz, two antennas
in air and one receiver below the target. Reference: the same DG band and
elements with a uniform 1 cm Yee grid and the model itself.

| | target response error | total field error | s per 2200 steps |
|---|---|---|---|
| hybrid (3 cm Yee) | 42 / 42 / 43 % | 15 / 18 / 15 % | 5.4 |
| hybrid + subgrid, `time_ratio=3` | 31 / 30 / 30 % | 15 / 18 / 14 % | 6.7 |
| hybrid + subgrid, `time_ratio=1` | 31 / 30 / 30 % | 15 / 18 / 14 % | 6.0 |
| reference (1 cm Yee everywhere) | - | - | 13.6 |

* Without subgrids the split step equals `HybridMaxwell3D` to 1.5e-15.
* Transparency of the subgrid in the target-free model: 2.7e-4 / 5.4e-5 at the
  surface receivers, 0.66 % at the receiver below the box.
* `time_ratio=1` against 3: 1e-4.
* The remaining 30 % target-response error is the 3 cm Yee propagation between
  the band and the box (about 7 points per wavelength at the top of the
  spectrum).

FWI (`examples/subgrid/fwi3d_dgtd_subgrid.py`):

* model: rugged surface, two 4 cm anomalies at 0.585 m (A: eps 16, 20 mS/m;
  B: 40 mS/m);
* 4 shots, 5 x 5 receivers in air, 300 MHz, 1200 steps (dt 1.2e-11 s, set by
  the DG band);
* data: the anomaly response of the DG + 1 cm Yee reference;
* compared: hybrid, hybrid + subgrid, and FDTD with the surface as a 3 cm
  staircase plus the same subgrid.

Parametric (anomaly values, shapes known; 15 L-BFGS iterations):

| Solver | modelling floor | eps A (true 16) | sigma A (true 20 mS/m) | sigma B (true 40 mS/m) | misfit |
|---|---|---|---|---|---|
| hybrid | 0.190 | 13.77 | 0 (bound) | 18.4 | 0.070 |
| hybrid + subgrid | 0.060 | **15.69** | 0 (bound) | **30.8** | 0.043 |
| staircase surface + subgrid | 2.40 | 1.77 | 0 (bound) | 0 (bound) | 0.72 |

Values are bounded (eps >= 1, sigma >= 0). Without the bounds both hybrids
drove sigma_A to about -60 mS/m.

What these runs show:

* **The DG band is essential.** With a 3 cm staircase surface, the anomaly
  response is misrepresented by more than 100 % (floor 2.4), and the
  parametric inversion cannot fit it (misfit 0.72).
* **The subgrid helps on top of the DG band.** It lowers the floor 3.2x and
  recovers eps_A almost exactly (13.8 -> 15.7, true 16) and sigma_B much
  better (18 -> 31 mS/m, true 40) at a lower misfit.
* **sigma_A is not recovered by either.** It reaches the bound, traded off
  against eps_A.

Voxel FWI (background start, TV, 15 iterations):

| Solver | floor | final misfit | eps err (6 cm) | int dEps A | int dSig A / B | s / gradient | peak |
|---|---|---|---|---|---|---|---|
| hybrid | 0.190 | 0.0048 | 0.984 (0.989) | 0.04 | 0.02 / 0.01 | 16.1* | 8.7 GB |
| hybrid + subgrid | 0.060 | 0.0048 | 0.984 (0.979) | 0.05 | 0.03 / 0.02 | 13.2 | 10.3 GB |
| staircase surface + subgrid | 2.40 | 0.019 | 1.021 (0.993) | -0.02 | -0.09 / -0.03 | 4.7 | 4.7 GB |

\* Measured while another job shared the GPU. The staircase solver runs at the
DG band's `dt` for the comparison; on its own it could use a 4.5x larger step.

All three fit the anomaly response with small, smeared perturbations; the
4 cm anomalies 0.4 m below the antennas are not recovered in 15 iterations.
The staircase solver fits its data below its own floor (2.4) with artifacts:
the sign of A's eps anomaly is wrong. Voxel images therefore do not separate
the hybrid with and without the subgrid in this test, as in Section 4; the
modelling floors and the parametric test do.

Figure `examples/subgrid/layout_figure.py` (`results/subgrid_layout.png`)
shows the layout of Section 6's accuracy model:

* the main 3 cm grid and the subgrid's 1 cm grid;
* the inner and outer surfaces, the subgrid CPML and the DG band;
* Ez at 9.6 ns, with the subgrid's total field drawn inside the inner surface.

## 7. Usage

```python
from DeepGPR.subgrid import Subgrid, SubgridMaxwell3D
from DeepGPR.subgrid.adapt import plan_refinement
from DeepGPR.subgrid.adjoint import SubgridPropagator

plan = plan_refinement(eps, sigma, h=0.01, f_max=4e8, ppw=10, keep_points=antennas)
sol = SubgridMaxwell3D(eps, sigma, 0.01, coarsen=plan.coarsen, subgrids=plan.subgrids,
                       pmlthick=10, device="cuda", dtype=torch.float32)
data = sol.run(wavelet, src_xyz, rec_xyz)["receivers"]            # forward
prop = SubgridPropagator(sol, src_xyz, rec_xyz)                    # exact adjoint
d = prop(eps_param, sigma_param, wavelet); loss(d).backward()

from DeepGPR.subgrid.hybrid import HybridSubgridMaxwell3D          # with a DG surface band
sol = HybridSubgridMaxwell3D(eps, sigma, 0.01, coarsen=3, surface=surf, dg_band=(0.09, 0.24),
                             subgrids=[Subgrid(box, is_os_sep=2, pml_cells=4)], device="cuda")
```

## 8. Not implemented, limitations and open issues

* **Stability.** HSG has no discrete energy estimate. Stability was tested
  numerically: 30 000 steps, filter on and off. The switched (SHSG)
  stabilisation is not implemented.
* **Subcycling under the DG band.** By default the subgrid subcycles
  (`dt/r`). Under a DG band the main `dt` is already DG-limited, about 4.5x
  below the Yee limit, so `Subgrid(..., time_ratio=1)` runs the subgrid at the
  main `dt`. It is refused where `dt` exceeds the subgrid's CFL limit. It
  agrees with the subcycled subgrid to 1e-4 on the target response of
  Section 6 and halves the subgrid's added cost there.
* **Fine-ring materials.** The fine materials between the IS and the subgrid
  CPML come from the model. Where they differ from the main grid's block
  averages (an interface crossing the ring), the TF/SF cancellation degrades.
  A "main-material ring" option is not implemented.
* **Adjoint cost.** There is no hand-written VJP of the whole step; the
  gradient costs about five forward steps.
* **Graph replay.** It copies the state back after each step. An in-place
  no-grad step without the copy is not implemented.
* **Antennas with the DG band.** In `HybridSubgridMaxwell3D` antennas must be
  outside the subgrids.
* **Indicator.** The temporal term is skipped at checkpoint-segment starts.
  The temporal residual ignores conductivity in the leapfrog's truncation.
  Inside subgrids only the main grid is assessed: refinement persists, but
  there is no coarsening and no higher ratios.
* **FWI.** The forward-accuracy gain is verified (modelling floor 5.6-12x
  lower), but voxel FWI images are not better in the tests of Sections 4-5;
  only the parametric test shows less bias (eps_A).
* **CI.** `DeepGPR.subgrid` imports `DeepGPR.hybrid`, which needs scipy. The
  CPU job of `.github/workflows/quality.yml` does not install it (the same
  open issue as the hybrid's tests).

## 9. FWI ablation experiments (`examples/subgrid/fwi_ablation.py`)

Sections 4-6 compared solvers on data that each solver partly reproduced
itself (transplanted background) and on sub-resolution targets. The ablation
study isolates the effect of each improvement on the inversion.

**Data.** Observed data come from an independent, more accurate solver:
the uniform 1 cm grid, or, with a surface, the DG band with 1 cm Yee.

**Acquisition.** Nine shots on a plane above and a plane below the targets,
with 6 x 6 receivers on each plane, gives reflection and transmission aperture.
Six shots are used for the inversion; three are held out.

**Inversion settings.** All solvers of an experiment share the same settings:
L-BFGS on eps and sigma on the 1 cm model grid, the same update region, start
model, TV weight and number of iterations.

**Metrics.**

* errors against the true model, point-wise and smoothed (5 cm Gaussian);
* integrated anomaly around each target;
* the **held-out prediction error** PE: the three held-out shots simulated
  with the accurate solver through the inverted model,
  `PE = ||d(m) - d_obs||^2 / ||d(m0) - d_obs||^2` (0 perfect, 1 = start
  model). PE rewards a model that is physically right, not one that fits the
  inversion's own solver;
* cost: seconds per gradient, gradient evaluations, peak memory, and
  failed line searches.

**Pre-study** (`prestudy`; benchmark 0.75 m cube, eps 4, a 4 cm eps-16 target
and a 4 cm 100 mS/m target 15 cm apart at 0.42 m). Values are energies
relative to the target signal:

| f0 | target signal / data | 3 cm: background error | 3 cm: target-response error | 3 cm + subgrid: response error | 5 cm: background error |
|---|---|---|---|---|---|
| 200 MHz | 1.0e-4 | 40 | 0.028 | 0.002 | 637 |
| 300 MHz | 8.4e-4 | 5.3 | 0.086 | 0.024 | 67 |
| 400 MHz | 3.2e-3 | 5.1 | 0.276 | 0.119 | 56 |

With raw data, the 3 cm grid's dispersion of the direct and transmitted waves
is 5-40 times the target signal. That error arises outside any box, so
local refinement cannot remove it from raw-data FWI.

The experiments therefore use **differential data** (`--data diff`, the
default): a known baseline survey (the target-free model) is subtracted, and
each solver predicts `d(m) - d(m0)` from its own baseline run. Only local
modelling errors remain, which isolates the improvements. One raw-data run of
experiment 1 (`--data raw`) shows the limit. All runs use f0 = 300 MHz and
14 ns.

| Experiment | Question | Solvers |
|---|---|---|
| exp1 | Does forward accuracy change the FWI result? | 3 cm main grid; 3 cm + ratio-3 subgrid (box from the planner); uniform 1 cm (no modelling error; the data's own solver) |
| exp2 | Does the exact discrete adjoint matter? | same subgridded forward model with the exact adjoint, or with the 3 cm grid's adjoint (what one gets without the subgrid's own adjoint) |
| exp3 | Is the planner's grid size right? | 5 cm main grid (too coarse) against exp1's 3 cm, 3 cm + subgrid and 1 cm |
| exp4 | Without knowing where the targets are, does adjoint-driven AMR find the refinement? | three targets; 3 cm only; a box over the whole update region (equal here to the planner's oracle box on the true model); AMR (10 iterations per cycle, Doerfler 0.5, differential DWR indicator) |
| exp5 | Rugged surface: what do the DG band and the subgrid each contribute? | 3 cm staircase surface; staircase + subgrid; DG band (3 cm Yee); DG band + subgrid; data from DG band + 1 cm Yee |

### Results (300 MHz, 30 L-BFGS iterations; experiment 5: 20 iterations)

Notation: PE = held-out prediction error (accurate solver; 1 = start model).
"int dEps / dSig" = integrated anomaly relative to the true one, per target
(target 0: eps + sigma anomaly; target 1: sigma only; target 2: eps only).
Full table: `results/ablation/report.md`. Figures: `results/ablation/<exp>.png`
(misfit curves, PE, smoothed eps error, cost), `<exp>_models.png` (slices
through the targets) and `summary.png` (cost against PE).

| Exp | Solver | PE | train misfit | eps err (smoothed) | sig err (smoothed) | int dEps / dSig per target | s / gradient |
|---|---|---|---|---|---|---|---|
| 1 diff | 3 cm | 0.039 | 0.0101 | 0.868 (0.410) | 0.938 (0.499) | 0.46 / -0.43 ; - / 0.28 | 0.31 |
| | 3 cm + subgrid | **0.031** | 0.0097 | 0.849 (0.385) | 0.933 (0.472) | 0.50 / -0.41 ; - / 0.30 | 1.48 |
| | 1 cm (no modelling error) | 0.018 | 0.0073 | 0.840 (0.382) | 0.928 (0.455) | 0.50 / **+0.33** ; - / 0.29 | 19.0 |
| 1 raw | 3 cm | 2.64 | 2.77 | 2.02 (0.827) | 1.25 (1.60) | 0.30 / -0.86 ; - / -0.03 | 0.33 |
| | 3 cm + subgrid | 4.25 | 2.64 | 2.14 (0.886) | 1.26 (1.51) | 0.27 / -0.86 ; - / 0.08 | 1.61 |
| | 1 cm | 0.018 | 0.0073 | 0.840 (0.382) | 0.928 (0.455) | 0.50 / 0.33 ; - / 0.29 | 19.4 |
| 2 diff | subgrid, exact adjoint | 0.031 | 0.0096 | 0.849 (0.384) | 0.933 (0.472) | 0.50 / -0.41 ; - / 0.30 | 1.50 |
| | subgrid, 3 cm adjoint | 0.031 | 0.0097 | 0.856 (0.380) | 0.934 (0.477) | 0.50 / -0.44 ; - / 0.29 | 0.63 |
| 3 diff | 5 cm | 0.229 | 0.101 | 0.946 (0.641) | 0.962 (0.718) | 0.25 / -0.83 ; - / 0.21 | 0.28 |
| 4 diff | 3 cm | 0.033 | 0.0115 | 0.869 (0.405) | 0.936 (0.480) | 0.48 / -0.70 ; - / 0.32 ; 0.38 / - | 0.32 |
| | box over the region (= oracle) | 0.024 | 0.0100 | 0.850 (0.379) | 0.938 (0.497) | 0.52 / -0.65 ; - / 0.30 ; 0.38 / - | 1.97 |
| | AMR (box found after cycle 0) | **0.025** | 0.0104 | 0.857 (0.382) | 0.939 (0.507) | 0.51 / -0.64 ; - / 0.29 ; 0.37 / - | **1.47** |
| 5 diff | staircase surface | 1.31 | 0.219 | 0.989 (0.964) | 1.05 (1.45) | 0.28 / -1.59 ; - / 0.06 | 1.3 |
| | staircase + subgrid | 1.50 | 0.198 | 1.02 (1.03) | 1.04 (1.46) | 0.35 / -1.59 ; - / 0.09 | 6.0 |
| | DG band | 0.069 | 0.0182 | 0.859 (0.396) | 0.940 (0.480) | 0.47 / -1.41 ; - / 0.31 | 11.5 |
| | DG band + subgrid | 0.070 | 0.0165 | 0.842 (0.383) | 0.945 (0.484) | 0.51 / -1.38 ; - / 0.31 | 12.6 |

Modelling floors (true-model misfit relative to the target signal) in
experiment 5: staircase + subgrid 1.10, DG band 0.18, DG band + subgrid 0.09.
No run had a failed line search.

**What each improvement does to the FWI result**

1. **Subgridding (exp1).** With differential data, the subgrid lowers the
   held-out prediction error from 0.039 to 0.031. That closes 40 % of the gap
   to the error-free 1 cm inversion (0.018), at 1/13 of its cost per gradient.
   The smoothed eps error reaches the 1 cm value (0.385 vs 0.382).
   Not fixed: target 0's conductivity has the wrong sign on both 3 cm solvers
   and the right sign only at 1 cm. The residual modelling error outside the
   box (about 15 % of the target response at 300 MHz) moves the eps-sigma
   trade-off.
2. **Raw data (exp1 raw).** The 3 cm background error is 5x the target
   signal. Both 3 cm inversions end *worse* than the start model on held-out
   shots, the subgrid more so (4.25 vs 2.64): a more accurate box lets the
   inversion explain the background error with sharper artifacts. Local
   refinement does not substitute for an adequate main grid.
3. **Exact discrete adjoint (exp2).** The 3 cm adjoint used with the
   subgridded forward model gives a gradient with cosine 0.97 to the exact one
   (24 % relative difference). The first iteration is worse (misfit 0.78 vs
   0.50); from the second iteration on, L-BFGS with the strong-Wolfe line
   search (on the true objective) makes the curves coincide, and the end
   result is the same. In this test exactness buys a better first step, not
   a better model, and the approximate gradient is 2.4x cheaper. The exact
   adjoint remains what the gradient tests verify; it matters where the
   subgrid's sensitivity dominates.
4. **Grid size (exp3).** A 5 cm main grid gives PE 0.229, six times the 3 cm
   value, and it is not even cheaper at this size (GPU-overhead bound). The
   planner picks 3 cm with one box over the targets for `f_max = 1.5 f0`,
   which is the configuration of exp1. For `f_max = 2.5 f0` it picks a
   uniform 1 cm grid, consistent with the raw-data pre-study.
5. **Adjoint-driven AMR (exp4).** Without prior knowledge of the targets, the
   differential DWR indicator places, after one coarse cycle, the same box the
   oracle plan puts on the true model, and keeps it. AMR reaches the oracle's
   quality (PE 0.025 vs 0.024; 3 cm only: 0.033) at 25 % less cost per
   gradient, because the first cycle runs on the main grid.
   (In this 0.75 m domain the three targets merge into one box: separate boxes
   need about 24 cm between their inner surfaces.)
6. **DG surface band (exp5).** With a rugged surface the staircase solvers'
   modelling floor exceeds the target signal. Their inversions fit the data
   better than the start model but predict held-out shots worse (PE 1.3-1.5),
   and the subgrid does not help them. The DG band brings PE to 0.069 and the
   eps anomaly to 47 % of the true integral. The subgrid on top halves the
   floor (0.18 to 0.09) and improves eps slightly (0.396 to 0.383 smoothed,
   target 0: 0.47 to 0.51), but not PE (0.069 vs 0.070).

**Overall.** The DG band for the surface and an adequate main grid size have
the largest effect on the FWI result. The subgrid and AMR give consistent but
modest gains (20-30 % lower held-out prediction error) at a fraction of the
uniform-grid cost. Exactness of the adjoint did not change the converged
result here. Conductivities stay the weakest parameter in all runs.

**Limitations of these experiments.**

* One model family, one frequency, no noise.
* Differential data assume a known baseline.
* The 1 cm upper bound is the data's own solver (inverse crime by design).
* 20-30 iterations; longer runs may separate the solvers further.

Planner fixes found while setting up the experiments (covered by the tests):

* the admissible range now matches the solver's check;
* flag groups near the margin are clipped instead of discarded, and only
  groups touching the domain sides count as unrefinable;
* boxes that cannot be enlarged to take in nearby antennas are shrunk away
  from them;
* refinement persists across AMR regrids without growing.


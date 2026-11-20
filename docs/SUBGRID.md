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

Since Section 13 the whole step has a hand-written transpose (`fused.py`) and the
sweeps run as CUDA Graphs (`sweep.py`); the autograd sweep described here remains
the reference (`DEEPGPR_SUBGRID_AUTOGRAD_VJP=1`).

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


## 10. Equal-budget study: AMR vs static refinement vs uniform FDTD (`examples/subgrid/amr_budget_study.py`)

Question: at the same wall-clock budget, does adjoint-driven AMR give better FWI
models than static local refinement and uniform FDTD across models, frequencies and
noise? Run with `examples/subgrid/run_amr_budget_study.sh` (round 1) and
`run_amr_budget_study2.sh` (round 2); `amr_budget_study.py report` writes
`results/amr_budget/report.md`, summary figures and per-scenario GIFs
(`evolution.gif`: all methods at the same compute time; `depth_sweep.gif`;
`iso3d.gif`; `fdtd1_timeline.gif`).

Design: 0.75 x 1.2 x 1.2 m domain; models M1-M4 (compact sphere pair, pipe + air
void, dipping plate + boulder, random inclusions) and, in round 2, the fresh random
models M5/M6 used for validation only; two-plane acquisition, 7 training and 6
held-out shots; data from a uniform 0.5 cm grid (no inverse crime for any solver),
differential data plus Ricker-band noise; 200/300/400 MHz at 20 dB, 300 MHz also at
10 and 30 dB. Every method gets T = 150 s on an RTX 4090; its clock covers setup,
CUDA graphs, baseline runs, gradients, line searches, indicators and re-planning, and
the model scored at time t is the last iterate completed by t. Score: held-out
prediction error PE (noise-free responses, 1 cm evaluator; a 0.5 cm re-evaluation
gives the same ranking, Spearman 0.89-1.0), plus PE at 4/3 f0 and model errors.

Methods: uniform 3 cm and 1 cm FDTD; static 1 cm box over the whole update region or
at the survey centre (same fine-node count as AMR); oracle boxes from the true model;
AMR v1 (round 1: Doerfler marking of the differential DWR indicator, persistent
boxes, a priori criteria of the current model) dynamic and "adaptive once" (one
re-plan after 10 coarse iterations, then fixed); AMR v2 (round 2, "finer"):
indicator of the 3 cm grid alone, `regrid(gamma=0.2, persistent=False,
hysteresis=0.9)`, no a priori criteria, so boxes can shrink and move.

Results (24 scenarios):

1. **Adaptive once (v1) is the best method without prior knowledge** (best in 14 of
   24). On the 20 round-1 scenarios its PE is 0.75x that of 3 cm FDTD (16 wins) and
   0.74x that of the whole-region box (19 wins); on the fresh models 0.51x (4/4) and
   0.80x (3/4), and better than the oracle boxes in all 4.
2. **Frequency decides against the main grid**: at 400 MHz every refined solver
   halves the 3 cm PE; at 200 MHz uniform 3 cm FDTD is best in all 4 models (accurate
   enough and about 5x more iterations in the budget).
3. **Dynamic re-planning does not pay.** v1's second re-plan grows the box to almost
   the whole region (Doerfler marking of the diffuse remainder, plus L-BFGS restarts):
   PE 1.19x that of adaptive once on round 1. v2 avoids the growth (boxes 0.12-0.38 M
   fine nodes instead of 0.3-0.8 M, about twice the iterations) but its indicator
   boxes only the dominant target and leaves the others on the 3 cm grid: PE 1.45x
   that of adaptive once v1 (round 1, 19 of 20 worse; fresh models 4 of 4 worse).
   Smaller boxes are not better unless they cover every target that matters.
4. **Noise** between 10 and 30 dB changes PE by a few percent; modelling error
   dominates.
5. **Uniform 1 cm FDTD** (29-40 s per gradient) completes no L-BFGS iteration within
   150 s. Run for 1800 s it is the most accurate solver (PE 0.008-0.045, typically
   1.5-2.5x below the best 150 s result), but it needs 620-1700 s (4-11x the budget)
   to reach the PE of adaptive once at 150 s, and more than 1800 s in 3 of the
   200 MHz scenarios.

Limits: one acquisition geometry, differential data with a known baseline, one GPU;
the v2 settings were chosen on round-1 snapshots of M1-M4 (only M5/M6 are an
independent test); the 1 cm evaluator slightly favours the 1 cm solver.

## 11. Dynamic mesh and time-step scheduling (`schedule.py`)

`DeepGPR.subgrid.schedule` turns the pieces above (a priori planner, DWR
indicator, `regrid`, the hybrid's local time stepping, the subgrids' time ratio)
into a scheduling strategy for adaptive FWI that balances inversion error and
cost:

* `TimeStepLadder.from_solver(solver)` reads the time-step ladder of a layout
  (Yee macro step `dt`, DG sub-steps `p` of the local time stepping,
  `docs/HYBRID_DGTD.md` Section 12, subgrid time ratios `r`) and the unknown
  counts of its parts.
* `CostModel` predicts the wall-clock cost of a forward run or gradient as a
  linear model in the unknowns (`yee·N_yee + dg·N_dg·(p + k) + sub·Σ N_g r_g`, with `k` the split-penalty damping sub-steps of 12.2 in HYBRID_DGTD.md; the earlier `(2p − 1)/2` undercounted the band: a macro step evaluates the operator `2p` times)
  per macro step, times the steps and shots, times `gradient_factor` for a
  gradient). The defaults are the RTX 4090 rates of the fused kernels;
  `calibrate`/`measure` rescale them to the machine at hand.
  `fine_node_budget` inverts the model: the subgrid nodes affordable per
  gradient given the Yee/DG parts and a time budget.
* `Scheduler` keeps the misfit history and decides between cycles:
  **keep** while the misfit still falls (`plateau` test over `window`
  iterations) or fewer than `min_cycle` iterations passed since the last
  change; otherwise **refine** when the DWR estimate of the modelling error
  (`DWRIndicator.estimate`, same units as the misfit) is at least `balance`
  (default 0.25) of the misfit, **coarsen** when it is below
  `coarsen_factor × balance` and boxes exist, **keep** in between (error
  balancing: refinement only pays while the modelling error is a sizeable part
  of the residual). With `urgent = u` (opt-in, 0.5 is sensible) the layout is
  refined at once, without the plateau and `min_cycle` waits, when the estimate
  is at least `u` of the misfit: its own modelling error is then most of the
  residual, and iterating on it would mostly fit that error (Section 12: a check
  before the first iteration leaves the 2 cm level after 44 s instead of after
  four iterations, about 24 min). `plan` then calls `regrid` (maximum strategy `gamma`,
  hysteresis, non-persistent boxes) under the fine-node budget of the remaining
  time and iterations: the marking threshold is raised until the planned boxes
  fit, and the boxes with the least indicator mass are dropped if they still do
  not. Every decision is logged (`Scheduler.log`).

Tests: `tests/test_subgrid_schedule.py` (ladders of all solvers, calibration,
budget monotonicity, decision rules, planning under a budget).

With local time stepping in the hybrid (`HybridSubgridMaxwell3D(..., lts="auto",
flux="central")`) the main `dt` is the Yee step, so subgrids under the band must
sub-cycle (`time_ratio = ratio`, the default; `time_ratio = 1` only fits when the
band sets the global step). The split step of Section 6 shares the LTS half step
of the hybrid (`functional3d.h_half_step`), so without subgrids it reproduces
`HybridMaxwell3D` with LTS to round-off (`tests/test_hybrid_lts.py`).
`examples/subgrid/amr_scheduled.py` runs the scheduler on the sand-pit benchmark
(`amr_sandpit3d.py`, method `amr_sched*`: same model, data, optimiser, v3 marking
and scoring as `amr_dyn_v3`; only the decisions come from the scheduler). Findings
on M1 (2400 s budget, RTX 4090, 2026-11-07):

* The DWR estimate of the differential objective must be scaled like the misfit
  (`residual = 2 (d − obs) / norm`); so scaled, it is 1–4 % of the misfit during
  the first 45 iterations and 30–70 % after the first refinement. With the
  default `balance = 0.25` the scheduler therefore keeps the coarse grid at the
  first plateau (iteration 45, ratio 0.02): the differential estimate measures the
  change of the *anomaly response*, which is small while the model is still
  close to the background, although the coarse grid cannot fit the total data
  at all (forward floor 1.2). `balance = 0.01` refines at the first plateau and
  reproduces the layout of `adaptive once v3` (516 k fine nodes, two boxes).
* Held-out prediction error at equal wall time (`amr_sched_b`, balance 0.01,
  check every 5 iterations, min cycle 5):

  | t [s] | scheduled | dynamic v3 | adaptive once v3 | static whole | coarse | uniform fine |
  |---|---|---|---|---|---|---|
  | 75 | **0.216** | 0.230 | 0.232 | 1.000 | 0.203 | 1.000 |
  | 300 | **0.180** | 0.190 | 0.192 | 0.215 | 0.186 | 1.000 |
  | 600 | **0.160** | 0.170 | 0.187 | 0.185 | 0.176 | 0.215 |
  | 1200 | 0.155 | **0.153** | 0.168 | 0.164 | 0.167 | 0.210 |
  | 2400 | 0.159 | **0.144** | 0.159 | – | – | 0.182 |

  Staying coarse until the first plateau makes the scheduled run the best
  method up to 600 s and ties dynamic v3 at 1200 s; afterwards the cost
  control (the affordable cost per gradient shrinks with the remaining time at
  a fixed iteration target) made the layout oscillate (one box, two boxes, no
  box: 7 changes in 39 checks, each restarting L-BFGS), which cost the
  2400 s result. Two rules were added after this run: the plateau is counted
  on the current layout only with `min_cycle = 10`, and no layout change is
  made once less than `end_fraction = 0.1` of the budget remains. With them
  (`amr_sched_c`) the layout settles after two changes (one box of 0.6 M fine
  nodes from iteration ~50 on): PE 0.181 / 0.176 / 0.165 / 0.154 at
  300 / 600 / 1200 / 2400 s, i.e. a steadier run that ends better than the
  oscillating one (0.159) and than adaptive once v3 (0.159), but loses the
  early advantage of the coarse phase (600–1200 s now at the level of the
  static and coarse runs) and still trails dynamic v3 at 2400 s (0.144). On
  this benchmark the scheduler is therefore a robust, cost-aware default rather
  than a better strategy than the tuned fixed-interval re-planning; its
  advantage is that it needs no re-plan interval and no node budget as inputs.
* Cost model: calibrated on the measured gradient time of the current layout
  (0.9 s coarse, 6–7 s with two boxes), the fine-node budget of a 200-iteration
  plan is 0.79 M nodes, the cap of the whole-region box.

## 12. Scheduled refinement for 3D joint ε/σ FWI on rugged terrain (`examples/subgrid/amr_terrain3d.py`)

The scheduler of Section 11 on the hybrid with local time stepping: the deep
variant of the rugged-terrain model of `docs/HYBRID_DGTD.md` 12.6, deep enough for
Huygens subgrids under the DG band (a box needs 7 main cells of padding below the
band's two-cell gap).

**Set-up.** 2 cm model grid, 1.0 m deep, 0.8 m × 0.8 m, 8-cell absorbers; surface
0.17 m ± 7 cm (11 × 11 samples, sine pattern plus random roughness, PCHIP); P1
DG band 0.08–0.28 m (flux 0.1, with LTS the split penalty); ground εr 5, σ 3 mS/m;
an εr 7.5 sphere and, 24 cm away, a 40 mS/m sphere (radius 8 cm, centres 0.60 m
deep, about 0.43 m below the surface); a known reflector εr 8 below 0.86 m (in the
initial model); 3 × 3 sources, 7 × 7 receivers 4 cm below the top, 400 MHz Ricker,
16 ns, offsets below 12 cm muted; observed data from native DeepGPR on 4 mm; time
gain `(t − 1 ns)^2`; misfit, parametrisation and optimiser of HYBRID_DGTD.md 12.6
(L-BFGS, stage 1 εr only at 250 MHz, stage 2 joint, full band).

Levels (RTX 4090, float32; modelling error = data error of the level at the true
model against the 4 mm data, gained; gradient = nine shots):

| level | Yee | band stepping | macro dt | modelling error | gradient |
|---|---|---|---|---|---|
| coarse | 2 cm | LTS, p = 4 | 2.50e-11 s | 26.8 % | 96 s |
| whole-region box | 2 cm + one box (0.82 M nodes at 2/3 cm, 0.48–0.78 m deep, time ratio 3) | LTS, p = 4 | 2.50e-11 s | 23.8 % | 112 s |
| fine | 2/3 cm everywhere | global | 7.45e-12 s | 2.1 % | 216 s |

The anomaly signal of the observed data is 10.8 %: on the coarse level the
modelling error is 2.5 times the signal, and the largest admissible box removes
a tenth of it. 89 % of the gained data energy is the direct wave and the surface
reflection; the error sits in the 2 cm Yee cells around the antennas and in the
20 cm of 2 cm ground between the band (0.28 m) and the shallowest admissible box
top (0.48 m), where boxes cannot go. The fine level runs with global stepping:
at P1 with Yee 2/3 cm the LTS costs more than it saves (HYBRID_DGTD.md 12.6;
`lts="cost"` makes the same choice).

Methods, all from the same initial model:

* `fine`: the fine level throughout, 8 + 24 iterations (reference);
* `coarse`, `static`: the coarse level and the whole-region box, 8 + 8;
* `sched`: the scheduler, 8 + 8; absolute DWR estimate of the coarse level on
  shots 0, 4, 8; `balance = 0.01`, `urgent = 0.5`, budget 6000 s; a check at the
  start of every stage; when more than half of the indicator mass lies where
  boxes cannot go, a refinement becomes the switch to the fine level (global
  refinement).

The methods' own misfits are not comparable (each contains its own level's
modelling error), so `amr_terrain3d.py evaluate` simulates the model every method
holds at a given wall time with the fine level and scores it against the data
(`results/amr_terrain3d/evaluate.json`, `report.md`, `curves.png`).

**Results** (2026-11-08): data error with the fine level / εr error / σ error of
the model held at each wall time.

| method | 20 min | 40 min | 60 min | 90 min | 120 min | 180 min | end |
|---|---|---|---|---|---|---|---|
| fine | 10.1 % / 0.995 / 1.000 | 7.7 % / 0.983 / 1.000 | 6.6 % / 0.980 / 1.000 | 6.3 % / 0.978 / 1.000 | 2.2 % / 0.967 / 0.998 | 1.2 % / 0.961 / 0.999 | 0.87 % / 0.954 / 0.995 (281 min, 72 gradients) |
| sched | 10.9 % / 1.000 / 1.000 | 7.7 % / 0.983 / 1.000 | 6.8 % / 0.980 / 1.000 | 6.2 % / 0.977 / 1.000 | 2.2 % / 0.966 / 0.998 | 1.45 % / 0.963 / 0.998 | same (155 min, 39 gradients, 16 iterations) |
| static | 8.0 % / 1.002 / 1.000 | 12.5 % / 1.177 / 1.000 | 9.9 % / 1.245 / 1.015 | 9.0 % / 1.219 / 1.073 | – | – | 9.0 % / 1.219 / 1.073 (81 min) |
| coarse | 10.4 % / 1.079 / 1.000 | 14.4 % / 1.286 / 1.000 | 13.5 % / 1.261 / 1.074 | 13.3 % / 1.260 / 1.080 | – | – | 13.3 % / 1.260 / 1.080 (66 min) |

The initial model has a data error of 10.9 % (εr and σ errors 1).

* **The scheduler leaves the hopeless level before the first iteration.** At the
  check of iteration 0 the absolute estimate was −0.139 against a misfit of
  0.0339 (ratio 4.1; the actual difference of the two levels at the initial
  model is −0.028, ratio 0.83: the linearised estimate overshoots fivefold at
  this error level, but sign and order are right), with 98 % of |η| outside the
  region boxes can cover, so it switched to the fine level after a 44 s check.
  From then on the run is the fine reference shifted by about one minute: equal
  data errors at 40, 60, 90 and 120 min, the same model errors.
* **The cheap levels fit their own modelling error.** Coarse and static reduce
  their own misfits by 33 % and 25 % while the εr error rises to 1.26 and 1.22 and
  the σ error to 1.08 and 1.07 (worse than the initial model); with the fine level
  their data error ends at 13.3 % and 9.0 %. The box gives a short head start in
  the 250 MHz stage (8.0 % at 20 min against 10.1 %), lost as soon as the
  iterations exploit the modelling error.
* **The plateau rule alone would have wasted a sixth of the run.** A first
  `sched` run without the urgent rule (stopped, `run_sched_plateau_only.*`) spent
  8 gradient evaluations (13.6 min) on its first coarse iteration and changed the
  misfit by 0.006 %; the plateau test and `min_cycle = 4` would have kept the
  2 cm level for four iterations (about 24 min, judging by the `coarse` run, whose
  full-band misfit rose over them) before the same switch. This is why
  `Scheduler(urgent=...)` exists (Section 11).
* **Local refinement cannot help on this layout.** The error of the cheap level
  lies near the surface, above and in the band's neighbourhood, where Huygens
  boxes cannot go; the scheduler's correct decision is global refinement, and
  its benefit over the uniform fine grid is nil here (but it avoids the
  traps of the cheap levels at the cost of one indicator evaluation, 44 s). Boxes
  pay when the modelling error is at depth (Sections 10 and 11: the sand-pit
  benchmark); on rugged terrain they would need to reach the band, which the
  present subgrid–band coupling does not allow.
* **Deep targets stay weakly resolved.** After 32 iterations on the fine level
  the εr sphere reaches 5.32 (true 7.5), the σ sphere 3.9 mS/m (true 40); 62 % of
  the εr update energy lies at the targets' depth (0.48–0.66 m) and the update
  correlates with the true εr perturbation at 0.27; σ is updated mostly near the
  reflector. The data error with the fine level (0.87 %) ends below that level's
  own modelling error (2.1 %): the last iterations fit solver error.

## 13. Fused step, hand-written transpose and whole-sweep CUDA Graphs (`fused.py`, `sweep.py`)

Before this change the forward step called its kernels from Python (the fused Yee
updates of every grid, the precursor and correction kernels of `cuda.py`) and was
replayed as one CUDA Graph per step, with the state copied in and out
(`graphs.py`). The reverse sweep differentiated `hsg.step` (or `hybrid.step`)
with autograd, step by step and eagerly: the kernels' transposes as autograd
functions plus the clones, zero tensors and additions autograd needs. With the
DG band, the hybrid halves went through the PyTorch glue of
`functional3d.h_half_step` / `e_half_step`.

**What was changed.**
* `fused.py`: `step_into(st, c, u, out, ...)` writes `u^(n+1)` into given
  buffers; `vjp_into(...)` applies the hand-written transpose of the step and
  writes the cotangent of `u^n` into given buffers (destroying the input
  cotangent). It serves both solvers. For `HybridSubgridMaxwell3D` the main step
  is the fused global step of `DeepGPR.hybrid.fused3d`, split into its H and E
  halves (`step_h_part` / `step_e_part`, `vjp_e_part` / `vjp_h_part`) so that the
  subgrids' outer-surface H correction falls in between, as in `hybrid.step`.
* Order of the transpose of one main step (`r` sub-steps per subgrid):
  receivers^T; per subgrid OS E^T and the hsg_1 sub-steps^T; `pe_next`^T; the
  main E half^T (sources, Yee and DG E); per subgrid OS H^T, subgrid
  receivers^T and the hsg_2 sub-steps^T; `ph_cur`^T; the main H half^T;
  `pe_cur`^T and `ph_prev`^T into the cotangent of `u^n`.
* The E updates' transposes need their inputs for the `ca`/`cb` cotangents, so
  the reverse step first recomputes the `r − 1` intermediate states of every
  subgrid (fields and CPML auxiliaries) into scratch buffers; the main grid
  needs only `u^n` and `u^(n+1)`. The precursors of the new main fields
  (`ph_cur`, `pe_next`) are taken from `u^(n+1)`: `fused.available()` checks
  once per static that no precursor stencil reads a main node changed by an
  outer-surface correction (true for `is_os_sep ≥ 2` with the filter and
  disjoint subgrids); otherwise the autograd sweep is used.
* Point sources and receivers on every grid use the point kernels of
  `hybrid/csrc/glue3d.cu`. Subgrid receivers are sampled straight into the data
  (the receiver buffer of the state is not written on this path).
* `sweep.py`: the CUDA Graph arrangement of `hybrid/sweep3d.py` (`k` forward
  graphs `F_j`, `k` reverse graphs `R_j`, two ping-pong graphs for forward-only
  runs) with per-step buffers for the main-rate and subgrid-rate source
  samples, the main-grid and subgrid receiver data (subgrid receivers one step
  late: `nt + 1` steps) and their cotangents. Buffers belong to the solver,
  graphs to each propagator; the coefficients are refreshed by copy;
  `fit_k`, fall-back to the eager fused steps on out-of-memory, checkpoints on
  the device when they fit. `SubgridPropagator.release()` frees them.
* `DEEPGPR_SUBGRID_DISABLE_GRAPHS=1` runs the same fused steps eagerly. With
  `indicator=True` the reverse sweep runs eagerly on the fused transpose and
  copies the cotangents of the main E and H updates for the DWR indicator, on
  the segments of the autograd sweep (whose time term is skipped at each segment
  start, so the segment length is kept). `DEEPGPR_SUBGRID_AUTOGRAD_VJP=1` runs the
  autograd sweep.
* Not fused: a hybrid main step with local time stepping (`lts > 1`) keeps the
  autograd sweep.

**Numerical validation** (float64).
* One step, fused against `hsg.step` / `hybrid.step`: Yee main identical,
  hybrid main ≤ 3e-16. Fused transpose against reverse-mode differentiation of
  the reference step: ≤ 1.4e-15 for every state, coefficient (main and subgrid
  `ca`/`cb`), wavelet (main-rate and subgrid-rate) and indicator cotangent.
  Configurations: Yee main with one `r = 3` box, two boxes, an `r = 5` box
  (sources and receivers in the main grid and in the boxes, 2 shots); hybrid
  main with an `r = 3` box and with `time_ratio = 1` (2 shots).
* Propagator data and ε/σ/wavelet gradients (graph sweep, eager fused, autograd
  sweep; the models of `tests/test_subgrid_adjoint.py` and
  `tests/test_subgrid_hybrid.py`): ≤ 7e-15; DWR indicator 3e-16 (Yee main) and
  3e-14 (hybrid main).
* float32 at full length against the code before the change: data 3e-8–5e-8
  (Yee main) and 1e-6–2e-6 (hybrid main), ε gradient 1e-6–2e-6 and 8e-6.

**Speed.** RTX 4090 (server, idle card), float32, minimum of three runs after a
warm-up, ms per simulated ns of one propagation
(`examples/subgrid/subgrid_benchmark.py`).
* SP: the 3D sand-pit benchmark of Section 11 (`amr_sandpit3d.py`): Yee main
  grid 2.25 cm, 0.75 cm boxes, the 9 training shots in one propagation, 44
  receivers each, 16 ns (389 steps). `static`: one box over the whole update
  region (1.13 M fine nodes); `oracle`: the boxes around the targets of M1
  (three boxes, 0.73 M); `coarse`: the main grid alone; `fine`: uniform
  0.75 cm FDTD, one shot per propagation (as in the study).
* T: the rugged-terrain model of Section 12 (`amr_terrain3d.py`) with the
  global time step: Yee 2 cm, P1 band, one shot, 16 ns (2147 steps). `static`:
  one 2/3 cm box over the whole admissible region (0.82 M fine nodes);
  `coarse`: no box.

| model | forward before | forward after | gradient before | gradient after |
|---|---|---|---|---|
| SP static (9 shots) | 120.0 | **94.1** (1.28×) | 694.6 | **521.0** (1.33×) |
| SP oracle (9 shots) | 88.3 | **67.3** (1.31×) | 548.9 | **358.1** (1.53×) |
| SP coarse 2.25 cm (9 shots) | 14.0 | **6.9** (2.01×) | 54.6 | **30.8** (1.77×) |
| SP fine 0.75 cm (1 shot) | 63.1 | **36.7** (1.72×) | 290.8 | **170.5** (1.71×) |
| T static (hybrid + box) | 171.5 | **123.6** (1.39×) | 1921.9 | **563.2** (3.41×) |
| T coarse (hybrid) | 112.6 | **76.3** (1.48×) | 1078.4 | **320.1** (3.37×) |

Peak memory is unchanged except for the forward pass with boxes (+0.5 GB of
scratch for the recomputed intermediates).

After the change the boxes cost, relative to the coarse main grid alone, 13.6×
(forward) and 16.9× (gradient) for the whole-region box and 9.8× and 11.6× for
the oracle boxes; relative to uniform 0.75 cm FDTD for the same 9 shots
(9 × the one-shot time) the whole-region box is 3.5× (forward) and 2.9×
(gradient) faster, the oracle boxes 4.9× and 4.3×. Before the change these
advantages were 4.7× and 3.8× (whole region): uniform FDTD gained more from the
change than the boxes did. On the terrain model the box costs 1.62× (forward)
and 1.76× (gradient) of the hybrid alone.

**Where the time goes now** (SP static, gradient, eager profile on the local
RTX 4090, GPU time 23.8 ms per step): Yee transposes 49 %, Yee forward updates 42 % (forward pass,
segment recomputation and the in-step recomputation of the intermediates, the
last about 10 % of the total), correction, precursor and point kernels 9 %. The
remaining lever is the Yee transpose kernel itself (two passes through a
six-component scratch array and per-shot coefficient accumulators), shared with
the hybrid and the pure FDTD paths (not done).

**Follow-up: single-pass Yee transposes** (`docs/HYBRID_DGTD.md` 12.13). The Yee
transpose of every grid is now one kernel without the scratch array, and the
`ca`/`cb` cotangents are summed over the shots in the kernel (the `main_ca:upd`
and `sub_ca{g}:upd` accumulators hold `N` values instead of `S × N`). Exactness
as before (step transposes ≤ 1.8e-15, propagator gradients ≤ 5e-15, indicator
3.7e-14 in float64). Same protocol, before and after measured in one session on
another idle card (so the "before" column differs slightly from the "after"
column of the table above); forward times are unchanged:

| model | gradient before | gradient after |
|---|---|---|
| SP static (9 shots) | 520.6 | **416.7** (1.25×) |
| SP oracle (9 shots) | 359.6 | **296.2** (1.21×) |
| SP coarse 2.25 cm (9 shots) | 31.1 | **25.2** (1.23×) |
| SP fine 0.75 cm (1 shot) | 171.0 | **132.6** (1.29×) |
| T static (hybrid + box) | 563.1 | **511.9** (1.10×) |
| T coarse (hybrid) | 319.9 | **316.5** (1.01×) |

Uniform FDTD gained as much as the boxes: the whole-region box is now 2.86×
faster than uniform 0.75 cm FDTD for the 9 shots in the gradient (1193 against
417 ms/ns; 2.96× before), the oracle boxes 4.03× (4.28×), and the box costs
16.5× the coarse grid alone (16.7×). On the terrain model the box costs 1.62× of
the hybrid alone (1.76×); the hybrid alone is DG-bound and barely changes. The
SP static gradient now spends 54 % of its GPU time in the Yee updates (forward
pass and recomputation), 35 % in the Yee transposes and 11 % in the correction,
precursor and point kernels (eager profile, server, 16.5 ms per step against
21.1 before).

**Follow-up: compile-time term tables in the Yee update kernel**
(`docs/HYBRID_DGTD.md` 12.14). The Yee update kernel of every grid (forward pass
and recomputation) has no local-memory stack frame any more; its outputs are
bitwise identical. Same protocol, before and after alternating per model on one
idle card:

| model | forward before → after | gradient before → after |
|---|---|---|
| SP static (9 shots) | 94.2 → **90.4** (1.04×) | 411.6 → **402.8** (1.02×) |
| SP oracle (9 shots) | 67.9 → **59.3** (1.14×) | 296.1 → **275.3** (1.08×) |
| SP coarse 2.25 cm (9 shots) | 7.0 → **5.6** (1.26×) | 25.3 → **22.5** (1.12×) |
| SP fine 0.75 cm (1 shot) | 36.8 → **35.0** (1.05×) | 132.4 → **128.9** (1.03×) |
| T static (hybrid + box) | 124.2 → **112.7** (1.10×) | 512.3 → **482.8** (1.06×) |
| T coarse (hybrid) | 76.3 → **74.5** (1.02×) | 316.8 → **313.0** (1.01×) |

The small grids gain most (the update kernel of the 2.25 cm main grid with 9
shots 1.20–1.22×); the kernels of the whole-region box run close to the memory
bandwidth and gain 2–4 %. Against uniform 0.75 cm FDTD for the 9 shots, the
whole-region box is now 2.88× faster in the gradient (2.90× before, same session)
and 3.48× in the forward pass (3.52×), the oracle boxes 4.21× (4.02×) and 5.31×
(4.88×); in the gradient the box costs 17.9× the coarse grid alone (16.3×), and on
the terrain model 1.54× of the hybrid alone (1.62×). The SP static gradient now
spends 52 % of its GPU time in the Yee updates, 35 % in the Yee transposes and
13 % in the correction, precursor and point kernels (16.2 ms per step). The
gradients of the models with boxes are not bitwise reproducible from run to run
(the transposes of the correction accumulate with atomics; two runs of one code
differ by 2e-6 to 5e-6, relative L2, in float32).

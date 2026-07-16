#include <stddef.h>

#if defined(_OPENMP) || defined(DEEPGPR_USE_OPENMP)
#include <omp.h>
#define DEEPGPR_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#define DEEPGPR_OMP_ATOMIC_UPDATE _Pragma("omp atomic update")
#else
#define DEEPGPR_OMP_PARALLEL_FOR
#define DEEPGPR_OMP_ATOMIC_UPDATE
#endif

#ifdef _WIN32
#define DEEPGPR_API __declspec(dllexport)
#define RESTRICT __restrict
#else
#define DEEPGPR_API __attribute__((visibility("default")))
#define RESTRICT restrict
#endif

static const float E0 = 8.8541878128e-12f;
static const float M0 = 1.25663706212e-06f;

static int g_fdtd_order = 2;

DEEPGPR_API int deepgpr_abi_version(void)
{
    return 2;
}

/*
 * Set the spatial FDTD finite-difference order.
 *
 * Parameters:
 *   order: Requested order; 2 is used unless order is 4 or 8.
 */
DEEPGPR_API void set_fdtd_order(int order)
{
    g_fdtd_order = (order == 4 || order == 8) ? order : 2;
}

/*
 * Return the finite-difference stencil radius for an FDTD order.
 *
 * Parameters:
 *   order: Spatial finite-difference order.
 */
static int fdtd_radius_for_order(int order)
{
    if (order >= 8) return 4;
    if (order >= 4) return 2;
    return 1;
}

/*
 * Return one staggered finite-difference coefficient.
 *
 * Parameters:
 *   radius: Stencil radius.
 *   r: Coefficient index within the stencil.
 */
static float fdtd_coeff(int radius, int r)
{
    if (radius >= 4) {
        if (r == 1) return 1225.0f / 1024.0f;
        if (r == 2) return -245.0f / 3072.0f;
        if (r == 3) return 49.0f / 5120.0f;
        return -5.0f / 7168.0f;
    }
    if (radius == 3) {
        if (r == 1) return 75.0f / 64.0f;
        if (r == 2) return -25.0f / 384.0f;
        return 3.0f / 640.0f;
    }
    if (radius == 2) {
        if (r == 1) return 9.0f / 8.0f;
        return -1.0f / 24.0f;
    }
    return 1.0f;
}

/*
 * Clamp the backward-difference radius near model boundaries.
 *
 * Parameters:
 *   coord: Current grid coordinate along the derivative axis.
 *   n: Grid size along the derivative axis.
 *   requested: Requested stencil radius.
 */
static int usable_backward_radius(long long coord, long long n, int requested)
{
    int radius = requested;
    while (radius > 1 && (coord < radius || coord + radius - 1 >= n)) {
        --radius;
    }
    return radius;
}

/*
 * Clamp the forward-difference radius near model boundaries.
 *
 * Parameters:
 *   coord: Current grid coordinate along the derivative axis.
 *   n: Grid size along the derivative axis.
 *   requested: Requested stencil radius.
 */
static int usable_forward_radius(long long coord, long long n, int requested)
{
    int radius = requested;
    while (radius > 1 && (coord - radius + 1 < 0 || coord + radius >= n)) {
        --radius;
    }
    return radius;
}

/*
 * Compute a staggered backward spatial difference.
 *
 * Parameters:
 *   f: Field array to differentiate.
 *   id: Linear index of the current field sample.
 *   stride: Linear stride along the derivative axis.
 *   coord: Current grid coordinate along the derivative axis.
 *   n: Grid size along the derivative axis.
 *   order: Spatial finite-difference order.
 */
static float staggered_backward_diff(
    const float* RESTRICT f, long long id, long long stride,
    long long coord, long long n, int order)
{
    if (order <= 2) {
        return f[id] - f[id - stride];
    }

    int radius = usable_backward_radius(coord, n, fdtd_radius_for_order(order));
    float acc = 0.0f;
    for (int r = 1; r <= radius; ++r) {
        acc += fdtd_coeff(radius, r) * (f[id + (long long)(r - 1) * stride] - f[id - (long long)r * stride]);
    }
    return acc;
}

/*
 * Compute a staggered forward spatial difference.
 *
 * Parameters:
 *   f: Field array to differentiate.
 *   id: Linear index of the current field sample.
 *   stride: Linear stride along the derivative axis.
 *   coord: Current grid coordinate along the derivative axis.
 *   n: Grid size along the derivative axis.
 *   order: Spatial finite-difference order.
 */
static float staggered_forward_diff(
    const float* RESTRICT f, long long id, long long stride,
    long long coord, long long n, int order)
{
    if (order <= 2) {
        return f[id + stride] - f[id];
    }

    int radius = usable_forward_radius(coord, n, fdtd_radius_for_order(order));
    float acc = 0.0f;
    for (int r = 1; r <= radius; ++r) {
        acc += fdtd_coeff(radius, r) * (f[id + (long long)r * stride] - f[id - (long long)(r - 1) * stride]);
    }
    return acc;
}

/* Scatter the transpose of a staggered backward derivative. */
static void add_staggered_backward_adjoint_cpu(
    float* RESTRICT gradient, long long id, long long stride,
    long long coord, long long n, int order, float weight)
{
    int radius = order <= 2 ? 1 : usable_backward_radius(coord, n, fdtd_radius_for_order(order));
    for (int r = 1; r <= radius; ++r) {
        float value = weight * fdtd_coeff(radius, r);
        DEEPGPR_OMP_ATOMIC_UPDATE
        gradient[id + (long long)(r - 1) * stride] += value;
        DEEPGPR_OMP_ATOMIC_UPDATE
        gradient[id - (long long)r * stride] -= value;
    }
}

/* Scatter the transpose of a staggered forward derivative. */
static void add_staggered_forward_adjoint_cpu(
    float* RESTRICT gradient, long long id, long long stride,
    long long coord, long long n, int order, float weight)
{
    int radius = order <= 2 ? 1 : usable_forward_radius(coord, n, fdtd_radius_for_order(order));
    for (int r = 1; r <= radius; ++r) {
        float value = weight * fdtd_coeff(radius, r);
        DEEPGPR_OMP_ATOMIC_UPDATE
        gradient[id + (long long)r * stride] += value;
        DEEPGPR_OMP_ATOMIC_UPDATE
        gradient[id - (long long)(r - 1) * stride] -= value;
    }
}

/*
 * Build forward-update coefficients for electric and magnetic fields.
 *
 * Parameters:
 *   er: Padded relative permittivity array.
 *   se: Padded electrical conductivity array.
 *   mr: Padded relative permeability array.
 *   uE0, uE1, uE4: Output electric-field update coefficient arrays.
 *   uH0, uH1, uH4: Output magnetic-field update coefficient arrays.
 *   NX_FIELDS, NY_FIELDS, NZ_FIELDS: Padded field grid sizes.
 *   dt: Time step size.
 *   dx: Spatial grid spacing.
 */
static void ucgetforward_cpu(const float* RESTRICT er, const float* RESTRICT se, const float* RESTRICT mr,
    float* RESTRICT uE0, float* RESTRICT uE1, float* RESTRICT uE4,
    float* RESTRICT uH0, float* RESTRICT uH1, float* RESTRICT uH4,
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS, float dt, float dx)
{
    long long total = (long long)NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long idx;

    DEEPGPR_OMP_PARALLEL_FOR
    for (idx = 0; idx < total; ++idx) {
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ_FIELDS;
        long long k = rem % NZ_FIELDS;

        if (i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1)) {
            float HA = M0 * mr[idx] / dt;
            uH0[idx] = 1.0f;
            uH1[idx] = (1.0f / dx) / HA;
            uH4[idx] = 1.0f / HA;

            if (se[idx] > 100.0f) {
                uE0[idx] = 0.0f;
                uE1[idx] = 0.0f;
                uE4[idx] = 0.0f;
            } else {
                float e_term = E0 * er[idx] / dt;
                float s_term = 0.5f * se[idx];
                float EA = e_term + s_term;
                float EB = e_term - s_term;
                uE0[idx] = EB / EA;
                uE1[idx] = (1.0f / dx) / EA;
                uE4[idx] = 1.0f / EA;
            }
        }
    }
}

/*
 * Store receiver samples for all six field components.
 *
 * Parameters:
 *   step: Number of shots or simulations in the batch.
 *   NRX: Number of receivers per shot.
 *   iteration: Current time-step index.
 *   receiverlocation: Receiver coordinates with shape (step, NRX, 3).
 *   rxs: Output receiver sample array.
 *   Ex, Ey, Ez: Electric field component arrays.
 *   Hx, Hy, Hz: Magnetic field component arrays.
 *   NX, NY, NZ: Padded field grid sizes.
 *   N_ITER: Total number of time steps.
 */
static void store_outputs_cpu(
    int step, int NRX, int iteration,
    const int* RESTRICT receiverlocation, float* RESTRICT rxs,
    const float* RESTRICT Ex, const float* RESTRICT Ey, const float* RESTRICT Ez,
    const float* RESTRICT Hx, const float* RESTRICT Hy, const float* RESTRICT Hz,
    int NX, int NY, int NZ, int N_ITER)
{
    long long field_stride = (long long)NX * NY * NZ;
    long long total = (long long)step * NRX;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total; ++work) {
        int s = (int)(work / NRX);
        long long rx = work % NRX;
        long long i = receiverlocation[s * NRX * 3 + rx * 3 + 0];
        long long j = receiverlocation[s * NRX * 3 + rx * 3 + 1];
        long long k = receiverlocation[s * NRX * 3 + rx * 3 + 2];

        long long id4 = (long long)s * field_stride + i * NY * NZ + j * NZ + k;

        rxs[((s * 6 + 0) * N_ITER + iteration) * NRX + rx] = Ex[id4];
        rxs[((s * 6 + 1) * N_ITER + iteration) * NRX + rx] = Ey[id4];
        rxs[((s * 6 + 2) * N_ITER + iteration) * NRX + rx] = Ez[id4];
        rxs[((s * 6 + 3) * N_ITER + iteration) * NRX + rx] = Hx[id4];
        rxs[((s * 6 + 4) * N_ITER + iteration) * NRX + rx] = Hy[id4];
        rxs[((s * 6 + 5) * N_ITER + iteration) * NRX + rx] = Hz[id4];
    }
}

/*
 * Inject a Hertzian dipole source into one electric-field component.
 *
 * Parameters:
 *   step: Number of shots or simulations in the batch.
 *   iteration: Current time-step index.
 *   dx: Spatial grid spacing.
 *   sourcelocation: Source coordinates with shape (step, nsrc, 3).
 *   srcwaveforms: Source waveform array.
 *   Ex, Ey, Ez: Electric field component arrays to update.
 *   uE4: Electric source scaling coefficient array.
 *   NX, NY, NZ: Padded field grid sizes.
 *   nsrc: Number of sources per shot.
 *   polarisation: Source component, 0 for x, 1 for y, 2 for z.
 *   nt: Total number of time steps.
 */
static void Update_hertzian_dipole_cpu(
    int step, int iteration, float dx,
    const int* RESTRICT sourcelocation, const float* RESTRICT srcwaveforms,
    float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez, const float* RESTRICT uE4,
    int NX, int NY, int NZ, int nsrc, int polarisation, int nt)
{
    long long field_stride = (long long)NX * NY * NZ;
    long long total = (long long)step * nsrc;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total; ++work) {
        int s = (int)(work / nsrc);
        long long src = work % nsrc;
        float waveform_value = srcwaveforms[src * nt + iteration];
        float scale = waveform_value * dx / (dx * dx * dx);

        long long i = sourcelocation[s * nsrc * 3 + src * 3 + 0];
        long long j = sourcelocation[s * nsrc * 3 + src * 3 + 1];
        long long k = sourcelocation[s * nsrc * 3 + src * 3 + 2];

        long long id3 = i * NY * NZ + j * NZ + k;
        long long id4 = (long long)s * field_stride + id3;

        if (polarisation == 0) {
            DEEPGPR_OMP_ATOMIC_UPDATE
            Ex[id4] -= uE4[id3] * scale;
        } else if (polarisation == 1) {
            DEEPGPR_OMP_ATOMIC_UPDATE
            Ey[id4] -= uE4[id3] * scale;
        } else if (polarisation == 2) {
            DEEPGPR_OMP_ATOMIC_UPDATE
            Ez[id4] -= uE4[id3] * scale;
        }
    }
}

static void e_fields_base_update_cpu(
    const float* RESTRICT uE0, const float* RESTRICT uE1,
    float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez,
    const float* RESTRICT Hx, const float* RESTRICT Hy, const float* RESTRICT Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
    int fdtd_order)
{
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long field_stride = (long long)NX_FIELDS * ny_nz;
    long long total_work = (long long)step * field_stride;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ_FIELDS;
        long long k = rem % NZ_FIELDS;
        long long id4 = work;

        float ue0 = uE0[idx];
        float ue1 = uE1[idx];

        int do_ex = (((NY_FIELDS - 1) != 1 || (NZ_FIELDS - 1) != 1) && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));
        int do_ey = (((NX_FIELDS - 1) != 1 || (NZ_FIELDS - 1) != 1) && i > 0 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));
        int do_ez = (((NX_FIELDS - 1) != 1 || (NY_FIELDS - 1) != 1) && i > 0 && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));

        if (do_ex) {
            float dHz_dy = staggered_backward_diff(Hz, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order);
            float dHy_dz = staggered_backward_diff(Hy, id4, 1, k, NZ_FIELDS, fdtd_order);
            Ex[id4] = ue0 * Ex[id4] + ue1 * dHz_dy - ue1 * dHy_dz;
        }
        if (do_ey) {
            float dHx_dz = staggered_backward_diff(Hx, id4, 1, k, NZ_FIELDS, fdtd_order);
            float dHz_dx = staggered_backward_diff(Hz, id4, ny_nz, i, NX_FIELDS, fdtd_order);
            Ey[id4] = ue0 * Ey[id4] + ue1 * dHx_dz - ue1 * dHz_dx;
        }
        if (do_ez) {
            float dHy_dx = staggered_backward_diff(Hy, id4, ny_nz, i, NX_FIELDS, fdtd_order);
            float dHx_dy = staggered_backward_diff(Hx, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order);
            Ez[id4] = ue0 * Ez[id4] + ue1 * dHy_dx - ue1 * dHx_dy;
        }
    }
}

/*
 * Apply electric CPML boundary corrections after the base electric-field update.
 */
static void pml_e_fields_update_cpu(
    float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez,
    const float* RESTRICT Hx, const float* RESTRICT Hy, const float* RESTRICT Hz,
    float dx, float dy, float dz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* RESTRICT x0ER, const float* RESTRICT xmER,
    const float* RESTRICT y0ER, const float* RESTRICT ymER,
    const float* RESTRICT z0ER, const float* RESTRICT zmER,
    const float* RESTRICT updatecoeffsE,
    float* RESTRICT x0EPhi1, float* RESTRICT x0EPhi2,
    float* RESTRICT xmEPhi1, float* RESTRICT xmEPhi2,
    float* RESTRICT y0EPhi1, float* RESTRICT y0EPhi2,
    float* RESTRICT ymEPhi1, float* RESTRICT ymEPhi2,
    float* RESTRICT z0EPhi1, float* RESTRICT z0EPhi2,
    float* RESTRICT zmEPhi1, float* RESTRICT zmEPhi2,
    int fdtd_order)
{
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long field_stride = (long long)NX_FIELDS * ny_nz;
    long long total_work = (long long)step * field_stride;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        int s = (int)(work / field_stride);
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ_FIELDS;
        long long k = rem % NZ_FIELDS;

        int in_x0 = (pml0 > 0 && i > 0 && i <= pml0 && j < NY_FIELDS && k < NZ_FIELDS);
        int in_xm = (pml1 > 0 && i >= NX_FIELDS - 1 - pml1 && i < NX_FIELDS - 1 && j < NY_FIELDS && k < NZ_FIELDS);
        int in_y0 = (pml2 > 0 && i < NX_FIELDS && j > 0 && j <= pml2 && k < NZ_FIELDS);
        int in_ym = (pml3 > 0 && i < NX_FIELDS && j >= NY_FIELDS - 1 - pml3 && j < NY_FIELDS - 1 && k < NZ_FIELDS);
        int in_z0 = (pml4 > 0 && i < NX_FIELDS && j < NY_FIELDS && k > 0 && k <= pml4);
        int in_zm = (pml5 > 0 && i < NX_FIELDS && j < NY_FIELDS && k >= NZ_FIELDS - 1 - pml5 && k < NZ_FIELDS - 1);
        if (!(in_x0 || in_xm || in_y0 || in_ym || in_z0 || in_zm)) {
            continue;
        }

        float upd = updatecoeffsE[idx];
        long long id4 = work;

        {
            if (in_x0) {
                long long i1 = pml0 - i;
                float RA01 = x0ER[i1] - 1.0f, RB0 = x0ER[pml0 + i1], RE0 = x0ER[2 * pml0 + i1], RF0 = x0ER[3 * pml0 + i1];
                if (j < NY_FIELDS - 1 && i > 0) {
                    float dHz = staggered_backward_diff(Hz, id4, ny_nz, i, NX_FIELDS, fdtd_order) / dx;
                    long long p_idx = ((long long)s * (pml0 + 1) * (NY_FIELDS - 1) * NZ_FIELDS) + i1 * (NY_FIELDS - 1) * NZ_FIELDS + j * NZ_FIELDS + k;
                    float phi = x0EPhi1[p_idx];
                    Ey[id4] -= upd * (RA01 * dHz + RB0 * phi);
                    x0EPhi1[p_idx] = RE0 * phi - RF0 * dHz;
                }
                if (k < NZ_FIELDS - 1 && i > 0) {
                    float dHy = staggered_backward_diff(Hy, id4, ny_nz, i, NX_FIELDS, fdtd_order) / dx;
                    long long p_idx = ((long long)s * (pml0 + 1) * NY_FIELDS * (NZ_FIELDS - 1)) + i1 * NY_FIELDS * (NZ_FIELDS - 1) + j * (NZ_FIELDS - 1) + k;
                    float phi = x0EPhi2[p_idx];
                    Ez[id4] += upd * (RA01 * dHy + RB0 * phi);
                    x0EPhi2[p_idx] = RE0 * phi - RF0 * dHy;
                }
            }

            if (in_xm) {
                long long i1 = i - (NX_FIELDS - 1 - pml1);
                float RA01 = xmER[i1] - 1.0f, RB0 = xmER[pml1 + i1], RE0 = xmER[2 * pml1 + i1], RF0 = xmER[3 * pml1 + i1];
                if (j < NY_FIELDS - 1 && i > 0) {
                    float dHz = staggered_backward_diff(Hz, id4, ny_nz, i, NX_FIELDS, fdtd_order) / dx;
                    long long p_idx = ((long long)s * (pml1 + 1) * (NY_FIELDS - 1) * NZ_FIELDS) + i1 * (NY_FIELDS - 1) * NZ_FIELDS + j * NZ_FIELDS + k;
                    float phi = xmEPhi1[p_idx];
                    Ey[id4] -= upd * (RA01 * dHz + RB0 * phi);
                    xmEPhi1[p_idx] = RE0 * phi - RF0 * dHz;
                }
                if (k < NZ_FIELDS - 1 && i > 0) {
                    float dHy = staggered_backward_diff(Hy, id4, ny_nz, i, NX_FIELDS, fdtd_order) / dx;
                    long long p_idx = ((long long)s * (pml1 + 1) * NY_FIELDS * (NZ_FIELDS - 1)) + i1 * NY_FIELDS * (NZ_FIELDS - 1) + j * (NZ_FIELDS - 1) + k;
                    float phi = xmEPhi2[p_idx];
                    Ez[id4] += upd * (RA01 * dHy + RB0 * phi);
                    xmEPhi2[p_idx] = RE0 * phi - RF0 * dHy;
                }
            }

            if (in_y0) {
                long long j1 = pml2 - j;
                float RA01 = y0ER[j1] - 1.0f, RB0 = y0ER[pml2 + j1], RE0 = y0ER[2 * pml2 + j1], RF0 = y0ER[3 * pml2 + j1];
                if (i < NX_FIELDS - 1 && j > 0) {
                    float dHz = staggered_backward_diff(Hz, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order) / dy;
                    long long p_idx = ((long long)s * (NX_FIELDS - 1) * (pml2 + 1) * NZ_FIELDS) + i * (pml2 + 1) * NZ_FIELDS + j1 * NZ_FIELDS + k;
                    float phi = y0EPhi1[p_idx];
                    Ex[id4] += upd * (RA01 * dHz + RB0 * phi);
                    y0EPhi1[p_idx] = RE0 * phi - RF0 * dHz;
                }
                if (k < NZ_FIELDS - 1 && j > 0) {
                    float dHx = staggered_backward_diff(Hx, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order) / dy;
                    long long p_idx = ((long long)s * NX_FIELDS * (pml2 + 1) * (NZ_FIELDS - 1)) + i * (pml2 + 1) * (NZ_FIELDS - 1) + j1 * (NZ_FIELDS - 1) + k;
                    float phi = y0EPhi2[p_idx];
                    Ez[id4] -= upd * (RA01 * dHx + RB0 * phi);
                    y0EPhi2[p_idx] = RE0 * phi - RF0 * dHx;
                }
            }

            if (in_ym) {
                long long j1 = j - (NY_FIELDS - 1 - pml3);
                float RA01 = ymER[j1] - 1.0f, RB0 = ymER[pml3 + j1], RE0 = ymER[2 * pml3 + j1], RF0 = ymER[3 * pml3 + j1];
                if (i < NX_FIELDS - 1 && j > 0) {
                    float dHz = staggered_backward_diff(Hz, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order) / dy;
                    long long p_idx = ((long long)s * (NX_FIELDS - 1) * (pml3 + 1) * NZ_FIELDS) + i * (pml3 + 1) * NZ_FIELDS + j1 * NZ_FIELDS + k;
                    float phi = ymEPhi1[p_idx];
                    Ex[id4] += upd * (RA01 * dHz + RB0 * phi);
                    ymEPhi1[p_idx] = RE0 * phi - RF0 * dHz;
                }
                if (k < NZ_FIELDS - 1 && j > 0) {
                    float dHx = staggered_backward_diff(Hx, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order) / dy;
                    long long p_idx = ((long long)s * NX_FIELDS * (pml3 + 1) * (NZ_FIELDS - 1)) + i * (pml3 + 1) * (NZ_FIELDS - 1) + j1 * (NZ_FIELDS - 1) + k;
                    float phi = ymEPhi2[p_idx];
                    Ez[id4] -= upd * (RA01 * dHx + RB0 * phi);
                    ymEPhi2[p_idx] = RE0 * phi - RF0 * dHx;
                }
            }

            if (in_z0) {
                long long k1 = pml4 - k;
                float RA01 = z0ER[k1] - 1.0f, RB0 = z0ER[pml4 + k1], RE0 = z0ER[2 * pml4 + k1], RF0 = z0ER[3 * pml4 + k1];
                if (i < NX_FIELDS - 1 && k > 0) {
                    float dHy = staggered_backward_diff(Hy, id4, 1, k, NZ_FIELDS, fdtd_order) / dz;
                    long long p_idx = ((long long)s * (NX_FIELDS - 1) * NY_FIELDS * (pml4 + 1)) + i * NY_FIELDS * (pml4 + 1) + j * (pml4 + 1) + k1;
                    float phi = z0EPhi1[p_idx];
                    Ex[id4] -= upd * (RA01 * dHy + RB0 * phi);
                    z0EPhi1[p_idx] = RE0 * phi - RF0 * dHy;
                }
                if (j < NY_FIELDS - 1 && k > 0) {
                    float dHx = staggered_backward_diff(Hx, id4, 1, k, NZ_FIELDS, fdtd_order) / dz;
                    long long p_idx = ((long long)s * NX_FIELDS * (NY_FIELDS - 1) * (pml4 + 1)) + i * (NY_FIELDS - 1) * (pml4 + 1) + j * (pml4 + 1) + k1;
                    float phi = z0EPhi2[p_idx];
                    Ey[id4] += upd * (RA01 * dHx + RB0 * phi);
                    z0EPhi2[p_idx] = RE0 * phi - RF0 * dHx;
                }
            }

            if (in_zm) {
                long long k1 = k - (NZ_FIELDS - 1 - pml5);
                float RA01 = zmER[k1] - 1.0f, RB0 = zmER[pml5 + k1], RE0 = zmER[2 * pml5 + k1], RF0 = zmER[3 * pml5 + k1];
                if (i < NX_FIELDS - 1 && k > 0) {
                    float dHy = staggered_backward_diff(Hy, id4, 1, k, NZ_FIELDS, fdtd_order) / dz;
                    long long p_idx = ((long long)s * (NX_FIELDS - 1) * NY_FIELDS * (pml5 + 1)) + i * NY_FIELDS * (pml5 + 1) + j * (pml5 + 1) + k1;
                    float phi = zmEPhi1[p_idx];
                    Ex[id4] -= upd * (RA01 * dHy + RB0 * phi);
                    zmEPhi1[p_idx] = RE0 * phi - RF0 * dHy;
                }
                if (j < NY_FIELDS - 1 && k > 0) {
                    float dHx = staggered_backward_diff(Hx, id4, 1, k, NZ_FIELDS, fdtd_order) / dz;
                    long long p_idx = ((long long)s * NX_FIELDS * (NY_FIELDS - 1) * (pml5 + 1)) + i * (NY_FIELDS - 1) * (pml5 + 1) + j * (pml5 + 1) + k1;
                    float phi = zmEPhi2[p_idx];
                    Ey[id4] += upd * (RA01 * dHx + RB0 * phi);
                    zmEPhi2[p_idx] = RE0 * phi - RF0 * dHx;
                }
            }
        }
    }
}

static void pml_backward_derivative_adjoint_cpu(
    float lambda_field, float* RESTRICT lambda_source,
    long long source_id, long long stride, long long coord, long long n, int order,
    float inverse_spacing, float update_coeff, float sign,
    float ra_minus_one, float rb, float re, float rf,
    float* RESTRICT lambda_phi);

/* Apply the exact transpose of the electric CPML correction. */
static void pml_e_fields_adjoint_cpu(
    float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez,
    float* RESTRICT Hx, float* RESTRICT Hy, float* RESTRICT Hz,
    float dx, float dy, float dz, int step, int NX, int NY, int NZ,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* RESTRICT x0R, const float* RESTRICT xmR,
    const float* RESTRICT y0R, const float* RESTRICT ymR,
    const float* RESTRICT z0R, const float* RESTRICT zmR,
    const float* RESTRICT update,
    float* RESTRICT x0P1, float* RESTRICT x0P2,
    float* RESTRICT xmP1, float* RESTRICT xmP2,
    float* RESTRICT y0P1, float* RESTRICT y0P2,
    float* RESTRICT ymP1, float* RESTRICT ymP2,
    float* RESTRICT z0P1, float* RESTRICT z0P2,
    float* RESTRICT zmP1, float* RESTRICT zmP2, int order)
{
    long long ny_nz = (long long)NY * NZ;
    long long field_stride = (long long)NX * ny_nz;
    long long total_work = (long long)step * field_stride;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        int s = (int)(work / field_stride);
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ;
        long long k = rem % NZ;
        float upd = update[idx];

#define APPLY_E_PML(R, P, p, q, stride, coord, n, spacing, field, source, sign) \
        do { \
            float ra = (R)[q] - 1.0f; \
            float rb = (R)[(p) + (q)]; \
            float re = (R)[2 * (p) + (q)]; \
            float rf = (R)[3 * (p) + (q)]; \
            pml_backward_derivative_adjoint_cpu( \
                (field)[work], (source), work, (stride), (coord), (n), order, \
                1.0f / (spacing), upd, (sign), ra, rb, re, rf, &(P)[p_idx]); \
        } while (0)

        if (pml0 > 0 && i > 0 && i <= pml0) {
            long long q = pml0 - i;
            if (j < NY - 1) {
                long long p_idx = ((long long)s * (pml0 + 1) * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
                APPLY_E_PML(x0R, x0P1, pml0, q, ny_nz, i, NX, dx, Ey, Hz, -1.0f);
            }
            if (k < NZ - 1) {
                long long p_idx = ((long long)s * (pml0 + 1) * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
                APPLY_E_PML(x0R, x0P2, pml0, q, ny_nz, i, NX, dx, Ez, Hy, 1.0f);
            }
        }
        if (pml1 > 0 && i >= NX - 1 - pml1 && i < NX - 1) {
            long long q = i - (NX - 1 - pml1);
            if (j < NY - 1 && i > 0) {
                long long p_idx = ((long long)s * (pml1 + 1) * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
                APPLY_E_PML(xmR, xmP1, pml1, q, ny_nz, i, NX, dx, Ey, Hz, -1.0f);
            }
            if (k < NZ - 1 && i > 0) {
                long long p_idx = ((long long)s * (pml1 + 1) * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
                APPLY_E_PML(xmR, xmP2, pml1, q, ny_nz, i, NX, dx, Ez, Hy, 1.0f);
            }
        }
        if (pml2 > 0 && j > 0 && j <= pml2) {
            long long q = pml2 - j;
            if (i < NX - 1) {
                long long p_idx = ((long long)s * (NX - 1) * (pml2 + 1) * NZ) + i * (pml2 + 1) * NZ + q * NZ + k;
                APPLY_E_PML(y0R, y0P1, pml2, q, NZ, j, NY, dy, Ex, Hz, 1.0f);
            }
            if (k < NZ - 1) {
                long long p_idx = ((long long)s * NX * (pml2 + 1) * (NZ - 1)) + i * (pml2 + 1) * (NZ - 1) + q * (NZ - 1) + k;
                APPLY_E_PML(y0R, y0P2, pml2, q, NZ, j, NY, dy, Ez, Hx, -1.0f);
            }
        }
        if (pml3 > 0 && j >= NY - 1 - pml3 && j < NY - 1) {
            long long q = j - (NY - 1 - pml3);
            if (i < NX - 1 && j > 0) {
                long long p_idx = ((long long)s * (NX - 1) * (pml3 + 1) * NZ) + i * (pml3 + 1) * NZ + q * NZ + k;
                APPLY_E_PML(ymR, ymP1, pml3, q, NZ, j, NY, dy, Ex, Hz, 1.0f);
            }
            if (k < NZ - 1 && j > 0) {
                long long p_idx = ((long long)s * NX * (pml3 + 1) * (NZ - 1)) + i * (pml3 + 1) * (NZ - 1) + q * (NZ - 1) + k;
                APPLY_E_PML(ymR, ymP2, pml3, q, NZ, j, NY, dy, Ez, Hx, -1.0f);
            }
        }
        if (pml4 > 0 && k > 0 && k <= pml4) {
            long long q = pml4 - k;
            if (i < NX - 1) {
                long long p_idx = ((long long)s * (NX - 1) * NY * (pml4 + 1)) + i * NY * (pml4 + 1) + j * (pml4 + 1) + q;
                APPLY_E_PML(z0R, z0P1, pml4, q, 1, k, NZ, dz, Ex, Hy, -1.0f);
            }
            if (j < NY - 1) {
                long long p_idx = ((long long)s * NX * (NY - 1) * (pml4 + 1)) + i * (NY - 1) * (pml4 + 1) + j * (pml4 + 1) + q;
                APPLY_E_PML(z0R, z0P2, pml4, q, 1, k, NZ, dz, Ey, Hx, 1.0f);
            }
        }
        if (pml5 > 0 && k >= NZ - 1 - pml5 && k < NZ - 1) {
            long long q = k - (NZ - 1 - pml5);
            if (i < NX - 1 && k > 0) {
                long long p_idx = ((long long)s * (NX - 1) * NY * (pml5 + 1)) + i * NY * (pml5 + 1) + j * (pml5 + 1) + q;
                APPLY_E_PML(zmR, zmP1, pml5, q, 1, k, NZ, dz, Ex, Hy, -1.0f);
            }
            if (j < NY - 1 && k > 0) {
                long long p_idx = ((long long)s * NX * (NY - 1) * (pml5 + 1)) + i * (NY - 1) * (pml5 + 1) + j * (pml5 + 1) + q;
                APPLY_E_PML(zmR, zmP2, pml5, q, 1, k, NZ, dz, Ey, Hx, 1.0f);
            }
        }
#undef APPLY_E_PML
    }
}

static void h_fields_base_update_cpu(
    const float* RESTRICT uH0, const float* RESTRICT uH1,
    const float* RESTRICT Ex, const float* RESTRICT Ey, const float* RESTRICT Ez,
    float* RESTRICT Hx, float* RESTRICT Hy, float* RESTRICT Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
    int fdtd_order)
{
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long field_stride = (long long)NX_FIELDS * ny_nz;
    long long total_work = (long long)step * field_stride;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ_FIELDS;
        long long k = rem % NZ_FIELDS;
        long long id4 = work;

        float uh0 = uH0[idx];
        float uh1 = uH1[idx];

        int do_hx = ((NX_FIELDS - 1) != 1 && i > 0 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));
        int do_hy = ((NY_FIELDS - 1) != 1 && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));
        int do_hz = ((NZ_FIELDS - 1) != 1 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));

        if (do_hx) {
            float dEz_dy = staggered_forward_diff(Ez, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order);
            float dEy_dz = staggered_forward_diff(Ey, id4, 1, k, NZ_FIELDS, fdtd_order);
            Hx[id4] = uh0 * Hx[id4] - uh1 * dEz_dy + uh1 * dEy_dz;
        }
        if (do_hy) {
            float dEx_dz = staggered_forward_diff(Ex, id4, 1, k, NZ_FIELDS, fdtd_order);
            float dEz_dx = staggered_forward_diff(Ez, id4, ny_nz, i, NX_FIELDS, fdtd_order);
            Hy[id4] = uh0 * Hy[id4] - uh1 * dEx_dz + uh1 * dEz_dx;
        }
        if (do_hz) {
            float dEy_dx = staggered_forward_diff(Ey, id4, ny_nz, i, NX_FIELDS, fdtd_order);
            float dEx_dy = staggered_forward_diff(Ex, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order);
            Hz[id4] = uh0 * Hz[id4] - uh1 * dEy_dx + uh1 * dEx_dy;
        }
    }
}

/* Apply the exact transpose of the electric-field base update. */
static void e_fields_base_adjoint_cpu(
    const float* RESTRICT uE0, const float* RESTRICT uE1,
    float* RESTRICT lambda_Ex, float* RESTRICT lambda_Ey, float* RESTRICT lambda_Ez,
    float* RESTRICT lambda_Hx, float* RESTRICT lambda_Hy, float* RESTRICT lambda_Hz,
    int step, int NX, int NY, int NZ, int order)
{
    long long ny_nz = (long long)NY * NZ;
    long long field_stride = (long long)NX * ny_nz;
    long long total_work = (long long)step * field_stride;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ;
        long long k = rem % NZ;
        int do_ex = (((NY - 1) != 1 || (NZ - 1) != 1) && i < NX - 1 && j > 0 && j < NY - 1 && k > 0 && k < NZ - 1);
        int do_ey = (((NX - 1) != 1 || (NZ - 1) != 1) && i > 0 && i < NX - 1 && j < NY - 1 && k > 0 && k < NZ - 1);
        int do_ez = (((NX - 1) != 1 || (NY - 1) != 1) && i > 0 && i < NX - 1 && j > 0 && j < NY - 1 && k < NZ - 1);
        float coeff = uE1[idx];

        if (do_ex) {
            float value = lambda_Ex[work];
            add_staggered_backward_adjoint_cpu(lambda_Hz, work, NZ, j, NY, order, coeff * value);
            add_staggered_backward_adjoint_cpu(lambda_Hy, work, 1, k, NZ, order, -coeff * value);
            lambda_Ex[work] = uE0[idx] * value;
        }
        if (do_ey) {
            float value = lambda_Ey[work];
            add_staggered_backward_adjoint_cpu(lambda_Hx, work, 1, k, NZ, order, coeff * value);
            add_staggered_backward_adjoint_cpu(lambda_Hz, work, ny_nz, i, NX, order, -coeff * value);
            lambda_Ey[work] = uE0[idx] * value;
        }
        if (do_ez) {
            float value = lambda_Ez[work];
            add_staggered_backward_adjoint_cpu(lambda_Hy, work, ny_nz, i, NX, order, coeff * value);
            add_staggered_backward_adjoint_cpu(lambda_Hx, work, NZ, j, NY, order, -coeff * value);
            lambda_Ez[work] = uE0[idx] * value;
        }
    }
}

/* Apply the exact transpose of the magnetic-field base update. */
static void h_fields_base_adjoint_cpu(
    const float* RESTRICT uH0, const float* RESTRICT uH1,
    float* RESTRICT lambda_Ex, float* RESTRICT lambda_Ey, float* RESTRICT lambda_Ez,
    float* RESTRICT lambda_Hx, float* RESTRICT lambda_Hy, float* RESTRICT lambda_Hz,
    int step, int NX, int NY, int NZ, int order)
{
    long long ny_nz = (long long)NY * NZ;
    long long field_stride = (long long)NX * ny_nz;
    long long total_work = (long long)step * field_stride;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ;
        long long k = rem % NZ;
        int do_hx = ((NX - 1) != 1 && i > 0 && i < NX - 1 && j < NY - 1 && k < NZ - 1);
        int do_hy = ((NY - 1) != 1 && i < NX - 1 && j > 0 && j < NY - 1 && k < NZ - 1);
        int do_hz = ((NZ - 1) != 1 && i < NX - 1 && j < NY - 1 && k > 0 && k < NZ - 1);
        float coeff = uH1[idx];

        if (do_hx) {
            float value = lambda_Hx[work];
            add_staggered_forward_adjoint_cpu(lambda_Ez, work, NZ, j, NY, order, -coeff * value);
            add_staggered_forward_adjoint_cpu(lambda_Ey, work, 1, k, NZ, order, coeff * value);
            lambda_Hx[work] = uH0[idx] * value;
        }
        if (do_hy) {
            float value = lambda_Hy[work];
            add_staggered_forward_adjoint_cpu(lambda_Ex, work, 1, k, NZ, order, -coeff * value);
            add_staggered_forward_adjoint_cpu(lambda_Ez, work, ny_nz, i, NX, order, coeff * value);
            lambda_Hy[work] = uH0[idx] * value;
        }
        if (do_hz) {
            float value = lambda_Hz[work];
            add_staggered_forward_adjoint_cpu(lambda_Ey, work, ny_nz, i, NX, order, -coeff * value);
            add_staggered_forward_adjoint_cpu(lambda_Ex, work, NZ, j, NY, order, coeff * value);
            lambda_Hz[work] = uH0[idx] * value;
        }
    }
}

static void pml_backward_derivative_adjoint_cpu(
    float lambda_field, float* RESTRICT lambda_source,
    long long source_id, long long stride, long long coord, long long n, int order,
    float inverse_spacing, float update_coeff, float sign,
    float ra_minus_one, float rb, float re, float rf,
    float* RESTRICT lambda_phi)
{
    float phi_new = *lambda_phi;
    float derivative_weight =
        (sign * update_coeff * ra_minus_one * lambda_field - rf * phi_new) * inverse_spacing;
    add_staggered_backward_adjoint_cpu(
        lambda_source, source_id, stride, coord, n, order, derivative_weight);
    *lambda_phi = sign * update_coeff * rb * lambda_field + re * phi_new;
}

static void pml_forward_derivative_adjoint_cpu(
    float lambda_field, float* RESTRICT lambda_source,
    long long source_id, long long stride, long long coord, long long n, int order,
    float inverse_spacing, float update_coeff, float sign,
    float ra_minus_one, float rb, float re, float rf,
    float* RESTRICT lambda_phi)
{
    float phi_new = *lambda_phi;
    float derivative_weight =
        (sign * update_coeff * ra_minus_one * lambda_field - rf * phi_new) * inverse_spacing;
    add_staggered_forward_adjoint_cpu(
        lambda_source, source_id, stride, coord, n, order, derivative_weight);
    *lambda_phi = sign * update_coeff * rb * lambda_field + re * phi_new;
}

/*
 * Apply magnetic CPML boundary corrections after the base magnetic-field update.
 */
static void pml_h_fields_update_cpu(
    const float* RESTRICT Ex, const float* RESTRICT Ey, const float* RESTRICT Ez,
    float* RESTRICT Hx, float* RESTRICT Hy, float* RESTRICT Hz,
    float dx, float dy, float dz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* RESTRICT x0HR, const float* RESTRICT xmHR,
    const float* RESTRICT y0HR, const float* RESTRICT ymHR,
    const float* RESTRICT z0HR, const float* RESTRICT zmHR,
    const float* RESTRICT updatecoeffsH,
    float* RESTRICT x0HPhi1, float* RESTRICT x0HPhi2,
    float* RESTRICT xmHPhi1, float* RESTRICT xmHPhi2,
    float* RESTRICT y0HPhi1, float* RESTRICT y0HPhi2,
    float* RESTRICT ymHPhi1, float* RESTRICT ymHPhi2,
    float* RESTRICT z0HPhi1, float* RESTRICT z0HPhi2,
    float* RESTRICT zmHPhi1, float* RESTRICT zmHPhi2,
    int fdtd_order)
{
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long field_stride = (long long)NX_FIELDS * ny_nz;
    long long total_work = (long long)step * field_stride;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        int s = (int)(work / field_stride);
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ_FIELDS;
        long long k = rem % NZ_FIELDS;

        int in_x0 = (pml0 > 0 && i < pml0 && j < NY_FIELDS && k < NZ_FIELDS);
        int in_xm = (pml1 > 0 && i >= NX_FIELDS - 1 - pml1 && i < NX_FIELDS - 1 && j < NY_FIELDS && k < NZ_FIELDS);
        int in_y0 = (pml2 > 0 && i < NX_FIELDS && j < pml2 && k < NZ_FIELDS);
        int in_ym = (pml3 > 0 && i < NX_FIELDS && j >= NY_FIELDS - 1 - pml3 && j < NY_FIELDS - 1 && k < NZ_FIELDS);
        int in_z0 = (pml4 > 0 && i < NX_FIELDS && j < NY_FIELDS && k < pml4);
        int in_zm = (pml5 > 0 && i < NX_FIELDS && j < NY_FIELDS && k >= NZ_FIELDS - 1 - pml5 && k < NZ_FIELDS - 1);
        if (!(in_x0 || in_xm || in_y0 || in_ym || in_z0 || in_zm)) {
            continue;
        }

        float upd = updatecoeffsH[idx];
        long long id4 = work;

        {
            if (in_x0) {
                long long i1 = pml0 - 1 - i;
                float RA01 = x0HR[i1] - 1.0f, RB0 = x0HR[pml0 + i1], RE0 = x0HR[2 * pml0 + i1], RF0 = x0HR[3 * pml0 + i1];
                if (k < NZ_FIELDS - 1) {
                    float dEz = staggered_forward_diff(Ez, id4, ny_nz, i, NX_FIELDS, fdtd_order) / dx;
                    long long p_idx = ((long long)s * pml0 * NY_FIELDS * (NZ_FIELDS - 1)) + i1 * NY_FIELDS * (NZ_FIELDS - 1) + j * (NZ_FIELDS - 1) + k;
                    float phi = x0HPhi1[p_idx];
                    Hy[id4] += upd * (RA01 * dEz + RB0 * phi);
                    x0HPhi1[p_idx] = RE0 * phi - RF0 * dEz;
                }
                if (j < NY_FIELDS - 1) {
                    float dEy = staggered_forward_diff(Ey, id4, ny_nz, i, NX_FIELDS, fdtd_order) / dx;
                    long long p_idx = ((long long)s * pml0 * (NY_FIELDS - 1) * NZ_FIELDS) + i1 * (NY_FIELDS - 1) * NZ_FIELDS + j * NZ_FIELDS + k;
                    float phi = x0HPhi2[p_idx];
                    Hz[id4] -= upd * (RA01 * dEy + RB0 * phi);
                    x0HPhi2[p_idx] = RE0 * phi - RF0 * dEy;
                }
            }

            if (in_xm) {
                long long i1 = i - (NX_FIELDS - 1 - pml1);
                float RA01 = xmHR[i1] - 1.0f, RB0 = xmHR[pml1 + i1], RE0 = xmHR[2 * pml1 + i1], RF0 = xmHR[3 * pml1 + i1];
                if (k < NZ_FIELDS - 1) {
                    float dEz = staggered_forward_diff(Ez, id4, ny_nz, i, NX_FIELDS, fdtd_order) / dx;
                    long long p_idx = ((long long)s * pml1 * NY_FIELDS * (NZ_FIELDS - 1)) + i1 * NY_FIELDS * (NZ_FIELDS - 1) + j * (NZ_FIELDS - 1) + k;
                    float phi = xmHPhi1[p_idx];
                    Hy[id4] += upd * (RA01 * dEz + RB0 * phi);
                    xmHPhi1[p_idx] = RE0 * phi - RF0 * dEz;
                }
                if (j < NY_FIELDS - 1) {
                    float dEy = staggered_forward_diff(Ey, id4, ny_nz, i, NX_FIELDS, fdtd_order) / dx;
                    long long p_idx = ((long long)s * pml1 * (NY_FIELDS - 1) * NZ_FIELDS) + i1 * (NY_FIELDS - 1) * NZ_FIELDS + j * NZ_FIELDS + k;
                    float phi = xmHPhi2[p_idx];
                    Hz[id4] -= upd * (RA01 * dEy + RB0 * phi);
                    xmHPhi2[p_idx] = RE0 * phi - RF0 * dEy;
                }
            }

            if (in_y0) {
                long long j1 = pml2 - 1 - j;
                float RA01 = y0HR[j1] - 1.0f, RB0 = y0HR[pml2 + j1], RE0 = y0HR[2 * pml2 + j1], RF0 = y0HR[3 * pml2 + j1];
                if (i < NX_FIELDS && k < NZ_FIELDS - 1) {
                    float dEz = staggered_forward_diff(Ez, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order) / dy;
                    long long p_idx = ((long long)s * NX_FIELDS * pml2 * (NZ_FIELDS - 1)) + i * pml2 * (NZ_FIELDS - 1) + j1 * (NZ_FIELDS - 1) + k;
                    float phi = y0HPhi1[p_idx];
                    Hx[id4] -= upd * (RA01 * dEz + RB0 * phi);
                    y0HPhi1[p_idx] = RE0 * phi - RF0 * dEz;
                }
                if (i < NX_FIELDS - 1 && k < NZ_FIELDS) {
                    float dEx = staggered_forward_diff(Ex, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order) / dy;
                    long long p_idx = ((long long)s * (NX_FIELDS - 1) * pml2 * NZ_FIELDS) + i * pml2 * NZ_FIELDS + j1 * NZ_FIELDS + k;
                    float phi = y0HPhi2[p_idx];
                    Hz[id4] += upd * (RA01 * dEx + RB0 * phi);
                    y0HPhi2[p_idx] = RE0 * phi - RF0 * dEx;
                }
            }

            if (in_ym) {
                long long j1 = j - (NY_FIELDS - 1 - pml3);
                float RA01 = ymHR[j1] - 1.0f, RB0 = ymHR[pml3 + j1], RE0 = ymHR[2 * pml3 + j1], RF0 = ymHR[3 * pml3 + j1];
                if (i < NX_FIELDS && k < NZ_FIELDS - 1) {
                    float dEz = staggered_forward_diff(Ez, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order) / dy;
                    long long p_idx = ((long long)s * NX_FIELDS * pml3 * (NZ_FIELDS - 1)) + i * pml3 * (NZ_FIELDS - 1) + j1 * (NZ_FIELDS - 1) + k;
                    float phi = ymHPhi1[p_idx];
                    Hx[id4] -= upd * (RA01 * dEz + RB0 * phi);
                    ymHPhi1[p_idx] = RE0 * phi - RF0 * dEz;
                }
                if (i < NX_FIELDS - 1 && k < NZ_FIELDS) {
                    float dEx = staggered_forward_diff(Ex, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order) / dy;
                    long long p_idx = ((long long)s * (NX_FIELDS - 1) * pml3 * NZ_FIELDS) + i * pml3 * NZ_FIELDS + j1 * NZ_FIELDS + k;
                    float phi = ymHPhi2[p_idx];
                    Hz[id4] += upd * (RA01 * dEx + RB0 * phi);
                    ymHPhi2[p_idx] = RE0 * phi - RF0 * dEx;
                }
            }

            if (in_z0) {
                long long k1 = pml4 - 1 - k;
                float RA01 = z0HR[k1] - 1.0f, RB0 = z0HR[pml4 + k1], RE0 = z0HR[2 * pml4 + k1], RF0 = z0HR[3 * pml4 + k1];
                if (i < NX_FIELDS && j < NY_FIELDS - 1) {
                    float dEy = staggered_forward_diff(Ey, id4, 1, k, NZ_FIELDS, fdtd_order) / dz;
                    long long p_idx = ((long long)s * NX_FIELDS * (NY_FIELDS - 1) * pml4) + i * (NY_FIELDS - 1) * pml4 + j * pml4 + k1;
                    float phi = z0HPhi1[p_idx];
                    Hx[id4] += upd * (RA01 * dEy + RB0 * phi);
                    z0HPhi1[p_idx] = RE0 * phi - RF0 * dEy;
                }
                if (i < NX_FIELDS - 1 && j < NY_FIELDS) {
                    float dEx = staggered_forward_diff(Ex, id4, 1, k, NZ_FIELDS, fdtd_order) / dz;
                    long long p_idx = ((long long)s * (NX_FIELDS - 1) * NY_FIELDS * pml4) + i * NY_FIELDS * pml4 + j * pml4 + k1;
                    float phi = z0HPhi2[p_idx];
                    Hy[id4] -= upd * (RA01 * dEx + RB0 * phi);
                    z0HPhi2[p_idx] = RE0 * phi - RF0 * dEx;
                }
            }

            if (in_zm) {
                long long k1 = k - (NZ_FIELDS - 1 - pml5);
                float RA01 = zmHR[k1] - 1.0f, RB0 = zmHR[pml5 + k1], RE0 = zmHR[2 * pml5 + k1], RF0 = zmHR[3 * pml5 + k1];
                if (i < NX_FIELDS && j < NY_FIELDS - 1) {
                    float dEy = staggered_forward_diff(Ey, id4, 1, k, NZ_FIELDS, fdtd_order) / dz;
                    long long p_idx = ((long long)s * NX_FIELDS * (NY_FIELDS - 1) * pml5) + i * (NY_FIELDS - 1) * pml5 + j * pml5 + k1;
                    float phi = zmHPhi1[p_idx];
                    Hx[id4] += upd * (RA01 * dEy + RB0 * phi);
                    zmHPhi1[p_idx] = RE0 * phi - RF0 * dEy;
                }
                if (i < NX_FIELDS - 1 && j < NY_FIELDS) {
                    float dEx = staggered_forward_diff(Ex, id4, 1, k, NZ_FIELDS, fdtd_order) / dz;
                    long long p_idx = ((long long)s * (NX_FIELDS - 1) * NY_FIELDS * pml5) + i * NY_FIELDS * pml5 + j * pml5 + k1;
                    float phi = zmHPhi2[p_idx];
                    Hy[id4] -= upd * (RA01 * dEx + RB0 * phi);
                    zmHPhi2[p_idx] = RE0 * phi - RF0 * dEx;
                }
            }
        }
    }
}

/* Apply the exact transpose of the magnetic CPML correction. */
static void pml_h_fields_adjoint_cpu(
    float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez,
    float* RESTRICT Hx, float* RESTRICT Hy, float* RESTRICT Hz,
    float dx, float dy, float dz, int step, int NX, int NY, int NZ,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* RESTRICT x0R, const float* RESTRICT xmR,
    const float* RESTRICT y0R, const float* RESTRICT ymR,
    const float* RESTRICT z0R, const float* RESTRICT zmR,
    const float* RESTRICT update,
    float* RESTRICT x0P1, float* RESTRICT x0P2,
    float* RESTRICT xmP1, float* RESTRICT xmP2,
    float* RESTRICT y0P1, float* RESTRICT y0P2,
    float* RESTRICT ymP1, float* RESTRICT ymP2,
    float* RESTRICT z0P1, float* RESTRICT z0P2,
    float* RESTRICT zmP1, float* RESTRICT zmP2, int order)
{
    long long ny_nz = (long long)NY * NZ;
    long long field_stride = (long long)NX * ny_nz;
    long long total_work = (long long)step * field_stride;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        int s = (int)(work / field_stride);
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ;
        long long k = rem % NZ;
        float upd = update[idx];

#define APPLY_H_PML(R, P, p, q, stride, coord, n, spacing, field, source, sign) \
        do { \
            float ra = (R)[q] - 1.0f; \
            float rb = (R)[(p) + (q)]; \
            float re = (R)[2 * (p) + (q)]; \
            float rf = (R)[3 * (p) + (q)]; \
            pml_forward_derivative_adjoint_cpu( \
                (field)[work], (source), work, (stride), (coord), (n), order, \
                1.0f / (spacing), upd, (sign), ra, rb, re, rf, &(P)[p_idx]); \
        } while (0)

        if (pml0 > 0 && i < pml0) {
            long long q = pml0 - 1 - i;
            if (k < NZ - 1) {
                long long p_idx = ((long long)s * pml0 * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
                APPLY_H_PML(x0R, x0P1, pml0, q, ny_nz, i, NX, dx, Hy, Ez, 1.0f);
            }
            if (j < NY - 1) {
                long long p_idx = ((long long)s * pml0 * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
                APPLY_H_PML(x0R, x0P2, pml0, q, ny_nz, i, NX, dx, Hz, Ey, -1.0f);
            }
        }
        if (pml1 > 0 && i >= NX - 1 - pml1 && i < NX - 1) {
            long long q = i - (NX - 1 - pml1);
            if (k < NZ - 1) {
                long long p_idx = ((long long)s * pml1 * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
                APPLY_H_PML(xmR, xmP1, pml1, q, ny_nz, i, NX, dx, Hy, Ez, 1.0f);
            }
            if (j < NY - 1) {
                long long p_idx = ((long long)s * pml1 * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
                APPLY_H_PML(xmR, xmP2, pml1, q, ny_nz, i, NX, dx, Hz, Ey, -1.0f);
            }
        }
        if (pml2 > 0 && j < pml2) {
            long long q = pml2 - 1 - j;
            if (k < NZ - 1) {
                long long p_idx = ((long long)s * NX * pml2 * (NZ - 1)) + i * pml2 * (NZ - 1) + q * (NZ - 1) + k;
                APPLY_H_PML(y0R, y0P1, pml2, q, NZ, j, NY, dy, Hx, Ez, -1.0f);
            }
            if (i < NX - 1) {
                long long p_idx = ((long long)s * (NX - 1) * pml2 * NZ) + i * pml2 * NZ + q * NZ + k;
                APPLY_H_PML(y0R, y0P2, pml2, q, NZ, j, NY, dy, Hz, Ex, 1.0f);
            }
        }
        if (pml3 > 0 && j >= NY - 1 - pml3 && j < NY - 1) {
            long long q = j - (NY - 1 - pml3);
            if (k < NZ - 1) {
                long long p_idx = ((long long)s * NX * pml3 * (NZ - 1)) + i * pml3 * (NZ - 1) + q * (NZ - 1) + k;
                APPLY_H_PML(ymR, ymP1, pml3, q, NZ, j, NY, dy, Hx, Ez, -1.0f);
            }
            if (i < NX - 1) {
                long long p_idx = ((long long)s * (NX - 1) * pml3 * NZ) + i * pml3 * NZ + q * NZ + k;
                APPLY_H_PML(ymR, ymP2, pml3, q, NZ, j, NY, dy, Hz, Ex, 1.0f);
            }
        }
        if (pml4 > 0 && k < pml4) {
            long long q = pml4 - 1 - k;
            if (j < NY - 1) {
                long long p_idx = ((long long)s * NX * (NY - 1) * pml4) + i * (NY - 1) * pml4 + j * pml4 + q;
                APPLY_H_PML(z0R, z0P1, pml4, q, 1, k, NZ, dz, Hx, Ey, 1.0f);
            }
            if (i < NX - 1) {
                long long p_idx = ((long long)s * (NX - 1) * NY * pml4) + i * NY * pml4 + j * pml4 + q;
                APPLY_H_PML(z0R, z0P2, pml4, q, 1, k, NZ, dz, Hy, Ex, -1.0f);
            }
        }
        if (pml5 > 0 && k >= NZ - 1 - pml5 && k < NZ - 1) {
            long long q = k - (NZ - 1 - pml5);
            if (j < NY - 1) {
                long long p_idx = ((long long)s * NX * (NY - 1) * pml5) + i * (NY - 1) * pml5 + j * pml5 + q;
                APPLY_H_PML(zmR, zmP1, pml5, q, 1, k, NZ, dz, Hx, Ey, 1.0f);
            }
            if (i < NX - 1) {
                long long p_idx = ((long long)s * (NX - 1) * NY * pml5) + i * NY * pml5 + j * pml5 + q;
                APPLY_H_PML(zmR, zmP2, pml5, q, 1, k, NZ, dz, Hy, Ex, -1.0f);
            }
        }
#undef APPLY_H_PML
    }
}

/*
 * Inject the adjoint source into one electric-field component.
 *
 * Parameters:
 *   step: Number of shots or simulations in the batch.
 *   iteration: Current reverse time-step index.
 *   sourcelocation: Adjoint source coordinates with shape (step, nsr, 3).
 *   srcwaveforms: Adjoint source waveform array.
 *   Ex, Ey, Ez: Electric field component arrays to update.
 *   NX, NY, NZ: Padded field grid sizes.
 *   nsr: Number of adjoint sources per shot.
 *   polarisation: Source component, 0 for x, 1 for y, 2 for z.
 *   iterations: Total number of time steps.
 */
static void add_adjoint_sources_cpu(
    int step, int iteration,
    const int* RESTRICT sourcelocation, const float* RESTRICT srcwaveforms,
    float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez,
    int NX, int NY, int NZ, int nsr, int polarisation, int iterations)
{
    long long field_stride = (long long)NX * NY * NZ;
    long long total = (long long)step * nsr;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total; ++work) {
        int s = (int)(work / nsr);
        long long src = work % nsr;
        long long index = (long long)s * iterations * nsr + (long long)iteration * nsr + src;

        long long i = sourcelocation[s * nsr * 3 + src * 3 + 0];
        long long j = sourcelocation[s * nsr * 3 + src * 3 + 1];
        long long k = sourcelocation[s * nsr * 3 + src * 3 + 2];

        float waveform_value = srcwaveforms[index];
        long long id4 = (long long)s * field_stride + i * NY * NZ + j * NZ + k;

        if (polarisation == 0) {
            DEEPGPR_OMP_ATOMIC_UPDATE
            Ex[id4] += waveform_value;
        } else if (polarisation == 1) {
            DEEPGPR_OMP_ATOMIC_UPDATE
            Ey[id4] += waveform_value;
        } else if (polarisation == 2) {
            DEEPGPR_OMP_ATOMIC_UPDATE
            Ez[id4] += waveform_value;
        }
    }
}

/*
 * Save the effective electric-field right-hand side R from
 * E^(n+1) = ca E^n + cb R. Eold_ptr contains E^n for this saved step.
 * This definition includes CPML corrections and the material-dependent
 * source injection, so the cb gradient follows the executed discrete scheme.
 */
static void copy_to_Rall_single_cpu(
    float* RESTRICT dst_ptr, int t_idx,
    const float* RESTRICT E, const float* RESTRICT Eold_ptr,
    const float* RESTRICT ca, const float* RESTRICT cb,
    int step, int NX, int NY, int NZ)
{
    long long nx1 = NX - 1, ny1 = NY - 1, nz1 = NZ - 1;
    long long total = nx1 * ny1 * nz1;
    long long field_stride = (long long)NX * NY * NZ;
    long long snap_stride = (long long)step * total;
    long long total_work = (long long)step * total;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        int s = (int)(work / total);
        long long idx = work % total;
        long long i = idx / (ny1 * nz1);
        long long rem = idx % (ny1 * nz1);
        long long j = rem / nz1;
        long long k = rem % nz1;
        long long field_idx = (long long)s * field_stride + i * NY * NZ + j * NZ + k;
        long long saved_idx = (long long)t_idx * snap_stride + work;
        float cb_value = cb[i * NY * NZ + j * NZ + k];

        dst_ptr[saved_idx] = cb_value != 0.0f
            ? (E[field_idx] - ca[i * NY * NZ + j * NZ + k] * Eold_ptr[saved_idx]) / cb_value
            : 0.0f;
    }
}

/*
 * Copy one electric-field component snapshot into the saved wavefield buffer.
 *
 * Parameters:
 *   dst_ptr: Destination wavefield buffer.
 *   t_idx: Saved time index.
 *   E: Electric field component array to copy.
 *   step: Number of shots or simulations in the batch.
 *   NX, NY, NZ: Padded field grid sizes.
 */
static void copy_to_Eall_single_cpu(
    float* RESTRICT dst_ptr, int t_idx, const float* RESTRICT E,
    int step, int NX, int NY, int NZ)
{
    long long nx1 = NX - 1, ny1 = NY - 1, nz1 = NZ - 1;
    long long total = nx1 * ny1 * nz1;
    long long field_stride = (long long)NX * NY * NZ;
    long long total_work = (long long)step * total;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        int s = (int)(work / total);
        long long idx = work % total;
        long long i = idx / (ny1 * nz1);
        long long rem = idx % (ny1 * nz1);
        long long j = rem / nz1;
        long long k = rem % nz1;

        long long src_idx = (long long)s * field_stride + i * NY * NZ + j * NZ + k;
        long long dst_idx = (long long)t_idx * step * total + (long long)s * total + idx;
        dst_ptr[dst_idx] = E[src_idx];
    }
}

/*
 * Accumulate model gradients from saved forward fields and adjoint fields.
 *
 * Parameters:
 *   Ex, Ey, Ez: Adjoint electric field component arrays.
 *   Eall_ptr: Saved pre-update electric field E^n.
 *   Rall_ptr: Saved effective right-hand side R^n.
 *   ca, cb: Discrete electric update coefficients.
 *   grader: Output relative permittivity gradient array.
 *   gradse: Output conductivity gradient array.
 *   i: Current reverse time-step index.
 *   step: Number of shots or simulations in the batch.
 *   NX, NY, NZ: Padded field grid sizes.
 *   dt: Time step size.
 *   errequiregrad: Whether to accumulate grader.
 *   serequiregrad: Whether to accumulate gradse.
 *   S: Forward wavefield sampling interval.
 *   nt_saved: Number of saved forward snapshots.
 *   fwi_mode: Gradient mode; 2 uses Ez only, 3 uses Ex, Ey, and Ez.
 */
static void accumulate_gradients_cpu(
    const float* RESTRICT Ex, const float* RESTRICT Ey, const float* RESTRICT Ez,
    const float* RESTRICT Eall_ptr, const float* RESTRICT Rall_ptr,
    const float* RESTRICT ca, const float* RESTRICT cb,
    const float* RESTRICT se,
    float* RESTRICT grader, float* RESTRICT gradse,
    int i, int step, int NX, int NY, int NZ, float dt,
    int errequiregrad, int serequiregrad, int S, int nt_saved, int fwi_mode)
{
    long long sx = NX - 1, sy = NY - 1, sz = NZ - 1;
    long long total_cells = sx * sy * sz;
    long long snap_stride = (long long)step * total_cells;
    long long component_stride = (long long)nt_saved * snap_stride;
    long long total_work = (long long)step * total_cells;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        int s = (int)(work / total_cells);
        long long idx = work % total_cells;
        long long ix = idx / (sy * sz);
        long long rem = idx % (sy * sz);
        long long iy = rem / sz;
        long long iz = rem % sz;

        long long e_stride = (long long)NX * NY * NZ;
        long long idx_E = (long long)s * e_stride + ix * NY * NZ + iy * NZ + iz;
        long long material_idx = ix * NY * NZ + iy * NZ + iz;
        float local_grader = 0.0f;
        float local_gradse = 0.0f;

        long long base_idx = (long long)s * total_cells + idx;
        int components = (fwi_mode == 3) ? 3 : 1;
        float adjoint_values[3];
        adjoint_values[0] = Ex[idx_E];
        adjoint_values[1] = Ey[idx_E];
        adjoint_values[2] = Ez[idx_E];

        for (int c = 0; c < components; ++c) {
            long long comp_offset = (fwi_mode == 3) ? (long long)c * component_stride : 0;
            long long saved_idx = comp_offset + (long long)(i / S) * snap_stride + base_idx;
            float e_old = Eall_ptr[saved_idx];
            float rhs = Rall_ptr[saved_idx];
            float adjoint_val = adjoint_values[(fwi_mode == 3) ? c : 2];
            float grad_ca = adjoint_val * e_old * (float)S;
            float grad_cb = adjoint_val * rhs * (float)S;
            float ca_value = ca[material_idx];
            float cb_value = cb[material_idx];

            if (se[material_idx] <= 100.0f) {
                if (errequiregrad == 1) {
                    float dca_der = E0 * (1.0f - ca_value) * cb_value / dt;
                    float dcb_der = -E0 * cb_value * cb_value / dt;
                    local_grader += grad_ca * dca_der + grad_cb * dcb_der;
                }
                if (serequiregrad == 1) {
                    float dca_dse = -0.5f * (1.0f + ca_value) * cb_value;
                    float dcb_dse = -0.5f * cb_value * cb_value;
                    local_gradse += grad_ca * dca_dse + grad_cb * dcb_dse;
                }
            }
        }

        if (errequiregrad == 1) {
            DEEPGPR_OMP_ATOMIC_UPDATE
            grader[idx] += local_grader;
        }
        if (serequiregrad == 1) {
            DEEPGPR_OMP_ATOMIC_UPDATE
            gradse[idx] += local_gradse;
        }
    }
}

/*
 * Run CPU forward FDTD modeling.
 *
 * Parameters:
 *   er, se, mr: Padded material property arrays.
 *   Eall_ptr: Saved pre-update electric field history.
 *   Rall_ptr: Saved effective electric right-hand sides.
 *   Ex, Ey, Ez: Electric field component arrays.
 *   Hx, Hy, Hz: Magnetic field component arrays.
 *   uE0, uE1, uE4: Electric-field update coefficient arrays.
 *   uH0, uH1, uH4: Magnetic-field update coefficient arrays.
 *   x0EPhi1, x0EPhi2, x0HPhi1, x0HPhi2: Low-x PML auxiliary arrays.
 *   xmEPhi1, xmEPhi2, xmHPhi1, xmHPhi2: High-x PML auxiliary arrays.
 *   y0EPhi1, y0EPhi2, y0HPhi1, y0HPhi2: Low-y PML auxiliary arrays.
 *   ymEPhi1, ymEPhi2, ymHPhi1, ymHPhi2: High-y PML auxiliary arrays.
 *   z0EPhi1, z0EPhi2, z0HPhi1, z0HPhi2: Low-z PML auxiliary arrays.
 *   zmEPhi1, zmEPhi2, zmHPhi1, zmHPhi2: High-z PML auxiliary arrays.
 *   pml0, pml1, pml2, pml3, pml4, pml5: PML thickness for each boundary.
 *   x0ER, xmER, y0ER, ymER, z0ER, zmER: Electric PML coefficient arrays.
 *   x0HR, xmHR, y0HR, ymHR, z0HR, zmHR: Magnetic PML coefficient arrays.
 *   dt: Time step size.
 *   nt: Number of time steps.
 *   step: Number of shots or simulations in the batch.
 *   nrx: Number of receivers per shot.
 *   dx: Spatial grid spacing.
 *   receiverlocation: Receiver coordinates with shape (step, nrx, 3).
 *   rxs: Output receiver sample array.
 *   NX_FIELDS, NY_FIELDS, NZ_FIELDS: Padded field grid sizes.
 *   nsrc: Number of sources per shot.
 *   sourcelocation: Source coordinates with shape (step, nsrc, 3).
 *   srcwaveforms: Source waveform array.
 *   polarisation: Source component, 0 for x, 1 for y, 2 for z.
 *   sampling_interval: Forward wavefield sampling interval.
 *   fwi_mode: Gradient mode; 2 saves Ez only, 3 saves Ex, Ey, and Ez.
 */
DEEPGPR_API void forward(const float* RESTRICT er, const float* RESTRICT se, const float* RESTRICT mr,
             float* RESTRICT Eall_ptr, float* RESTRICT Rall_ptr,
             float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez,
             float* RESTRICT Hx, float* RESTRICT Hy, float* RESTRICT Hz,
             float* RESTRICT uE0, float* RESTRICT uE1, float* RESTRICT uE4,
             float* RESTRICT uH0, float* RESTRICT uH1, float* RESTRICT uH4,

            float* RESTRICT x0EPhi1,float* RESTRICT x0EPhi2, float* RESTRICT x0HPhi1,float* RESTRICT x0HPhi2,
            float* RESTRICT xmEPhi1,float* RESTRICT xmEPhi2, float* RESTRICT xmHPhi1,float* RESTRICT xmHPhi2,
            float* RESTRICT y0EPhi1,float* RESTRICT y0EPhi2, float* RESTRICT y0HPhi1,float* RESTRICT y0HPhi2,
            float* RESTRICT ymEPhi1,float* RESTRICT ymEPhi2, float* RESTRICT ymHPhi1,float* RESTRICT ymHPhi2,
            float* RESTRICT z0EPhi1,float* RESTRICT z0EPhi2, float* RESTRICT z0HPhi1,float* RESTRICT z0HPhi2,
            float* RESTRICT zmEPhi1,float* RESTRICT zmEPhi2, float* RESTRICT zmHPhi1,float* RESTRICT zmHPhi2,

            int pml0,int pml1,int pml2,int pml3,int pml4,int pml5,

            const float* RESTRICT x0ER,const float* RESTRICT xmER, const float* RESTRICT y0ER,const float* RESTRICT ymER,
            const float* RESTRICT z0ER,const float* RESTRICT zmER, const float* RESTRICT x0HR,const float* RESTRICT xmHR,
            const float* RESTRICT y0HR,const float* RESTRICT ymHR, const float* RESTRICT z0HR,const float* RESTRICT zmHR,

             float dt, int nt, int step, int nrx, float dx,
             const int* RESTRICT receiverlocation, float* RESTRICT rxs,

             int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS, int nsrc,
             const int* RESTRICT sourcelocation, const float* RESTRICT srcwaveforms, int polarisation,
             int sampling_interval, int fwi_mode)
{
    int fdtd_order = g_fdtd_order;
    int nt_saved = (nt + sampling_interval - 1) / sampling_interval;
    long long snap_size = (long long)step * (NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1);
    long long component_stride = (long long)nt_saved * snap_size;

    ucgetforward_cpu(er, se, mr, uE0, uE1, uE4, uH0, uH1, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);

    for (int i = 0; i < nt; i++) {
        if (i % sampling_interval == 0) {
            int t_saved = i / sampling_interval;
            if (fwi_mode == 3) {
                copy_to_Eall_single_cpu(Eall_ptr, t_saved, Ex, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                copy_to_Eall_single_cpu(Eall_ptr + component_stride, t_saved, Ey, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                copy_to_Eall_single_cpu(Eall_ptr + 2 * component_stride, t_saved, Ez, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            } else {
                copy_to_Eall_single_cpu(Eall_ptr, t_saved, Ez, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            }
        }

        h_fields_base_update_cpu(uH0, uH1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, fdtd_order);
        pml_h_fields_update_cpu(
            Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, uH4,
            x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2,
            fdtd_order);

        e_fields_base_update_cpu(uE0, uE1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, fdtd_order);
        pml_e_fields_update_cpu(
            Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, uE4,
            x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2,
            fdtd_order);

        Update_hertzian_dipole_cpu(step, i, dx, sourcelocation, srcwaveforms, Ex, Ey, Ez, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nsrc, polarisation, nt);

        if (i % sampling_interval == 0) {
            int t_saved = i / sampling_interval;
            if (fwi_mode == 3) {
                copy_to_Rall_single_cpu(Rall_ptr, t_saved, Ex, Eall_ptr, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                copy_to_Rall_single_cpu(Rall_ptr + component_stride, t_saved, Ey, Eall_ptr + component_stride, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                copy_to_Rall_single_cpu(Rall_ptr + 2 * component_stride, t_saved, Ez, Eall_ptr + 2 * component_stride, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            } else {
                copy_to_Rall_single_cpu(Rall_ptr, t_saved, Ez, Eall_ptr, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            }
        }

        store_outputs_cpu(step, nrx, i, receiverlocation, rxs, Ex, Ey, Ez, Hx, Hy, Hz, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nt);
    }
}

/*
 * Run CPU adjoint FDTD modeling and accumulate model gradients.
 *
 * Parameters:
 *   er, se, mr: Padded material property arrays.
 *   Eall_ptr: Saved pre-update electric field history.
 *   Rall_ptr: Saved effective electric right-hand sides.
 *   Ex, Ey, Ez: Adjoint electric field component arrays.
 *   Hx, Hy, Hz: Adjoint magnetic field component arrays.
 *   uE0, uE1, uE4: Electric-field update coefficient arrays.
 *   uH0, uH1, uH4: Magnetic-field update coefficient arrays.
 *   x0EPhi1, x0EPhi2, x0HPhi1, x0HPhi2: Low-x PML auxiliary arrays.
 *   xmEPhi1, xmEPhi2, xmHPhi1, xmHPhi2: High-x PML auxiliary arrays.
 *   y0EPhi1, y0EPhi2, y0HPhi1, y0HPhi2: Low-y PML auxiliary arrays.
 *   ymEPhi1, ymEPhi2, ymHPhi1, ymHPhi2: High-y PML auxiliary arrays.
 *   z0EPhi1, z0EPhi2, z0HPhi1, z0HPhi2: Low-z PML auxiliary arrays.
 *   zmEPhi1, zmEPhi2, zmHPhi1, zmHPhi2: High-z PML auxiliary arrays.
 *   pml0, pml1, pml2, pml3, pml4, pml5: PML thickness for each boundary.
 *   x0ER, xmER, y0ER, ymER, z0ER, zmER: Electric PML coefficient arrays.
 *   x0HR, xmHR, y0HR, ymHR, z0HR, zmHR: Magnetic PML coefficient arrays.
 *   dt: Time step size.
 *   nt: Number of time steps.
 *   step: Number of shots or simulations in the batch.
 *   nrx: Number of receivers per shot.
 *   dx: Spatial grid spacing.
 *   NX_FIELDS, NY_FIELDS, NZ_FIELDS: Padded field grid sizes.
 *   nsrc: Number of adjoint sources per shot.
 *   sourcelocation: Adjoint source coordinates with shape (step, nsrc, 3).
 *   srcwaveforms: Adjoint source waveform array.
 *   polarisation: Adjoint source component, 0 for x, 1 for y, 2 for z.
 *   grad_er: Output relative permittivity gradient array.
 *   grad_se: Output conductivity gradient array.
 *   errequiregrad: Whether grad_er should be accumulated.
 *   serequiregrad: Whether grad_se should be accumulated.
 *   sampling_interval: Forward wavefield sampling interval.
 *   fwi_mode: Gradient mode; 2 uses Ez only, 3 uses Ex, Ey, and Ez.
 */
DEEPGPR_API void backward(const float* RESTRICT er, const float* RESTRICT se, const float* RESTRICT mr,
             const float* RESTRICT Eall_ptr, const float* RESTRICT Rall_ptr,
             float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez,
             float* RESTRICT Hx, float* RESTRICT Hy, float* RESTRICT Hz,
             float* RESTRICT uE0, float* RESTRICT uE1, float* RESTRICT uE4,
             float* RESTRICT uH0, float* RESTRICT uH1, float* RESTRICT uH4,

            float* RESTRICT x0EPhi1,float* RESTRICT x0EPhi2, float* RESTRICT x0HPhi1,float* RESTRICT x0HPhi2,
            float* RESTRICT xmEPhi1,float* RESTRICT xmEPhi2, float* RESTRICT xmHPhi1,float* RESTRICT xmHPhi2,
            float* RESTRICT y0EPhi1,float* RESTRICT y0EPhi2, float* RESTRICT y0HPhi1,float* RESTRICT y0HPhi2,
            float* RESTRICT ymEPhi1,float* RESTRICT ymEPhi2, float* RESTRICT ymHPhi1,float* RESTRICT ymHPhi2,
            float* RESTRICT z0EPhi1,float* RESTRICT z0EPhi2, float* RESTRICT z0HPhi1,float* RESTRICT z0HPhi2,
            float* RESTRICT zmEPhi1,float* RESTRICT zmEPhi2, float* RESTRICT zmHPhi1,float* RESTRICT zmHPhi2,

            int pml0,int pml1,int pml2,int pml3,int pml4,int pml5,

            float* RESTRICT x0ER,float* RESTRICT xmER, float* RESTRICT y0ER,float* RESTRICT ymER,
            float* RESTRICT z0ER,float* RESTRICT zmER, float* RESTRICT x0HR,float* RESTRICT xmHR,
            float* RESTRICT y0HR,float* RESTRICT ymHR, float* RESTRICT z0HR,float* RESTRICT zmHR,

             float dt, int nt, int step, int nrx, float dx,
             int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
             int nsrc, const int* RESTRICT sourcelocation, const float* RESTRICT srcwaveforms,
             int polarisation,
             float* RESTRICT grad_er,float* RESTRICT grad_se, int errequiregrad, int serequiregrad,
             int sampling_interval, int fwi_mode)
{
    (void)nrx;

    int fdtd_order = g_fdtd_order;
    int nt_saved = (nt + sampling_interval - 1) / sampling_interval;

    ucgetforward_cpu(er, se, mr, uE0, uE1, uE4, uH0, uH1, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);

    for (int i = nt - 1; i >= 0; i--) {
        add_adjoint_sources_cpu(step, i, sourcelocation, srcwaveforms, Ex, Ey, Ez, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nsrc, polarisation, nt);

        if (i % sampling_interval == 0) {
            accumulate_gradients_cpu(Ex, Ey, Ez, Eall_ptr, Rall_ptr, uE0, uE4, se,
                grad_er, grad_se, i, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt,
                errequiregrad, serequiregrad, sampling_interval, nt_saved, fwi_mode);
        }

        pml_e_fields_adjoint_cpu(
            Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, uE4,
            x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2,
            z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2, fdtd_order);
        e_fields_base_adjoint_cpu(uE0, uE1, Ex, Ey, Ez, Hx, Hy, Hz,
            step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, fdtd_order);
        pml_h_fields_adjoint_cpu(
            Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, uH4,
            x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2,
            z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2, fdtd_order);
        h_fields_base_adjoint_cpu(uH0, uH1, Ex, Ey, Ez, Hx, Hy, Hz,
            step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, fdtd_order);
    }
}

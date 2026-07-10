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

DEEPGPR_API void set_fdtd_order(int order)
{
    g_fdtd_order = (order == 4 || order == 8) ? order : 2;
}

static int fdtd_radius_for_order(int order)
{
    if (order >= 8) return 4;
    if (order >= 4) return 2;
    return 1;
}

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

static int usable_backward_radius(long long coord, long long n, int requested)
{
    int radius = requested;
    while (radius > 1 && (coord < radius || coord + radius - 1 >= n)) {
        --radius;
    }
    return radius;
}

static int usable_forward_radius(long long coord, long long n, int requested)
{
    int radius = requested;
    while (radius > 1 && (coord - radius + 1 < 0 || coord + radius >= n)) {
        --radius;
    }
    return radius;
}

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

static void ucgetbackward_cpu(const float* RESTRICT er, const float* RESTRICT se, const float* RESTRICT mr,
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
                float EA = (E0 * er[idx] / dt) + 0.5f * se[idx];
                uE0[idx] = (2.0f * E0 * er[idx]) / (2.0f * E0 * er[idx] + se[idx] * dt);
                uE1[idx] = (1.0f / dx) / EA;
                uE4[idx] = 1.0f / EA;
            }
        }
    }
}

static void store_outputs_cpu(
    int step, int NRX, int iteration,
    const int* RESTRICT receiverlocation, float* RESTRICT rxs,
    const float* RESTRICT Ex, const float* RESTRICT Ey, const float* RESTRICT Ez,
    const float* RESTRICT Hx, const float* RESTRICT Hy, const float* RESTRICT Hz,
    int NX, int NY, int NZ, int N_ITER)
{
    long long field_stride = (long long)NX * NY * NZ;
    long long rx;

    DEEPGPR_OMP_PARALLEL_FOR
    for (rx = 0; rx < NRX; ++rx) {
        for (int s = 0; s < step; ++s) {
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
}

static void Update_hertzian_dipole_cpu(
    int step, int iteration, float dx,
    const int* RESTRICT sourcelocation, const float* RESTRICT srcwaveforms,
    float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez, const float* RESTRICT uE4,
    int NX, int NY, int NZ, int nsrc, int polarisation, int nt)
{
    long long field_stride = (long long)NX * NY * NZ;
    long long src;

    DEEPGPR_OMP_PARALLEL_FOR
    for (src = 0; src < nsrc; ++src) {
        float waveform_value = srcwaveforms[src * nt + iteration];
        float scale = waveform_value * dx / (dx * dx * dx);

        for (int s = 0; s < step; ++s) {
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
}

static void fused_e_fields_updates_cpu(
    const float* RESTRICT uE0, const float* RESTRICT uE1,
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
    long long idx;

    DEEPGPR_OMP_PARALLEL_FOR
    for (idx = 0; idx < field_stride; ++idx) {
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ_FIELDS;
        long long k = rem % NZ_FIELDS;

        int do_ex = (((NY_FIELDS - 1) != 1 || (NZ_FIELDS - 1) != 1) && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));
        int do_ey = (((NX_FIELDS - 1) != 1 || (NZ_FIELDS - 1) != 1) && i > 0 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));
        int do_ez = (((NX_FIELDS - 1) != 1 || (NY_FIELDS - 1) != 1) && i > 0 && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));

        int in_x0 = (pml0 > 0 && i > 0 && i <= pml0 && j < NY_FIELDS && k < NZ_FIELDS);
        int in_xm = (pml1 > 0 && i >= NX_FIELDS - 1 - pml1 && i < NX_FIELDS - 1 && j < NY_FIELDS && k < NZ_FIELDS);
        int in_y0 = (pml2 > 0 && i < NX_FIELDS && j > 0 && j <= pml2 && k < NZ_FIELDS);
        int in_ym = (pml3 > 0 && i < NX_FIELDS && j >= NY_FIELDS - 1 - pml3 && j < NY_FIELDS - 1 && k < NZ_FIELDS);
        int in_z0 = (pml4 > 0 && i < NX_FIELDS && j < NY_FIELDS && k > 0 && k <= pml4);
        int in_zm = (pml5 > 0 && i < NX_FIELDS && j < NY_FIELDS && k >= NZ_FIELDS - 1 - pml5 && k < NZ_FIELDS - 1);

        float ue0 = uE0[idx];
        float ue1 = uE1[idx];
        float upd = updatecoeffsE[idx];
        long long id4 = idx;

        for (int s = 0; s < step; ++s) {
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

            id4 += field_stride;
        }
    }
}

static void fused_h_fields_updates_cpu(
    const float* RESTRICT uH0, const float* RESTRICT uH1,
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
    long long idx;

    DEEPGPR_OMP_PARALLEL_FOR
    for (idx = 0; idx < field_stride; ++idx) {
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ_FIELDS;
        long long k = rem % NZ_FIELDS;

        int do_hx = ((NX_FIELDS - 1) != 1 && i > 0 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));
        int do_hy = ((NY_FIELDS - 1) != 1 && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));
        int do_hz = ((NZ_FIELDS - 1) != 1 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));

        int in_x0 = (pml0 > 0 && i < pml0 && j < NY_FIELDS && k < NZ_FIELDS);
        int in_xm = (pml1 > 0 && i >= NX_FIELDS - 1 - pml1 && i < NX_FIELDS - 1 && j < NY_FIELDS && k < NZ_FIELDS);
        int in_y0 = (pml2 > 0 && i < NX_FIELDS && j < pml2 && k < NZ_FIELDS);
        int in_ym = (pml3 > 0 && i < NX_FIELDS && j >= NY_FIELDS - 1 - pml3 && j < NY_FIELDS - 1 && k < NZ_FIELDS);
        int in_z0 = (pml4 > 0 && i < NX_FIELDS && j < NY_FIELDS && k < pml4);
        int in_zm = (pml5 > 0 && i < NX_FIELDS && j < NY_FIELDS && k >= NZ_FIELDS - 1 - pml5 && k < NZ_FIELDS - 1);

        float uh0 = uH0[idx];
        float uh1 = uH1[idx];
        float upd = updatecoeffsH[idx];
        long long id4 = idx;

        for (int s = 0; s < step; ++s) {
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

            id4 += field_stride;
        }
    }
}

static void Back_source_cpu(
    int step, int iteration,
    const int* RESTRICT sourcelocation, const float* RESTRICT srcwaveforms,
    float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez,
    int NX, int NY, int NZ, int nsr, int polarisation, int iterations)
{
    long long field_stride = (long long)NX * NY * NZ;
    long long index_stride = (long long)iterations * nsr;
    long long src;

    DEEPGPR_OMP_PARALLEL_FOR
    for (src = 0; src < nsr; ++src) {
        long long index = (long long)iteration * nsr + src;

        for (int s = 0; s < step; ++s) {
            long long i = sourcelocation[s * nsr * 3 + src * 3 + 0];
            long long j = sourcelocation[s * nsr * 3 + src * 3 + 1];
            long long k = sourcelocation[s * nsr * 3 + src * 3 + 2];

            float waveform_value = srcwaveforms[index];
            long long id4 = (long long)s * field_stride + i * NY * NZ + j * NZ + k;

            if (polarisation == 0) {
                DEEPGPR_OMP_ATOMIC_UPDATE
                Ex[id4] -= waveform_value;
            } else if (polarisation == 1) {
                DEEPGPR_OMP_ATOMIC_UPDATE
                Ey[id4] -= waveform_value;
            } else if (polarisation == 2) {
                DEEPGPR_OMP_ATOMIC_UPDATE
                Ez[id4] -= waveform_value;
            }

            index += index_stride;
        }
    }
}

static void copy_to_Eall_single_cpu(
    float* RESTRICT dst_ptr, int t_idx, const float* RESTRICT E,
    int step, int NX, int NY, int NZ)
{
    long long nx1 = NX - 1, ny1 = NY - 1, nz1 = NZ - 1;
    long long total = nx1 * ny1 * nz1;
    long long field_stride = (long long)NX * NY * NZ;
    long long idx;

    DEEPGPR_OMP_PARALLEL_FOR
    for (idx = 0; idx < total; ++idx) {
        long long i = idx / (ny1 * nz1);
        long long rem = idx % (ny1 * nz1);
        long long j = rem / nz1;
        long long k = rem % nz1;

        long long src_idx = i * NY * NZ + j * NZ + k;
        long long dst_idx = (long long)t_idx * step * total + idx;

        for (int s = 0; s < step; ++s) {
            dst_ptr[dst_idx] = E[src_idx];
            src_idx += field_stride;
            dst_idx += total;
        }
    }
}

static long long ll_min(long long a, long long b)
{
    return a < b ? a : b;
}

static void accumulate_gradients_cpu(
    const float* RESTRICT Ez, const float* RESTRICT Eall_ptr,
    float* RESTRICT grader, float* RESTRICT gradse,
    int i, int step, int NX, int NY, int NZ, float dt,
    int errequiregrad, int serequiregrad, int S, int nt_saved)
{
    long long sx = NX - 1, sy = NY - 1, sz = NZ - 1;
    long long total_cells = sx * sy * sz;
    long long idx;

    DEEPGPR_OMP_PARALLEL_FOR
    for (idx = 0; idx < total_cells; ++idx) {
        long long ix = idx / (sy * sz);
        long long rem = idx % (sy * sz);
        long long iy = rem / sz;
        long long iz = rem % sz;

        long long idx0_curr = i / S;
        long long idx1_curr = ll_min(idx0_curr + 1, (long long)nt_saved - 1);
        float w1_curr = (float)(i % S) / (float)S;
        float w0_curr = 1.0f - w1_curr;

        long long idx0_prev = (i - 1) / S;
        long long idx1_prev = ll_min(idx0_prev + 1, (long long)nt_saved - 1);
        float w1_prev = (float)((i - 1) % S) / (float)S;
        float w0_prev = 1.0f - w1_prev;

        long long idx_Ez = ix * NY * NZ + iy * NZ + iz;
        long long ez_stride = (long long)NX * NY * NZ;
        float local_grader = 0.0f;
        float local_gradse = 0.0f;

        for (int s = 0; s < step; ++s) {
            long long base_idx = (long long)s * total_cells + idx;
            float e0_c = Eall_ptr[idx0_curr * step * total_cells + base_idx];
            float e1_c = Eall_ptr[idx1_curr * step * total_cells + base_idx];
            float e0_p = Eall_ptr[idx0_prev * step * total_cells + base_idx];
            float e1_p = Eall_ptr[idx1_prev * step * total_cells + base_idx];

            float e_curr = e0_c * w0_curr + e1_c * w1_curr;
            float e_prev = e0_p * w0_prev + e1_p * w1_prev;
            float ez_val = Ez[idx_Ez];

            if (errequiregrad == 1) local_grader += (e_curr - e_prev) * ez_val / dt;
            if (serequiregrad == 1) local_gradse += e_curr * ez_val * dt;

            idx_Ez += ez_stride;
        }

        if (errequiregrad == 1) grader[idx] += local_grader;
        if (serequiregrad == 1) gradse[idx] += local_gradse;
    }
}

DEEPGPR_API void forward(const float* RESTRICT er, const float* RESTRICT se, const float* RESTRICT mr,
             float* RESTRICT Eall_ptr,
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
             int sampling_interval)
{
    int fdtd_order = g_fdtd_order;

    ucgetforward_cpu(er, se, mr, uE0, uE1, uE4, uH0, uH1, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);

    for (int i = 0; i < nt; i++) {
        store_outputs_cpu(step, nrx, i, receiverlocation, rxs, Ex, Ey, Ez, Hx, Hy, Hz, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nt);

        fused_h_fields_updates_cpu(
            uH0, uH1, Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, uH4,
            x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2,
            fdtd_order);

        fused_e_fields_updates_cpu(
            uE0, uE1, Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, uE4,
            x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2,
            fdtd_order);

        Update_hertzian_dipole_cpu(step, i, dx, sourcelocation, srcwaveforms, Ex, Ey, Ez, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nsrc, polarisation, nt);

        if (i % sampling_interval == 0) {
            copy_to_Eall_single_cpu(Eall_ptr, i / sampling_interval, Ez, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
        }
    }
}

DEEPGPR_API void backward(const float* RESTRICT er, const float* RESTRICT se, const float* RESTRICT mr,
             const float* RESTRICT Eall_ptr,
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
             int sampling_interval)
{
    int fdtd_order = g_fdtd_order;
    int nt_saved = (nt + sampling_interval - 1) / sampling_interval;

    ucgetbackward_cpu(er, se, mr, uE0, uE1, uE4, uH0, uH1, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);

    for (int i = nt - 1; i > 0; i--) {
        Back_source_cpu(step, i, sourcelocation, srcwaveforms, Ex, Ey, Ez, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nsrc, polarisation, nt);

        fused_e_fields_updates_cpu(
            uE0, uE1, Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, uE4,
            x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2,
            fdtd_order);

        fused_h_fields_updates_cpu(
            uH0, uH1, Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, uH4,
            x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2,
            fdtd_order);

        accumulate_gradients_cpu(Ez, Eall_ptr, grad_er, grad_se, i, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, errequiregrad, serequiregrad, sampling_interval, nt_saved);
    }
}

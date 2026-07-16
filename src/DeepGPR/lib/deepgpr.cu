#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cfloat>


#ifdef _WIN32
#define DEEPGPR_API extern "C" __declspec(dllexport)
#else
#define DEEPGPR_API extern "C" __attribute__((visibility("default")))
#endif

__constant__ float e0 = 8.8541878128e-12;
__constant__ float m0 = 1.25663706212e-06;

#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err__) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return; \
    } \
} while (0)

#define CUDA_CHECK_LAST() do { \
    cudaError_t err__ = cudaGetLastError(); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err__) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return; \
    } \
} while (0)

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
__device__ __forceinline__ int fdtd_radius_for_order(int order)
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
__device__ __forceinline__ float fdtd_coeff(int radius, int r)
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
__device__ __forceinline__ int usable_backward_radius(long long coord, long long n, int requested)
{
    int radius = requested;
    while (radius > 1 && (coord < radius || coord + radius - 1 >= n)) --radius;
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
__device__ __forceinline__ int usable_forward_radius(long long coord, long long n, int requested)
{
    int radius = requested;
    while (radius > 1 && (coord - radius + 1 < 0 || coord + radius >= n)) --radius;
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
__device__ __forceinline__ float staggered_backward_diff(
    const float* __restrict__ f, long long id, long long stride,
    long long coord, long long n, int order)
{
    if (order <= 2) return f[id] - f[id - stride];

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
__device__ __forceinline__ float staggered_forward_diff(
    const float* __restrict__ f, long long id, long long stride,
    long long coord, long long n, int order)
{
    if (order <= 2) return f[id + stride] - f[id];

    int radius = usable_forward_radius(coord, n, fdtd_radius_for_order(order));
    float acc = 0.0f;
    for (int r = 1; r <= radius; ++r) {
        acc += fdtd_coeff(radius, r) * (f[id + (long long)r * stride] - f[id - (long long)(r - 1) * stride]);
    }
    return acc;
}

template<int ORDER>
struct FdtdStaticRadius
{
    enum { value = (ORDER >= 8) ? 4 : ((ORDER >= 4) ? 2 : 1) };
};

template<int ORDER>
__device__ __forceinline__ float staggered_backward_diff_static(
    const float* __restrict__ f, long long id, long long stride,
    long long coord, long long n)
{
    const int requested = FdtdStaticRadius<ORDER>::value;
    if (requested == 1) return f[id] - f[id - stride];

    int radius = usable_backward_radius(coord, n, requested);
    float acc = 0.0f;
#pragma unroll
    for (int r = 1; r <= requested; ++r) {
        if (r <= radius) {
            acc += fdtd_coeff(radius, r) * (f[id + (long long)(r - 1) * stride] - f[id - (long long)r * stride]);
        }
    }
    return acc;
}

template<int ORDER>
__device__ __forceinline__ float staggered_forward_diff_static(
    const float* __restrict__ f, long long id, long long stride,
    long long coord, long long n)
{
    const int requested = FdtdStaticRadius<ORDER>::value;
    if (requested == 1) return f[id + stride] - f[id];

    int radius = usable_forward_radius(coord, n, requested);
    float acc = 0.0f;
#pragma unroll
    for (int r = 1; r <= requested; ++r) {
        if (r <= radius) {
            acc += fdtd_coeff(radius, r) * (f[id + (long long)r * stride] - f[id - (long long)(r - 1) * stride]);
        }
    }
    return acc;
}

__device__ __forceinline__ void add_staggered_backward_adjoint(
    float* gradient, long long id, long long stride,
    long long coord, long long n, int order, float weight)
{
    int radius = order <= 2 ? 1 : usable_backward_radius(coord, n, fdtd_radius_for_order(order));
    for (int r = 1; r <= radius; ++r) {
        float value = weight * fdtd_coeff(radius, r);
        atomicAdd(&gradient[id + (long long)(r - 1) * stride], value);
        atomicAdd(&gradient[id - (long long)r * stride], -value);
    }
}

__device__ __forceinline__ void add_staggered_forward_adjoint(
    float* gradient, long long id, long long stride,
    long long coord, long long n, int order, float weight)
{
    int radius = order <= 2 ? 1 : usable_forward_radius(coord, n, fdtd_radius_for_order(order));
    for (int r = 1; r <= radius; ++r) {
        float value = weight * fdtd_coeff(radius, r);
        atomicAdd(&gradient[id + (long long)r * stride], value);
        atomicAdd(&gradient[id - (long long)(r - 1) * stride], -value);
    }
}

__device__ __forceinline__ void pml_backward_derivative_adjoint(
    float lambda_field, float* lambda_source,
    long long source_id, long long stride, long long coord, long long n, int order,
    float inverse_spacing, float update_coeff, float sign,
    float ra_minus_one, float rb, float re, float rf, float* lambda_phi)
{
    float phi_new = *lambda_phi;
    float derivative_weight =
        (sign * update_coeff * ra_minus_one * lambda_field - rf * phi_new) * inverse_spacing;
    add_staggered_backward_adjoint(
        lambda_source, source_id, stride, coord, n, order, derivative_weight);
    *lambda_phi = sign * update_coeff * rb * lambda_field + re * phi_new;
}

__device__ __forceinline__ void pml_forward_derivative_adjoint(
    float lambda_field, float* lambda_source,
    long long source_id, long long stride, long long coord, long long n, int order,
    float inverse_spacing, float update_coeff, float sign,
    float ra_minus_one, float rb, float re, float rf, float* lambda_phi)
{
    float phi_new = *lambda_phi;
    float derivative_weight =
        (sign * update_coeff * ra_minus_one * lambda_field - rf * phi_new) * inverse_spacing;
    add_staggered_forward_adjoint(
        lambda_source, source_id, stride, coord, n, order, derivative_weight);
    *lambda_phi = sign * update_coeff * rb * lambda_field + re * phi_new;
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
__global__ void ucgetforward(const float* __restrict__ er, const float* __restrict__ se, const float* __restrict__ mr,
    float* __restrict__ uE0, float* __restrict__ uE1, float* __restrict__ uE4,
    float* __restrict__ uH0, float* __restrict__ uH1, float* __restrict__ uH4,
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS, float dt, float dx) 
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    if (idx >= (long long)NX_FIELDS * ny_nz) return;

    long long i = idx / ny_nz;
    long long rem = idx % ny_nz;
    long long j = rem / NZ_FIELDS;
    long long k = rem % NZ_FIELDS;

    if (i < (NX_FIELDS-1) && j < (NY_FIELDS-1) && k < (NZ_FIELDS-1) ) {
        float HA = m0 * mr[idx] / dt;
        uH0[idx] = 1.0f;
        uH1[idx] = (1.0f / dx) / HA;
        uH4[idx] = 1.0f / HA;

        if (se[idx] > 100.0f) {
            uE0[idx] = 0.0f; uE1[idx] = 0.0f; uE4[idx] = 0.0f;
        } else {
            float e_term = e0 * er[idx] / dt;
            float s_term = 0.5f * se[idx];
            float EA = e_term + s_term;
            float EB = e_term - s_term;
            uE0[idx] = EB / EA;
            uE1[idx] = (1.0f / dx) / EA;
            uE4[idx] = 1.0f / EA;
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
__global__ void store_outputs(
    int step, int NRX, int iteration,
    const int* __restrict__ receiverlocation, float* __restrict__ rxs,
    const float* __restrict__ Ex, const float* __restrict__ Ey, const float* __restrict__ Ez, 
    const float* __restrict__ Hx, const float* __restrict__ Hy, const float* __restrict__ Hz,
    int NX, int NY, int NZ, int N_ITER) 
{
    long long field_stride = (long long)NX * NY * NZ;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)step * NRX;
    if (work >= total) return;

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
__global__ void Update_hertzian_dipole(
    int step, int iteration, float dx, 
    const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms,
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez, const float* __restrict__ uE4,
    int NX, int NY, int NZ, int nsrc, int polarisation, int nt) 
{
    long long field_stride = (long long)NX * NY * NZ;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)step * nsrc;
    if (work >= total) return;

    int s = (int)(work / nsrc);
    long long src = work % nsrc;
    float waveform_value = srcwaveforms[src * nt + iteration];
    float scale = waveform_value * dx / (dx * dx * dx);  

    long long i = sourcelocation[s * nsrc * 3 + src * 3 + 0];
    long long j = sourcelocation[s * nsrc * 3 + src * 3 + 1];
    long long k = sourcelocation[s * nsrc * 3 + src * 3 + 2];

    long long id3 = i * NY * NZ + j * NZ + k;
    long long id4 = (long long)s * field_stride + id3;

    if (polarisation == 0) Ex[id4] -= uE4[id3] * scale;
    else if (polarisation == 1) Ey[id4] -= uE4[id3] * scale;
    else if (polarisation == 2) Ez[id4] -= uE4[id3] * scale;
}


template<int ORDER>
__global__ void e_fields_base_update_gpu(
    const float* __restrict__ uE0, const float* __restrict__ uE1,
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    const float* __restrict__ Hx, const float* __restrict__ Hy, const float* __restrict__ Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS)
{
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long field_stride = (long long)NX_FIELDS * ny_nz;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= (long long)step * field_stride) return;

    long long idx = work % field_stride;
    long long i = idx / ny_nz;
    long long rem = idx % ny_nz;
    long long j = rem / NZ_FIELDS;
    long long k = rem % NZ_FIELDS;
    long long id4 = work;

    float ue0 = uE0[idx];
    float ue1 = uE1[idx];

    bool do_ex = (((NY_FIELDS - 1) != 1 || (NZ_FIELDS - 1) != 1) && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));
    bool do_ey = (((NX_FIELDS - 1) != 1 || (NZ_FIELDS - 1) != 1) && i > 0 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));
    bool do_ez = (((NX_FIELDS - 1) != 1 || (NY_FIELDS - 1) != 1) && i > 0 && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));

    if (do_ex) {
        float dHz_dy = staggered_backward_diff_static<ORDER>(Hz, id4, NZ_FIELDS, j, NY_FIELDS);
        float dHy_dz = staggered_backward_diff_static<ORDER>(Hy, id4, 1, k, NZ_FIELDS);
        Ex[id4] = ue0 * Ex[id4] + ue1 * dHz_dy - ue1 * dHy_dz;
    }
    if (do_ey) {
        float dHx_dz = staggered_backward_diff_static<ORDER>(Hx, id4, 1, k, NZ_FIELDS);
        float dHz_dx = staggered_backward_diff_static<ORDER>(Hz, id4, ny_nz, i, NX_FIELDS);
        Ey[id4] = ue0 * Ey[id4] + ue1 * dHx_dz - ue1 * dHz_dx;
    }
    if (do_ez) {
        float dHy_dx = staggered_backward_diff_static<ORDER>(Hy, id4, ny_nz, i, NX_FIELDS);
        float dHx_dy = staggered_backward_diff_static<ORDER>(Hx, id4, NZ_FIELDS, j, NY_FIELDS);
        Ez[id4] = ue0 * Ez[id4] + ue1 * dHy_dx - ue1 * dHx_dy;
    }
}

/*
 * Apply electric CPML boundary corrections after the base electric-field update.
 */
template<int ORDER>
__global__ void pml_e_fields_update_gpu(
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,  
    const float* __restrict__ Hx, const float* __restrict__ Hy, const float* __restrict__ Hz,
    float dx, float dy, float dz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* __restrict__ x0ER, const float* __restrict__ xmER,
    const float* __restrict__ y0ER, const float* __restrict__ ymER,
    const float* __restrict__ z0ER, const float* __restrict__ zmER,
    const float* __restrict__ updatecoeffsE,
    float* __restrict__ x0EPhi1, float* __restrict__ x0EPhi2,
    float* __restrict__ xmEPhi1, float* __restrict__ xmEPhi2,
    float* __restrict__ y0EPhi1, float* __restrict__ y0EPhi2,
    float* __restrict__ ymEPhi1, float* __restrict__ ymEPhi2,
    float* __restrict__ z0EPhi1, float* __restrict__ z0EPhi2,
    float* __restrict__ zmEPhi1, float* __restrict__ zmEPhi2)
{
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long field_stride = (long long)NX_FIELDS * ny_nz;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= (long long)step * field_stride) return;

    int s = (int)(work / field_stride);
    long long idx = work % field_stride;

    long long i = idx / ny_nz;
    long long rem = idx % ny_nz;
    long long j = rem / NZ_FIELDS;
    long long k = rem % NZ_FIELDS;

    bool in_x0 = (pml0 > 0 && i > 0 && i <= pml0 && j < NY_FIELDS && k < NZ_FIELDS);
    bool in_xm = (pml1 > 0 && i >= NX_FIELDS - 1 - pml1 && i < NX_FIELDS - 1 && j < NY_FIELDS && k < NZ_FIELDS);
    bool in_y0 = (pml2 > 0 && i < NX_FIELDS && j > 0 && j <= pml2 && k < NZ_FIELDS);
    bool in_ym = (pml3 > 0 && i < NX_FIELDS && j >= NY_FIELDS - 1 - pml3 && j < NY_FIELDS - 1 && k < NZ_FIELDS);
    bool in_z0 = (pml4 > 0 && i < NX_FIELDS && j < NY_FIELDS && k > 0 && k <= pml4);
    bool in_zm = (pml5 > 0 && i < NX_FIELDS && j < NY_FIELDS && k >= NZ_FIELDS - 1 - pml5 && k < NZ_FIELDS - 1);
    if (!(in_x0 || in_xm || in_y0 || in_ym || in_z0 || in_zm)) return;

    float upd = updatecoeffsE[idx];

    long long id4 = work;

    {
        if (in_x0) {
            long long i1 = pml0 - i;
            float RA01 = x0ER[i1] - 1.0f, RB0 = x0ER[pml0 + i1], RE0 = x0ER[2 * pml0 + i1], RF0 = x0ER[3 * pml0 + i1];
            if (j < NY_FIELDS - 1 && i > 0) {
                float dHz = staggered_backward_diff_static<ORDER>(Hz, id4, ny_nz, i, NX_FIELDS) / dx;
                long long p_idx = ((long long)s * (pml0+1) * (NY_FIELDS-1) * NZ_FIELDS) + i1 * (NY_FIELDS-1) * NZ_FIELDS + j * NZ_FIELDS + k;
                float phi = x0EPhi1[p_idx];
                Ey[id4] -= upd * (RA01 * dHz + RB0 * phi);
                x0EPhi1[p_idx] = RE0 * phi - RF0 * dHz;
            }
            if (k < NZ_FIELDS - 1 && i > 0) {
                float dHy = staggered_backward_diff_static<ORDER>(Hy, id4, ny_nz, i, NX_FIELDS) / dx;
                long long p_idx = ((long long)s * (pml0+1) * NY_FIELDS * (NZ_FIELDS-1)) + i1 * NY_FIELDS * (NZ_FIELDS-1) + j * (NZ_FIELDS-1) + k;
                float phi = x0EPhi2[p_idx];
                Ez[id4] += upd * (RA01 * dHy + RB0 * phi);
                x0EPhi2[p_idx] = RE0 * phi - RF0 * dHy;
            }
        }

        if (in_xm) {
            long long i1 = i - (NX_FIELDS - 1 - pml1);
            float RA01 = xmER[i1] - 1.0f, RB0 = xmER[pml1 + i1], RE0 = xmER[2 * pml1 + i1], RF0 = xmER[3 * pml1 + i1];
            if (j < NY_FIELDS - 1 && i > 0) {
                float dHz = staggered_backward_diff_static<ORDER>(Hz, id4, ny_nz, i, NX_FIELDS) / dx;
                long long p_idx = ((long long)s * (pml1+1) * (NY_FIELDS-1) * NZ_FIELDS) + i1 * (NY_FIELDS-1) * NZ_FIELDS + j * NZ_FIELDS + k;
                float phi = xmEPhi1[p_idx];
                Ey[id4] -= upd * (RA01 * dHz + RB0 * phi);
                xmEPhi1[p_idx] = RE0 * phi - RF0 * dHz;
            }
            if (k < NZ_FIELDS - 1 && i > 0) {
                float dHy = staggered_backward_diff_static<ORDER>(Hy, id4, ny_nz, i, NX_FIELDS) / dx;
                long long p_idx = ((long long)s * (pml1+1) * NY_FIELDS * (NZ_FIELDS-1)) + i1 * NY_FIELDS * (NZ_FIELDS-1) + j * (NZ_FIELDS-1) + k;
                float phi = xmEPhi2[p_idx];
                Ez[id4] += upd * (RA01 * dHy + RB0 * phi);
                xmEPhi2[p_idx] = RE0 * phi - RF0 * dHy;
            }
        }

        if (in_y0) {
            long long j1 = pml2 - j;
            float RA01 = y0ER[j1] - 1.0f, RB0 = y0ER[pml2 + j1], RE0 = y0ER[2 * pml2 + j1], RF0 = y0ER[3 * pml2 + j1];
            if (i < NX_FIELDS - 1 && j > 0) {
                float dHz = staggered_backward_diff_static<ORDER>(Hz, id4, NZ_FIELDS, j, NY_FIELDS) / dy;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * (pml2+1) * NZ_FIELDS) + i * (pml2+1) * NZ_FIELDS + j1 * NZ_FIELDS + k;
                float phi = y0EPhi1[p_idx];
                Ex[id4] += upd * (RA01 * dHz + RB0 * phi);
                y0EPhi1[p_idx] = RE0 * phi - RF0 * dHz;
            }
            if (k < NZ_FIELDS - 1 && j > 0) {
                float dHx = staggered_backward_diff_static<ORDER>(Hx, id4, NZ_FIELDS, j, NY_FIELDS) / dy;
                long long p_idx = ((long long)s * NX_FIELDS * (pml2+1) * (NZ_FIELDS-1)) + i * (pml2+1) * (NZ_FIELDS-1) + j1 * (NZ_FIELDS-1) + k;
                float phi = y0EPhi2[p_idx];
                Ez[id4] -= upd * (RA01 * dHx + RB0 * phi);
                y0EPhi2[p_idx] = RE0 * phi - RF0 * dHx;
            }
        }

        if (in_ym) {
            long long j1 = j - (NY_FIELDS - 1 - pml3);
            float RA01 = ymER[j1] - 1.0f, RB0 = ymER[pml3 + j1], RE0 = ymER[2 * pml3 + j1], RF0 = ymER[3 * pml3 + j1];
            if (i < NX_FIELDS - 1 && j > 0) {
                float dHz = staggered_backward_diff_static<ORDER>(Hz, id4, NZ_FIELDS, j, NY_FIELDS) / dy;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * (pml3+1) * NZ_FIELDS) + i * (pml3+1) * NZ_FIELDS + j1 * NZ_FIELDS + k;
                float phi = ymEPhi1[p_idx];
                Ex[id4] += upd * (RA01 * dHz + RB0 * phi);
                ymEPhi1[p_idx] = RE0 * phi - RF0 * dHz;
            }
            if (k < NZ_FIELDS - 1 && j > 0) {
                float dHx = staggered_backward_diff_static<ORDER>(Hx, id4, NZ_FIELDS, j, NY_FIELDS) / dy;
                long long p_idx = ((long long)s * NX_FIELDS * (pml3+1) * (NZ_FIELDS-1)) + i * (pml3+1) * (NZ_FIELDS-1) + j1 * (NZ_FIELDS-1) + k;
                float phi = ymEPhi2[p_idx];
                Ez[id4] -= upd * (RA01 * dHx + RB0 * phi);
                ymEPhi2[p_idx] = RE0 * phi - RF0 * dHx;
            }
        }

        if (in_z0) {
            long long k1 = pml4 - k;
            float RA01 = z0ER[k1] - 1.0f, RB0 = z0ER[pml4 + k1], RE0 = z0ER[2 * pml4 + k1], RF0 = z0ER[3 * pml4 + k1];
            if (i < NX_FIELDS - 1 && k > 0) {
                float dHy = staggered_backward_diff_static<ORDER>(Hy, id4, 1, k, NZ_FIELDS) / dz;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * NY_FIELDS * (pml4+1)) + i * NY_FIELDS * (pml4+1) + j * (pml4+1) + k1;
                float phi = z0EPhi1[p_idx];
                Ex[id4] -= upd * (RA01 * dHy + RB0 * phi);
                z0EPhi1[p_idx] = RE0 * phi - RF0 * dHy;
            }
            if (j < NY_FIELDS - 1 && k > 0) {
                float dHx = staggered_backward_diff_static<ORDER>(Hx, id4, 1, k, NZ_FIELDS) / dz;
                long long p_idx = ((long long)s * NX_FIELDS * (NY_FIELDS-1) * (pml4+1)) + i * (NY_FIELDS-1) * (pml4+1) + j * (pml4+1) + k1;
                float phi = z0EPhi2[p_idx];
                Ey[id4] += upd * (RA01 * dHx + RB0 * phi);
                z0EPhi2[p_idx] = RE0 * phi - RF0 * dHx;
            }
        }

        if (in_zm) {
            long long k1 = k - (NZ_FIELDS - 1 - pml5);
            float RA01 = zmER[k1] - 1.0f, RB0 = zmER[pml5 + k1], RE0 = zmER[2 * pml5 + k1], RF0 = zmER[3 * pml5 + k1];
            if (i < NX_FIELDS - 1 && k > 0) {
                float dHy = staggered_backward_diff_static<ORDER>(Hy, id4, 1, k, NZ_FIELDS) / dz;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * NY_FIELDS * (pml5+1)) + i * NY_FIELDS * (pml5+1) + j * (pml5+1) + k1;
                float phi = zmEPhi1[p_idx];
                Ex[id4] -= upd * (RA01 * dHy + RB0 * phi);
                zmEPhi1[p_idx] = RE0 * phi - RF0 * dHy;
            }
            if (j < NY_FIELDS - 1 && k > 0) {
                float dHx = staggered_backward_diff_static<ORDER>(Hx, id4, 1, k, NZ_FIELDS) / dz;
                long long p_idx = ((long long)s * NX_FIELDS * (NY_FIELDS-1) * (pml5+1)) + i * (NY_FIELDS-1) * (pml5+1) + j * (pml5+1) + k1;
                float phi = zmEPhi2[p_idx];
                Ey[id4] += upd * (RA01 * dHx + RB0 * phi);
                zmEPhi2[p_idx] = RE0 * phi - RF0 * dHx;
            }
        }
    }
}


template<int ORDER>
__global__ void h_fields_base_update_gpu(
    const float* __restrict__ uH0, const float* __restrict__ uH1,
    const float* __restrict__ Ex, const float* __restrict__ Ey, const float* __restrict__ Ez,
    float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS)
{
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long field_stride = (long long)NX_FIELDS * ny_nz;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= (long long)step * field_stride) return;

    long long idx = work % field_stride;
    long long i = idx / ny_nz;
    long long rem = idx % ny_nz;
    long long j = rem / NZ_FIELDS;
    long long k = rem % NZ_FIELDS;
    long long id4 = work;

    float uh0 = uH0[idx];
    float uh1 = uH1[idx];

    bool do_hx = ((NX_FIELDS - 1) != 1 && i > 0 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));
    bool do_hy = ((NY_FIELDS - 1) != 1 && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));
    bool do_hz = ((NZ_FIELDS - 1) != 1 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));

    if (do_hx) {
        float dEz_dy = staggered_forward_diff_static<ORDER>(Ez, id4, NZ_FIELDS, j, NY_FIELDS);
        float dEy_dz = staggered_forward_diff_static<ORDER>(Ey, id4, 1, k, NZ_FIELDS);
        Hx[id4] = uh0 * Hx[id4] - uh1 * dEz_dy + uh1 * dEy_dz;
    }
    if (do_hy) {
        float dEx_dz = staggered_forward_diff_static<ORDER>(Ex, id4, 1, k, NZ_FIELDS);
        float dEz_dx = staggered_forward_diff_static<ORDER>(Ez, id4, ny_nz, i, NX_FIELDS);
        Hy[id4] = uh0 * Hy[id4] - uh1 * dEx_dz + uh1 * dEz_dx;
    }
    if (do_hz) {
        float dEy_dx = staggered_forward_diff_static<ORDER>(Ey, id4, ny_nz, i, NX_FIELDS);
        float dEx_dy = staggered_forward_diff_static<ORDER>(Ex, id4, NZ_FIELDS, j, NY_FIELDS);
        Hz[id4] = uh0 * Hz[id4] - uh1 * dEy_dx + uh1 * dEx_dy;
    }
}

__global__ void e_fields_base_adjoint_gpu(
    const float* uE0, const float* uE1,
    float* Ex, float* Ey, float* Ez, float* Hx, float* Hy, float* Hz,
    int step, int NX, int NY, int NZ, int order)
{
    long long ny_nz = (long long)NY * NZ;
    long long field_stride = (long long)NX * ny_nz;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= (long long)step * field_stride) return;
    long long idx = work % field_stride;
    long long i = idx / ny_nz;
    long long rem = idx % ny_nz;
    long long j = rem / NZ;
    long long k = rem % NZ;
    bool do_ex = (((NY - 1) != 1 || (NZ - 1) != 1) && i < NX - 1 && j > 0 && j < NY - 1 && k > 0 && k < NZ - 1);
    bool do_ey = (((NX - 1) != 1 || (NZ - 1) != 1) && i > 0 && i < NX - 1 && j < NY - 1 && k > 0 && k < NZ - 1);
    bool do_ez = (((NX - 1) != 1 || (NY - 1) != 1) && i > 0 && i < NX - 1 && j > 0 && j < NY - 1 && k < NZ - 1);
    float coeff = uE1[idx];

    if (do_ex) {
        float value = Ex[work];
        add_staggered_backward_adjoint(Hz, work, NZ, j, NY, order, coeff * value);
        add_staggered_backward_adjoint(Hy, work, 1, k, NZ, order, -coeff * value);
        Ex[work] = uE0[idx] * value;
    }
    if (do_ey) {
        float value = Ey[work];
        add_staggered_backward_adjoint(Hx, work, 1, k, NZ, order, coeff * value);
        add_staggered_backward_adjoint(Hz, work, ny_nz, i, NX, order, -coeff * value);
        Ey[work] = uE0[idx] * value;
    }
    if (do_ez) {
        float value = Ez[work];
        add_staggered_backward_adjoint(Hy, work, ny_nz, i, NX, order, coeff * value);
        add_staggered_backward_adjoint(Hx, work, NZ, j, NY, order, -coeff * value);
        Ez[work] = uE0[idx] * value;
    }
}

__global__ void h_fields_base_adjoint_gpu(
    const float* uH0, const float* uH1,
    float* Ex, float* Ey, float* Ez, float* Hx, float* Hy, float* Hz,
    int step, int NX, int NY, int NZ, int order)
{
    long long ny_nz = (long long)NY * NZ;
    long long field_stride = (long long)NX * ny_nz;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= (long long)step * field_stride) return;
    long long idx = work % field_stride;
    long long i = idx / ny_nz;
    long long rem = idx % ny_nz;
    long long j = rem / NZ;
    long long k = rem % NZ;
    bool do_hx = ((NX - 1) != 1 && i > 0 && i < NX - 1 && j < NY - 1 && k < NZ - 1);
    bool do_hy = ((NY - 1) != 1 && i < NX - 1 && j > 0 && j < NY - 1 && k < NZ - 1);
    bool do_hz = ((NZ - 1) != 1 && i < NX - 1 && j < NY - 1 && k > 0 && k < NZ - 1);
    float coeff = uH1[idx];

    if (do_hx) {
        float value = Hx[work];
        add_staggered_forward_adjoint(Ez, work, NZ, j, NY, order, -coeff * value);
        add_staggered_forward_adjoint(Ey, work, 1, k, NZ, order, coeff * value);
        Hx[work] = uH0[idx] * value;
    }
    if (do_hy) {
        float value = Hy[work];
        add_staggered_forward_adjoint(Ex, work, 1, k, NZ, order, -coeff * value);
        add_staggered_forward_adjoint(Ez, work, ny_nz, i, NX, order, coeff * value);
        Hy[work] = uH0[idx] * value;
    }
    if (do_hz) {
        float value = Hz[work];
        add_staggered_forward_adjoint(Ey, work, ny_nz, i, NX, order, -coeff * value);
        add_staggered_forward_adjoint(Ex, work, NZ, j, NY, order, coeff * value);
        Hz[work] = uH0[idx] * value;
    }
}

/*
 * Apply magnetic CPML boundary corrections after the base magnetic-field update.
 */
template<int ORDER>
__global__ void pml_h_fields_update_gpu(
    const float* __restrict__ Ex, const float* __restrict__ Ey, const float* __restrict__ Ez,
    float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
    float dx, float dy, float dz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* __restrict__ x0HR, const float* __restrict__ xmHR,
    const float* __restrict__ y0HR, const float* __restrict__ ymHR,
    const float* __restrict__ z0HR, const float* __restrict__ zmHR,
    const float* __restrict__ updatecoeffsH,
    float* __restrict__ x0HPhi1, float* __restrict__ x0HPhi2,
    float* __restrict__ xmHPhi1, float* __restrict__ xmHPhi2,
    float* __restrict__ y0HPhi1, float* __restrict__ y0HPhi2,
    float* __restrict__ ymHPhi1, float* __restrict__ ymHPhi2,
    float* __restrict__ z0HPhi1, float* __restrict__ z0HPhi2,
    float* __restrict__ zmHPhi1, float* __restrict__ zmHPhi2)
{
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long field_stride = (long long)NX_FIELDS * ny_nz;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= (long long)step * field_stride) return;

    int s = (int)(work / field_stride);
    long long idx = work % field_stride;

    long long i = idx / ny_nz;
    long long rem = idx % ny_nz;
    long long j = rem / NZ_FIELDS;
    long long k = rem % NZ_FIELDS;

    bool in_x0 = (pml0 > 0 && i < pml0 && j < NY_FIELDS && k < NZ_FIELDS);
    bool in_xm = (pml1 > 0 && i >= NX_FIELDS - 1 - pml1 && i < NX_FIELDS - 1 && j < NY_FIELDS && k < NZ_FIELDS);
    bool in_y0 = (pml2 > 0 && i < NX_FIELDS && j < pml2 && k < NZ_FIELDS);
    bool in_ym = (pml3 > 0 && i < NX_FIELDS && j >= NY_FIELDS - 1 - pml3 && j < NY_FIELDS - 1 && k < NZ_FIELDS);
    bool in_z0 = (pml4 > 0 && i < NX_FIELDS && j < NY_FIELDS && k < pml4);
    bool in_zm = (pml5 > 0 && i < NX_FIELDS && j < NY_FIELDS && k >= NZ_FIELDS - 1 - pml5 && k < NZ_FIELDS - 1);
    if (!(in_x0 || in_xm || in_y0 || in_ym || in_z0 || in_zm)) return;

    float upd = updatecoeffsH[idx];

    long long id4 = work;

    {
        if (in_x0) {
            long long i1 = pml0 - 1 - i;
            float RA01 = x0HR[i1] - 1.0f, RB0 = x0HR[pml0 + i1], RE0 = x0HR[2 * pml0 + i1], RF0 = x0HR[3 * pml0 + i1];
            if (k < NZ_FIELDS - 1) {
                float dEz = staggered_forward_diff_static<ORDER>(Ez, id4, ny_nz, i, NX_FIELDS) / dx;
                long long p_idx = ((long long)s * pml0 * NY_FIELDS * (NZ_FIELDS-1)) + i1 * NY_FIELDS * (NZ_FIELDS-1) + j * (NZ_FIELDS-1) + k;
                float phi = x0HPhi1[p_idx];
                Hy[id4] += upd * (RA01 * dEz + RB0 * phi);
                x0HPhi1[p_idx] = RE0 * phi - RF0 * dEz;
            }
            if (j < NY_FIELDS - 1) {
                float dEy = staggered_forward_diff_static<ORDER>(Ey, id4, ny_nz, i, NX_FIELDS) / dx;
                long long p_idx = ((long long)s * pml0 * (NY_FIELDS-1) * NZ_FIELDS) + i1 * (NY_FIELDS-1) * NZ_FIELDS + j * NZ_FIELDS + k;
                float phi = x0HPhi2[p_idx];
                Hz[id4] -= upd * (RA01 * dEy + RB0 * phi);
                x0HPhi2[p_idx] = RE0 * phi - RF0 * dEy;
            }
        }

        if (in_xm) {
            long long i1 = i - (NX_FIELDS - 1 - pml1);
            float RA01 = xmHR[i1] - 1.0f, RB0 = xmHR[pml1 + i1], RE0 = xmHR[2 * pml1 + i1], RF0 = xmHR[3 * pml1 + i1];
            if (k < NZ_FIELDS - 1) {
                float dEz = staggered_forward_diff_static<ORDER>(Ez, id4, ny_nz, i, NX_FIELDS) / dx;
                long long p_idx = ((long long)s * pml1 * NY_FIELDS * (NZ_FIELDS-1)) + i1 * NY_FIELDS * (NZ_FIELDS-1) + j * (NZ_FIELDS-1) + k;
                float phi = xmHPhi1[p_idx];
                Hy[id4] += upd * (RA01 * dEz + RB0 * phi);
                xmHPhi1[p_idx] = RE0 * phi - RF0 * dEz;
            }
            if (j < NY_FIELDS - 1) {
                float dEy = staggered_forward_diff_static<ORDER>(Ey, id4, ny_nz, i, NX_FIELDS) / dx;
                long long p_idx = ((long long)s * pml1 * (NY_FIELDS-1) * NZ_FIELDS) + i1 * (NY_FIELDS-1) * NZ_FIELDS + j * NZ_FIELDS + k;
                float phi = xmHPhi2[p_idx];
                Hz[id4] -= upd * (RA01 * dEy + RB0 * phi);
                xmHPhi2[p_idx] = RE0 * phi - RF0 * dEy;
            }
        }

        if (in_y0) {
            long long j1 = pml2 - 1 - j;
            float RA01 = y0HR[j1] - 1.0f, RB0 = y0HR[pml2 + j1], RE0 = y0HR[2 * pml2 + j1], RF0 = y0HR[3 * pml2 + j1];
            if (i < NX_FIELDS && k < NZ_FIELDS - 1) {
                float dEz = staggered_forward_diff_static<ORDER>(Ez, id4, NZ_FIELDS, j, NY_FIELDS) / dy;
                long long p_idx = ((long long)s * NX_FIELDS * pml2 * (NZ_FIELDS-1)) + i * pml2 * (NZ_FIELDS-1) + j1 * (NZ_FIELDS-1) + k;
                float phi = y0HPhi1[p_idx];
                Hx[id4] -= upd * (RA01 * dEz + RB0 * phi);
                y0HPhi1[p_idx] = RE0 * phi - RF0 * dEz;
            }
            if (i < NX_FIELDS - 1 && k < NZ_FIELDS) {
                float dEx = staggered_forward_diff_static<ORDER>(Ex, id4, NZ_FIELDS, j, NY_FIELDS) / dy;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * pml2 * NZ_FIELDS) + i * pml2 * NZ_FIELDS + j1 * NZ_FIELDS + k;
                float phi = y0HPhi2[p_idx];
                Hz[id4] += upd * (RA01 * dEx + RB0 * phi);
                y0HPhi2[p_idx] = RE0 * phi - RF0 * dEx;
            }
        }

        if (in_ym) {
            long long j1 = j - (NY_FIELDS - 1 - pml3);
            float RA01 = ymHR[j1] - 1.0f, RB0 = ymHR[pml3 + j1], RE0 = ymHR[2 * pml3 + j1], RF0 = ymHR[3 * pml3 + j1];
            if (i < NX_FIELDS && k < NZ_FIELDS - 1) {
                float dEz = staggered_forward_diff_static<ORDER>(Ez, id4, NZ_FIELDS, j, NY_FIELDS) / dy;
                long long p_idx = ((long long)s * NX_FIELDS * pml3 * (NZ_FIELDS-1)) + i * pml3 * (NZ_FIELDS-1) + j1 * (NZ_FIELDS-1) + k;
                float phi = ymHPhi1[p_idx];
                Hx[id4] -= upd * (RA01 * dEz + RB0 * phi);
                ymHPhi1[p_idx] = RE0 * phi - RF0 * dEz;
            }
            if (i < NX_FIELDS - 1 && k < NZ_FIELDS) {
                float dEx = staggered_forward_diff_static<ORDER>(Ex, id4, NZ_FIELDS, j, NY_FIELDS) / dy;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * pml3 * NZ_FIELDS) + i * pml3 * NZ_FIELDS + j1 * NZ_FIELDS + k;
                float phi = ymHPhi2[p_idx];
                Hz[id4] += upd * (RA01 * dEx + RB0 * phi);
                ymHPhi2[p_idx] = RE0 * phi - RF0 * dEx;
            }
        }

        if (in_z0) {
            long long k1 = pml4 - 1 - k;
            float RA01 = z0HR[k1] - 1.0f, RB0 = z0HR[pml4 + k1], RE0 = z0HR[2 * pml4 + k1], RF0 = z0HR[3 * pml4 + k1];
            if (i < NX_FIELDS && j < NY_FIELDS - 1) {
                float dEy = staggered_forward_diff_static<ORDER>(Ey, id4, 1, k, NZ_FIELDS) / dz;
                long long p_idx = ((long long)s * NX_FIELDS * (NY_FIELDS-1) * pml4) + i * (NY_FIELDS-1) * pml4 + j * pml4 + k1;
                float phi = z0HPhi1[p_idx];
                Hx[id4] += upd * (RA01 * dEy + RB0 * phi);
                z0HPhi1[p_idx] = RE0 * phi - RF0 * dEy;
            }
            if (i < NX_FIELDS - 1 && j < NY_FIELDS) {
                float dEx = staggered_forward_diff_static<ORDER>(Ex, id4, 1, k, NZ_FIELDS) / dz;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * NY_FIELDS * pml4) + i * NY_FIELDS * pml4 + j * pml4 + k1;
                float phi = z0HPhi2[p_idx];
                Hy[id4] -= upd * (RA01 * dEx + RB0 * phi);
                z0HPhi2[p_idx] = RE0 * phi - RF0 * dEx;
            }
        }

        if (in_zm) {
            long long k1 = k - (NZ_FIELDS - 1 - pml5);
            float RA01 = zmHR[k1] - 1.0f, RB0 = zmHR[pml5 + k1], RE0 = zmHR[2 * pml5 + k1], RF0 = zmHR[3 * pml5 + k1];
            if (i < NX_FIELDS && j < NY_FIELDS - 1) {
                float dEy = staggered_forward_diff_static<ORDER>(Ey, id4, 1, k, NZ_FIELDS) / dz;
                long long p_idx = ((long long)s * NX_FIELDS * (NY_FIELDS-1) * pml5) + i * (NY_FIELDS-1) * pml5 + j * pml5 + k1;
                float phi = zmHPhi1[p_idx];
                Hx[id4] += upd * (RA01 * dEy + RB0 * phi);
                zmHPhi1[p_idx] = RE0 * phi - RF0 * dEy;
            }
            if (i < NX_FIELDS - 1 && j < NY_FIELDS) {
                float dEx = staggered_forward_diff_static<ORDER>(Ex, id4, 1, k, NZ_FIELDS) / dz;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * NY_FIELDS * pml5) + i * NY_FIELDS * pml5 + j * pml5 + k1;
                float phi = zmHPhi2[p_idx];
                Hy[id4] -= upd * (RA01 * dEx + RB0 * phi);
                zmHPhi2[p_idx] = RE0 * phi - RF0 * dEx;
            }
        }
    }
}


__global__ void pml_e_fields_adjoint_gpu(
    float* Ex, float* Ey, float* Ez, float* Hx, float* Hy, float* Hz,
    float dx, float dy, float dz, int step, int NX, int NY, int NZ,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* x0R, const float* xmR, const float* y0R, const float* ymR,
    const float* z0R, const float* zmR, const float* update,
    float* x0P1, float* x0P2, float* xmP1, float* xmP2,
    float* y0P1, float* y0P2, float* ymP1, float* ymP2,
    float* z0P1, float* z0P2, float* zmP1, float* zmP2, int order)
{
    long long ny_nz = (long long)NY * NZ;
    long long field_stride = (long long)NX * ny_nz;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= (long long)step * field_stride) return;
    int s = (int)(work / field_stride);
    long long idx = work % field_stride;
    long long i = idx / ny_nz;
    long long rem = idx % ny_nz;
    long long j = rem / NZ;
    long long k = rem % NZ;
    float upd = update[idx];

#define APPLY_E_PML_GPU(R, P, p, q, stride, coord, n, spacing, field, source, sign) \
    do { \
        pml_backward_derivative_adjoint( \
            (field)[work], (source), work, (stride), (coord), (n), order, \
            1.0f / (spacing), upd, (sign), (R)[q] - 1.0f, (R)[(p) + (q)], \
            (R)[2 * (p) + (q)], (R)[3 * (p) + (q)], &(P)[p_idx]); \
    } while (0)

    if (pml0 > 0 && i > 0 && i <= pml0) {
        long long q = pml0 - i;
        if (j < NY - 1) {
            long long p_idx = ((long long)s * (pml0 + 1) * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
            APPLY_E_PML_GPU(x0R, x0P1, pml0, q, ny_nz, i, NX, dx, Ey, Hz, -1.0f);
        }
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * (pml0 + 1) * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
            APPLY_E_PML_GPU(x0R, x0P2, pml0, q, ny_nz, i, NX, dx, Ez, Hy, 1.0f);
        }
    }
    if (pml1 > 0 && i >= NX - 1 - pml1 && i < NX - 1) {
        long long q = i - (NX - 1 - pml1);
        if (j < NY - 1 && i > 0) {
            long long p_idx = ((long long)s * (pml1 + 1) * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
            APPLY_E_PML_GPU(xmR, xmP1, pml1, q, ny_nz, i, NX, dx, Ey, Hz, -1.0f);
        }
        if (k < NZ - 1 && i > 0) {
            long long p_idx = ((long long)s * (pml1 + 1) * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
            APPLY_E_PML_GPU(xmR, xmP2, pml1, q, ny_nz, i, NX, dx, Ez, Hy, 1.0f);
        }
    }
    if (pml2 > 0 && j > 0 && j <= pml2) {
        long long q = pml2 - j;
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * (pml2 + 1) * NZ) + i * (pml2 + 1) * NZ + q * NZ + k;
            APPLY_E_PML_GPU(y0R, y0P1, pml2, q, NZ, j, NY, dy, Ex, Hz, 1.0f);
        }
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * NX * (pml2 + 1) * (NZ - 1)) + i * (pml2 + 1) * (NZ - 1) + q * (NZ - 1) + k;
            APPLY_E_PML_GPU(y0R, y0P2, pml2, q, NZ, j, NY, dy, Ez, Hx, -1.0f);
        }
    }
    if (pml3 > 0 && j >= NY - 1 - pml3 && j < NY - 1) {
        long long q = j - (NY - 1 - pml3);
        if (i < NX - 1 && j > 0) {
            long long p_idx = ((long long)s * (NX - 1) * (pml3 + 1) * NZ) + i * (pml3 + 1) * NZ + q * NZ + k;
            APPLY_E_PML_GPU(ymR, ymP1, pml3, q, NZ, j, NY, dy, Ex, Hz, 1.0f);
        }
        if (k < NZ - 1 && j > 0) {
            long long p_idx = ((long long)s * NX * (pml3 + 1) * (NZ - 1)) + i * (pml3 + 1) * (NZ - 1) + q * (NZ - 1) + k;
            APPLY_E_PML_GPU(ymR, ymP2, pml3, q, NZ, j, NY, dy, Ez, Hx, -1.0f);
        }
    }
    if (pml4 > 0 && k > 0 && k <= pml4) {
        long long q = pml4 - k;
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * NY * (pml4 + 1)) + i * NY * (pml4 + 1) + j * (pml4 + 1) + q;
            APPLY_E_PML_GPU(z0R, z0P1, pml4, q, 1, k, NZ, dz, Ex, Hy, -1.0f);
        }
        if (j < NY - 1) {
            long long p_idx = ((long long)s * NX * (NY - 1) * (pml4 + 1)) + i * (NY - 1) * (pml4 + 1) + j * (pml4 + 1) + q;
            APPLY_E_PML_GPU(z0R, z0P2, pml4, q, 1, k, NZ, dz, Ey, Hx, 1.0f);
        }
    }
    if (pml5 > 0 && k >= NZ - 1 - pml5 && k < NZ - 1) {
        long long q = k - (NZ - 1 - pml5);
        if (i < NX - 1 && k > 0) {
            long long p_idx = ((long long)s * (NX - 1) * NY * (pml5 + 1)) + i * NY * (pml5 + 1) + j * (pml5 + 1) + q;
            APPLY_E_PML_GPU(zmR, zmP1, pml5, q, 1, k, NZ, dz, Ex, Hy, -1.0f);
        }
        if (j < NY - 1 && k > 0) {
            long long p_idx = ((long long)s * NX * (NY - 1) * (pml5 + 1)) + i * (NY - 1) * (pml5 + 1) + j * (pml5 + 1) + q;
            APPLY_E_PML_GPU(zmR, zmP2, pml5, q, 1, k, NZ, dz, Ey, Hx, 1.0f);
        }
    }
#undef APPLY_E_PML_GPU
}

__global__ void pml_h_fields_adjoint_gpu(
    float* Ex, float* Ey, float* Ez, float* Hx, float* Hy, float* Hz,
    float dx, float dy, float dz, int step, int NX, int NY, int NZ,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* x0R, const float* xmR, const float* y0R, const float* ymR,
    const float* z0R, const float* zmR, const float* update,
    float* x0P1, float* x0P2, float* xmP1, float* xmP2,
    float* y0P1, float* y0P2, float* ymP1, float* ymP2,
    float* z0P1, float* z0P2, float* zmP1, float* zmP2, int order)
{
    long long ny_nz = (long long)NY * NZ;
    long long field_stride = (long long)NX * ny_nz;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= (long long)step * field_stride) return;
    int s = (int)(work / field_stride);
    long long idx = work % field_stride;
    long long i = idx / ny_nz;
    long long rem = idx % ny_nz;
    long long j = rem / NZ;
    long long k = rem % NZ;
    float upd = update[idx];

#define APPLY_H_PML_GPU(R, P, p, q, stride, coord, n, spacing, field, source, sign) \
    do { \
        pml_forward_derivative_adjoint( \
            (field)[work], (source), work, (stride), (coord), (n), order, \
            1.0f / (spacing), upd, (sign), (R)[q] - 1.0f, (R)[(p) + (q)], \
            (R)[2 * (p) + (q)], (R)[3 * (p) + (q)], &(P)[p_idx]); \
    } while (0)

    if (pml0 > 0 && i < pml0) {
        long long q = pml0 - 1 - i;
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * pml0 * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
            APPLY_H_PML_GPU(x0R, x0P1, pml0, q, ny_nz, i, NX, dx, Hy, Ez, 1.0f);
        }
        if (j < NY - 1) {
            long long p_idx = ((long long)s * pml0 * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
            APPLY_H_PML_GPU(x0R, x0P2, pml0, q, ny_nz, i, NX, dx, Hz, Ey, -1.0f);
        }
    }
    if (pml1 > 0 && i >= NX - 1 - pml1 && i < NX - 1) {
        long long q = i - (NX - 1 - pml1);
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * pml1 * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
            APPLY_H_PML_GPU(xmR, xmP1, pml1, q, ny_nz, i, NX, dx, Hy, Ez, 1.0f);
        }
        if (j < NY - 1) {
            long long p_idx = ((long long)s * pml1 * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
            APPLY_H_PML_GPU(xmR, xmP2, pml1, q, ny_nz, i, NX, dx, Hz, Ey, -1.0f);
        }
    }
    if (pml2 > 0 && j < pml2) {
        long long q = pml2 - 1 - j;
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * NX * pml2 * (NZ - 1)) + i * pml2 * (NZ - 1) + q * (NZ - 1) + k;
            APPLY_H_PML_GPU(y0R, y0P1, pml2, q, NZ, j, NY, dy, Hx, Ez, -1.0f);
        }
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * pml2 * NZ) + i * pml2 * NZ + q * NZ + k;
            APPLY_H_PML_GPU(y0R, y0P2, pml2, q, NZ, j, NY, dy, Hz, Ex, 1.0f);
        }
    }
    if (pml3 > 0 && j >= NY - 1 - pml3 && j < NY - 1) {
        long long q = j - (NY - 1 - pml3);
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * NX * pml3 * (NZ - 1)) + i * pml3 * (NZ - 1) + q * (NZ - 1) + k;
            APPLY_H_PML_GPU(ymR, ymP1, pml3, q, NZ, j, NY, dy, Hx, Ez, -1.0f);
        }
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * pml3 * NZ) + i * pml3 * NZ + q * NZ + k;
            APPLY_H_PML_GPU(ymR, ymP2, pml3, q, NZ, j, NY, dy, Hz, Ex, 1.0f);
        }
    }
    if (pml4 > 0 && k < pml4) {
        long long q = pml4 - 1 - k;
        if (j < NY - 1) {
            long long p_idx = ((long long)s * NX * (NY - 1) * pml4) + i * (NY - 1) * pml4 + j * pml4 + q;
            APPLY_H_PML_GPU(z0R, z0P1, pml4, q, 1, k, NZ, dz, Hx, Ey, 1.0f);
        }
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * NY * pml4) + i * NY * pml4 + j * pml4 + q;
            APPLY_H_PML_GPU(z0R, z0P2, pml4, q, 1, k, NZ, dz, Hy, Ex, -1.0f);
        }
    }
    if (pml5 > 0 && k >= NZ - 1 - pml5 && k < NZ - 1) {
        long long q = k - (NZ - 1 - pml5);
        if (j < NY - 1) {
            long long p_idx = ((long long)s * NX * (NY - 1) * pml5) + i * (NY - 1) * pml5 + j * pml5 + q;
            APPLY_H_PML_GPU(zmR, zmP1, pml5, q, 1, k, NZ, dz, Hx, Ey, 1.0f);
        }
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * NY * pml5) + i * NY * pml5 + j * pml5 + q;
            APPLY_H_PML_GPU(zmR, zmP2, pml5, q, 1, k, NZ, dz, Hy, Ex, -1.0f);
        }
    }
#undef APPLY_H_PML_GPU
}

/*
 * Inject the adjoint source into one electric-field component.
 *
 * Parameters:
 *   step: Number of shots or simulations in the batch.
 *   iteration: Current reverse time-step index.
 *   dx: Spatial grid spacing.
 *   sourcelocation: Adjoint source coordinates with shape (step, nsr, 3).
 *   srcwaveforms: Adjoint source waveform array.
 *   Ex, Ey, Ez: Electric field component arrays to update.
 *   uE4: Electric source scaling coefficient array.
 *   NX, NY, NZ: Padded field grid sizes.
 *   nsr: Number of adjoint sources per shot.
 *   polarisation: Source component, 0 for x, 1 for y, 2 for z.
 *   iterations: Total number of time steps.
 */
__global__ void add_adjoint_sources(
    int step, int iteration,
    const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms,
    float* Ex, float* Ey, float* Ez,
    int NX, int NY, int NZ, int nsr, int polarisation, int iterations
){
    long long field_stride = (long long)NX * NY * NZ;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)step * nsr;
    if (work >= total) return;

    int s = (int)(work / nsr);
    long long src = work % nsr;

    long long i = sourcelocation[s * nsr * 3 + src * 3 + 0];
    long long j = sourcelocation[s * nsr * 3 + src * 3 + 1];
    long long k = sourcelocation[s * nsr * 3 + src * 3 + 2];

    long long index = (long long)s * iterations * nsr + (long long)iteration * nsr + src;
    float waveform_value = srcwaveforms[index];
    long long id4 = (long long)s * field_stride + i * NY * NZ + j * NZ + k;

    if (polarisation == 0) Ex[id4] += waveform_value;
    else if (polarisation == 1) Ey[id4] += waveform_value;
    else if (polarisation == 2) Ez[id4] += waveform_value;
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
__global__ void copy_to_Eall_single(
    float* __restrict__ dst_ptr, int t_idx, const float* __restrict__ E, 
    int step, int NX, int NY, int NZ)
{
    long long nx1 = NX - 1, ny1 = NY - 1, nz1 = NZ - 1;
    long long total = nx1 * ny1 * nz1;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= (long long)step * total) return;

    int s = (int)(work / total);
    long long idx = work % total;

    long long i = idx / (ny1 * nz1);
    long long rem = idx % (ny1 * nz1);
    long long j = rem / nz1;
    long long k = rem % nz1;

    long long field_stride = (long long)NX * NY * NZ;
    long long src_idx = (long long)s * field_stride + i * NY * NZ + j * NZ + k;
    long long dst_idx = (long long)t_idx * step * total + (long long)s * total + idx;
    dst_ptr[dst_idx] = E[src_idx];
}


/* Save R from the executed update E^(n+1) = ca E^n + cb R. */
__global__ void copy_to_Rall_single(
    float* __restrict__ dst_ptr, int t_idx,
    const float* __restrict__ E, const float* __restrict__ Eold_ptr,
    const float* __restrict__ ca, const float* __restrict__ cb,
    int step, int NX, int NY, int NZ)
{
    long long nx1 = NX - 1, ny1 = NY - 1, nz1 = NZ - 1;
    long long total = nx1 * ny1 * nz1;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= (long long)step * total) return;

    int s = (int)(work / total);
    long long idx = work % total;
    long long i = idx / (ny1 * nz1);
    long long rem = idx % (ny1 * nz1);
    long long j = rem / nz1;
    long long k = rem % nz1;
    long long material_idx = i * NY * NZ + j * NZ + k;
    long long field_idx = (long long)s * NX * NY * NZ + material_idx;
    long long snap_stride = (long long)step * total;
    long long saved_idx = (long long)t_idx * snap_stride + work;
    float cb_value = cb[material_idx];

    dst_ptr[saved_idx] = cb_value != 0.0f
        ? (E[field_idx] - ca[material_idx] * Eold_ptr[saved_idx]) / cb_value
        : 0.0f;
}


/*
 * Accumulate model gradients from saved forward fields and adjoint fields.
 *
 * Parameters:
 *   Ex, Ey, Ez: Adjoint electric field component arrays.
 *   Eall_ptr: Saved pre-update electric field E^n.
 *   Rall_ptr: Saved effective right-hand side R^n.
 *   d_E_buf, d_R_buf: Device staging buffers for host-offloaded snapshots.
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
 *   use_async_offload: Whether Eall is read through d_E_buf.
 *   fwi_mode: Gradient mode; 2 uses Ez only, 3 uses Ex, Ey, and Ez.
 */
__global__ void accumulate_gradients(
    const float* __restrict__ Ex, const float* __restrict__ Ey, const float* __restrict__ Ez,
    const float* __restrict__ Eall_ptr, const float* __restrict__ Rall_ptr,
    const float* __restrict__ d_E_buf, const float* __restrict__ d_R_buf,
    const float* __restrict__ ca, const float* __restrict__ cb,
    const float* __restrict__ se,
    float* __restrict__ grader, float* __restrict__ gradse,
    int i, int step, int NX, int NY, int NZ, float dt,int errequiregrad,int serequiregrad,
    int S, int nt_saved, int use_async_offload, int fwi_mode
) {
    long long sx = (NX - 1), sy = (NY - 1), sz = (NZ - 1);
    long long total_cells = sx * sy * sz;
    long long snap_stride = (long long)step * total_cells;
    long long component_stride = (long long)nt_saved * snap_stride;
    int components = (fwi_mode == 3) ? 3 : 1;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;

    if (work >= (long long)step * total_cells) return;

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
    float adjoint_values[3];
    adjoint_values[0] = Ex[idx_E];
    adjoint_values[1] = Ey[idx_E];
    adjoint_values[2] = Ez[idx_E];

    for (int c = 0; c < components; ++c) {
        long long comp_offset = (fwi_mode == 3) ? (long long)c * component_stride : 0;
        float e_old, rhs;

        if (use_async_offload) {
            long long device_idx = (long long)c * snap_stride + base_idx;
            e_old = d_E_buf[device_idx];
            rhs = d_R_buf[device_idx];
        } else {
            long long saved_idx = comp_offset + (long long)(i / S) * snap_stride + base_idx;
            e_old = Eall_ptr[saved_idx];
            rhs = Rall_ptr[saved_idx];
        }

        float adjoint_val = adjoint_values[(fwi_mode == 3) ? c : 2];
        float grad_ca = adjoint_val * e_old * (float)S;
        float grad_cb = adjoint_val * rhs * (float)S;
        float ca_value = ca[material_idx];
        float cb_value = cb[material_idx];

        if (se[material_idx] <= 100.0f) {
            if (errequiregrad == 1) {
                float dca_der = e0 * (1.0f - ca_value) * cb_value / dt;
                float dcb_der = -e0 * cb_value * cb_value / dt;
                local_grader += grad_ca * dca_der + grad_cb * dcb_der;
            }
            if (serequiregrad == 1) {
                float dca_dse = -0.5f * (1.0f + ca_value) * cb_value;
                float dcb_dse = -0.5f * cb_value * cb_value;
                local_gradse += grad_ca * dca_dse + grad_cb * dcb_dse;
            }
        }
    }

    if (errequiregrad == 1) atomicAdd(&grader[idx], local_grader);
    if (serequiregrad == 1) atomicAdd(&gradse[idx], local_gradse);
}


/*
 * Run CUDA forward FDTD modeling.
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
DEEPGPR_API void forward(const float* __restrict__ er, const float* __restrict__ se, const float* __restrict__ mr,  
             float* __restrict__ Eall_ptr, float* __restrict__ Rall_ptr,
             float* __restrict__ Ex,  float* __restrict__ Ey, float* __restrict__ Ez,  
             float* __restrict__ Hx, float* __restrict__ Hy,  float* __restrict__ Hz,
             float* __restrict__ uE0, float* __restrict__ uE1, float* __restrict__ uE4,
             float* __restrict__ uH0, float* __restrict__ uH1, float* __restrict__ uH4,

            float* __restrict__ x0EPhi1,float* __restrict__ x0EPhi2, float* __restrict__ x0HPhi1,float* __restrict__ x0HPhi2,
            float* __restrict__ xmEPhi1,float* __restrict__ xmEPhi2, float* __restrict__ xmHPhi1,float* __restrict__ xmHPhi2,
            float* __restrict__ y0EPhi1,float* __restrict__ y0EPhi2, float* __restrict__ y0HPhi1,float* __restrict__ y0HPhi2,
            float* __restrict__ ymEPhi1,float* __restrict__ ymEPhi2, float* __restrict__ ymHPhi1,float* __restrict__ ymHPhi2,
            float* __restrict__ z0EPhi1,float* __restrict__ z0EPhi2, float* __restrict__ z0HPhi1,float* __restrict__ z0HPhi2,
            float* __restrict__ zmEPhi1,float* __restrict__ zmEPhi2, float* __restrict__ zmHPhi1,float* __restrict__ zmHPhi2,

            int pml0,int pml1,int pml2,int pml3,int pml4,int pml5,

            const float* __restrict__ x0ER,const float* __restrict__ xmER, const float* __restrict__ y0ER,const float* __restrict__ ymER,
            const float* __restrict__ z0ER,const float* __restrict__ zmER, const float* __restrict__ x0HR,const float* __restrict__ xmHR,
            const float* __restrict__ y0HR,const float* __restrict__ ymHR, const float* __restrict__ z0HR,const float* __restrict__ zmHR,

             float dt, int nt, int step, int nrx, float dx,
             const int* __restrict__ receiverlocation, float* __restrict__ rxs, 

             int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS, int nsrc, 
             const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms, int polarisation,
             int sampling_interval, int fwi_mode)
{
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, Eall_ptr);
    int use_async = 0;
    if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
        use_async = 0; 
    } else {
        cudaGetLastError(); 
        use_async = 1; 
    }
    int fdtd_order = g_fdtd_order;
    int e_components = (fwi_mode == 3) ? 3 : 1;
    int nt_saved = (nt + sampling_interval - 1) / sampling_interval;

    float* d_E_buf = nullptr;
    float* d_R_buf = nullptr;
    long long snap_size = (long long)step * (NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1);
    long long component_stride = (long long)nt_saved * snap_size;
    
    cudaStream_t stream_comp = 0, stream_trans = 0;
    cudaEvent_t event_comp;
    if (use_async) {
        CUDA_CHECK(cudaStreamCreate(&stream_comp));
        CUDA_CHECK(cudaStreamCreate(&stream_trans));
        CUDA_CHECK(cudaEventCreate(&event_comp));
        CUDA_CHECK(cudaMalloc(&d_E_buf, 2 * e_components * snap_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_R_buf, 2 * e_components * snap_size * sizeof(float)));
    }

    long long blockSize = 256;
    long long total_fields = (long long)NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    dim3 grid_material(CEIL_DIV(total_fields, blockSize));
    dim3 grid_fields(CEIL_DIV((long long)step * total_fields, blockSize));

    ucgetforward<<<grid_material, blockSize, 0, stream_comp>>>(er, se, mr, uE0, uE1, uE4, uH0, uH1, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);
    CUDA_CHECK_LAST();
  
    dim3 grid_rx(CEIL_DIV((long long)step * nrx, blockSize));
    dim3 grid_src(CEIL_DIV((long long)step * nsrc, blockSize));
    long long total_copy = (long long)(NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1); 
    dim3 grid_copy(CEIL_DIV((long long)step * total_copy, blockSize));

    for (int i = 0; i < nt; i++) {
        if (i % sampling_interval == 0) {
            int t_saved = i / sampling_interval;
            if (use_async) {
                int buf_idx = t_saved % 2;
                long long buf_base = (long long)buf_idx * e_components * snap_size;
                CUDA_CHECK(cudaStreamSynchronize(stream_trans));
                if (fwi_mode == 3) {
                    copy_to_Eall_single<<<grid_copy, blockSize, 0, stream_comp>>>(d_E_buf + buf_base, 0, Ex, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    copy_to_Eall_single<<<grid_copy, blockSize, 0, stream_comp>>>(d_E_buf + buf_base + snap_size, 0, Ey, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    copy_to_Eall_single<<<grid_copy, blockSize, 0, stream_comp>>>(d_E_buf + buf_base + 2 * snap_size, 0, Ez, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                } else {
                    copy_to_Eall_single<<<grid_copy, blockSize, 0, stream_comp>>>(d_E_buf + buf_base, 0, Ez, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                }
                CUDA_CHECK_LAST();
            } else if (fwi_mode == 3) {
                copy_to_Eall_single<<<grid_copy, blockSize, 0, stream_comp>>>(Eall_ptr, t_saved, Ex, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                copy_to_Eall_single<<<grid_copy, blockSize, 0, stream_comp>>>(Eall_ptr + component_stride, t_saved, Ey, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                copy_to_Eall_single<<<grid_copy, blockSize, 0, stream_comp>>>(Eall_ptr + 2 * component_stride, t_saved, Ez, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                CUDA_CHECK_LAST();
            } else {
                copy_to_Eall_single<<<grid_copy, blockSize, 0, stream_comp>>>(Eall_ptr, t_saved, Ez, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                CUDA_CHECK_LAST();
            }
        }

        if (fdtd_order == 8) {
            h_fields_base_update_gpu<8><<<grid_fields, blockSize, 0, stream_comp>>>(uH0, uH1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            CUDA_CHECK_LAST();
            pml_h_fields_update_gpu<8><<<grid_fields, blockSize, 0, stream_comp>>>(
                Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, uH4,
                x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2);
            CUDA_CHECK_LAST();
            e_fields_base_update_gpu<8><<<grid_fields, blockSize, 0, stream_comp>>>(uE0, uE1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            CUDA_CHECK_LAST();
            pml_e_fields_update_gpu<8><<<grid_fields, blockSize, 0, stream_comp>>>(
                Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, uE4,
                x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2);
            CUDA_CHECK_LAST();
        } else if (fdtd_order == 4) {
            h_fields_base_update_gpu<4><<<grid_fields, blockSize, 0, stream_comp>>>(uH0, uH1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            CUDA_CHECK_LAST();
            pml_h_fields_update_gpu<4><<<grid_fields, blockSize, 0, stream_comp>>>(
                Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, uH4,
                x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2);
            CUDA_CHECK_LAST();
            e_fields_base_update_gpu<4><<<grid_fields, blockSize, 0, stream_comp>>>(uE0, uE1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            CUDA_CHECK_LAST();
            pml_e_fields_update_gpu<4><<<grid_fields, blockSize, 0, stream_comp>>>(
                Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, uE4,
                x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2);
            CUDA_CHECK_LAST();
        } else {
            h_fields_base_update_gpu<2><<<grid_fields, blockSize, 0, stream_comp>>>(uH0, uH1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            CUDA_CHECK_LAST();
            pml_h_fields_update_gpu<2><<<grid_fields, blockSize, 0, stream_comp>>>(
                Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, uH4,
                x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2);
            CUDA_CHECK_LAST();
            e_fields_base_update_gpu<2><<<grid_fields, blockSize, 0, stream_comp>>>(uE0, uE1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            CUDA_CHECK_LAST();
            pml_e_fields_update_gpu<2><<<grid_fields, blockSize, 0, stream_comp>>>(
                Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, uE4,
                x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2);
            CUDA_CHECK_LAST();
        }

        Update_hertzian_dipole<<<grid_src, blockSize, 0, stream_comp>>>(step, i, dx, sourcelocation, srcwaveforms, Ex, Ey, Ez, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nsrc, polarisation, nt);
        CUDA_CHECK_LAST();
        
        if (i % sampling_interval == 0) {
            int t_saved = i / sampling_interval;
            if (use_async) {
                int buf_idx = t_saved % 2;
                long long buf_base = (long long)buf_idx * e_components * snap_size;
                if (fwi_mode == 3) {
                    copy_to_Rall_single<<<grid_copy, blockSize, 0, stream_comp>>>(d_R_buf + buf_base, 0, Ex, d_E_buf + buf_base, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    copy_to_Rall_single<<<grid_copy, blockSize, 0, stream_comp>>>(d_R_buf + buf_base + snap_size, 0, Ey, d_E_buf + buf_base + snap_size, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    copy_to_Rall_single<<<grid_copy, blockSize, 0, stream_comp>>>(d_R_buf + buf_base + 2 * snap_size, 0, Ez, d_E_buf + buf_base + 2 * snap_size, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                } else {
                    copy_to_Rall_single<<<grid_copy, blockSize, 0, stream_comp>>>(d_R_buf + buf_base, 0, Ez, d_E_buf + buf_base, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                }
                CUDA_CHECK_LAST();
                CUDA_CHECK(cudaEventRecord(event_comp, stream_comp));
                CUDA_CHECK(cudaStreamWaitEvent(stream_trans, event_comp, 0));
                if (fwi_mode == 3) {
                    CUDA_CHECK(cudaMemcpyAsync(Eall_ptr + t_saved * snap_size, d_E_buf + buf_base, snap_size * sizeof(float), cudaMemcpyDeviceToHost, stream_trans));
                    CUDA_CHECK(cudaMemcpyAsync(Eall_ptr + component_stride + t_saved * snap_size, d_E_buf + buf_base + snap_size, snap_size * sizeof(float), cudaMemcpyDeviceToHost, stream_trans));
                    CUDA_CHECK(cudaMemcpyAsync(Eall_ptr + 2 * component_stride + t_saved * snap_size, d_E_buf + buf_base + 2 * snap_size, snap_size * sizeof(float), cudaMemcpyDeviceToHost, stream_trans));
                    CUDA_CHECK(cudaMemcpyAsync(Rall_ptr + t_saved * snap_size, d_R_buf + buf_base, snap_size * sizeof(float), cudaMemcpyDeviceToHost, stream_trans));
                    CUDA_CHECK(cudaMemcpyAsync(Rall_ptr + component_stride + t_saved * snap_size, d_R_buf + buf_base + snap_size, snap_size * sizeof(float), cudaMemcpyDeviceToHost, stream_trans));
                    CUDA_CHECK(cudaMemcpyAsync(Rall_ptr + 2 * component_stride + t_saved * snap_size, d_R_buf + buf_base + 2 * snap_size, snap_size * sizeof(float), cudaMemcpyDeviceToHost, stream_trans));
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(Eall_ptr + t_saved * snap_size, d_E_buf + buf_base, snap_size * sizeof(float), cudaMemcpyDeviceToHost, stream_trans));
                    CUDA_CHECK(cudaMemcpyAsync(Rall_ptr + t_saved * snap_size, d_R_buf + buf_base, snap_size * sizeof(float), cudaMemcpyDeviceToHost, stream_trans));
                }
            } else {
                if (fwi_mode == 3) {
                    copy_to_Rall_single<<<grid_copy, blockSize, 0, stream_comp>>>(Rall_ptr, t_saved, Ex, Eall_ptr, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    copy_to_Rall_single<<<grid_copy, blockSize, 0, stream_comp>>>(Rall_ptr + component_stride, t_saved, Ey, Eall_ptr + component_stride, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    copy_to_Rall_single<<<grid_copy, blockSize, 0, stream_comp>>>(Rall_ptr + 2 * component_stride, t_saved, Ez, Eall_ptr + 2 * component_stride, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                } else {
                    copy_to_Rall_single<<<grid_copy, blockSize, 0, stream_comp>>>(Rall_ptr, t_saved, Ez, Eall_ptr, uE0, uE4, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                }
                CUDA_CHECK_LAST();
            }
        }

        store_outputs<<<grid_rx, blockSize, 0, stream_comp>>>(step, nrx, i, receiverlocation, rxs, Ex, Ey, Ez, Hx, Hy, Hz, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nt);
        CUDA_CHECK_LAST();
    }

    if (use_async) {
        CUDA_CHECK(cudaStreamSynchronize(stream_comp));
        CUDA_CHECK(cudaStreamSynchronize(stream_trans));
        CUDA_CHECK(cudaFree(d_E_buf));
        CUDA_CHECK(cudaFree(d_R_buf));
        CUDA_CHECK(cudaEventDestroy(event_comp));
        CUDA_CHECK(cudaStreamDestroy(stream_comp));
        CUDA_CHECK(cudaStreamDestroy(stream_trans));
    }
}

/*
 * Run CUDA adjoint FDTD modeling and accumulate model gradients.
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
DEEPGPR_API void backward(const float* __restrict__ er, const float* __restrict__ se, const float* __restrict__ mr,
             const float* __restrict__ Eall_ptr, const float* __restrict__ Rall_ptr,
             float* __restrict__ Ex,  float* __restrict__ Ey, float* __restrict__ Ez,  
             float* __restrict__ Hx, float* __restrict__ Hy,  float* __restrict__ Hz,
             float* __restrict__ uE0, float* __restrict__ uE1, float* __restrict__ uE4,
             float* __restrict__ uH0, float* __restrict__ uH1, float* __restrict__ uH4,

            float* __restrict__ x0EPhi1,float* __restrict__ x0EPhi2, float* __restrict__ x0HPhi1,float* __restrict__ x0HPhi2,
            float* __restrict__ xmEPhi1,float* __restrict__ xmEPhi2, float* __restrict__ xmHPhi1,float* __restrict__ xmHPhi2,
            float* __restrict__ y0EPhi1,float* __restrict__ y0EPhi2, float* __restrict__ y0HPhi1,float* __restrict__ y0HPhi2,
            float* __restrict__ ymEPhi1,float* __restrict__ ymEPhi2, float* __restrict__ ymHPhi1,float* __restrict__ ymHPhi2,
            float* __restrict__ z0EPhi1,float* __restrict__ z0EPhi2, float* __restrict__ z0HPhi1,float* __restrict__ z0HPhi2,
            float* __restrict__ zmEPhi1,float* __restrict__ zmEPhi2, float* __restrict__ zmHPhi1,float* __restrict__ zmHPhi2,

            int pml0,int pml1,int pml2,int pml3,int pml4,int pml5,

            float* __restrict__ x0ER,float* __restrict__ xmER, float* __restrict__ y0ER,float* __restrict__ ymER,
            float* __restrict__ z0ER,float* __restrict__ zmER, float* __restrict__ x0HR,float* __restrict__ xmHR,
            float* __restrict__ y0HR,float* __restrict__ ymHR, float* __restrict__ z0HR,float* __restrict__ zmHR,

             float dt, int nt, int step, int nrx, float dx,
             int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
             int nsrc, const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms,
             int polarisation, 
             float*__restrict__ grad_er,float*__restrict__ grad_se, int errequiregrad, int serequiregrad,
             int sampling_interval, int fwi_mode)
{
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, Eall_ptr);
    int use_async = 0;
    if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
        use_async = 0;
    } else {
        cudaGetLastError(); 
        use_async = 1;
    }
    int fdtd_order = g_fdtd_order;
    int e_components = (fwi_mode == 3) ? 3 : 1;

    float* d_E_buf = nullptr;
    float* d_R_buf = nullptr;
    long long snap_size = (long long)step * (NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1);
    
    cudaStream_t stream_comp = 0, stream_trans = 0;
    cudaEvent_t event_trans;
    if (use_async) {
        CUDA_CHECK(cudaStreamCreate(&stream_comp));
        CUDA_CHECK(cudaStreamCreate(&stream_trans));
        CUDA_CHECK(cudaEventCreate(&event_trans));
        CUDA_CHECK(cudaMalloc(&d_E_buf, e_components * snap_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_R_buf, e_components * snap_size * sizeof(float)));
    }

    long long blockSize = 256;
    long long total_fields = (long long)NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    dim3 grid_material(CEIL_DIV(total_fields, blockSize));
    dim3 grid_fields(CEIL_DIV((long long)step * total_fields, blockSize));

    ucgetforward<<<grid_material, blockSize, 0, stream_comp>>>(er, se, mr, uE0, uE1, uE4, uH0, uH1, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);
    CUDA_CHECK_LAST();

    long long total_src = (long long)step * nsrc;
    long long src_blocks = (total_src + blockSize - 1) / blockSize;        
    dim3 grid_src(src_blocks);

    long long total_grad = (long long)(NX_FIELDS-1) * (NY_FIELDS-1) * (NZ_FIELDS-1);
    dim3 grid_grad(CEIL_DIV((long long)step * total_grad, blockSize));
  
    int nt_saved = (nt + sampling_interval - 1) / sampling_interval;
    long long component_stride = (long long)nt_saved * snap_size;

    for (int i = nt - 1; i >= 0; i--) {
        if (use_async && i % sampling_interval == 0) {
            int t_saved = i / sampling_interval;
            CUDA_CHECK(cudaStreamSynchronize(stream_trans));
            for (int c = 0; c < e_components; ++c) {
                long long comp_offset = (fwi_mode == 3) ? (long long)c * component_stride : 0;
                CUDA_CHECK(cudaMemcpyAsync(d_E_buf + (long long)c * snap_size, Eall_ptr + comp_offset + (long long)t_saved * snap_size, snap_size * sizeof(float), cudaMemcpyHostToDevice, stream_trans));
                CUDA_CHECK(cudaMemcpyAsync(d_R_buf + (long long)c * snap_size, Rall_ptr + comp_offset + (long long)t_saved * snap_size, snap_size * sizeof(float), cudaMemcpyHostToDevice, stream_trans));
            }
            CUDA_CHECK(cudaEventRecord(event_trans, stream_trans));
            CUDA_CHECK(cudaStreamWaitEvent(stream_comp, event_trans, 0));
        }
        
        add_adjoint_sources<<<grid_src, blockSize, 0, stream_comp>>>(
            step, i, sourcelocation, srcwaveforms, Ex, Ey, Ez,
            NX_FIELDS, NY_FIELDS, NZ_FIELDS, nsrc, polarisation, nt);
        CUDA_CHECK_LAST();

        if (i % sampling_interval == 0) {
            accumulate_gradients<<<grid_grad, blockSize, 0, stream_comp>>>(
                Ex, Ey, Ez, Eall_ptr, Rall_ptr, d_E_buf, d_R_buf, uE0, uE4, se,
                grad_er, grad_se, i, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt,
                errequiregrad, serequiregrad, sampling_interval, nt_saved, use_async, fwi_mode);
            CUDA_CHECK_LAST();
        }

        pml_e_fields_adjoint_gpu<<<grid_fields, blockSize, 0, stream_comp>>>(
            Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, uE4,
            x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2,
            z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2, fdtd_order);
        CUDA_CHECK_LAST();
        e_fields_base_adjoint_gpu<<<grid_fields, blockSize, 0, stream_comp>>>(
            uE0, uE1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, fdtd_order);
        CUDA_CHECK_LAST();
        pml_h_fields_adjoint_gpu<<<grid_fields, blockSize, 0, stream_comp>>>(
            Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, uH4,
            x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2,
            z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2, fdtd_order);
        CUDA_CHECK_LAST();
        h_fields_base_adjoint_gpu<<<grid_fields, blockSize, 0, stream_comp>>>(
            uH0, uH1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, fdtd_order);
        CUDA_CHECK_LAST();
    }

    if (use_async) {
        CUDA_CHECK(cudaStreamSynchronize(stream_comp));
        CUDA_CHECK(cudaStreamSynchronize(stream_trans));
        CUDA_CHECK(cudaFree(d_E_buf));
        CUDA_CHECK(cudaFree(d_R_buf));
        CUDA_CHECK(cudaEventDestroy(event_trans));
        CUDA_CHECK(cudaStreamDestroy(stream_comp));
        CUDA_CHECK(cudaStreamDestroy(stream_trans));
    }
}

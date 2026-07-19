#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cfloat>

#define DEEPGPR_BUILD
#include "deepgpr.h"

__constant__ float e0 = 8.8541878128e-12;
__constant__ float m0 = 1.25663706212e-06;

enum {
    WAVEFIELD_FLOAT32 = 0,
    WAVEFIELD_FLOAT16 = 1,
    WAVEFIELD_BFLOAT16 = 2
};

static size_t wavefield_element_size_host(int storage_type)
{
    return storage_type == WAVEFIELD_FLOAT32 ? sizeof(float) : sizeof(unsigned short);
}

static void* wavefield_offset_host(void* pointer, long long offset, int storage_type)
{
    return (void*)((unsigned char*)pointer + offset * (long long)wavefield_element_size_host(storage_type));
}

static const void* wavefield_const_offset_host(const void* pointer, long long offset, int storage_type)
{
    return (const void*)((const unsigned char*)pointer + offset * (long long)wavefield_element_size_host(storage_type));
}

__device__ __forceinline__ unsigned short float_to_half_bits_device(float value)
{
    unsigned int bits = __float_as_uint(value);
    unsigned int sign = (bits >> 16) & 0x8000u;
    unsigned int exponent = (bits >> 23) & 0xffu;
    unsigned int mantissa = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        return (unsigned short)(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
    }

    int half_exponent = (int)exponent - 127 + 15;
    if (half_exponent >= 31) return (unsigned short)(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return (unsigned short)sign;
        mantissa |= 0x800000u;
        int shift = 14 - half_exponent;
        unsigned int half_mantissa = mantissa >> shift;
        unsigned int remainder = mantissa & ((1u << shift) - 1u);
        unsigned int halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) {
            ++half_mantissa;
        }
        return (unsigned short)(sign | half_mantissa);
    }

    unsigned int half_mantissa = mantissa >> 13;
    unsigned int remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u))) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0u;
            ++half_exponent;
            if (half_exponent >= 31) return (unsigned short)(sign | 0x7c00u);
        }
    }
    return (unsigned short)(sign | ((unsigned int)half_exponent << 10) | half_mantissa);
}

__device__ __forceinline__ float half_bits_to_float_device(unsigned short half)
{
    unsigned int sign = ((unsigned int)half & 0x8000u) << 16;
    unsigned int exponent = ((unsigned int)half >> 10) & 0x1fu;
    unsigned int mantissa = (unsigned int)half & 0x3ffu;
    unsigned int bits;

    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            int normalized_exponent = -14;
            while ((mantissa & 0x400u) == 0u) {
                mantissa <<= 1;
                --normalized_exponent;
            }
            mantissa &= 0x3ffu;
            bits = sign | ((unsigned int)(normalized_exponent + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    return __uint_as_float(bits);
}

__device__ __forceinline__ unsigned short float_to_bfloat16_bits_device(float value)
{
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        return (unsigned short)((bits >> 16) | 0x0040u);
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short)(bits >> 16);
}

template<int STORAGE_TYPE>
__device__ __forceinline__ void store_wavefield_value_device(
    void* pointer, long long index, float value)
{
    if (STORAGE_TYPE == WAVEFIELD_FLOAT16) {
        ((unsigned short*)pointer)[index] = float_to_half_bits_device(value);
    } else if (STORAGE_TYPE == WAVEFIELD_BFLOAT16) {
        ((unsigned short*)pointer)[index] = float_to_bfloat16_bits_device(value);
    } else {
        ((float*)pointer)[index] = value;
    }
}

template<int STORAGE_TYPE>
__device__ __forceinline__ float load_wavefield_value_device(
    const void* pointer, long long index)
{
    if (STORAGE_TYPE == WAVEFIELD_FLOAT16) {
        return half_bits_to_float_device(((const unsigned short*)pointer)[index]);
    }
    if (STORAGE_TYPE == WAVEFIELD_BFLOAT16) {
        unsigned int bits = (unsigned int)((const unsigned short*)pointer)[index] << 16;
        return __uint_as_float(bits);
    }
    return ((const float*)pointer)[index];
}

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

/* Keep storage-format decisions on the host so FP32 kernels contain no dtype branch. */
#define LAUNCH_STORAGE_KERNEL(kernel, grid, block, stream, storage_type, ...) do { \
    if ((storage_type) == WAVEFIELD_FLOAT16) { \
        kernel<WAVEFIELD_FLOAT16><<<(grid), (block), 0, (stream)>>>(__VA_ARGS__); \
    } else if ((storage_type) == WAVEFIELD_BFLOAT16) { \
        kernel<WAVEFIELD_BFLOAT16><<<(grid), (block), 0, (stream)>>>(__VA_ARGS__); \
    } else { \
        kernel<WAVEFIELD_FLOAT32><<<(grid), (block), 0, (stream)>>>(__VA_ARGS__); \
    } \
} while (0)

/* Compile each stencil order independently to unroll adjoint curl loops. */
#define LAUNCH_ORDER_KERNEL(kernel, grid, block, stream, order, ...) do { \
    if ((order) == 8) { \
        kernel<8><<<(grid), (block), 0, (stream)>>>(__VA_ARGS__); \
    } else if ((order) == 4) { \
        kernel<4><<<(grid), (block), 0, (stream)>>>(__VA_ARGS__); \
    } else { \
        kernel<2><<<(grid), (block), 0, (stream)>>>(__VA_ARGS__); \
    } \
} while (0)

static int g_fdtd_order = 2;

DEEPGPR_API int deepgpr_abi_version(void)
{
    return DEEPGPR_ABI_VERSION;
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

template<int ORDER>
__device__ __forceinline__ void add_staggered_backward_adjoint(
    float* gradient, long long id, long long stride,
    long long coord, long long n, float weight)
{
    const int requested = FdtdStaticRadius<ORDER>::value;
    int radius = requested == 1 ? 1 : usable_backward_radius(coord, n, requested);
#pragma unroll
    for (int r = 1; r <= requested; ++r) {
        if (r <= radius) {
            float value = weight * fdtd_coeff(radius, r);
            atomicAdd(&gradient[id + (long long)(r - 1) * stride], value);
            atomicAdd(&gradient[id - (long long)r * stride], -value);
        }
    }
}

template<int ORDER>
__device__ __forceinline__ void add_staggered_forward_adjoint(
    float* gradient, long long id, long long stride,
    long long coord, long long n, float weight)
{
    const int requested = FdtdStaticRadius<ORDER>::value;
    int radius = requested == 1 ? 1 : usable_forward_radius(coord, n, requested);
#pragma unroll
    for (int r = 1; r <= requested; ++r) {
        if (r <= radius) {
            float value = weight * fdtd_coeff(radius, r);
            atomicAdd(&gradient[id + (long long)r * stride], value);
            atomicAdd(&gradient[id - (long long)(r - 1) * stride], -value);
        }
    }
}

template<int ORDER>
__device__ __forceinline__ void pml_backward_derivative_adjoint(
    float lambda_field, float* lambda_source,
    long long source_id, long long stride, long long coord, long long n,
    float inverse_spacing, float update_coeff, float sign,
    float ra_minus_one, float rb, float re, float rf, float* lambda_phi)
{
    float phi_new = *lambda_phi;
    float derivative_weight =
        (sign * update_coeff * ra_minus_one * lambda_field - rf * phi_new) * inverse_spacing;
    add_staggered_backward_adjoint<ORDER>(
        lambda_source, source_id, stride, coord, n, derivative_weight);
    *lambda_phi = sign * update_coeff * rb * lambda_field + re * phi_new;
}

template<int ORDER>
__device__ __forceinline__ void pml_forward_derivative_adjoint(
    float lambda_field, float* lambda_source,
    long long source_id, long long stride, long long coord, long long n,
    float inverse_spacing, float update_coeff, float sign,
    float ra_minus_one, float rb, float re, float rf, float* lambda_phi)
{
    float phi_new = *lambda_phi;
    float derivative_weight =
        (sign * update_coeff * ra_minus_one * lambda_field - rf * phi_new) * inverse_spacing;
    add_staggered_forward_adjoint<ORDER>(
        lambda_source, source_id, stride, coord, n, derivative_weight);
    *lambda_phi = sign * update_coeff * rb * lambda_field + re * phi_new;
}

/*
 * Build forward-update coefficients for electric and magnetic fields.
 *
 * Parameters:
 *   eps_r_pad: Padded relative permittivity array.
 *   sigma_pad: Padded electrical conductivity array.
 *   mu_r_pad: Padded relative permeability array.
 *   ce_hist, ce_curl, ce_rhs: Output electric-field update coefficient arrays.
 *   ch_hist, ch_curl, ch_rhs: Output magnetic-field update coefficient arrays.
 *   NX_FIELDS, NY_FIELDS, NZ_FIELDS: Padded field grid sizes.
 *   dt: Time step size.
 *   dx, dy, dz: Grid spacing along each axis.
 */
__global__ void build_update_coeffs_gpu(const float* __restrict__ eps_r_pad, const float* __restrict__ sigma_pad, const float* __restrict__ mu_r_pad,
    float* __restrict__ ce_hist, float* __restrict__ ce_curl, float* __restrict__ ce_rhs,
    float* __restrict__ ch_hist, float* __restrict__ ch_curl, float* __restrict__ ch_rhs,
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
        float HA = m0 * mu_r_pad[idx] / dt;
        ch_hist[idx] = 1.0f;
        ch_curl[idx] = (1.0f / dx) / HA;
        ch_rhs[idx] = 1.0f / HA;

        if (sigma_pad[idx] > 100.0f) {
            ce_hist[idx] = 0.0f; ce_curl[idx] = 0.0f; ce_rhs[idx] = 0.0f;
        } else {
            float e_term = e0 * eps_r_pad[idx] / dt;
            float s_term = 0.5f * sigma_pad[idx];
            float EA = e_term + s_term;
            float EB = e_term - s_term;
            ce_hist[idx] = EB / EA;
            ce_curl[idx] = (1.0f / dx) / EA;
            ce_rhs[idx] = 1.0f / EA;
        }
    }
}

/*
 * Store receiver samples for the requested electric-field component.
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
__global__ void sample_receivers_gpu(
    int step, int NRX, int iteration,
    const int* __restrict__ receiverlocation, float* __restrict__ rxs,
    const float* __restrict__ Ex, const float* __restrict__ Ey, const float* __restrict__ Ez,
    int NX, int NY, int NZ, int N_ITER, int receiver_component)
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

    const float* field = receiver_component == 0 ? Ex : (receiver_component == 1 ? Ey : Ez);
    rxs[((long long)s * N_ITER + iteration) * NRX + rx] = field[id4];
}


/*
 * Inject a Hertzian dipole source into one electric-field component.
 *
 * Parameters:
 *   step: Number of shots or simulations in the batch.
 *   iteration: Current time-step index.
 *   dx, dy, dz: Grid spacing along each axis.
 *   sourcelocation: Source coordinates with shape (step, nsrc, 3).
 *   srcwaveforms: Source waveform array.
 *   Ex, Ey, Ez: Electric field component arrays to update.
 *   ce_rhs: Electric source scaling coefficient array.
 *   NX, NY, NZ: Padded field grid sizes.
 *   nsrc: Number of sources per shot.
 *   polarisation: Source component, 0 for x, 1 for y, 2 for z.
 *   nt: Total number of time steps.
 */
__global__ void inject_sources_gpu(
    int step, int iteration, float dx, float dy, float dz,
    const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms,
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez, const float* __restrict__ ce_rhs,
    int NX, int NY, int NZ, int nsrc, int polarisation, int nt) 
{
    long long field_stride = (long long)NX * NY * NZ;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)step * nsrc;
    if (work >= total) return;

    int s = (int)(work / nsrc);
    long long src = work % nsrc;
    float waveform_value = srcwaveforms[src * nt + iteration];
    float dipole_length = polarisation == 0 ? dx : (polarisation == 1 ? dy : dz);
    float scale = waveform_value * dipole_length / (dx * dy * dz);

    long long i = sourcelocation[s * nsrc * 3 + src * 3 + 0];
    long long j = sourcelocation[s * nsrc * 3 + src * 3 + 1];
    long long k = sourcelocation[s * nsrc * 3 + src * 3 + 2];

    long long id3 = i * NY * NZ + j * NZ + k;
    long long id4 = (long long)s * field_stride + id3;

    if (polarisation == 0) Ex[id4] -= ce_rhs[id3] * scale;
    else if (polarisation == 1) Ey[id4] -= ce_rhs[id3] * scale;
    else if (polarisation == 2) Ez[id4] -= ce_rhs[id3] * scale;
}


/* Inject all sources, then sample the selected receiver component. */
__global__ void inject_sources_and_sample_gpu(
    int step, int iteration, float dx, float dy, float dz,
    const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms,
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    const float* __restrict__ ce_rhs,
    int NX, int NY, int NZ, int nsrc, int source_component, int nt,
    int nrx, const int* __restrict__ receiverlocation,
    float* __restrict__ receiver_data, int receiver_component)
{
    long long field_stride = (long long)NX * NY * NZ;
    long long total_sources = (long long)step * nsrc;
    for (long long work = threadIdx.x; work < total_sources; work += blockDim.x) {
        int s = (int)(work / nsrc);
        long long src = work % nsrc;
        long long i = sourcelocation[s * nsrc * 3 + src * 3 + 0];
        long long j = sourcelocation[s * nsrc * 3 + src * 3 + 1];
        long long k = sourcelocation[s * nsrc * 3 + src * 3 + 2];
        long long material_idx = i * NY * NZ + j * NZ + k;
        long long field_idx = (long long)s * field_stride + material_idx;
        float dipole_length = source_component == 0 ? dx : (source_component == 1 ? dy : dz);
        float scale = srcwaveforms[src * nt + iteration] * dipole_length / (dx * dy * dz);
        float value = ce_rhs[material_idx] * scale;

        if (source_component == 0) Ex[field_idx] -= value;
        else if (source_component == 1) Ey[field_idx] -= value;
        else Ez[field_idx] -= value;
    }

    __syncthreads();

    long long total_receivers = (long long)step * nrx;
    for (long long work = threadIdx.x; work < total_receivers; work += blockDim.x) {
        int s = (int)(work / nrx);
        long long rx = work % nrx;
        long long i = receiverlocation[s * nrx * 3 + rx * 3 + 0];
        long long j = receiverlocation[s * nrx * 3 + rx * 3 + 1];
        long long k = receiverlocation[s * nrx * 3 + rx * 3 + 2];
        long long field_idx = (long long)s * field_stride + i * NY * NZ + j * NZ + k;
        const float* field = receiver_component == 0 ? Ex : (receiver_component == 1 ? Ey : Ez);
        receiver_data[((long long)s * nt + iteration) * nrx + rx] = field[field_idx];
    }
}


template<int ORDER>
__global__ void update_e_gpu(
    const float* __restrict__ ce_hist, const float* __restrict__ ce_curl,
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    const float* __restrict__ Hx, const float* __restrict__ Hy, const float* __restrict__ Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
    float dx, float dy, float dz)
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

    float ue0 = ce_hist[idx];
    float ue1 = ce_curl[idx];
    float ue_y = dy == dx ? ue1 : ue1 * dx / dy;
    float ue_z = dz == dx ? ue1 : ue1 * dx / dz;

    bool do_ex = (((NY_FIELDS - 1) != 1 || (NZ_FIELDS - 1) != 1) && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));
    bool do_ey = (((NX_FIELDS - 1) != 1 || (NZ_FIELDS - 1) != 1) && i > 0 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));
    bool do_ez = (((NX_FIELDS - 1) != 1 || (NY_FIELDS - 1) != 1) && i > 0 && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));

    if (do_ex) {
        float dHz_dy = staggered_backward_diff_static<ORDER>(Hz, id4, NZ_FIELDS, j, NY_FIELDS);
        float dHy_dz = staggered_backward_diff_static<ORDER>(Hy, id4, 1, k, NZ_FIELDS);
        Ex[id4] = ue0 * Ex[id4] + ue_y * dHz_dy - ue_z * dHy_dz;
    }
    if (do_ey) {
        float dHx_dz = staggered_backward_diff_static<ORDER>(Hx, id4, 1, k, NZ_FIELDS);
        float dHz_dx = staggered_backward_diff_static<ORDER>(Hz, id4, ny_nz, i, NX_FIELDS);
        Ey[id4] = ue0 * Ey[id4] + ue_z * dHx_dz - ue1 * dHz_dx;
    }
    if (do_ez) {
        float dHy_dx = staggered_backward_diff_static<ORDER>(Hy, id4, ny_nz, i, NX_FIELDS);
        float dHx_dy = staggered_backward_diff_static<ORDER>(Hx, id4, NZ_FIELDS, j, NY_FIELDS);
        Ez[id4] = ue0 * Ez[id4] + ue1 * dHy_dx - ue_y * dHx_dy;
    }
}

/*
 * Apply electric CPML boundary corrections after the base electric-field update.
 */
template<int ORDER>
__global__ void cpml_e_gpu(
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
__global__ void update_h_gpu(
    const float* __restrict__ ch_hist, const float* __restrict__ ch_curl,
    const float* __restrict__ Ex, const float* __restrict__ Ey, const float* __restrict__ Ez,
    float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
    float dx, float dy, float dz)
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

    float uh0 = ch_hist[idx];
    float uh1 = ch_curl[idx];
    float uh_y = dy == dx ? uh1 : uh1 * dx / dy;
    float uh_z = dz == dx ? uh1 : uh1 * dx / dz;

    bool do_hx = ((NX_FIELDS - 1) != 1 && i > 0 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));
    bool do_hy = ((NY_FIELDS - 1) != 1 && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));
    bool do_hz = ((NZ_FIELDS - 1) != 1 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));

    if (do_hx) {
        float dEz_dy = staggered_forward_diff_static<ORDER>(Ez, id4, NZ_FIELDS, j, NY_FIELDS);
        float dEy_dz = staggered_forward_diff_static<ORDER>(Ey, id4, 1, k, NZ_FIELDS);
        Hx[id4] = uh0 * Hx[id4] - uh_y * dEz_dy + uh_z * dEy_dz;
    }
    if (do_hy) {
        float dEx_dz = staggered_forward_diff_static<ORDER>(Ex, id4, 1, k, NZ_FIELDS);
        float dEz_dx = staggered_forward_diff_static<ORDER>(Ez, id4, ny_nz, i, NX_FIELDS);
        Hy[id4] = uh0 * Hy[id4] - uh_z * dEx_dz + uh1 * dEz_dx;
    }
    if (do_hz) {
        float dEy_dx = staggered_forward_diff_static<ORDER>(Ey, id4, ny_nz, i, NX_FIELDS);
        float dEx_dy = staggered_forward_diff_static<ORDER>(Ex, id4, NZ_FIELDS, j, NY_FIELDS);
        Hz[id4] = uh0 * Hz[id4] - uh1 * dEy_dx + uh_y * dEx_dy;
    }
}

template<int ORDER>
__global__ void adjoint_e_gpu(
    const float* ce_hist, const float* ce_curl,
    float* lambda_ex, float* lambda_ey, float* lambda_ez, float* lambda_hx, float* lambda_hy, float* lambda_hz,
    int step, int NX, int NY, int NZ, float dx, float dy, float dz)
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
    float coeff = ce_curl[idx];
    float coeff_y = dy == dx ? coeff : coeff * dx / dy;
    float coeff_z = dz == dx ? coeff : coeff * dx / dz;

    if (do_ex) {
        float value = lambda_ex[work];
        add_staggered_backward_adjoint<ORDER>(lambda_hz, work, NZ, j, NY, coeff_y * value);
        add_staggered_backward_adjoint<ORDER>(lambda_hy, work, 1, k, NZ, -coeff_z * value);
        lambda_ex[work] = ce_hist[idx] * value;
    }
    if (do_ey) {
        float value = lambda_ey[work];
        add_staggered_backward_adjoint<ORDER>(lambda_hx, work, 1, k, NZ, coeff_z * value);
        add_staggered_backward_adjoint<ORDER>(lambda_hz, work, ny_nz, i, NX, -coeff * value);
        lambda_ey[work] = ce_hist[idx] * value;
    }
    if (do_ez) {
        float value = lambda_ez[work];
        add_staggered_backward_adjoint<ORDER>(lambda_hy, work, ny_nz, i, NX, coeff * value);
        add_staggered_backward_adjoint<ORDER>(lambda_hx, work, NZ, j, NY, -coeff_y * value);
        lambda_ez[work] = ce_hist[idx] * value;
    }
}

template<int ORDER>
__global__ void adjoint_h_gpu(
    const float* ch_hist, const float* ch_curl,
    float* lambda_ex, float* lambda_ey, float* lambda_ez, float* lambda_hx, float* lambda_hy, float* lambda_hz,
    int step, int NX, int NY, int NZ, float dx, float dy, float dz)
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
    float coeff = ch_curl[idx];
    float coeff_y = dy == dx ? coeff : coeff * dx / dy;
    float coeff_z = dz == dx ? coeff : coeff * dx / dz;

    if (do_hx) {
        float value = lambda_hx[work];
        add_staggered_forward_adjoint<ORDER>(lambda_ez, work, NZ, j, NY, -coeff_y * value);
        add_staggered_forward_adjoint<ORDER>(lambda_ey, work, 1, k, NZ, coeff_z * value);
        lambda_hx[work] = ch_hist[idx] * value;
    }
    if (do_hy) {
        float value = lambda_hy[work];
        add_staggered_forward_adjoint<ORDER>(lambda_ex, work, 1, k, NZ, -coeff_z * value);
        add_staggered_forward_adjoint<ORDER>(lambda_ez, work, ny_nz, i, NX, coeff * value);
        lambda_hy[work] = ch_hist[idx] * value;
    }
    if (do_hz) {
        float value = lambda_hz[work];
        add_staggered_forward_adjoint<ORDER>(lambda_ey, work, ny_nz, i, NX, -coeff * value);
        add_staggered_forward_adjoint<ORDER>(lambda_ex, work, NZ, j, NY, coeff_y * value);
        lambda_hz[work] = ch_hist[idx] * value;
    }
}

/*
 * Apply magnetic CPML boundary corrections after the base magnetic-field update.
 */
template<int ORDER>
__global__ void cpml_h_gpu(
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


template<int ORDER>
__global__ void adjoint_cpml_e_gpu(
    float* lambda_ex, float* lambda_ey, float* lambda_ez, float* lambda_hx, float* lambda_hy, float* lambda_hz,
    float dx, float dy, float dz, int step, int NX, int NY, int NZ,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* x0R, const float* xmR, const float* y0R, const float* ymR,
    const float* z0R, const float* zmR, const float* update,
    float* x0P1, float* x0P2, float* xmP1, float* xmP2,
    float* y0P1, float* y0P2, float* ymP1, float* ymP2,
    float* z0P1, float* z0P2, float* zmP1, float* zmP2)
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
        pml_backward_derivative_adjoint<ORDER>( \
            (field)[work], (source), work, (stride), (coord), (n), \
            1.0f / (spacing), upd, (sign), (R)[q] - 1.0f, (R)[(p) + (q)], \
            (R)[2 * (p) + (q)], (R)[3 * (p) + (q)], &(P)[p_idx]); \
    } while (0)

    if (pml0 > 0 && i > 0 && i <= pml0) {
        long long q = pml0 - i;
        if (j < NY - 1) {
            long long p_idx = ((long long)s * (pml0 + 1) * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
            APPLY_E_PML_GPU(x0R, x0P1, pml0, q, ny_nz, i, NX, dx, lambda_ey, lambda_hz, -1.0f);
        }
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * (pml0 + 1) * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
            APPLY_E_PML_GPU(x0R, x0P2, pml0, q, ny_nz, i, NX, dx, lambda_ez, lambda_hy, 1.0f);
        }
    }
    if (pml1 > 0 && i >= NX - 1 - pml1 && i < NX - 1) {
        long long q = i - (NX - 1 - pml1);
        if (j < NY - 1 && i > 0) {
            long long p_idx = ((long long)s * (pml1 + 1) * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
            APPLY_E_PML_GPU(xmR, xmP1, pml1, q, ny_nz, i, NX, dx, lambda_ey, lambda_hz, -1.0f);
        }
        if (k < NZ - 1 && i > 0) {
            long long p_idx = ((long long)s * (pml1 + 1) * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
            APPLY_E_PML_GPU(xmR, xmP2, pml1, q, ny_nz, i, NX, dx, lambda_ez, lambda_hy, 1.0f);
        }
    }
    if (pml2 > 0 && j > 0 && j <= pml2) {
        long long q = pml2 - j;
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * (pml2 + 1) * NZ) + i * (pml2 + 1) * NZ + q * NZ + k;
            APPLY_E_PML_GPU(y0R, y0P1, pml2, q, NZ, j, NY, dy, lambda_ex, lambda_hz, 1.0f);
        }
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * NX * (pml2 + 1) * (NZ - 1)) + i * (pml2 + 1) * (NZ - 1) + q * (NZ - 1) + k;
            APPLY_E_PML_GPU(y0R, y0P2, pml2, q, NZ, j, NY, dy, lambda_ez, lambda_hx, -1.0f);
        }
    }
    if (pml3 > 0 && j >= NY - 1 - pml3 && j < NY - 1) {
        long long q = j - (NY - 1 - pml3);
        if (i < NX - 1 && j > 0) {
            long long p_idx = ((long long)s * (NX - 1) * (pml3 + 1) * NZ) + i * (pml3 + 1) * NZ + q * NZ + k;
            APPLY_E_PML_GPU(ymR, ymP1, pml3, q, NZ, j, NY, dy, lambda_ex, lambda_hz, 1.0f);
        }
        if (k < NZ - 1 && j > 0) {
            long long p_idx = ((long long)s * NX * (pml3 + 1) * (NZ - 1)) + i * (pml3 + 1) * (NZ - 1) + q * (NZ - 1) + k;
            APPLY_E_PML_GPU(ymR, ymP2, pml3, q, NZ, j, NY, dy, lambda_ez, lambda_hx, -1.0f);
        }
    }
    if (pml4 > 0 && k > 0 && k <= pml4) {
        long long q = pml4 - k;
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * NY * (pml4 + 1)) + i * NY * (pml4 + 1) + j * (pml4 + 1) + q;
            APPLY_E_PML_GPU(z0R, z0P1, pml4, q, 1, k, NZ, dz, lambda_ex, lambda_hy, -1.0f);
        }
        if (j < NY - 1) {
            long long p_idx = ((long long)s * NX * (NY - 1) * (pml4 + 1)) + i * (NY - 1) * (pml4 + 1) + j * (pml4 + 1) + q;
            APPLY_E_PML_GPU(z0R, z0P2, pml4, q, 1, k, NZ, dz, lambda_ey, lambda_hx, 1.0f);
        }
    }
    if (pml5 > 0 && k >= NZ - 1 - pml5 && k < NZ - 1) {
        long long q = k - (NZ - 1 - pml5);
        if (i < NX - 1 && k > 0) {
            long long p_idx = ((long long)s * (NX - 1) * NY * (pml5 + 1)) + i * NY * (pml5 + 1) + j * (pml5 + 1) + q;
            APPLY_E_PML_GPU(zmR, zmP1, pml5, q, 1, k, NZ, dz, lambda_ex, lambda_hy, -1.0f);
        }
        if (j < NY - 1 && k > 0) {
            long long p_idx = ((long long)s * NX * (NY - 1) * (pml5 + 1)) + i * (NY - 1) * (pml5 + 1) + j * (pml5 + 1) + q;
            APPLY_E_PML_GPU(zmR, zmP2, pml5, q, 1, k, NZ, dz, lambda_ey, lambda_hx, 1.0f);
        }
    }
#undef APPLY_E_PML_GPU
}

template<int ORDER>
__global__ void adjoint_cpml_h_gpu(
    float* lambda_ex, float* lambda_ey, float* lambda_ez, float* lambda_hx, float* lambda_hy, float* lambda_hz,
    float dx, float dy, float dz, int step, int NX, int NY, int NZ,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* x0R, const float* xmR, const float* y0R, const float* ymR,
    const float* z0R, const float* zmR, const float* update,
    float* x0P1, float* x0P2, float* xmP1, float* xmP2,
    float* y0P1, float* y0P2, float* ymP1, float* ymP2,
    float* z0P1, float* z0P2, float* zmP1, float* zmP2)
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
        pml_forward_derivative_adjoint<ORDER>( \
            (field)[work], (source), work, (stride), (coord), (n), \
            1.0f / (spacing), upd, (sign), (R)[q] - 1.0f, (R)[(p) + (q)], \
            (R)[2 * (p) + (q)], (R)[3 * (p) + (q)], &(P)[p_idx]); \
    } while (0)

    if (pml0 > 0 && i < pml0) {
        long long q = pml0 - 1 - i;
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * pml0 * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
            APPLY_H_PML_GPU(x0R, x0P1, pml0, q, ny_nz, i, NX, dx, lambda_hy, lambda_ez, 1.0f);
        }
        if (j < NY - 1) {
            long long p_idx = ((long long)s * pml0 * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
            APPLY_H_PML_GPU(x0R, x0P2, pml0, q, ny_nz, i, NX, dx, lambda_hz, lambda_ey, -1.0f);
        }
    }
    if (pml1 > 0 && i >= NX - 1 - pml1 && i < NX - 1) {
        long long q = i - (NX - 1 - pml1);
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * pml1 * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
            APPLY_H_PML_GPU(xmR, xmP1, pml1, q, ny_nz, i, NX, dx, lambda_hy, lambda_ez, 1.0f);
        }
        if (j < NY - 1) {
            long long p_idx = ((long long)s * pml1 * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
            APPLY_H_PML_GPU(xmR, xmP2, pml1, q, ny_nz, i, NX, dx, lambda_hz, lambda_ey, -1.0f);
        }
    }
    if (pml2 > 0 && j < pml2) {
        long long q = pml2 - 1 - j;
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * NX * pml2 * (NZ - 1)) + i * pml2 * (NZ - 1) + q * (NZ - 1) + k;
            APPLY_H_PML_GPU(y0R, y0P1, pml2, q, NZ, j, NY, dy, lambda_hx, lambda_ez, -1.0f);
        }
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * pml2 * NZ) + i * pml2 * NZ + q * NZ + k;
            APPLY_H_PML_GPU(y0R, y0P2, pml2, q, NZ, j, NY, dy, lambda_hz, lambda_ex, 1.0f);
        }
    }
    if (pml3 > 0 && j >= NY - 1 - pml3 && j < NY - 1) {
        long long q = j - (NY - 1 - pml3);
        if (k < NZ - 1) {
            long long p_idx = ((long long)s * NX * pml3 * (NZ - 1)) + i * pml3 * (NZ - 1) + q * (NZ - 1) + k;
            APPLY_H_PML_GPU(ymR, ymP1, pml3, q, NZ, j, NY, dy, lambda_hx, lambda_ez, -1.0f);
        }
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * pml3 * NZ) + i * pml3 * NZ + q * NZ + k;
            APPLY_H_PML_GPU(ymR, ymP2, pml3, q, NZ, j, NY, dy, lambda_hz, lambda_ex, 1.0f);
        }
    }
    if (pml4 > 0 && k < pml4) {
        long long q = pml4 - 1 - k;
        if (j < NY - 1) {
            long long p_idx = ((long long)s * NX * (NY - 1) * pml4) + i * (NY - 1) * pml4 + j * pml4 + q;
            APPLY_H_PML_GPU(z0R, z0P1, pml4, q, 1, k, NZ, dz, lambda_hx, lambda_ey, 1.0f);
        }
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * NY * pml4) + i * NY * pml4 + j * pml4 + q;
            APPLY_H_PML_GPU(z0R, z0P2, pml4, q, 1, k, NZ, dz, lambda_hy, lambda_ex, -1.0f);
        }
    }
    if (pml5 > 0 && k >= NZ - 1 - pml5 && k < NZ - 1) {
        long long q = k - (NZ - 1 - pml5);
        if (j < NY - 1) {
            long long p_idx = ((long long)s * NX * (NY - 1) * pml5) + i * (NY - 1) * pml5 + j * pml5 + q;
            APPLY_H_PML_GPU(zmR, zmP1, pml5, q, 1, k, NZ, dz, lambda_hx, lambda_ey, 1.0f);
        }
        if (i < NX - 1) {
            long long p_idx = ((long long)s * (NX - 1) * NY * pml5) + i * NY * pml5 + j * pml5 + q;
            APPLY_H_PML_GPU(zmR, zmP2, pml5, q, 1, k, NZ, dz, lambda_hy, lambda_ex, -1.0f);
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
 *   sourcelocation: Adjoint source coordinates with shape (step, nsr, 3).
 *   srcwaveforms: Adjoint source waveform array.
 *   lambda_ex, lambda_ey, lambda_ez: Electric field component arrays to update.
 *   ce_rhs: Electric source scaling coefficient array.
 *   NX, NY, NZ: Padded field grid sizes.
 *   nsr: Number of adjoint sources per shot.
 *   polarisation: Source component, 0 for x, 1 for y, 2 for z.
 *   iterations: Total number of time steps.
 */
__global__ void adjoint_receivers_gpu(
    int step, int iteration,
    const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms,
    float* lambda_ex, float* lambda_ey, float* lambda_ez,
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

    if (polarisation == 0) lambda_ex[id4] += waveform_value;
    else if (polarisation == 1) lambda_ey[id4] += waveform_value;
    else if (polarisation == 2) lambda_ez[id4] += waveform_value;
}


/* Accumulate the transpose of the forward source injection into its waveform. */
__global__ void adjoint_source_injection_gpu(
    int step, int iteration, float dx, float dy, float dz,
    const int* __restrict__ source_location,
    const float* __restrict__ lambda_ex, const float* __restrict__ lambda_ey,
    const float* __restrict__ lambda_ez, const float* __restrict__ ce_rhs,
    int NX, int NY, int NZ, int nsrc, int source_component, int nt,
    float* __restrict__ grad_source)
{
    long long field_stride = (long long)NX * NY * NZ;
    long long work = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)step * nsrc;
    if (work >= total) return;

    int s = (int)(work / nsrc);
    long long src = work % nsrc;
    long long i = source_location[s * nsrc * 3 + src * 3 + 0];
    long long j = source_location[s * nsrc * 3 + src * 3 + 1];
    long long k = source_location[s * nsrc * 3 + src * 3 + 2];
    long long material_idx = i * NY * NZ + j * NZ + k;
    long long field_idx = (long long)s * field_stride + material_idx;
    float lambda_e = source_component == 0 ? lambda_ex[field_idx]
        : (source_component == 1 ? lambda_ey[field_idx] : lambda_ez[field_idx]);
    float dipole_length = source_component == 0 ? dx : (source_component == 1 ? dy : dz);
    float value = -ce_rhs[material_idx] * dipole_length / (dx * dy * dz) * lambda_e;
    atomicAdd(&grad_source[src * nt + iteration], value);
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
template<int STORAGE_TYPE>
__global__ void save_e_snapshot_gpu(
    void* __restrict__ dst_ptr, int t_idx, const float* __restrict__ E,
    float* __restrict__ exact_Eold,
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
    float value = E[src_idx];
    store_wavefield_value_device<STORAGE_TYPE>(dst_ptr, dst_idx, value);
    if (STORAGE_TYPE != WAVEFIELD_FLOAT32 && exact_Eold != nullptr) {
        exact_Eold[work] = value;
    }
}


/* Save R from the executed update E^(n+1) = ca E^n + cb R. */
template<int STORAGE_TYPE>
__global__ void save_rhs_snapshot_gpu(
    void* __restrict__ dst_ptr, int t_idx,
    const float* __restrict__ E, const void* __restrict__ Eold_ptr,
    const float* __restrict__ exact_Eold,
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

    float e_old = STORAGE_TYPE == WAVEFIELD_FLOAT32
        ? load_wavefield_value_device<STORAGE_TYPE>(Eold_ptr, saved_idx)
        : exact_Eold[work];
    float rhs = cb_value != 0.0f
        ? (E[field_idx] - ca[material_idx] * e_old) / cb_value
        : 0.0f;
    store_wavefield_value_device<STORAGE_TYPE>(dst_ptr, saved_idx, rhs);
}


/*
 * Accumulate model gradients from saved forward fields and adjoint fields.
 *
 * Parameters:
 *   lambda_ex, lambda_ey, lambda_ez: Adjoint electric field component arrays.
 *   E_saved: Saved pre-update electric field E^n.
 *   R_saved: Saved effective right-hand side R^n.
 *   d_E_buf, d_R_buf: Device staging buffers for host-offloaded snapshots.
 *   grad_eps_r: Output relative permittivity gradient array.
 *   grad_sigma: Output conductivity gradient array.
 *   i: Current reverse time-step index.
 *   step: Number of shots or simulations in the batch.
 *   NX, NY, NZ: Padded field grid sizes.
 *   pml0..pml5: CPML thicknesses; gradients are excluded from these cells.
 *   dt: Time step size.
 *   eps_r_requires_grad: Whether to accumulate grad_eps_r.
 *   sigma_requires_grad: Whether to accumulate grad_sigma.
 *   S: Forward wavefield sampling interval.
 *   nt_saved: Number of saved forward snapshots.
 *   use_async_offload: Whether E_saved is read through d_E_buf.
 *   fwi_mode: Gradient mode; 2 uses lambda_ez only, 3 uses lambda_ex, lambda_ey, and lambda_ez.
 */
template<int STORAGE_TYPE>
__global__ void accumulate_material_gradients_gpu(
    const float* __restrict__ lambda_ex, const float* __restrict__ lambda_ey, const float* __restrict__ lambda_ez,
    const void* __restrict__ E_saved, const void* __restrict__ R_saved,
    const void* __restrict__ d_E_buf, const void* __restrict__ d_R_buf,
    const float* __restrict__ ca, const float* __restrict__ cb,
    const float* __restrict__ sigma_pad,
    float* __restrict__ grad_eps_r, float* __restrict__ grad_sigma,
    int i, int step, int NX, int NY, int NZ,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    float dt,int eps_r_requires_grad,int sigma_requires_grad,
    int S, int sample_weight, int nt_saved, int use_async_offload,
    int fwi_mode
) {
    long long sx = (NX - 1), sy = (NY - 1), sz = (NZ - 1);
    long long total_cells = sx * sy * sz;
    long long snap_stride = (long long)step * total_cells;
    long long component_stride = (long long)nt_saved * snap_stride;
    int components = (fwi_mode == 3) ? 3 : 1;
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_cells) return;

    long long ix = idx / (sy * sz);
    long long rem = idx % (sy * sz);
    long long iy = rem / sz;
    long long iz = rem % sz;

    /* CPML is a numerical boundary, not part of the invertible model. */
    if ((pml0 > 0 && ix <= pml0) ||
        (pml1 > 0 && ix >= sx - pml1) ||
        (pml2 > 0 && iy <= pml2) ||
        (pml3 > 0 && iy >= sy - pml3) ||
        (pml4 > 0 && iz <= pml4) ||
        (pml5 > 0 && iz >= sz - pml5)) {
        return;
    }

    long long material_idx = ix * NY * NZ + iy * NZ + iz;
    float local_grader = 0.0f;
    float local_gradse = 0.0f;

    long long e_stride = (long long)NX * NY * NZ;
    float ca_value = ca[material_idx];
    float cb_value = cb[material_idx];
    bool active_material = sigma_pad[material_idx] <= 100.0f;

    for (int s = 0; s < step; ++s) {
        long long idx_E = (long long)s * e_stride + material_idx;
        long long base_idx = (long long)s * total_cells + idx;
        float adjoint_values[3];
        adjoint_values[0] = lambda_ex[idx_E];
        adjoint_values[1] = lambda_ey[idx_E];
        adjoint_values[2] = lambda_ez[idx_E];

        for (int c = 0; c < components; ++c) {
            long long comp_offset = (fwi_mode == 3) ? (long long)c * component_stride : 0;
            float e_old, rhs;

            if (use_async_offload) {
                long long device_idx = (long long)c * snap_stride + base_idx;
                e_old = load_wavefield_value_device<STORAGE_TYPE>(d_E_buf, device_idx);
                rhs = load_wavefield_value_device<STORAGE_TYPE>(d_R_buf, device_idx);
            } else {
                long long saved_idx = comp_offset + (long long)(i / S) * snap_stride + base_idx;
                e_old = load_wavefield_value_device<STORAGE_TYPE>(E_saved, saved_idx);
                rhs = load_wavefield_value_device<STORAGE_TYPE>(R_saved, saved_idx);
            }

            float adjoint_val = adjoint_values[(fwi_mode == 3) ? c : 2];
            float grad_ca = adjoint_val * e_old * (float)sample_weight;
            float grad_cb = adjoint_val * rhs * (float)sample_weight;

            if (active_material) {
                if (eps_r_requires_grad == 1) {
                    float dca_der = e0 * (1.0f - ca_value) * cb_value / dt;
                    float dcb_der = -e0 * cb_value * cb_value / dt;
                    local_grader += grad_ca * dca_der + grad_cb * dcb_der;
                }
                if (sigma_requires_grad == 1) {
                    float dca_dse = -0.5f * (1.0f + ca_value) * cb_value;
                    float dcb_dse = -0.5f * cb_value * cb_value;
                    local_gradse += grad_ca * dca_dse + grad_cb * dcb_dse;
                }
            }
        }
    }

    if (eps_r_requires_grad == 1) grad_eps_r[idx] += local_grader;
    if (sigma_requires_grad == 1) grad_sigma[idx] += local_gradse;
}


/*
 * Run CUDA forward FDTD modeling.
 *
 * Parameters:
 *   eps_r_pad, sigma_pad, mu_r_pad: Padded material property arrays.
 *   E_saved: Saved pre-update electric field history.
 *   R_saved: Saved effective electric right-hand sides.
 *   Ex, Ey, Ez: Electric field component arrays.
 *   Hx, Hy, Hz: Magnetic field component arrays.
 *   ce_hist, ce_curl, ce_rhs: Electric-field update coefficient arrays.
 *   ch_hist, ch_curl, ch_rhs: Magnetic-field update coefficient arrays.
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
 *   dx, dy, dz: Grid spacing along each axis.
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
DEEPGPR_API void forward(const float* __restrict__ eps_r_pad, const float* __restrict__ sigma_pad, const float* __restrict__ mu_r_pad,
             void* __restrict__ E_saved, void* __restrict__ R_saved,
             float* __restrict__ Ex,  float* __restrict__ Ey, float* __restrict__ Ez,  
             float* __restrict__ Hx, float* __restrict__ Hy,  float* __restrict__ Hz,
             float* __restrict__ ce_hist, float* __restrict__ ce_curl, float* __restrict__ ce_rhs,
             float* __restrict__ ch_hist, float* __restrict__ ch_curl, float* __restrict__ ch_rhs,

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

             float dt, int nt, int step, int nrx, float dx, float dy, float dz,
             const int* __restrict__ receiverlocation, float* __restrict__ rxs, int receiver_component,

             int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS, int nsrc, 
             const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms, int polarisation,
             int sampling_interval, int fwi_mode, int storage_type, int save_model_history,
             int use_async_offload)
{
    int use_async = use_async_offload != 0;
    int fdtd_order = g_fdtd_order;
    int e_components = (fwi_mode == 3) ? 3 : 1;
    int has_cpml = pml0 || pml1 || pml2 || pml3 || pml4 || pml5;
    int nt_saved = (nt + sampling_interval - 1) / sampling_interval;

    unsigned char* d_E_buf = nullptr;
    unsigned char* d_R_buf = nullptr;
    float* d_exact_Eold = nullptr;
    long long snap_size = (long long)step * (NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1);
    long long component_stride = (long long)nt_saved * snap_size;
    size_t storage_size = wavefield_element_size_host(storage_type);
    
    cudaStream_t stream_comp = 0, stream_trans = 0;
    cudaEvent_t event_input, event_comp, event_transfer[2];
    int transfer_buffer_in_use[2] = {0, 0};
    if (use_async) {
        CUDA_CHECK(cudaStreamCreate(&stream_comp));
        CUDA_CHECK(cudaStreamCreate(&stream_trans));
        CUDA_CHECK(cudaEventCreate(&event_input));
        CUDA_CHECK(cudaEventCreate(&event_comp));
        CUDA_CHECK(cudaEventCreate(&event_transfer[0]));
        CUDA_CHECK(cudaEventCreate(&event_transfer[1]));
        CUDA_CHECK(cudaMalloc(&d_E_buf, 2 * e_components * snap_size * storage_size));
        if (save_model_history) {
            CUDA_CHECK(cudaMalloc(&d_R_buf, 2 * e_components * snap_size * storage_size));
        }
        CUDA_CHECK(cudaEventRecord(event_input, 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream_comp, event_input, 0));
    }
    if (save_model_history && storage_type != WAVEFIELD_FLOAT32) {
        CUDA_CHECK(cudaMalloc(&d_exact_Eold, e_components * snap_size * sizeof(float)));
    }

    long long blockSize = 256;
    long long total_fields = (long long)NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    dim3 grid_material(CEIL_DIV(total_fields, blockSize));
    dim3 grid_fields(CEIL_DIV((long long)step * total_fields, blockSize));

    build_update_coeffs_gpu<<<grid_material, blockSize, 0, stream_comp>>>(eps_r_pad, sigma_pad, mu_r_pad, ce_hist, ce_curl, ce_rhs, ch_hist, ch_curl, ch_rhs, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);
    CUDA_CHECK_LAST();
  
    long long total_copy = (long long)(NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1); 
    dim3 grid_copy(CEIL_DIV((long long)step * total_copy, blockSize));

    for (int i = 0; i < nt; i++) {
        if (i % sampling_interval == 0) {
            int t_saved = i / sampling_interval;
            if (use_async) {
                int buf_idx = t_saved % 2;
                long long buf_base = (long long)buf_idx * e_components * snap_size;
                unsigned char* buffer = d_E_buf + buf_base * storage_size;
                if (transfer_buffer_in_use[buf_idx]) {
                    CUDA_CHECK(cudaStreamWaitEvent(stream_comp, event_transfer[buf_idx], 0));
                }
                if (fwi_mode == 3) {
                    LAUNCH_STORAGE_KERNEL(save_e_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        buffer, 0, Ex, d_exact_Eold, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    LAUNCH_STORAGE_KERNEL(save_e_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        buffer + snap_size * storage_size, 0, Ey,
                        d_exact_Eold != nullptr ? d_exact_Eold + snap_size : nullptr, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    LAUNCH_STORAGE_KERNEL(save_e_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        buffer + 2 * snap_size * storage_size, 0, Ez,
                        d_exact_Eold != nullptr ? d_exact_Eold + 2 * snap_size : nullptr, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                } else {
                    LAUNCH_STORAGE_KERNEL(save_e_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        buffer, 0, Ez, d_exact_Eold, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                }
                CUDA_CHECK_LAST();
            } else if (fwi_mode == 3) {
                LAUNCH_STORAGE_KERNEL(save_e_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                    E_saved, t_saved, Ex, d_exact_Eold, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                LAUNCH_STORAGE_KERNEL(save_e_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                    wavefield_offset_host(E_saved, component_stride, storage_type), t_saved, Ey,
                    d_exact_Eold != nullptr ? d_exact_Eold + snap_size : nullptr, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                LAUNCH_STORAGE_KERNEL(save_e_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                    wavefield_offset_host(E_saved, 2 * component_stride, storage_type), t_saved, Ez,
                    d_exact_Eold != nullptr ? d_exact_Eold + 2 * snap_size : nullptr, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                CUDA_CHECK_LAST();
            } else {
                LAUNCH_STORAGE_KERNEL(save_e_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                    E_saved, t_saved, Ez, d_exact_Eold, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                CUDA_CHECK_LAST();
            }
        }

        if (fdtd_order == 8) {
            update_h_gpu<8><<<grid_fields, blockSize, 0, stream_comp>>>(ch_hist, ch_curl, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dx, dy, dz);
            CUDA_CHECK_LAST();
            if (has_cpml) {
                cpml_h_gpu<8><<<grid_fields, blockSize, 0, stream_comp>>>(
                    Ex, Ey, Ez, Hx, Hy, Hz, dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                    pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, ch_rhs,
                    x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2);
                CUDA_CHECK_LAST();
            }
            update_e_gpu<8><<<grid_fields, blockSize, 0, stream_comp>>>(ce_hist, ce_curl, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dx, dy, dz);
            CUDA_CHECK_LAST();
            if (has_cpml) {
                cpml_e_gpu<8><<<grid_fields, blockSize, 0, stream_comp>>>(
                    Ex, Ey, Ez, Hx, Hy, Hz, dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                    pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, ce_rhs,
                    x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2);
                CUDA_CHECK_LAST();
            }
        } else if (fdtd_order == 4) {
            update_h_gpu<4><<<grid_fields, blockSize, 0, stream_comp>>>(ch_hist, ch_curl, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dx, dy, dz);
            CUDA_CHECK_LAST();
            if (has_cpml) {
                cpml_h_gpu<4><<<grid_fields, blockSize, 0, stream_comp>>>(
                    Ex, Ey, Ez, Hx, Hy, Hz, dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                    pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, ch_rhs,
                    x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2);
                CUDA_CHECK_LAST();
            }
            update_e_gpu<4><<<grid_fields, blockSize, 0, stream_comp>>>(ce_hist, ce_curl, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dx, dy, dz);
            CUDA_CHECK_LAST();
            if (has_cpml) {
                cpml_e_gpu<4><<<grid_fields, blockSize, 0, stream_comp>>>(
                    Ex, Ey, Ez, Hx, Hy, Hz, dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                    pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, ce_rhs,
                    x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2);
                CUDA_CHECK_LAST();
            }
        } else {
            update_h_gpu<2><<<grid_fields, blockSize, 0, stream_comp>>>(ch_hist, ch_curl, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dx, dy, dz);
            CUDA_CHECK_LAST();
            if (has_cpml) {
                cpml_h_gpu<2><<<grid_fields, blockSize, 0, stream_comp>>>(
                    Ex, Ey, Ez, Hx, Hy, Hz, dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                    pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, ch_rhs,
                    x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2);
                CUDA_CHECK_LAST();
            }
            update_e_gpu<2><<<grid_fields, blockSize, 0, stream_comp>>>(ce_hist, ce_curl, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dx, dy, dz);
            CUDA_CHECK_LAST();
            if (has_cpml) {
                cpml_e_gpu<2><<<grid_fields, blockSize, 0, stream_comp>>>(
                    Ex, Ey, Ez, Hx, Hy, Hz, dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                    pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, ce_rhs,
                    x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2);
                CUDA_CHECK_LAST();
            }
        }

        inject_sources_and_sample_gpu<<<1, blockSize, 0, stream_comp>>>(
            step, i, dx, dy, dz, sourcelocation, srcwaveforms,
            Ex, Ey, Ez, ce_rhs, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            nsrc, polarisation, nt, nrx, receiverlocation, rxs, receiver_component);
        CUDA_CHECK_LAST();

        if (i % sampling_interval == 0) {
            int t_saved = i / sampling_interval;
            if (use_async) {
                int buf_idx = t_saved % 2;
                long long buf_base = (long long)buf_idx * e_components * snap_size;
                unsigned char* e_buffer = d_E_buf + buf_base * storage_size;
                unsigned char* r_buffer = save_model_history
                    ? d_R_buf + buf_base * storage_size : nullptr;
                if (save_model_history && fwi_mode == 3) {
                    LAUNCH_STORAGE_KERNEL(save_rhs_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        r_buffer, 0, Ex, e_buffer, d_exact_Eold,
                        ce_hist, ce_rhs, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    LAUNCH_STORAGE_KERNEL(save_rhs_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        r_buffer + snap_size * storage_size, 0, Ey,
                        e_buffer + snap_size * storage_size, d_exact_Eold != nullptr ? d_exact_Eold + snap_size : nullptr,
                        ce_hist, ce_rhs, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    LAUNCH_STORAGE_KERNEL(save_rhs_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        r_buffer + 2 * snap_size * storage_size, 0, Ez,
                        e_buffer + 2 * snap_size * storage_size, d_exact_Eold != nullptr ? d_exact_Eold + 2 * snap_size : nullptr,
                        ce_hist, ce_rhs, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                } else if (save_model_history) {
                    LAUNCH_STORAGE_KERNEL(save_rhs_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        r_buffer, 0, Ez, e_buffer, d_exact_Eold,
                        ce_hist, ce_rhs, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                }
                CUDA_CHECK_LAST();
                CUDA_CHECK(cudaEventRecord(event_comp, stream_comp));
                CUDA_CHECK(cudaStreamWaitEvent(stream_trans, event_comp, 0));
                if (fwi_mode == 3) {
                    CUDA_CHECK(cudaMemcpyAsync(wavefield_offset_host(E_saved, (long long)t_saved * snap_size, storage_type), e_buffer, snap_size * storage_size, cudaMemcpyDeviceToHost, stream_trans));
                    CUDA_CHECK(cudaMemcpyAsync(wavefield_offset_host(E_saved, component_stride + (long long)t_saved * snap_size, storage_type), e_buffer + snap_size * storage_size, snap_size * storage_size, cudaMemcpyDeviceToHost, stream_trans));
                    CUDA_CHECK(cudaMemcpyAsync(wavefield_offset_host(E_saved, 2 * component_stride + (long long)t_saved * snap_size, storage_type), e_buffer + 2 * snap_size * storage_size, snap_size * storage_size, cudaMemcpyDeviceToHost, stream_trans));
                    if (save_model_history) {
                        CUDA_CHECK(cudaMemcpyAsync(wavefield_offset_host(R_saved, (long long)t_saved * snap_size, storage_type), r_buffer, snap_size * storage_size, cudaMemcpyDeviceToHost, stream_trans));
                        CUDA_CHECK(cudaMemcpyAsync(wavefield_offset_host(R_saved, component_stride + (long long)t_saved * snap_size, storage_type), r_buffer + snap_size * storage_size, snap_size * storage_size, cudaMemcpyDeviceToHost, stream_trans));
                        CUDA_CHECK(cudaMemcpyAsync(wavefield_offset_host(R_saved, 2 * component_stride + (long long)t_saved * snap_size, storage_type), r_buffer + 2 * snap_size * storage_size, snap_size * storage_size, cudaMemcpyDeviceToHost, stream_trans));
                    }
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(wavefield_offset_host(E_saved, (long long)t_saved * snap_size, storage_type), e_buffer, snap_size * storage_size, cudaMemcpyDeviceToHost, stream_trans));
                    if (save_model_history) {
                        CUDA_CHECK(cudaMemcpyAsync(wavefield_offset_host(R_saved, (long long)t_saved * snap_size, storage_type), r_buffer, snap_size * storage_size, cudaMemcpyDeviceToHost, stream_trans));
                    }
                }
                CUDA_CHECK(cudaEventRecord(event_transfer[buf_idx], stream_trans));
                transfer_buffer_in_use[buf_idx] = 1;
            } else {
                if (save_model_history && fwi_mode == 3) {
                    LAUNCH_STORAGE_KERNEL(save_rhs_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        R_saved, t_saved, Ex, E_saved, d_exact_Eold,
                        ce_hist, ce_rhs, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    LAUNCH_STORAGE_KERNEL(save_rhs_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        wavefield_offset_host(R_saved, component_stride, storage_type), t_saved, Ey,
                        wavefield_const_offset_host(E_saved, component_stride, storage_type), d_exact_Eold != nullptr ? d_exact_Eold + snap_size : nullptr,
                        ce_hist, ce_rhs, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                    LAUNCH_STORAGE_KERNEL(save_rhs_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        wavefield_offset_host(R_saved, 2 * component_stride, storage_type), t_saved, Ez,
                        wavefield_const_offset_host(E_saved, 2 * component_stride, storage_type), d_exact_Eold != nullptr ? d_exact_Eold + 2 * snap_size : nullptr,
                        ce_hist, ce_rhs, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                } else if (save_model_history) {
                    LAUNCH_STORAGE_KERNEL(save_rhs_snapshot_gpu, grid_copy, blockSize, stream_comp, storage_type,
                        R_saved, t_saved, Ez, E_saved, d_exact_Eold,
                        ce_hist, ce_rhs, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                }
                CUDA_CHECK_LAST();
            }
        }

    }

    if (use_async) {
        CUDA_CHECK(cudaStreamSynchronize(stream_comp));
        CUDA_CHECK(cudaStreamSynchronize(stream_trans));
        CUDA_CHECK(cudaFree(d_E_buf));
        if (d_R_buf != nullptr) CUDA_CHECK(cudaFree(d_R_buf));
        CUDA_CHECK(cudaEventDestroy(event_input));
        CUDA_CHECK(cudaEventDestroy(event_comp));
        CUDA_CHECK(cudaEventDestroy(event_transfer[0]));
        CUDA_CHECK(cudaEventDestroy(event_transfer[1]));
        CUDA_CHECK(cudaStreamDestroy(stream_comp));
        CUDA_CHECK(cudaStreamDestroy(stream_trans));
    }
    if (d_exact_Eold != nullptr) CUDA_CHECK(cudaFree(d_exact_Eold));
}

/*
 * Run CUDA adjoint FDTD modeling and accumulate model gradients.
 *
 * Parameters:
 *   eps_r_pad, sigma_pad, mu_r_pad: Padded material property arrays.
 *   E_saved: Saved pre-update electric field history.
 *   R_saved: Saved effective electric right-hand sides.
 *   Ex, Ey, Ez: Adjoint electric field component arrays.
 *   Hx, Hy, Hz: Adjoint magnetic field component arrays.
 *   ce_hist, ce_curl, ce_rhs: Electric-field update coefficient arrays.
 *   ch_hist, ch_curl, ch_rhs: Magnetic-field update coefficient arrays.
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
 *   dx, dy, dz: Grid spacing along each axis.
 *   NX_FIELDS, NY_FIELDS, NZ_FIELDS: Padded field grid sizes.
 *   ndata_source: Number of receiver-data cotangent sources per shot.
 *   receiver_location: Receiver coordinates with shape (step, ndata_source, 3).
 *   data_grad: Receiver-data cotangent array.
 *   receiver_component: Sampled field component, 0 for x, 1 for y, 2 for z.
 *   nsource, source_location, source_component: Original forward-source definition.
 *   grad_source: Output source-waveform gradient array.
 *   grad_eps_r: Output relative permittivity gradient array.
 *   grad_sigma: Output conductivity gradient array.
 *   eps_r_requires_grad: Whether grad_eps_r should be accumulated.
 *   sigma_requires_grad: Whether grad_sigma should be accumulated.
 *   sampling_interval: Forward wavefield sampling interval.
 *   fwi_mode: Gradient mode; 2 uses Ez only, 3 uses Ex, Ey, and Ez.
 */
DEEPGPR_API void backward(const float* __restrict__ eps_r_pad, const float* __restrict__ sigma_pad, const float* __restrict__ mu_r_pad,
             const void* __restrict__ E_saved, const void* __restrict__ R_saved,
             float* __restrict__ lambda_ex, float* __restrict__ lambda_ey, float* __restrict__ lambda_ez,
             float* __restrict__ lambda_hx, float* __restrict__ lambda_hy,  float* __restrict__ lambda_hz,
             float* __restrict__ ce_hist, float* __restrict__ ce_curl, float* __restrict__ ce_rhs,
             float* __restrict__ ch_hist, float* __restrict__ ch_curl, float* __restrict__ ch_rhs,

            float* __restrict__ x0EPhi1,float* __restrict__ x0EPhi2, float* __restrict__ x0HPhi1,float* __restrict__ x0HPhi2,
            float* __restrict__ xmEPhi1,float* __restrict__ xmEPhi2, float* __restrict__ xmHPhi1,float* __restrict__ xmHPhi2,
            float* __restrict__ y0EPhi1,float* __restrict__ y0EPhi2, float* __restrict__ y0HPhi1,float* __restrict__ y0HPhi2,
            float* __restrict__ ymEPhi1,float* __restrict__ ymEPhi2, float* __restrict__ ymHPhi1,float* __restrict__ ymHPhi2,
            float* __restrict__ z0EPhi1,float* __restrict__ z0EPhi2, float* __restrict__ z0HPhi1,float* __restrict__ z0HPhi2,
            float* __restrict__ zmEPhi1,float* __restrict__ zmEPhi2, float* __restrict__ zmHPhi1,float* __restrict__ zmHPhi2,

            int pml0,int pml1,int pml2,int pml3,int pml4,int pml5,

            const float* __restrict__ x0ER,const float* __restrict__ xmER,
            const float* __restrict__ y0ER,const float* __restrict__ ymER,
            const float* __restrict__ z0ER,const float* __restrict__ zmER,
            const float* __restrict__ x0HR,const float* __restrict__ xmHR,
            const float* __restrict__ y0HR,const float* __restrict__ ymHR,
            const float* __restrict__ z0HR,const float* __restrict__ zmHR,

             float dt, int nt, int step, int nrx, float dx, float dy, float dz,
             int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
             int ndata_source, const int* __restrict__ receiver_location, const float* __restrict__ data_grad,
             int receiver_component,
             int nsource, const int* __restrict__ source_location,
             int source_component, float* __restrict__ grad_source, int source_requires_grad,
             float*__restrict__ grad_eps_r,float*__restrict__ grad_sigma, int eps_r_requires_grad, int sigma_requires_grad,
             int sampling_interval, int fwi_mode, int storage_type, int use_async_offload)
{
    int need_material_gradient = eps_r_requires_grad || sigma_requires_grad;
    int use_async = need_material_gradient && use_async_offload != 0;
    int fdtd_order = g_fdtd_order;
    int e_components = (fwi_mode == 3) ? 3 : 1;
    int has_cpml = pml0 || pml1 || pml2 || pml3 || pml4 || pml5;

    unsigned char* d_E_buf = nullptr;
    unsigned char* d_R_buf = nullptr;
    long long snap_size = (long long)step * (NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1);
    size_t storage_size = wavefield_element_size_host(storage_type);
    
    cudaStream_t stream_comp = 0, stream_trans = 0;
    cudaEvent_t event_input, event_trans, event_comp;
    int staging_buffer_in_use = 0;
    if (use_async) {
        CUDA_CHECK(cudaStreamCreate(&stream_comp));
        CUDA_CHECK(cudaStreamCreate(&stream_trans));
        CUDA_CHECK(cudaEventCreate(&event_input));
        CUDA_CHECK(cudaEventCreate(&event_trans));
        CUDA_CHECK(cudaEventCreate(&event_comp));
        CUDA_CHECK(cudaMalloc(&d_E_buf, e_components * snap_size * storage_size));
        CUDA_CHECK(cudaMalloc(&d_R_buf, e_components * snap_size * storage_size));
        CUDA_CHECK(cudaEventRecord(event_input, 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream_comp, event_input, 0));
    }

    long long blockSize = 256;
    long long total_fields = (long long)NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    dim3 grid_material(CEIL_DIV(total_fields, blockSize));
    dim3 grid_fields(CEIL_DIV((long long)step * total_fields, blockSize));

    build_update_coeffs_gpu<<<grid_material, blockSize, 0, stream_comp>>>(eps_r_pad, sigma_pad, mu_r_pad, ce_hist, ce_curl, ce_rhs, ch_hist, ch_curl, ch_rhs, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);
    CUDA_CHECK_LAST();

    long long total_data_source = (long long)step * ndata_source;
    dim3 grid_data_source(CEIL_DIV(total_data_source, blockSize));
    long long total_source = (long long)step * nsource;
    dim3 grid_source(CEIL_DIV(total_source, blockSize));

    long long total_grad = (long long)(NX_FIELDS-1) * (NY_FIELDS-1) * (NZ_FIELDS-1);
    dim3 grid_grad(CEIL_DIV(total_grad, blockSize));
  
    int nt_saved = (nt + sampling_interval - 1) / sampling_interval;
    long long component_stride = (long long)nt_saved * snap_size;

    for (int i = nt - 1; i >= 0; i--) {
        if (use_async && i % sampling_interval == 0) {
            int t_saved = i / sampling_interval;
            if (staging_buffer_in_use) {
                CUDA_CHECK(cudaStreamWaitEvent(stream_trans, event_comp, 0));
            }
            for (int c = 0; c < e_components; ++c) {
                long long comp_offset = (fwi_mode == 3) ? (long long)c * component_stride : 0;
                long long source_offset = comp_offset + (long long)t_saved * snap_size;
                CUDA_CHECK(cudaMemcpyAsync(d_E_buf + (long long)c * snap_size * storage_size,
                    wavefield_const_offset_host(E_saved, source_offset, storage_type), snap_size * storage_size,
                    cudaMemcpyHostToDevice, stream_trans));
                CUDA_CHECK(cudaMemcpyAsync(d_R_buf + (long long)c * snap_size * storage_size,
                    wavefield_const_offset_host(R_saved, source_offset, storage_type), snap_size * storage_size,
                    cudaMemcpyHostToDevice, stream_trans));
            }
            CUDA_CHECK(cudaEventRecord(event_trans, stream_trans));
            CUDA_CHECK(cudaStreamWaitEvent(stream_comp, event_trans, 0));
        }
        
        /* receiver sampling^T */
        adjoint_receivers_gpu<<<grid_data_source, blockSize, 0, stream_comp>>>(
            step, i, receiver_location, data_grad, lambda_ex, lambda_ey, lambda_ez,
            NX_FIELDS, NY_FIELDS, NZ_FIELDS, ndata_source, receiver_component, nt);
        CUDA_CHECK_LAST();

        /* source injection^T: identity on the state plus the waveform gradient */
        if (source_requires_grad == 1) {
            adjoint_source_injection_gpu<<<grid_source, blockSize, 0, stream_comp>>>(
                step, i, dx, dy, dz, source_location,
                lambda_ex, lambda_ey, lambda_ez, ce_rhs,
                NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                nsource, source_component, nt, grad_source);
            CUDA_CHECK_LAST();
        }

        if (need_material_gradient && i % sampling_interval == 0) {
            int sample_weight = sampling_interval;
            if (i + sample_weight > nt) sample_weight = nt - i;
            LAUNCH_STORAGE_KERNEL(accumulate_material_gradients_gpu, grid_grad, blockSize, stream_comp, storage_type,
                lambda_ex, lambda_ey, lambda_ez, E_saved, R_saved, d_E_buf, d_R_buf, ce_hist, ce_rhs, sigma_pad,
                grad_eps_r, grad_sigma, i, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, dt,
                eps_r_requires_grad, sigma_requires_grad, sampling_interval, sample_weight,
                nt_saved, use_async, fwi_mode);
            CUDA_CHECK_LAST();
            if (use_async) {
                CUDA_CHECK(cudaEventRecord(event_comp, stream_comp));
                staging_buffer_in_use = 1;
            }
        }

        /* E CPML^T -> E update^T -> H CPML^T -> H update^T. */
        if (has_cpml) {
            LAUNCH_ORDER_KERNEL(adjoint_cpml_e_gpu, grid_fields, blockSize, stream_comp, fdtd_order,
                lambda_ex, lambda_ey, lambda_ez, lambda_hx, lambda_hy, lambda_hz, dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, ce_rhs,
                x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2,
                z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2);
            CUDA_CHECK_LAST();
        }
        LAUNCH_ORDER_KERNEL(adjoint_e_gpu, grid_fields, blockSize, stream_comp, fdtd_order,
            ce_hist, ce_curl, lambda_ex, lambda_ey, lambda_ez, lambda_hx, lambda_hy, lambda_hz,
            step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dx, dy, dz);
        CUDA_CHECK_LAST();
        if (has_cpml) {
            LAUNCH_ORDER_KERNEL(adjoint_cpml_h_gpu, grid_fields, blockSize, stream_comp, fdtd_order,
                lambda_ex, lambda_ey, lambda_ez, lambda_hx, lambda_hy, lambda_hz, dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, ch_rhs,
                x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2,
                z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2);
            CUDA_CHECK_LAST();
        }
        LAUNCH_ORDER_KERNEL(adjoint_h_gpu, grid_fields, blockSize, stream_comp, fdtd_order,
            ch_hist, ch_curl, lambda_ex, lambda_ey, lambda_ez, lambda_hx, lambda_hy, lambda_hz,
            step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dx, dy, dz);
        CUDA_CHECK_LAST();
    }

    if (use_async) {
        CUDA_CHECK(cudaStreamSynchronize(stream_comp));
        CUDA_CHECK(cudaStreamSynchronize(stream_trans));
        CUDA_CHECK(cudaFree(d_E_buf));
        CUDA_CHECK(cudaFree(d_R_buf));
        CUDA_CHECK(cudaEventDestroy(event_input));
        CUDA_CHECK(cudaEventDestroy(event_trans));
        CUDA_CHECK(cudaEventDestroy(event_comp));
        CUDA_CHECK(cudaStreamDestroy(stream_comp));
        CUDA_CHECK(cudaStreamDestroy(stream_trans));
    }
}

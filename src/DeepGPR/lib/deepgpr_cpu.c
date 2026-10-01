#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEEPGPR_BUILD
#include "deepgpr.h"

#if defined(_OPENMP) || defined(DEEPGPR_USE_OPENMP)
#include <omp.h>
#define DEEPGPR_OMP_PARALLEL _Pragma("omp parallel")
#define DEEPGPR_OMP_PARALLEL_FOR _Pragma("omp for schedule(static)")
#define DEEPGPR_OMP_STANDALONE_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#else
#define DEEPGPR_OMP_PARALLEL
#define DEEPGPR_OMP_PARALLEL_FOR
#define DEEPGPR_OMP_STANDALONE_PARALLEL_FOR
#endif

/*
 * No OpenMP atomics are used anywhere in this backend: every parallel loop
 * writes disjoint addresses, so forward and adjoint results are bitwise
 * identical for any OMP_NUM_THREADS (see adjoint_e_step_cpu).
 */

#ifdef _WIN32
#define RESTRICT __restrict
#else
#define RESTRICT restrict
#endif

static const float E0 = DEEPGPR_EPSILON0_F;
static const float M0 = DEEPGPR_MU0_F;

#if defined(_MSC_VER)
#define DEEPGPR_THREAD_LOCAL __declspec(thread)
#else
#define DEEPGPR_THREAD_LOCAL __thread
#endif

/* Per-thread message of the last failed native call ("" after success). */
static DEEPGPR_THREAD_LOCAL char g_last_error[DEEPGPR_ERROR_BUFFER_SIZE];

static void deepgpr_set_error(const char* format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(g_last_error, sizeof(g_last_error), format, arguments);
    va_end(arguments);
}

DEEPGPR_API int deepgpr_supports_error_reporting(void)
{
    return 1;
}

DEEPGPR_API const char* deepgpr_last_error(void)
{
    return g_last_error;
}

DEEPGPR_API void deepgpr_clear_last_error(void)
{
    g_last_error[0] = '\0';
}

DEEPGPR_API int deepgpr_deterministic_adjoint(void)
{
    return 1;
}

enum {
    WAVEFIELD_FLOAT32 = 0,
    WAVEFIELD_FLOAT16 = 1,
    WAVEFIELD_BFLOAT16 = 2,
    WAVEFIELD_HISTORY_DISABLED = 1 << 26
};

static size_t wavefield_element_size(int storage_type)
{
    return storage_type == WAVEFIELD_FLOAT32 ? sizeof(float) : sizeof(uint16_t);
}

static void* wavefield_offset(void* pointer, long long offset, int storage_type)
{
    return (void*)((unsigned char*)pointer + offset * (long long)wavefield_element_size(storage_type));
}

static const void* wavefield_const_offset(const void* pointer, long long offset, int storage_type)
{
    return (const void*)((const unsigned char*)pointer + offset * (long long)wavefield_element_size(storage_type));
}

static uint16_t float_to_half_bits(float value)
{
    uint32_t bits;
    uint32_t sign, exponent, mantissa, half_mantissa, remainder, halfway;
    int half_exponent, shift;
    memcpy(&bits, &value, sizeof(bits));
    sign = (bits >> 16) & 0x8000u;
    exponent = (bits >> 23) & 0xffu;
    mantissa = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0u) return (uint16_t)(sign | 0x7c00u);
        return (uint16_t)(sign | 0x7e00u);
    }

    half_exponent = (int)exponent - 127 + 15;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return (uint16_t)sign;
        mantissa |= 0x800000u;
        shift = 14 - half_exponent;
        half_mantissa = mantissa >> shift;
        remainder = mantissa & ((1u << shift) - 1u);
        halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) {
            ++half_mantissa;
        }
        return (uint16_t)(sign | half_mantissa);
    }

    half_mantissa = mantissa >> 13;
    remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u))) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0u;
            ++half_exponent;
            if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10) | half_mantissa);
}

static float half_bits_to_float(uint16_t half)
{
    uint32_t sign = ((uint32_t)half & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)half >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)half & 0x3ffu;
    uint32_t bits;
    float value;

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
            bits = sign | ((uint32_t)(normalized_exponent + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t float_to_bfloat16_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        return (uint16_t)((bits >> 16) | 0x0040u);
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bfloat16_bits_to_float(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void store_wavefield_value(void* pointer, long long index, float value, int storage_type)
{
    if (storage_type == WAVEFIELD_FLOAT16) {
        ((uint16_t*)pointer)[index] = float_to_half_bits(value);
    } else if (storage_type == WAVEFIELD_BFLOAT16) {
        ((uint16_t*)pointer)[index] = float_to_bfloat16_bits(value);
    } else {
        ((float*)pointer)[index] = value;
    }
}

static float load_wavefield_value(const void* pointer, long long index, int storage_type)
{
    if (storage_type == WAVEFIELD_FLOAT16) {
        return half_bits_to_float(((const uint16_t*)pointer)[index]);
    }
    if (storage_type == WAVEFIELD_BFLOAT16) {
        return bfloat16_bits_to_float(((const uint16_t*)pointer)[index]);
    }
    return ((const float*)pointer)[index];
}

static int g_fdtd_order = 2;

/* External PML keeps the first physical cell at each low face invertible. */
DEEPGPR_API int deepgpr_supports_external_pml(void)
{
    return 1;
}

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

/*
 * Deterministic transposes of the staggered derivatives.
 *
 * The forward E update reads H through staggered *backward* differences and
 * the H update reads E through staggered *forward* differences. Their exact
 * transposes used to be applied by scattering every source point into its
 * stencil neighbours with OpenMP atomics, so the summation order - and the
 * last bits of every gradient - depended on thread scheduling.
 *
 * The adjoint is now evaluated in two passes. Every source point first
 * stores its stencil weight in one buffer per (target field, derivative axis)
 * "channel"; every target point then gathers the transposed stencil from those
 * weights in a fixed order. No two threads ever write the same address, so the
 * results are bitwise identical for any OMP_NUM_THREADS.
 *
 * Near the model faces the stencil radius of each *source* point is clamped
 * (usable_*_radius); the gather therefore evaluates the radius and the
 * coefficient at the source, exactly as the scatter did.
 */

/* Transpose of staggered_backward_diff, gathered at target point t. */
static float gather_backward_transpose(
    const float* RESTRICT weights, long long t, long long stride,
    long long coord, long long n, int order)
{
    if (order <= 2) {
        float acc = weights[t];
        if (coord + 1 < n) acc -= weights[t + stride];
        return acc;
    }

    int requested = fdtd_radius_for_order(order);
    float acc = 0.0f;
    if (coord >= 2 * requested - 1 && coord + 2 * requested - 1 < n) {
        /* Interior: every contributing source uses the full stencil. The
         * arithmetic is identical to the general branch below. */
        for (int r = 1; r <= requested; ++r) {
            float c = fdtd_coeff(requested, r);
            acc += c * weights[t - (long long)(r - 1) * stride];
            acc -= c * weights[t + (long long)r * stride];
        }
        return acc;
    }
    for (int r = 1; r <= requested; ++r) {
        long long plus_source = coord - (r - 1);
        if (plus_source >= 0) {
            int radius = usable_backward_radius(plus_source, n, requested);
            if (r <= radius) {
                acc += fdtd_coeff(radius, r) * weights[t - (long long)(r - 1) * stride];
            }
        }
        long long minus_source = coord + r;
        if (minus_source < n) {
            int radius = usable_backward_radius(minus_source, n, requested);
            if (r <= radius) {
                acc -= fdtd_coeff(radius, r) * weights[t + (long long)r * stride];
            }
        }
    }
    return acc;
}

/* Transpose of staggered_forward_diff, gathered at target point t. */
static float gather_forward_transpose(
    const float* RESTRICT weights, long long t, long long stride,
    long long coord, long long n, int order)
{
    if (order <= 2) {
        float acc = 0.0f;
        if (coord >= 1) acc += weights[t - stride];
        acc -= weights[t];
        return acc;
    }

    int requested = fdtd_radius_for_order(order);
    float acc = 0.0f;
    if (coord >= 2 * requested - 1 && coord + 2 * requested <= n) {
        /* Interior: every contributing source uses the full stencil. */
        for (int r = 1; r <= requested; ++r) {
            float c = fdtd_coeff(requested, r);
            acc += c * weights[t - (long long)r * stride];
            acc -= c * weights[t + (long long)(r - 1) * stride];
        }
        return acc;
    }
    for (int r = 1; r <= requested; ++r) {
        long long plus_source = coord - r;
        if (plus_source >= 0) {
            int radius = usable_forward_radius(plus_source, n, requested);
            if (r <= radius) {
                acc += fdtd_coeff(radius, r) * weights[t - (long long)r * stride];
            }
        }
        long long minus_source = coord + (r - 1);
        if (minus_source < n) {
            int radius = usable_forward_radius(minus_source, n, requested);
            if (r <= radius) {
                acc -= fdtd_coeff(radius, r) * weights[t + (long long)(r - 1) * stride];
            }
        }
    }
    return acc;
}

/*
 * Adjoint of one CPML auxiliary update. Returns the weight that the CPML
 * correction contributes to the transposed derivative of its source field and
 * advances the auxiliary cotangent in place.
 */
static float pml_adjoint_weight(
    float lambda_field, float inverse_spacing, float update_coeff, float sign,
    float ra_minus_one, float rb, float re, float rf, float* RESTRICT lambda_phi)
{
    float phi_new = *lambda_phi;
    float derivative_weight =
        (sign * update_coeff * ra_minus_one * lambda_field - rf * phi_new) * inverse_spacing;
    *lambda_phi = sign * update_coeff * rb * lambda_field + re * phi_new;
    return derivative_weight;
}

/* Weight channels: target field and derivative axis of each transposed stencil. */
enum {
    CH_A_X = 0, CH_A_Y, CH_B_X, CH_B_Z, CH_C_Y, CH_C_Z, ADJOINT_CHANNELS
};

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
static void build_update_coeffs_cpu(const float* RESTRICT eps_r_pad, const float* RESTRICT sigma_pad, const float* RESTRICT mu_r_pad,
    float* RESTRICT ce_hist, float* RESTRICT ce_curl, float* RESTRICT ce_rhs,
    float* RESTRICT ch_hist, float* RESTRICT ch_curl, float* RESTRICT ch_rhs,
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS, float dt, float dx)
{
    long long total = (long long)NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long idx;

    DEEPGPR_OMP_STANDALONE_PARALLEL_FOR
    for (idx = 0; idx < total; ++idx) {
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ_FIELDS;
        long long k = rem % NZ_FIELDS;

        if (i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1)) {
            float HA = M0 * mu_r_pad[idx] / dt;
            ch_hist[idx] = 1.0f;
            ch_curl[idx] = (1.0f / dx) / HA;
            ch_rhs[idx] = 1.0f / HA;

            if (sigma_pad[idx] > DEEPGPR_PEC_SIGMA_THRESHOLD_F) {
                ce_hist[idx] = 0.0f;
                ce_curl[idx] = 0.0f;
                ce_rhs[idx] = 0.0f;
            } else {
                float e_term = E0 * eps_r_pad[idx] / dt;
                float s_term = 0.5f * sigma_pad[idx];
                float EA = e_term + s_term;
                float EB = e_term - s_term;
                ce_hist[idx] = EB / EA;
                ce_curl[idx] = (1.0f / dx) / EA;
                ce_rhs[idx] = 1.0f / EA;
            }
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
static void sample_receivers_cpu(
    int step, int NRX, int iteration,
    const int* RESTRICT receiverlocation, float* RESTRICT rxs,
    const float* RESTRICT Ex, const float* RESTRICT Ey, const float* RESTRICT Ez,
    int NX, int NY, int NZ, int N_ITER, int receiver_component)
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

        const float* field = receiver_component == 0 ? Ex : (receiver_component == 1 ? Ey : Ez);
        rxs[((long long)s * N_ITER + iteration) * NRX + rx] = field[id4];
    }
}

/*
 * Inject a Hertzian dipole source into one electric-field component.
 *
 * Shots run in parallel; the sources of one shot are applied sequentially in
 * index order, so coincident sources accumulate deterministically.
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
static void inject_sources_cpu(
    int step, int iteration, float dx, float dy, float dz,
    const int* RESTRICT sourcelocation, const float* RESTRICT srcwaveforms,
    float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez, const float* RESTRICT ce_rhs,
    int NX, int NY, int NZ, int nsrc, int polarisation, int nt)
{
    long long field_stride = (long long)NX * NY * NZ;
    float* RESTRICT field = polarisation == 0 ? Ex : (polarisation == 1 ? Ey : Ez);
    float dipole_length = polarisation == 0 ? dx : (polarisation == 1 ? dy : dz);
    long long shot;

    if (polarisation < 0 || polarisation > 2) return;

    DEEPGPR_OMP_PARALLEL_FOR
    for (shot = 0; shot < step; ++shot) {
        for (long long src = 0; src < nsrc; ++src) {
            float waveform_value = srcwaveforms[src * nt + iteration];
            float scale = waveform_value * dipole_length / (dx * dy * dz);
            const int* location = sourcelocation + (shot * nsrc + src) * 3;
            long long id3 = (long long)location[0] * NY * NZ + (long long)location[1] * NZ + location[2];
            long long id4 = shot * field_stride + id3;
            field[id4] -= ce_rhs[id3] * scale;
        }
    }
}

static void update_e_cpu(
    const float* RESTRICT ce_hist, const float* RESTRICT ce_curl,
    float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez,
    const float* RESTRICT Hx, const float* RESTRICT Hy, const float* RESTRICT Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
    float dx, float dy, float dz,
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

        float ue0 = ce_hist[idx];
        float ue1 = ce_curl[idx];
        float ue_y = dy == dx ? ue1 : ue1 * dx / dy;
        float ue_z = dz == dx ? ue1 : ue1 * dx / dz;

        int do_ex = (((NY_FIELDS - 1) != 1 || (NZ_FIELDS - 1) != 1) && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));
        int do_ey = (((NX_FIELDS - 1) != 1 || (NZ_FIELDS - 1) != 1) && i > 0 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));
        int do_ez = (((NX_FIELDS - 1) != 1 || (NY_FIELDS - 1) != 1) && i > 0 && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));

        if (do_ex) {
            float dHz_dy = staggered_backward_diff(Hz, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order);
            float dHy_dz = staggered_backward_diff(Hy, id4, 1, k, NZ_FIELDS, fdtd_order);
            Ex[id4] = ue0 * Ex[id4] + ue_y * dHz_dy - ue_z * dHy_dz;
        }
        if (do_ey) {
            float dHx_dz = staggered_backward_diff(Hx, id4, 1, k, NZ_FIELDS, fdtd_order);
            float dHz_dx = staggered_backward_diff(Hz, id4, ny_nz, i, NX_FIELDS, fdtd_order);
            Ey[id4] = ue0 * Ey[id4] + ue_z * dHx_dz - ue1 * dHz_dx;
        }
        if (do_ez) {
            float dHy_dx = staggered_backward_diff(Hy, id4, ny_nz, i, NX_FIELDS, fdtd_order);
            float dHx_dy = staggered_backward_diff(Hx, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order);
            Ez[id4] = ue0 * Ez[id4] + ue1 * dHy_dx - ue_y * dHx_dy;
        }
    }
}

/*
 * Apply electric CPML boundary corrections after the base electric-field update.
 */
static void cpml_e_cpu(
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


static void update_h_cpu(
    const float* RESTRICT ch_hist, const float* RESTRICT ch_curl,
    const float* RESTRICT Ex, const float* RESTRICT Ey, const float* RESTRICT Ez,
    float* RESTRICT Hx, float* RESTRICT Hy, float* RESTRICT Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
    float dx, float dy, float dz,
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

        float uh0 = ch_hist[idx];
        float uh1 = ch_curl[idx];
        float uh_y = dy == dx ? uh1 : uh1 * dx / dy;
        float uh_z = dz == dx ? uh1 : uh1 * dx / dz;

        int do_hx = ((NX_FIELDS - 1) != 1 && i > 0 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));
        int do_hy = ((NY_FIELDS - 1) != 1 && i < (NX_FIELDS - 1) && j > 0 && j < (NY_FIELDS - 1) && k < (NZ_FIELDS - 1));
        int do_hz = ((NZ_FIELDS - 1) != 1 && i < (NX_FIELDS - 1) && j < (NY_FIELDS - 1) && k > 0 && k < (NZ_FIELDS - 1));

        if (do_hx) {
            float dEz_dy = staggered_forward_diff(Ez, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order);
            float dEy_dz = staggered_forward_diff(Ey, id4, 1, k, NZ_FIELDS, fdtd_order);
            Hx[id4] = uh0 * Hx[id4] - uh_y * dEz_dy + uh_z * dEy_dz;
        }
        if (do_hy) {
            float dEx_dz = staggered_forward_diff(Ex, id4, 1, k, NZ_FIELDS, fdtd_order);
            float dEz_dx = staggered_forward_diff(Ez, id4, ny_nz, i, NX_FIELDS, fdtd_order);
            Hy[id4] = uh0 * Hy[id4] - uh_z * dEx_dz + uh1 * dEz_dx;
        }
        if (do_hz) {
            float dEy_dx = staggered_forward_diff(Ey, id4, ny_nz, i, NX_FIELDS, fdtd_order);
            float dEx_dy = staggered_forward_diff(Ex, id4, NZ_FIELDS, j, NY_FIELDS, fdtd_order);
            Hz[id4] = uh0 * Hz[id4] - uh1 * dEy_dx + uh_y * dEx_dy;
        }
    }
}


/*
 * Apply magnetic CPML boundary corrections after the base magnetic-field update.
 */
static void cpml_h_cpu(
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

/*
 * Exact transpose of one electric half step: the CPML correction followed by
 * the base update E^(n+1) = ce_hist E^n + ce_curl curl(H).
 *
 * Pass 1 (per source point): CPML auxiliary cotangents are advanced, the
 * transposed-curl weights are stored per channel and lambda_E is scaled by
 * ce_hist. Pass 2 (per target point): lambda_H gathers the transposed
 * backward-difference stencils. Channels: A = lambda_hz (x, y),
 * B = lambda_hy (x, z), C = lambda_hx (y, z).
 */
static void adjoint_e_step_cpu(
    const float* RESTRICT ce_hist, const float* RESTRICT ce_curl,
    const float* RESTRICT update,
    float* RESTRICT lambda_ex, float* RESTRICT lambda_ey, float* RESTRICT lambda_ez,
    float* RESTRICT lambda_hx, float* RESTRICT lambda_hy, float* RESTRICT lambda_hz,
    float dx, float dy, float dz, int step, int NX, int NY, int NZ,
    int has_cpml, int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* RESTRICT x0R, const float* RESTRICT xmR,
    const float* RESTRICT y0R, const float* RESTRICT ymR,
    const float* RESTRICT z0R, const float* RESTRICT zmR,
    float* RESTRICT x0P1, float* RESTRICT x0P2,
    float* RESTRICT xmP1, float* RESTRICT xmP2,
    float* RESTRICT y0P1, float* RESTRICT y0P2,
    float* RESTRICT ymP1, float* RESTRICT ymP2,
    float* RESTRICT z0P1, float* RESTRICT z0P2,
    float* RESTRICT zmP1, float* RESTRICT zmP2,
    int order, float* RESTRICT weights)
{
    long long ny_nz = (long long)NY * NZ;
    long long field_stride = (long long)NX * ny_nz;
    long long total_work = (long long)step * field_stride;
    float* RESTRICT w_a_x = weights + CH_A_X * total_work;
    float* RESTRICT w_a_y = weights + CH_A_Y * total_work;
    float* RESTRICT w_b_x = weights + CH_B_X * total_work;
    float* RESTRICT w_b_z = weights + CH_B_Z * total_work;
    float* RESTRICT w_c_y = weights + CH_C_Y * total_work;
    float* RESTRICT w_c_z = weights + CH_C_Z * total_work;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        int s = (int)(work / field_stride);
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ;
        long long k = rem % NZ;
        float lex = lambda_ex[work];
        float ley = lambda_ey[work];
        float lez = lambda_ez[work];
        float hz_x = 0.0f, hz_y = 0.0f, hy_x = 0.0f, hy_z = 0.0f, hx_y = 0.0f, hx_z = 0.0f;

        if (has_cpml) {
            float upd = update[idx];
            long long p_idx;
#define E_PML(R, P, p, q, spacing, field, channel, sign) \
            do { \
                (channel) += pml_adjoint_weight( \
                    (field), 1.0f / (spacing), upd, (sign), (R)[q] - 1.0f, \
                    (R)[(p) + (q)], (R)[2 * (p) + (q)], (R)[3 * (p) + (q)], &(P)[p_idx]); \
            } while (0)

            if (pml0 > 0 && i > 0 && i <= pml0) {
                long long q = pml0 - i;
                if (j < NY - 1) {
                    p_idx = ((long long)s * (pml0 + 1) * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
                    E_PML(x0R, x0P1, pml0, q, dx, ley, hz_x, -1.0f);
                }
                if (k < NZ - 1) {
                    p_idx = ((long long)s * (pml0 + 1) * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
                    E_PML(x0R, x0P2, pml0, q, dx, lez, hy_x, 1.0f);
                }
            }
            if (pml1 > 0 && i >= NX - 1 - pml1 && i < NX - 1) {
                long long q = i - (NX - 1 - pml1);
                if (j < NY - 1 && i > 0) {
                    p_idx = ((long long)s * (pml1 + 1) * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
                    E_PML(xmR, xmP1, pml1, q, dx, ley, hz_x, -1.0f);
                }
                if (k < NZ - 1 && i > 0) {
                    p_idx = ((long long)s * (pml1 + 1) * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
                    E_PML(xmR, xmP2, pml1, q, dx, lez, hy_x, 1.0f);
                }
            }
            if (pml2 > 0 && j > 0 && j <= pml2) {
                long long q = pml2 - j;
                if (i < NX - 1) {
                    p_idx = ((long long)s * (NX - 1) * (pml2 + 1) * NZ) + i * (pml2 + 1) * NZ + q * NZ + k;
                    E_PML(y0R, y0P1, pml2, q, dy, lex, hz_y, 1.0f);
                }
                if (k < NZ - 1) {
                    p_idx = ((long long)s * NX * (pml2 + 1) * (NZ - 1)) + i * (pml2 + 1) * (NZ - 1) + q * (NZ - 1) + k;
                    E_PML(y0R, y0P2, pml2, q, dy, lez, hx_y, -1.0f);
                }
            }
            if (pml3 > 0 && j >= NY - 1 - pml3 && j < NY - 1) {
                long long q = j - (NY - 1 - pml3);
                if (i < NX - 1 && j > 0) {
                    p_idx = ((long long)s * (NX - 1) * (pml3 + 1) * NZ) + i * (pml3 + 1) * NZ + q * NZ + k;
                    E_PML(ymR, ymP1, pml3, q, dy, lex, hz_y, 1.0f);
                }
                if (k < NZ - 1 && j > 0) {
                    p_idx = ((long long)s * NX * (pml3 + 1) * (NZ - 1)) + i * (pml3 + 1) * (NZ - 1) + q * (NZ - 1) + k;
                    E_PML(ymR, ymP2, pml3, q, dy, lez, hx_y, -1.0f);
                }
            }
            if (pml4 > 0 && k > 0 && k <= pml4) {
                long long q = pml4 - k;
                if (i < NX - 1) {
                    p_idx = ((long long)s * (NX - 1) * NY * (pml4 + 1)) + i * NY * (pml4 + 1) + j * (pml4 + 1) + q;
                    E_PML(z0R, z0P1, pml4, q, dz, lex, hy_z, -1.0f);
                }
                if (j < NY - 1) {
                    p_idx = ((long long)s * NX * (NY - 1) * (pml4 + 1)) + i * (NY - 1) * (pml4 + 1) + j * (pml4 + 1) + q;
                    E_PML(z0R, z0P2, pml4, q, dz, ley, hx_z, 1.0f);
                }
            }
            if (pml5 > 0 && k >= NZ - 1 - pml5 && k < NZ - 1) {
                long long q = k - (NZ - 1 - pml5);
                if (i < NX - 1 && k > 0) {
                    p_idx = ((long long)s * (NX - 1) * NY * (pml5 + 1)) + i * NY * (pml5 + 1) + j * (pml5 + 1) + q;
                    E_PML(zmR, zmP1, pml5, q, dz, lex, hy_z, -1.0f);
                }
                if (j < NY - 1 && k > 0) {
                    p_idx = ((long long)s * NX * (NY - 1) * (pml5 + 1)) + i * (NY - 1) * (pml5 + 1) + j * (pml5 + 1) + q;
                    E_PML(zmR, zmP2, pml5, q, dz, ley, hx_z, 1.0f);
                }
            }
#undef E_PML
        }

        {
            int do_ex = (((NY - 1) != 1 || (NZ - 1) != 1) && i < NX - 1 && j > 0 && j < NY - 1 && k > 0 && k < NZ - 1);
            int do_ey = (((NX - 1) != 1 || (NZ - 1) != 1) && i > 0 && i < NX - 1 && j < NY - 1 && k > 0 && k < NZ - 1);
            int do_ez = (((NX - 1) != 1 || (NY - 1) != 1) && i > 0 && i < NX - 1 && j > 0 && j < NY - 1 && k < NZ - 1);
            float coeff = ce_curl[idx];
            float coeff_y = dy == dx ? coeff : coeff * dx / dy;
            float coeff_z = dz == dx ? coeff : coeff * dx / dz;

            if (do_ex) {
                hz_y += coeff_y * lex;
                hy_z += -coeff_z * lex;
                lambda_ex[work] = ce_hist[idx] * lex;
            }
            if (do_ey) {
                hx_z += coeff_z * ley;
                hz_x += -coeff * ley;
                lambda_ey[work] = ce_hist[idx] * ley;
            }
            if (do_ez) {
                hy_x += coeff * lez;
                hx_y += -coeff_y * lez;
                lambda_ez[work] = ce_hist[idx] * lez;
            }
        }

        w_a_x[work] = hz_x;
        w_a_y[work] = hz_y;
        w_b_x[work] = hy_x;
        w_b_z[work] = hy_z;
        w_c_y[work] = hx_y;
        w_c_z[work] = hx_z;
    }

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ;
        long long k = rem % NZ;

        lambda_hz[work] += gather_backward_transpose(w_a_x, work, ny_nz, i, NX, order);
        lambda_hz[work] += gather_backward_transpose(w_a_y, work, NZ, j, NY, order);
        lambda_hy[work] += gather_backward_transpose(w_b_x, work, ny_nz, i, NX, order);
        lambda_hy[work] += gather_backward_transpose(w_b_z, work, 1, k, NZ, order);
        lambda_hx[work] += gather_backward_transpose(w_c_y, work, NZ, j, NY, order);
        lambda_hx[work] += gather_backward_transpose(w_c_z, work, 1, k, NZ, order);
    }
}

/*
 * Exact transpose of one magnetic half step: the CPML correction followed by
 * the base update H^(n+1/2) = ch_hist H + ch_curl curl(E).
 *
 * Channels: A = lambda_ez (x, y), B = lambda_ey (x, z), C = lambda_ex (y, z);
 * lambda_E gathers the transposed forward-difference stencils.
 */
static void adjoint_h_step_cpu(
    const float* RESTRICT ch_hist, const float* RESTRICT ch_curl,
    const float* RESTRICT update,
    float* RESTRICT lambda_ex, float* RESTRICT lambda_ey, float* RESTRICT lambda_ez,
    float* RESTRICT lambda_hx, float* RESTRICT lambda_hy, float* RESTRICT lambda_hz,
    float dx, float dy, float dz, int step, int NX, int NY, int NZ,
    int has_cpml, int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    const float* RESTRICT x0R, const float* RESTRICT xmR,
    const float* RESTRICT y0R, const float* RESTRICT ymR,
    const float* RESTRICT z0R, const float* RESTRICT zmR,
    float* RESTRICT x0P1, float* RESTRICT x0P2,
    float* RESTRICT xmP1, float* RESTRICT xmP2,
    float* RESTRICT y0P1, float* RESTRICT y0P2,
    float* RESTRICT ymP1, float* RESTRICT ymP2,
    float* RESTRICT z0P1, float* RESTRICT z0P2,
    float* RESTRICT zmP1, float* RESTRICT zmP2,
    int order, float* RESTRICT weights)
{
    long long ny_nz = (long long)NY * NZ;
    long long field_stride = (long long)NX * ny_nz;
    long long total_work = (long long)step * field_stride;
    float* RESTRICT w_a_x = weights + CH_A_X * total_work;
    float* RESTRICT w_a_y = weights + CH_A_Y * total_work;
    float* RESTRICT w_b_x = weights + CH_B_X * total_work;
    float* RESTRICT w_b_z = weights + CH_B_Z * total_work;
    float* RESTRICT w_c_y = weights + CH_C_Y * total_work;
    float* RESTRICT w_c_z = weights + CH_C_Z * total_work;
    long long work;

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        int s = (int)(work / field_stride);
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ;
        long long k = rem % NZ;
        float lhx = lambda_hx[work];
        float lhy = lambda_hy[work];
        float lhz = lambda_hz[work];
        float ez_x = 0.0f, ez_y = 0.0f, ey_x = 0.0f, ey_z = 0.0f, ex_y = 0.0f, ex_z = 0.0f;

        if (has_cpml) {
            float upd = update[idx];
            long long p_idx;
#define H_PML(R, P, p, q, spacing, field, channel, sign) \
            do { \
                (channel) += pml_adjoint_weight( \
                    (field), 1.0f / (spacing), upd, (sign), (R)[q] - 1.0f, \
                    (R)[(p) + (q)], (R)[2 * (p) + (q)], (R)[3 * (p) + (q)], &(P)[p_idx]); \
            } while (0)

            if (pml0 > 0 && i < pml0) {
                long long q = pml0 - 1 - i;
                if (k < NZ - 1) {
                    p_idx = ((long long)s * pml0 * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
                    H_PML(x0R, x0P1, pml0, q, dx, lhy, ez_x, 1.0f);
                }
                if (j < NY - 1) {
                    p_idx = ((long long)s * pml0 * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
                    H_PML(x0R, x0P2, pml0, q, dx, lhz, ey_x, -1.0f);
                }
            }
            if (pml1 > 0 && i >= NX - 1 - pml1 && i < NX - 1) {
                long long q = i - (NX - 1 - pml1);
                if (k < NZ - 1) {
                    p_idx = ((long long)s * pml1 * NY * (NZ - 1)) + q * NY * (NZ - 1) + j * (NZ - 1) + k;
                    H_PML(xmR, xmP1, pml1, q, dx, lhy, ez_x, 1.0f);
                }
                if (j < NY - 1) {
                    p_idx = ((long long)s * pml1 * (NY - 1) * NZ) + q * (NY - 1) * NZ + j * NZ + k;
                    H_PML(xmR, xmP2, pml1, q, dx, lhz, ey_x, -1.0f);
                }
            }
            if (pml2 > 0 && j < pml2) {
                long long q = pml2 - 1 - j;
                if (k < NZ - 1) {
                    p_idx = ((long long)s * NX * pml2 * (NZ - 1)) + i * pml2 * (NZ - 1) + q * (NZ - 1) + k;
                    H_PML(y0R, y0P1, pml2, q, dy, lhx, ez_y, -1.0f);
                }
                if (i < NX - 1) {
                    p_idx = ((long long)s * (NX - 1) * pml2 * NZ) + i * pml2 * NZ + q * NZ + k;
                    H_PML(y0R, y0P2, pml2, q, dy, lhz, ex_y, 1.0f);
                }
            }
            if (pml3 > 0 && j >= NY - 1 - pml3 && j < NY - 1) {
                long long q = j - (NY - 1 - pml3);
                if (k < NZ - 1) {
                    p_idx = ((long long)s * NX * pml3 * (NZ - 1)) + i * pml3 * (NZ - 1) + q * (NZ - 1) + k;
                    H_PML(ymR, ymP1, pml3, q, dy, lhx, ez_y, -1.0f);
                }
                if (i < NX - 1) {
                    p_idx = ((long long)s * (NX - 1) * pml3 * NZ) + i * pml3 * NZ + q * NZ + k;
                    H_PML(ymR, ymP2, pml3, q, dy, lhz, ex_y, 1.0f);
                }
            }
            if (pml4 > 0 && k < pml4) {
                long long q = pml4 - 1 - k;
                if (j < NY - 1) {
                    p_idx = ((long long)s * NX * (NY - 1) * pml4) + i * (NY - 1) * pml4 + j * pml4 + q;
                    H_PML(z0R, z0P1, pml4, q, dz, lhx, ey_z, 1.0f);
                }
                if (i < NX - 1) {
                    p_idx = ((long long)s * (NX - 1) * NY * pml4) + i * NY * pml4 + j * pml4 + q;
                    H_PML(z0R, z0P2, pml4, q, dz, lhy, ex_z, -1.0f);
                }
            }
            if (pml5 > 0 && k >= NZ - 1 - pml5 && k < NZ - 1) {
                long long q = k - (NZ - 1 - pml5);
                if (j < NY - 1) {
                    p_idx = ((long long)s * NX * (NY - 1) * pml5) + i * (NY - 1) * pml5 + j * pml5 + q;
                    H_PML(zmR, zmP1, pml5, q, dz, lhx, ey_z, 1.0f);
                }
                if (i < NX - 1) {
                    p_idx = ((long long)s * (NX - 1) * NY * pml5) + i * NY * pml5 + j * pml5 + q;
                    H_PML(zmR, zmP2, pml5, q, dz, lhy, ex_z, -1.0f);
                }
            }
#undef H_PML
        }

        {
            int do_hx = ((NX - 1) != 1 && i > 0 && i < NX - 1 && j < NY - 1 && k < NZ - 1);
            int do_hy = ((NY - 1) != 1 && i < NX - 1 && j > 0 && j < NY - 1 && k < NZ - 1);
            int do_hz = ((NZ - 1) != 1 && i < NX - 1 && j < NY - 1 && k > 0 && k < NZ - 1);
            float coeff = ch_curl[idx];
            float coeff_y = dy == dx ? coeff : coeff * dx / dy;
            float coeff_z = dz == dx ? coeff : coeff * dx / dz;

            if (do_hx) {
                ez_y += -coeff_y * lhx;
                ey_z += coeff_z * lhx;
                lambda_hx[work] = ch_hist[idx] * lhx;
            }
            if (do_hy) {
                ex_z += -coeff_z * lhy;
                ez_x += coeff * lhy;
                lambda_hy[work] = ch_hist[idx] * lhy;
            }
            if (do_hz) {
                ey_x += -coeff * lhz;
                ex_y += coeff_y * lhz;
                lambda_hz[work] = ch_hist[idx] * lhz;
            }
        }

        w_a_x[work] = ez_x;
        w_a_y[work] = ez_y;
        w_b_x[work] = ey_x;
        w_b_z[work] = ey_z;
        w_c_y[work] = ex_y;
        w_c_z[work] = ex_z;
    }

    DEEPGPR_OMP_PARALLEL_FOR
    for (work = 0; work < total_work; ++work) {
        long long idx = work % field_stride;
        long long i = idx / ny_nz;
        long long rem = idx % ny_nz;
        long long j = rem / NZ;
        long long k = rem % NZ;

        lambda_ez[work] += gather_forward_transpose(w_a_x, work, ny_nz, i, NX, order);
        lambda_ez[work] += gather_forward_transpose(w_a_y, work, NZ, j, NY, order);
        lambda_ey[work] += gather_forward_transpose(w_b_x, work, ny_nz, i, NX, order);
        lambda_ey[work] += gather_forward_transpose(w_b_z, work, 1, k, NZ, order);
        lambda_ex[work] += gather_forward_transpose(w_c_y, work, NZ, j, NY, order);
        lambda_ex[work] += gather_forward_transpose(w_c_z, work, 1, k, NZ, order);
    }
}

/*
 * Adjoint of receiver sampling: add the receiver-data cotangent of the
 * current time step to the sampled electric-field component.
 *
 * Shots run in parallel; the receivers of one shot are applied sequentially
 * in index order, so coincident receivers accumulate deterministically.
 *
 * Parameters:
 *   step: Number of shots or simulations in the batch.
 *   iteration: Current reverse time-step index.
 *   sourcelocation: Receiver coordinates with shape (step, nsr, 3).
 *   srcwaveforms: Receiver-data cotangent with shape (step, iterations, nsr).
 *   lambda_ex, lambda_ey, lambda_ez: Electric field component arrays to update.
 *   NX, NY, NZ: Padded field grid sizes.
 *   nsr: Number of receivers per shot.
 *   polarisation: Sampled component, 0 for x, 1 for y, 2 for z.
 *   iterations: Total number of time steps.
 */
static void adjoint_receivers_cpu(
    int step, int iteration,
    const int* RESTRICT sourcelocation, const float* RESTRICT srcwaveforms,
    float* RESTRICT lambda_ex, float* RESTRICT lambda_ey, float* RESTRICT lambda_ez,
    int NX, int NY, int NZ, int nsr, int polarisation, int iterations)
{
    long long field_stride = (long long)NX * NY * NZ;
    float* RESTRICT field = polarisation == 0 ? lambda_ex : (polarisation == 1 ? lambda_ey : lambda_ez);
    long long shot;

    if (polarisation < 0 || polarisation > 2) return;

    DEEPGPR_OMP_PARALLEL_FOR
    for (shot = 0; shot < step; ++shot) {
        for (long long src = 0; src < nsr; ++src) {
            long long index = shot * iterations * nsr + (long long)iteration * nsr + src;
            const int* location = sourcelocation + (shot * nsr + src) * 3;
            long long id4 = shot * field_stride + (long long)location[0] * NY * NZ
                + (long long)location[1] * NZ + location[2];
            field[id4] += srcwaveforms[index];
        }
    }
}

/*
 * Accumulate the transpose of the forward source injection into the source
 * waveform gradient. Sources run in parallel; the shots contributing to one
 * waveform sample are summed sequentially in shot order (deterministic).
 */
static void adjoint_source_injection_cpu(
    int step, int iteration, float dx, float dy, float dz,
    const int* RESTRICT source_location,
    const float* RESTRICT lambda_ex, const float* RESTRICT lambda_ey,
    const float* RESTRICT lambda_ez, const float* RESTRICT ce_rhs,
    int NX, int NY, int NZ, int nsrc, int source_component, int nt,
    float* RESTRICT grad_source)
{
    long long field_stride = (long long)NX * NY * NZ;
    float dipole_length = source_component == 0 ? dx : (source_component == 1 ? dy : dz);
    float geometric_scale = dipole_length / (dx * dy * dz);
    const float* RESTRICT lambda_e = source_component == 0 ? lambda_ex
        : (source_component == 1 ? lambda_ey : lambda_ez);
    long long src;

    DEEPGPR_OMP_PARALLEL_FOR
    for (src = 0; src < nsrc; ++src) {
        for (long long shot = 0; shot < step; ++shot) {
            const int* location = source_location + (shot * nsrc + src) * 3;
            long long material_idx = (long long)location[0] * NY * NZ
                + (long long)location[1] * NZ + location[2];
            long long field_idx = shot * field_stride + material_idx;
            float value = -ce_rhs[material_idx] * geometric_scale * lambda_e[field_idx];
            grad_source[src * nt + iteration] += value;
        }
    }
}

/*
 * Save the effective electric-field right-hand side R from
 * E^(n+1) = ca E^n + cb R. Eold_ptr contains E^n for this saved step.
 * This definition includes CPML corrections and the material-dependent
 * source injection, so the cb gradient follows the executed discrete scheme.
 */
static void save_rhs_snapshot_cpu(
    void* RESTRICT dst_ptr, int t_idx,
    const float* RESTRICT E, const void* RESTRICT Eold_ptr,
    const float* RESTRICT exact_Eold,
    const float* RESTRICT ca, const float* RESTRICT cb,
    int step, int NX, int NY, int NZ, int storage_type)
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

        float e_old = exact_Eold != NULL
            ? exact_Eold[work]
            : load_wavefield_value(Eold_ptr, saved_idx, storage_type);
        float rhs = cb_value != 0.0f
            ? (E[field_idx] - ca[i * NY * NZ + j * NZ + k] * e_old) / cb_value
            : 0.0f;
        store_wavefield_value(dst_ptr, saved_idx, rhs, storage_type);
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
static void save_e_snapshot_cpu(
    void* RESTRICT dst_ptr, int t_idx, const float* RESTRICT E,
    float* RESTRICT exact_Eold,
    int step, int NX, int NY, int NZ, int storage_type)
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
        float value = E[src_idx];
        store_wavefield_value(dst_ptr, dst_idx, value, storage_type);
        if (exact_Eold != NULL) exact_Eold[work] = value;
    }
}

/*
 * Accumulate model gradients from saved forward fields and adjoint fields.
 *
 * Parameters:
 *   lambda_ex, lambda_ey, lambda_ez: Adjoint electric field component arrays.
 *   E_saved: Saved pre-update electric field E^n.
 *   R_saved: Saved effective right-hand side R^n.
 *   ca, cb: Discrete electric update coefficients.
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
 *   fwi_mode: Gradient mode; 2 uses lambda_ez only, 3 uses lambda_ex, lambda_ey, and lambda_ez.
 */
static void accumulate_material_gradients_cpu(
    const float* RESTRICT lambda_ex, const float* RESTRICT lambda_ey, const float* RESTRICT lambda_ez,
    const void* RESTRICT E_saved, const void* RESTRICT R_saved,
    const float* RESTRICT ca, const float* RESTRICT cb,
    const float* RESTRICT sigma_pad,
    float* RESTRICT grad_eps_r, float* RESTRICT grad_sigma,
    int i, int step, int NX, int NY, int NZ,
    int pml0, int pml1, int pml2, int pml3, int pml4, int pml5,
    float dt,
    int eps_r_requires_grad, int sigma_requires_grad, int S, int sample_weight,
    int nt_saved, int fwi_mode,
    int storage_type)
{
    long long sx = NX - 1, sy = NY - 1, sz = NZ - 1;
    long long total_cells = sx * sy * sz;
    long long snap_stride = (long long)step * total_cells;
    long long component_stride = (long long)nt_saved * snap_stride;
    long long idx;

    DEEPGPR_OMP_PARALLEL_FOR
    for (idx = 0; idx < total_cells; ++idx) {
        long long ix = idx / (sy * sz);
        long long rem = idx % (sy * sz);
        long long iy = rem / sz;
        long long iz = rem % sz;

        /* CPML is a numerical boundary, not part of the invertible model. */
        if ((pml0 > 0 && ix < pml0) ||
            (pml1 > 0 && ix >= sx - pml1) ||
            (pml2 > 0 && iy < pml2) ||
            (pml3 > 0 && iy >= sy - pml3) ||
            (pml4 > 0 && iz < pml4) ||
            (pml5 > 0 && iz >= sz - pml5)) {
            continue;
        }

        long long material_idx = ix * NY * NZ + iy * NZ + iz;
        float local_grader = 0.0f;
        float local_gradse = 0.0f;
        long long e_stride = (long long)NX * NY * NZ;
        int components = (fwi_mode == 3) ? 3 : 1;

        for (int s = 0; s < step; ++s) {
            long long idx_E = (long long)s * e_stride + material_idx;
            long long base_idx = (long long)s * total_cells + idx;
            float adjoint_values[3] = {
                lambda_ex[idx_E], lambda_ey[idx_E], lambda_ez[idx_E]
            };

            for (int c = 0; c < components; ++c) {
                long long comp_offset = (fwi_mode == 3) ? (long long)c * component_stride : 0;
                long long saved_idx = comp_offset + (long long)(i / S) * snap_stride + base_idx;
                float e_old = load_wavefield_value(E_saved, saved_idx, storage_type);
                float rhs = load_wavefield_value(R_saved, saved_idx, storage_type);
                float adjoint_val = adjoint_values[(fwi_mode == 3) ? c : 2];
                float grad_ca = adjoint_val * e_old * (float)sample_weight;
                float grad_cb = adjoint_val * rhs * (float)sample_weight;
                float ca_value = ca[material_idx];
                float cb_value = cb[material_idx];

                if (sigma_pad[material_idx] <= DEEPGPR_PEC_SIGMA_THRESHOLD_F) {
                    if (eps_r_requires_grad == 1) {
                        float dca_der = E0 * (1.0f - ca_value) * cb_value / dt;
                        float dcb_der = -E0 * cb_value * cb_value / dt;
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

        if (eps_r_requires_grad == 1) {
            grad_eps_r[idx] += local_grader;
        }
        if (sigma_requires_grad == 1) {
            grad_sigma[idx] += local_gradse;
        }
    }
}

/*
 * Run CPU forward FDTD modeling.
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
DEEPGPR_API void forward(const float* RESTRICT eps_r_pad, const float* RESTRICT sigma_pad, const float* RESTRICT mu_r_pad,
             void* RESTRICT E_saved, void* RESTRICT R_saved,
             float* RESTRICT Ex, float* RESTRICT Ey, float* RESTRICT Ez,
             float* RESTRICT Hx, float* RESTRICT Hy, float* RESTRICT Hz,
             float* RESTRICT ce_hist, float* RESTRICT ce_curl, float* RESTRICT ce_rhs,
             float* RESTRICT ch_hist, float* RESTRICT ch_curl, float* RESTRICT ch_rhs,

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

             float dt, int nt, int step, int nrx, float dx, float dy, float dz,
             const int* RESTRICT receiverlocation, float* RESTRICT rxs, int receiver_component,

             int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS, int nsrc,
             const int* RESTRICT sourcelocation, const float* RESTRICT srcwaveforms, int polarisation,
             int sampling_interval, int fwi_mode, int storage_type, int save_model_history,
             int use_async_offload)
{
    (void)use_async_offload;
    int save_wavefield_history = (storage_type & WAVEFIELD_HISTORY_DISABLED) == 0;
    int fdtd_order = g_fdtd_order;
    int e_components = (fwi_mode == 3) ? 3 : 1;
    int has_cpml = pml0 || pml1 || pml2 || pml3 || pml4 || pml5;
    int nt_saved = (nt + sampling_interval - 1) / sampling_interval;
    long long snap_size = (long long)step * (NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1);
    long long component_stride = (long long)nt_saved * snap_size;
    float* exact_Eold = NULL;

    deepgpr_clear_last_error();
    if (save_wavefield_history && save_model_history && storage_type != WAVEFIELD_FLOAT32) {
        size_t bytes;
        if (snap_size < 0 || (unsigned long long)snap_size > SIZE_MAX / ((size_t)e_components * sizeof(float))) {
            deepgpr_set_error("deepgpr_cpu forward: snapshot size overflows the address space");
            return;
        }
        bytes = (size_t)e_components * (size_t)snap_size * sizeof(float);
        exact_Eold = (float*)malloc(bytes);
        if (exact_Eold == NULL) {
            /* Previously this silently degraded the R-history accuracy. */
            deepgpr_set_error(
                "deepgpr_cpu forward: could not allocate %llu bytes for the exact "
                "E^n snapshot used with low-precision wavefield storage",
                (unsigned long long)bytes);
            return;
        }
    }

    build_update_coeffs_cpu(eps_r_pad, sigma_pad, mu_r_pad, ce_hist, ce_curl, ce_rhs, ch_hist, ch_curl, ch_rhs, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);

    DEEPGPR_OMP_PARALLEL
    {
    for (int i = 0; i < nt; i++) {
        if (save_wavefield_history && i % sampling_interval == 0) {
            int t_saved = i / sampling_interval;
            if (fwi_mode == 3) {
                save_e_snapshot_cpu(E_saved, t_saved, Ex, exact_Eold, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, storage_type);
                save_e_snapshot_cpu(wavefield_offset(E_saved, component_stride, storage_type), t_saved, Ey,
                    exact_Eold != NULL ? exact_Eold + snap_size : NULL, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, storage_type);
                save_e_snapshot_cpu(wavefield_offset(E_saved, 2 * component_stride, storage_type), t_saved, Ez,
                    exact_Eold != NULL ? exact_Eold + 2 * snap_size : NULL, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, storage_type);
            } else {
                save_e_snapshot_cpu(E_saved, t_saved, Ez, exact_Eold, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, storage_type);
            }
        }

        update_h_cpu(ch_hist, ch_curl, Ex, Ey, Ez, Hx, Hy, Hz,
            step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dx, dy, dz, fdtd_order);
        if (has_cpml) {
            cpml_h_cpu(
                Ex, Ey, Ez, Hx, Hy, Hz, dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, ch_rhs,
                x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2,
                fdtd_order);
        }

        update_e_cpu(ce_hist, ce_curl, Ex, Ey, Ez, Hx, Hy, Hz,
            step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dx, dy, dz, fdtd_order);
        if (has_cpml) {
            cpml_e_cpu(
                Ex, Ey, Ez, Hx, Hy, Hz, dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, ce_rhs,
                x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2,
                fdtd_order);
        }

        inject_sources_cpu(step, i, dx, dy, dz, sourcelocation, srcwaveforms,
            Ex, Ey, Ez, ce_rhs, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nsrc, polarisation, nt);

        if (save_wavefield_history && save_model_history && i % sampling_interval == 0) {
            int t_saved = i / sampling_interval;
            if (fwi_mode == 3) {
                save_rhs_snapshot_cpu(R_saved, t_saved, Ex, E_saved, exact_Eold, ce_hist, ce_rhs,
                    step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, storage_type);
                save_rhs_snapshot_cpu(wavefield_offset(R_saved, component_stride, storage_type), t_saved, Ey,
                    wavefield_const_offset(E_saved, component_stride, storage_type), exact_Eold != NULL ? exact_Eold + snap_size : NULL,
                    ce_hist, ce_rhs, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, storage_type);
                save_rhs_snapshot_cpu(wavefield_offset(R_saved, 2 * component_stride, storage_type), t_saved, Ez,
                    wavefield_const_offset(E_saved, 2 * component_stride, storage_type), exact_Eold != NULL ? exact_Eold + 2 * snap_size : NULL,
                    ce_hist, ce_rhs, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, storage_type);
            } else {
                save_rhs_snapshot_cpu(R_saved, t_saved, Ez, E_saved, exact_Eold, ce_hist, ce_rhs,
                    step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, storage_type);
            }
        }

        sample_receivers_cpu(step, nrx, i, receiverlocation, rxs, Ex, Ey, Ez, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nt, receiver_component);
    }
    }

    free(exact_Eold);
}

/*
 * Run CPU adjoint FDTD modeling and accumulate model gradients.
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
DEEPGPR_API void backward(const float* RESTRICT eps_r_pad, const float* RESTRICT sigma_pad, const float* RESTRICT mu_r_pad,
             const void* RESTRICT E_saved, const void* RESTRICT R_saved,
             float* RESTRICT lambda_ex, float* RESTRICT lambda_ey, float* RESTRICT lambda_ez,
             float* RESTRICT lambda_hx, float* RESTRICT lambda_hy, float* RESTRICT lambda_hz,
             float* RESTRICT ce_hist, float* RESTRICT ce_curl, float* RESTRICT ce_rhs,
             float* RESTRICT ch_hist, float* RESTRICT ch_curl, float* RESTRICT ch_rhs,

            float* RESTRICT x0EPhi1,float* RESTRICT x0EPhi2, float* RESTRICT x0HPhi1,float* RESTRICT x0HPhi2,
            float* RESTRICT xmEPhi1,float* RESTRICT xmEPhi2, float* RESTRICT xmHPhi1,float* RESTRICT xmHPhi2,
            float* RESTRICT y0EPhi1,float* RESTRICT y0EPhi2, float* RESTRICT y0HPhi1,float* RESTRICT y0HPhi2,
            float* RESTRICT ymEPhi1,float* RESTRICT ymEPhi2, float* RESTRICT ymHPhi1,float* RESTRICT ymHPhi2,
            float* RESTRICT z0EPhi1,float* RESTRICT z0EPhi2, float* RESTRICT z0HPhi1,float* RESTRICT z0HPhi2,
            float* RESTRICT zmEPhi1,float* RESTRICT zmEPhi2, float* RESTRICT zmHPhi1,float* RESTRICT zmHPhi2,

            int pml0,int pml1,int pml2,int pml3,int pml4,int pml5,

            const float* RESTRICT x0ER,const float* RESTRICT xmER,
            const float* RESTRICT y0ER,const float* RESTRICT ymER,
            const float* RESTRICT z0ER,const float* RESTRICT zmER,
            const float* RESTRICT x0HR,const float* RESTRICT xmHR,
            const float* RESTRICT y0HR,const float* RESTRICT ymHR,
            const float* RESTRICT z0HR,const float* RESTRICT zmHR,

             float dt, int nt, int step, int nrx, float dx, float dy, float dz,
             int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
             int ndata_source, const int* RESTRICT receiver_location, const float* RESTRICT data_grad,
             int receiver_component,
             int nsource, const int* RESTRICT source_location,
             int source_component, float* RESTRICT grad_source, int source_requires_grad,
             float* RESTRICT grad_eps_r,float* RESTRICT grad_sigma, int eps_r_requires_grad, int sigma_requires_grad,
             int sampling_interval, int fwi_mode, int storage_type, int use_async_offload)
{
    (void)nrx;
    (void)use_async_offload;

    int fdtd_order = g_fdtd_order;
    int nt_saved = (nt + sampling_interval - 1) / sampling_interval;
    int has_cpml = pml0 || pml1 || pml2 || pml3 || pml4 || pml5;
    long long total_work = (long long)step * NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    size_t weight_bytes;
    float* adjoint_weights;

    deepgpr_clear_last_error();
    if (total_work < 0 || (unsigned long long)total_work > SIZE_MAX / (ADJOINT_CHANNELS * sizeof(float))) {
        deepgpr_set_error("deepgpr_cpu backward: field size overflows the address space");
        return;
    }
    weight_bytes = (size_t)ADJOINT_CHANNELS * (size_t)total_work * sizeof(float);
    adjoint_weights = (float*)malloc(weight_bytes);
    if (adjoint_weights == NULL) {
        deepgpr_set_error(
            "deepgpr_cpu backward: could not allocate %llu bytes for the adjoint "
            "stencil weights", (unsigned long long)weight_bytes);
        return;
    }

    build_update_coeffs_cpu(eps_r_pad, sigma_pad, mu_r_pad, ce_hist, ce_curl, ce_rhs, ch_hist, ch_curl, ch_rhs, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);

    DEEPGPR_OMP_PARALLEL
    {
    for (int i = nt - 1; i >= 0; i--) {
        /* receiver sampling^T */
        adjoint_receivers_cpu(step, i, receiver_location, data_grad, lambda_ex, lambda_ey, lambda_ez,
            NX_FIELDS, NY_FIELDS, NZ_FIELDS, ndata_source, receiver_component, nt);

        /* source injection^T: identity on the state plus the waveform gradient */
        if (source_requires_grad == 1) {
            adjoint_source_injection_cpu(
                step, i, dx, dy, dz, source_location,
                lambda_ex, lambda_ey, lambda_ez, ce_rhs,
                NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                nsource, source_component, nt, grad_source);
        }

        if ((eps_r_requires_grad || sigma_requires_grad) && i % sampling_interval == 0) {
            int sample_weight = sampling_interval;
            if (i + sample_weight > nt) sample_weight = nt - i;
            accumulate_material_gradients_cpu(lambda_ex, lambda_ey, lambda_ez, E_saved, R_saved, ce_hist, ce_rhs, sigma_pad,
                grad_eps_r, grad_sigma, i, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                pml0, pml1, pml2, pml3, pml4, pml5, dt,
                eps_r_requires_grad, sigma_requires_grad, sampling_interval, sample_weight,
                nt_saved, fwi_mode, storage_type);
        }

        /* Strict reverse-mode order for the executed forward time step:
         * (E CPML + E update)^T, then (H CPML + H update)^T. */
        adjoint_e_step_cpu(ce_hist, ce_curl, ce_rhs,
            lambda_ex, lambda_ey, lambda_ez, lambda_hx, lambda_hy, lambda_hz,
            dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            has_cpml, pml0, pml1, pml2, pml3, pml4, pml5,
            x0ER, xmER, y0ER, ymER, z0ER, zmER,
            x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2,
            z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2, fdtd_order, adjoint_weights);
        adjoint_h_step_cpu(ch_hist, ch_curl, ch_rhs,
            lambda_ex, lambda_ey, lambda_ez, lambda_hx, lambda_hy, lambda_hz,
            dx, dy, dz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            has_cpml, pml0, pml1, pml2, pml3, pml4, pml5,
            x0HR, xmHR, y0HR, ymHR, z0HR, zmHR,
            x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2,
            z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2, fdtd_order, adjoint_weights);
    }
    }

    free(adjoint_weights);
}

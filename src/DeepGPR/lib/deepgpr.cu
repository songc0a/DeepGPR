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

#define CUDA_CHECK() {\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}

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

__global__ void ucgetbackward(const float* __restrict__ er, const float* __restrict__ se, const float* __restrict__ mr,
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
            float EA = (e0 * er[idx] / dt) + 0.5f * se[idx];
            uE0[idx] = (2.0f * e0 * er[idx]) / (2.0f * e0 * er[idx] + se[idx] * dt);
            uE1[idx] = (1.0f / dx) / EA;
            uE4[idx] = 1.0f / EA;
        }
    }
}


__global__ void store_outputs(
    int step, int NRX, int iteration,
    const int* __restrict__ receiverlocation, float* __restrict__ rxs,
    const float* __restrict__ Ex, const float* __restrict__ Ey, const float* __restrict__ Ez, 
    const float* __restrict__ Hx, const float* __restrict__ Hy, const float* __restrict__ Hz,
    int NX, int NY, int NZ, int N_ITER) 
{
    long long rx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rx >= NRX) return;

    long long field_stride = (long long)NX * NY * NZ;

    for (int s = 0; s < step; ++s) {
        long long i = receiverlocation[s * NRX * 3 + rx * 3 + 0];
        long long j = receiverlocation[s * NRX * 3 + rx * 3 + 1];
        long long k = receiverlocation[s * NRX * 3 + rx * 3 + 2];

        long long id4 = s * field_stride + i * NY * NZ + j * NZ + k;

        rxs[((s * 6 + 0) * N_ITER + iteration) * NRX + rx] = Ex[id4];
        rxs[((s * 6 + 1) * N_ITER + iteration) * NRX + rx] = Ey[id4];
        rxs[((s * 6 + 2) * N_ITER + iteration) * NRX + rx] = Ez[id4];
        rxs[((s * 6 + 3) * N_ITER + iteration) * NRX + rx] = Hx[id4];
        rxs[((s * 6 + 4) * N_ITER + iteration) * NRX + rx] = Hy[id4];
        rxs[((s * 6 + 5) * N_ITER + iteration) * NRX + rx] = Hz[id4];
    }
}


__global__ void Update_hertzian_dipole(
    int step, int iteration, float dx, 
    const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms,
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez, const float* __restrict__ uE4,
    int NX, int NY, int NZ, int nsrc, int polarisation, int nt) 
{
    long long src = blockIdx.x * blockDim.x + threadIdx.x; 
    if (src >= nsrc) return;

    float waveform_value = srcwaveforms[src * nt + iteration];
    float scale = waveform_value * dx / (dx * dx * dx);  
    long long field_stride = (long long)NX * NY * NZ;

    for (int s = 0; s < step; ++s) {
        long long i = sourcelocation[s * nsrc * 3 + src * 3 + 0];
        long long j = sourcelocation[s * nsrc * 3 + src * 3 + 1];
        long long k = sourcelocation[s * nsrc * 3 + src * 3 + 2];

        long long id3 = i * NY * NZ + j * NZ + k;
        long long id4 = s * field_stride + id3;

        if (polarisation == 0) Ex[id4] -= uE4[id3] * scale;
        else if (polarisation == 1) Ey[id4] -= uE4[id3] * scale;
        else if (polarisation == 2) Ez[id4] -= uE4[id3] * scale;
    }
}


__global__ void fused_e_fields_updates_gpu(
    const float* __restrict__ uE0, const float* __restrict__ uE1,  
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
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long field_stride = (long long)NX_FIELDS * ny_nz;
    if (idx >= field_stride) return;

    long long i = idx / ny_nz;
    long long rem = idx % ny_nz;
    long long j = rem / NZ_FIELDS;
    long long k = rem % NZ_FIELDS;

    bool do_ex = (((NY_FIELDS-1) != 1 || (NZ_FIELDS-1) != 1) && i < (NX_FIELDS-1) && j > 0 && j < (NY_FIELDS-1) && k > 0 && k < (NZ_FIELDS-1));
    bool do_ey = (((NX_FIELDS-1) != 1 || (NZ_FIELDS-1) != 1) && i > 0 && i < (NX_FIELDS-1) && j < (NY_FIELDS-1) && k > 0 && k < (NZ_FIELDS-1));
    bool do_ez = (((NX_FIELDS-1) != 1 || (NY_FIELDS-1) != 1) && i > 0 && i < (NX_FIELDS-1) && j > 0 && j < (NY_FIELDS-1) && k < (NZ_FIELDS-1));

    bool in_x0 = (pml0 > 0 && i <= pml0 && j < NY_FIELDS && k < NZ_FIELDS);
    bool in_xm = (pml1 > 0 && i >= NX_FIELDS - 1 - pml1 && i < NX_FIELDS && j < NY_FIELDS && k < NZ_FIELDS);
    bool in_y0 = (pml2 > 0 && i < NX_FIELDS && j <= pml2 && k < NZ_FIELDS);
    bool in_ym = (pml3 > 0 && i < NX_FIELDS && j >= NY_FIELDS - 1 - pml3 && j < NY_FIELDS && k < NZ_FIELDS);
    bool in_z0 = (pml4 > 0 && i < NX_FIELDS && j < NY_FIELDS && k <= pml4);
    bool in_zm = (pml5 > 0 && i < NX_FIELDS && j < NY_FIELDS && k >= NZ_FIELDS - 1 - pml5 && k < NZ_FIELDS);

    float ue0 = uE0[idx];
    float ue1 = uE1[idx];
    float upd = updatecoeffsE[idx];

    long long id4 = idx; 

    for (int s = 0; s < step; ++s) {
        if (do_ex) Ex[id4] = ue0 * Ex[id4] + ue1 * (Hz[id4] - Hz[id4 - NZ_FIELDS]) - ue1 * (Hy[id4] - Hy[id4 - 1]);
        if (do_ey) Ey[id4] = ue0 * Ey[id4] + ue1 * (Hx[id4] - Hx[id4 - 1]) - ue1 * (Hz[id4] - Hz[id4 - ny_nz]);
        if (do_ez) Ez[id4] = ue0 * Ez[id4] + ue1 * (Hy[id4] - Hy[id4 - ny_nz]) - ue1 * (Hx[id4] - Hx[id4 - NZ_FIELDS]);

        if (in_x0) {
            long long i1 = pml0 - i;
            float RA01 = x0ER[i1] - 1.0f, RB0 = x0ER[pml0 + i1], RE0 = x0ER[2 * pml0 + i1], RF0 = x0ER[3 * pml0 + i1];
            if (j < NY_FIELDS - 1 && i > 0) {
                float dHz = (Hz[id4] - Hz[id4 - ny_nz]) / dx;
                long long p_idx = ((long long)s * (pml0+1) * (NY_FIELDS-1) * NZ_FIELDS) + i1 * (NY_FIELDS-1) * NZ_FIELDS + j * NZ_FIELDS + k;
                float phi = x0EPhi1[p_idx];
                Ey[id4] -= upd * (RA01 * dHz + RB0 * phi);
                x0EPhi1[p_idx] = RE0 * phi - RF0 * dHz;
            }
            if (k < NZ_FIELDS - 1 && i > 0) {
                float dHy = (Hy[id4] - Hy[id4 - ny_nz]) / dx;
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
                float dHz = (Hz[id4] - Hz[id4 - ny_nz]) / dx;
                long long p_idx = ((long long)s * (pml1+1) * (NY_FIELDS-1) * NZ_FIELDS) + i1 * (NY_FIELDS-1) * NZ_FIELDS + j * NZ_FIELDS + k;
                float phi = xmEPhi1[p_idx];
                Ey[id4] -= upd * (RA01 * dHz + RB0 * phi);
                xmEPhi1[p_idx] = RE0 * phi - RF0 * dHz;
            }
            if (k < NZ_FIELDS - 1 && i > 0) {
                float dHy = (Hy[id4] - Hy[id4 - ny_nz]) / dx;
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
                float dHz = (Hz[id4] - Hz[id4 - NZ_FIELDS]) / dy;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * (pml2+1) * NZ_FIELDS) + i * (pml2+1) * NZ_FIELDS + j1 * NZ_FIELDS + k;
                float phi = y0EPhi1[p_idx];
                Ex[id4] += upd * (RA01 * dHz + RB0 * phi);
                y0EPhi1[p_idx] = RE0 * phi - RF0 * dHz;
            }
            if (k < NZ_FIELDS - 1 && j > 0) {
                float dHx = (Hx[id4] - Hx[id4 - NZ_FIELDS]) / dy;
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
                float dHz = (Hz[id4] - Hz[id4 - NZ_FIELDS]) / dy;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * (pml3+1) * NZ_FIELDS) + i * (pml3+1) * NZ_FIELDS + j1 * NZ_FIELDS + k;
                float phi = ymEPhi1[p_idx];
                Ex[id4] += upd * (RA01 * dHz + RB0 * phi);
                ymEPhi1[p_idx] = RE0 * phi - RF0 * dHz;
            }
            if (k < NZ_FIELDS - 1 && j > 0) {
                float dHx = (Hx[id4] - Hx[id4 - NZ_FIELDS]) / dy;
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
                float dHy = (Hy[id4] - Hy[id4 - 1]) / dz;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * NY_FIELDS * (pml4+1)) + i * NY_FIELDS * (pml4+1) + j * (pml4+1) + k1;
                float phi = z0EPhi1[p_idx];
                Ex[id4] -= upd * (RA01 * dHy + RB0 * phi);
                z0EPhi1[p_idx] = RE0 * phi - RF0 * dHy;
            }
            if (j < NY_FIELDS - 1 && k > 0) {
                float dHx = (Hx[id4] - Hx[id4 - 1]) / dz;
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
                float dHy = (Hy[id4] - Hy[id4 - 1]) / dz;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * NY_FIELDS * (pml5+1)) + i * NY_FIELDS * (pml5+1) + j * (pml5+1) + k1;
                float phi = zmEPhi1[p_idx];
                Ex[id4] -= upd * (RA01 * dHy + RB0 * phi);
                zmEPhi1[p_idx] = RE0 * phi - RF0 * dHy;
            }
            if (j < NY_FIELDS - 1 && k > 0) {
                float dHx = (Hx[id4] - Hx[id4 - 1]) / dz;
                long long p_idx = ((long long)s * NX_FIELDS * (NY_FIELDS-1) * (pml5+1)) + i * (NY_FIELDS-1) * (pml5+1) + j * (pml5+1) + k1;
                float phi = zmEPhi2[p_idx];
                Ey[id4] += upd * (RA01 * dHx + RB0 * phi);
                zmEPhi2[p_idx] = RE0 * phi - RF0 * dHx;
            }
        }

        id4 += field_stride;
    }
}


__global__ void fused_h_fields_updates_gpu(
    const float* __restrict__ uH0, const float* __restrict__ uH1,
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
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long ny_nz = (long long)NY_FIELDS * NZ_FIELDS;
    long long field_stride = (long long)NX_FIELDS * ny_nz;
    if (idx >= field_stride) return;

    long long i = idx / ny_nz;
    long long rem = idx % ny_nz;
    long long j = rem / NZ_FIELDS;
    long long k = rem % NZ_FIELDS;

    bool do_hx = ((NX_FIELDS-1) != 1 && i > 0 && i < (NX_FIELDS-1) && j < (NY_FIELDS-1) && k < (NZ_FIELDS-1));
    bool do_hy = ((NY_FIELDS-1) != 1 && i < (NX_FIELDS-1) && j > 0 && j < (NY_FIELDS-1) && k < (NZ_FIELDS-1));
    bool do_hz = ((NZ_FIELDS-1) != 1 && i < (NX_FIELDS-1) && j < (NY_FIELDS-1) && k > 0 && k < (NZ_FIELDS-1));

    bool in_x0 = (pml0 > 0 && i < pml0 && j < NY_FIELDS && k < NZ_FIELDS);
    bool in_xm = (pml1 > 0 && i >= NX_FIELDS - 1 - pml1 && i < NX_FIELDS - 1 && j < NY_FIELDS && k < NZ_FIELDS);
    bool in_y0 = (pml2 > 0 && i < NX_FIELDS && j < pml2 && k < NZ_FIELDS);
    bool in_ym = (pml3 > 0 && i < NX_FIELDS && j >= NY_FIELDS - 1 - pml3 && j < NY_FIELDS - 1 && k < NZ_FIELDS);
    bool in_z0 = (pml4 > 0 && i < NX_FIELDS && j < NY_FIELDS && k < pml4);
    bool in_zm = (pml5 > 0 && i < NX_FIELDS && j < NY_FIELDS && k >= NZ_FIELDS - 1 - pml5 && k < NZ_FIELDS - 1);

    float uh0 = uH0[idx];
    float uh1 = uH1[idx];
    float upd = updatecoeffsH[idx];

    long long id4 = idx; 

    for (int s = 0; s < step; ++s) {
        if (do_hx) Hx[id4] = uh0 * Hx[id4] - uh1 * (Ez[id4 + NZ_FIELDS] - Ez[id4]) + uh1 * (Ey[id4 + 1] - Ey[id4]);
        if (do_hy) Hy[id4] = uh0 * Hy[id4] - uh1 * (Ex[id4 + 1] - Ex[id4]) + uh1 * (Ez[id4 + ny_nz] - Ez[id4]);
        if (do_hz) Hz[id4] = uh0 * Hz[id4] - uh1 * (Ey[id4 + ny_nz] - Ey[id4]) + uh1 * (Ex[id4 + NZ_FIELDS] - Ex[id4]);

        if (in_x0) {
            long long i1 = pml0 - 1 - i;
            float RA01 = x0HR[i1] - 1.0f, RB0 = x0HR[pml0 + i1], RE0 = x0HR[2 * pml0 + i1], RF0 = x0HR[3 * pml0 + i1];
            if (k < NZ_FIELDS - 1) {
                float dEz = (Ez[id4 + ny_nz] - Ez[id4]) / dx;
                long long p_idx = ((long long)s * pml0 * NY_FIELDS * (NZ_FIELDS-1)) + i1 * NY_FIELDS * (NZ_FIELDS-1) + j * (NZ_FIELDS-1) + k;
                float phi = x0HPhi1[p_idx];
                Hy[id4] += upd * (RA01 * dEz + RB0 * phi);
                x0HPhi1[p_idx] = RE0 * phi - RF0 * dEz;
            }
            if (j < NY_FIELDS - 1) {
                float dEy = (Ey[id4 + ny_nz] - Ey[id4]) / dx;
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
                float dEz = (Ez[id4 + ny_nz] - Ez[id4]) / dx;
                long long p_idx = ((long long)s * pml1 * NY_FIELDS * (NZ_FIELDS-1)) + i1 * NY_FIELDS * (NZ_FIELDS-1) + j * (NZ_FIELDS-1) + k;
                float phi = xmHPhi1[p_idx];
                Hy[id4] += upd * (RA01 * dEz + RB0 * phi);
                xmHPhi1[p_idx] = RE0 * phi - RF0 * dEz;
            }
            if (j < NY_FIELDS - 1) {
                float dEy = (Ey[id4 + ny_nz] - Ey[id4]) / dx;
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
                float dEz = (Ez[id4 + NZ_FIELDS] - Ez[id4]) / dy;
                long long p_idx = ((long long)s * NX_FIELDS * pml2 * (NZ_FIELDS-1)) + i * pml2 * (NZ_FIELDS-1) + j1 * (NZ_FIELDS-1) + k;
                float phi = y0HPhi1[p_idx];
                Hx[id4] -= upd * (RA01 * dEz + RB0 * phi);
                y0HPhi1[p_idx] = RE0 * phi - RF0 * dEz;
            }
            if (i < NX_FIELDS - 1 && k < NZ_FIELDS) {
                float dEx = (Ex[id4 + NZ_FIELDS] - Ex[id4]) / dy;
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
                float dEz = (Ez[id4 + NZ_FIELDS] - Ez[id4]) / dy;
                long long p_idx = ((long long)s * NX_FIELDS * pml3 * (NZ_FIELDS-1)) + i * pml3 * (NZ_FIELDS-1) + j1 * (NZ_FIELDS-1) + k;
                float phi = ymHPhi1[p_idx];
                Hx[id4] -= upd * (RA01 * dEz + RB0 * phi);
                ymHPhi1[p_idx] = RE0 * phi - RF0 * dEz;
            }
            if (i < NX_FIELDS - 1 && k < NZ_FIELDS) {
                float dEx = (Ex[id4 + NZ_FIELDS] - Ex[id4]) / dy;
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
                float dEy = (Ey[id4 + 1] - Ey[id4]) / dz;
                long long p_idx = ((long long)s * NX_FIELDS * (NY_FIELDS-1) * pml4) + i * (NY_FIELDS-1) * pml4 + j * pml4 + k1;
                float phi = z0HPhi1[p_idx];
                Hx[id4] += upd * (RA01 * dEy + RB0 * phi);
                z0HPhi1[p_idx] = RE0 * phi - RF0 * dEy;
            }
            if (i < NX_FIELDS - 1 && j < NY_FIELDS) {
                float dEx = (Ex[id4 + 1] - Ex[id4]) / dz;
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
                float dEy = (Ey[id4 + 1] - Ey[id4]) / dz;
                long long p_idx = ((long long)s * NX_FIELDS * (NY_FIELDS-1) * pml5) + i * (NY_FIELDS-1) * pml5 + j * pml5 + k1;
                float phi = zmHPhi1[p_idx];
                Hx[id4] += upd * (RA01 * dEy + RB0 * phi);
                zmHPhi1[p_idx] = RE0 * phi - RF0 * dEy;
            }
            if (i < NX_FIELDS - 1 && j < NY_FIELDS) {
                float dEx = (Ex[id4 + 1] - Ex[id4]) / dz;
                long long p_idx = ((long long)s * (NX_FIELDS-1) * NY_FIELDS * pml5) + i * NY_FIELDS * pml5 + j * pml5 + k1;
                float phi = zmHPhi2[p_idx];
                Hy[id4] -= upd * (RA01 * dEx + RB0 * phi);
                zmHPhi2[p_idx] = RE0 * phi - RF0 * dEx;
            }
        }

        id4 += field_stride;
    }
}


__global__ void Back_source(
    int step, int iteration, float dx,
    const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms,
    float* Ex, float* Ey, float* Ez, float* uE4,
    int NX, int NY, int NZ, int nsr, int polarisation, int iterations
){
    long long src = blockIdx.x * blockDim.x + threadIdx.x;   
    if (src >= nsr) return;
    long long field_stride = (long long)NX * NY * NZ;
    long long index_stride = (long long)iterations * nsr;
    long long index = (long long)iteration * nsr + src;

    for (int s = 0; s < step; ++s) {
        long long i = sourcelocation[s * nsr * 3 + src * 3 + 0];
        long long j = sourcelocation[s * nsr * 3 + src * 3 + 1];
        long long k = sourcelocation[s * nsr * 3 + src * 3 + 2];

        float waveform_value = srcwaveforms[index];
        long long id4 = s * field_stride + i * NY * NZ + j * NZ + k;

        if (polarisation == 0) Ex[id4] -= waveform_value;
        else if (polarisation == 1) Ey[id4] -= waveform_value;
        else if (polarisation == 2) Ez[id4] -= waveform_value;

        index += index_stride;
    }
}


__global__ void copy_to_Eall_single(
    float* __restrict__ dst_ptr, int t_idx, const float* __restrict__ E, 
    int step, int NX, int NY, int NZ)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long nx1 = NX - 1, ny1 = NY - 1, nz1 = NZ - 1;
    long long total = nx1 * ny1 * nz1;
    if (idx >= total) return;

    long long i = idx / (ny1 * nz1);
    long long rem = idx % (ny1 * nz1);
    long long j = rem / nz1;
    long long k = rem % nz1;

    long long src_idx = i * NY * NZ + j * NZ + k;
    long long dst_idx = (long long)t_idx * step * total + idx; 
    long long field_stride = (long long)NX * NY * NZ;

    for (int s = 0; s < step; ++s) {
        dst_ptr[dst_idx] = E[src_idx];
        src_idx += field_stride;
        dst_idx += total;
    }
}


__global__ void accumulate_gradients(
    const float* __restrict__ Ez, const float* __restrict__ Eall_ptr, const float* __restrict__ d_E_buf,
    float* __restrict__ grader, float* __restrict__ gradse,
    int i, int step, int NX, int NY, int NZ, float dt,int errequiregrad,int serequiregrad,
    int S, int nt_saved, int use_async_offload
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long sx = (NX - 1), sy = (NY - 1), sz = (NZ - 1);
    long long total_cells = sx * sy * sz;

    if (idx >= total_cells) return;

    long long ix = idx / (sy * sz);
    long long rem = idx % (sy * sz);
    long long iy = rem / sz;
    long long iz = rem % sz;

    long long idx_Ez = ix * NY * NZ + iy * NZ + iz;
    
    long long idx0_curr = i / S;
    long long idx1_curr = min(idx0_curr + 1, (long long)nt_saved - 1);
    float w1_curr = (float)(i % S) / S;
    float w0_curr = 1.0f - w1_curr;

    long long idx0_prev = (i - 1) / S;
    long long idx1_prev = min(idx0_prev + 1, (long long)nt_saved - 1);
    float w1_prev = (float)((i - 1) % S) / S;
    float w0_prev = 1.0f - w1_prev;

    long long ez_stride = (long long)NX * NY * NZ;
    float local_grader = 0.0f;
    float local_gradse = 0.0f;

    for (int s = 0; s < step; ++s) {
        long long base_idx = (long long)s * total_cells + idx;
        float e0_c, e1_c, e0_p, e1_p;

        if (use_async_offload) {
            e0_c = d_E_buf[(idx0_curr % 3) * step * total_cells + base_idx];
            e1_c = d_E_buf[(idx1_curr % 3) * step * total_cells + base_idx];
            e0_p = d_E_buf[(idx0_prev % 3) * step * total_cells + base_idx];
            e1_p = d_E_buf[(idx1_prev % 3) * step * total_cells + base_idx];
        } else {
            e0_c = Eall_ptr[idx0_curr * step * total_cells + base_idx];
            e1_c = Eall_ptr[idx1_curr * step * total_cells + base_idx];
            e0_p = Eall_ptr[idx0_prev * step * total_cells + base_idx];
            e1_p = Eall_ptr[idx1_prev * step * total_cells + base_idx];
        }

        float e_curr = e0_c * w0_curr + e1_c * w1_curr;
        float e_prev = e0_p * w0_prev + e1_p * w1_prev;

        float ez_val = Ez[idx_Ez];

        if (errequiregrad == 1) local_grader += (e_curr - e_prev) * ez_val / dt;
        if (serequiregrad == 1) local_gradse += e_curr * ez_val * dt;

        idx_Ez += ez_stride;
    }

    if (errequiregrad == 1) atomicAdd(&grader[idx], local_grader);
    if (serequiregrad == 1) atomicAdd(&gradse[idx], local_gradse);
}


DEEPGPR_API void forward(const float* __restrict__ er, const float* __restrict__ se, const float* __restrict__ mr,  
             float* __restrict__ Eall_ptr, 
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
             int sampling_interval) 
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

    float* d_E_buf = nullptr;
    long long snap_size = (long long)step * (NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1);
    
    cudaStream_t stream_comp = 0, stream_trans = 0;
    cudaEvent_t event_comp;
    if (use_async) {
        cudaStreamCreate(&stream_comp);
        cudaStreamCreate(&stream_trans);
        cudaEventCreate(&event_comp);
        cudaMalloc(&d_E_buf, 2 * snap_size * sizeof(float)); 
    }

    long long blockSize = 256;
    long long total_fields = (long long)NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    dim3 grid_fields(CEIL_DIV(total_fields, blockSize));

    ucgetforward<<<grid_fields, blockSize, 0, stream_comp>>>(er, se, mr, uE0, uE1, uE4, uH0, uH1, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);
  
    dim3 grid_rx(CEIL_DIV(nrx, blockSize));
    dim3 grid_src(CEIL_DIV(nsrc, blockSize));
    long long total_copy = (long long)(NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1); 
    dim3 grid_copy(CEIL_DIV(total_copy, blockSize));

    for (int i = 0; i < nt; i++) {
        long long rx_total = step * nrx;
        long long gridSize_rx = (rx_total + blockSize - 1) / blockSize;
        store_outputs<<<gridSize_rx, blockSize, 0, stream_comp>>>(step, nrx, i, receiverlocation, rxs, Ex, Ey, Ez, Hx, Hy, Hz, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nt);

        fused_h_fields_updates_gpu<<<grid_fields, blockSize, 0, stream_comp>>>(
            uH0, uH1, Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, uH4, 
            x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2);

        fused_e_fields_updates_gpu<<<grid_fields, blockSize, 0, stream_comp>>>(
            uE0, uE1, Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, uE4,
            x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2);

        Update_hertzian_dipole<<<grid_src, blockSize, 0, stream_comp>>>(step, i, dx, sourcelocation, srcwaveforms, Ex, Ey, Ez, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nsrc, polarisation, nt);
        
        if (i % sampling_interval == 0) {
            int t_saved = i / sampling_interval;
            if (use_async) {
                int buf_idx = t_saved % 2; 
                cudaStreamSynchronize(stream_trans);
                copy_to_Eall_single<<<grid_copy, blockSize, 0, stream_comp>>>(d_E_buf, buf_idx, Ez, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
                cudaEventRecord(event_comp, stream_comp);
                cudaStreamWaitEvent(stream_trans, event_comp, 0);
                cudaMemcpyAsync(Eall_ptr + t_saved * snap_size, d_E_buf + buf_idx * snap_size, snap_size * sizeof(float), cudaMemcpyDeviceToHost, stream_trans);
            } else {
                copy_to_Eall_single<<<grid_copy, blockSize, 0, stream_comp>>>(Eall_ptr, t_saved, Ez, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            }
        }
    }

    if (use_async) {
        cudaStreamSynchronize(stream_comp);
        cudaStreamSynchronize(stream_trans);
        cudaFree(d_E_buf); 
        cudaEventDestroy(event_comp);
        cudaStreamDestroy(stream_comp);
        cudaStreamDestroy(stream_trans);
    }
}

DEEPGPR_API void backward(const float* __restrict__ er, const float* __restrict__ se, const float* __restrict__ mr,  
             const float* __restrict__ Eall_ptr,
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
             int sampling_interval) 
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

    float* d_E_buf = nullptr;
    long long snap_size = (long long)step * (NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1);
    
    cudaStream_t stream_comp = 0, stream_trans = 0;
    cudaEvent_t event_trans;
    if (use_async) {
        cudaStreamCreate(&stream_comp);
        cudaStreamCreate(&stream_trans);
        cudaEventCreate(&event_trans);
        cudaMalloc(&d_E_buf, 3 * snap_size * sizeof(float)); 
    }

    long long blockSize = 256;
    long long total_fields = (long long)NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    dim3 grid_fields(CEIL_DIV(total_fields, blockSize));

    ucgetbackward<<<grid_fields, blockSize, 0, stream_comp>>>(er, se, mr, uE0, uE1, uE4, uH0, uH1, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx);

    long long total_src = step * nsrc;                           
    long long src_blocks = (total_src + blockSize - 1) / blockSize;        
    dim3 grid_src(src_blocks);

    long long total_grad = (long long)(NX_FIELDS-1) * (NY_FIELDS-1) * (NZ_FIELDS-1);
    dim3 grid_grad(CEIL_DIV(total_grad, blockSize));
  
    int nt_saved = (nt + sampling_interval - 1) / sampling_interval;

    int max_t_needed = (nt - 1) / sampling_interval;
    max_t_needed = min(max_t_needed + 1, nt_saved - 1);
    int lowest_t_loaded = max_t_needed - 2;

    if (use_async) {
        for(int k = 0; k < 3; k++) {
            int t_load = max_t_needed - k;
            if(t_load >= 0) {
                cudaMemcpyAsync(d_E_buf + (t_load % 3) * snap_size, Eall_ptr + t_load * snap_size, snap_size * sizeof(float), cudaMemcpyHostToDevice, stream_trans);
            }
        }
        cudaStreamSynchronize(stream_trans);
    }

    for (int i = nt-1; i > 0; i--) {
        if (use_async) {
            int needed_t_min = (i - 1) / sampling_interval;
            if (needed_t_min < lowest_t_loaded && needed_t_min >= 0) {
                cudaStreamSynchronize(stream_trans); 
                cudaMemcpyAsync(d_E_buf + (needed_t_min % 3) * snap_size, Eall_ptr + needed_t_min * snap_size, snap_size * sizeof(float), cudaMemcpyHostToDevice, stream_trans);
                lowest_t_loaded = needed_t_min;
            }
            cudaEventRecord(event_trans, stream_trans);
            cudaStreamWaitEvent(stream_comp, event_trans, 0);
        }

        Back_source<<<grid_src, blockSize, 0, stream_comp>>>(step, i, dx, sourcelocation, srcwaveforms, Ex, Ey, Ez, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, nsrc, polarisation, nt);
        
        fused_e_fields_updates_gpu<<<grid_fields, blockSize, 0, stream_comp>>>(
            uE0, uE1, Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0ER, xmER, y0ER, ymER, z0ER, zmER, uE4,
            x0EPhi1, x0EPhi2, xmEPhi1, xmEPhi2, y0EPhi1, y0EPhi2, ymEPhi1, ymEPhi2, z0EPhi1, z0EPhi2, zmEPhi1, zmEPhi2);

        fused_h_fields_updates_gpu<<<grid_fields, blockSize, 0, stream_comp>>>(
            uH0, uH1, Ex, Ey, Ez, Hx, Hy, Hz, dx, dx, dx, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS,
            pml0, pml1, pml2, pml3, pml4, pml5, x0HR, xmHR, y0HR, ymHR, z0HR, zmHR, uH4, 
            x0HPhi1, x0HPhi2, xmHPhi1, xmHPhi2, y0HPhi1, y0HPhi2, ymHPhi1, ymHPhi2, z0HPhi1, z0HPhi2, zmHPhi1, zmHPhi2);

        accumulate_gradients<<<grid_grad, blockSize, 0, stream_comp>>>(Ez, Eall_ptr, d_E_buf, grad_er, grad_se, i, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, errequiregrad, serequiregrad, sampling_interval, nt_saved, use_async);
    }

    if (use_async) {
        cudaStreamSynchronize(stream_comp);
        cudaStreamSynchronize(stream_trans);
        cudaFree(d_E_buf); 
        cudaEventDestroy(event_trans);
        cudaStreamDestroy(stream_comp);
        cudaStreamDestroy(stream_trans);
    }
}
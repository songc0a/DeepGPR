// Fused 3D nodal DGTD operator on tetrahedra and its exact transpose.
//
// Forward (one thread per shot and element):
//
//   dF   = b0 [F] + b1 (n x [G]) + b2 F- + b3 (n x G-) + X        (face nodes)
//   T_j  = sum_q R[a_j, q] (D_q F_{f_j}) + LIFT (fscale n_{a_j} dF_{f_j})
//
// for the six curl terms j = (component c, derivative axis a_j, field
// component f_j) of DeepGPR.hybrid.functional3d.TERMS. [u] = u+ - u- with
// u- = u[vmapM], u+ = u[vmapP]. b2 has one value for the x component and one
// for y/z (interface faces); X holds the projected Yee trace (y, z).
//
// Backward: kernel 1 forms, per element, the volume transpose, the lifted
// face-node cotangent h = fscale sum_j n_(a_j) e_(f_j) (LIFT^T g_j) (3 values per
// face node, slot-major) and the coefficient cotangents; kernel 2 (one thread per
// element) rebuilds the cotangents of F-, F+, G-, G+ from h and the per-face
// coefficients of the element itself and of the mirror face (mirror-slot map
// slotP) and gathers them onto the volume nodes. No atomics: deterministic.
//
// Layouts (row-major, contiguous): F, G (S, 3, K, NP); X compact (S, 2, nX) on the
// interface face slots, xmap (K * NF) int32 = position in X or -1;
// R (3, 3, K); per face (affine elements): nrm (3, K, 4); fscale, b0, b1, b3 (K, 4);
// b2 (2, K, 4); vmapM, vmapP, slotP (K * NF) int32, fmask (NF) int32;
// D* (NP, NP); LIFT (NP, NF); out (S, 6, K, NP). Backward coefficient cotangents
// per face (S, [2,] 4, K), X cotangent compact (S, 2, nX).

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __constant__ int kA[6] = {1, 2, 2, 0, 0, 1};
__device__ __constant__ int kF[6] = {2, 1, 0, 2, 1, 0};

template <typename T>
__device__ inline void cross3(const T* n, const T* v, T* out) {
    out[0] = n[1] * v[2] - n[2] * v[1];
    out[1] = n[2] * v[0] - n[0] * v[2];
    out[2] = n[0] * v[1] - n[1] * v[0];
}

template <typename T, int NP, int NFP>
__global__ void terms_fwd(int S, int K, const T* __restrict__ F, const T* __restrict__ G,
                          const T* __restrict__ X, const T* __restrict__ R,
                          const T* __restrict__ nrm, const T* __restrict__ fscale,
                          const int* __restrict__ vmapM, const int* __restrict__ vmapP,
                          const T* __restrict__ b0, const T* __restrict__ b1,
                          const T* __restrict__ b2, const T* __restrict__ b3,
                          const T* __restrict__ Dr, const T* __restrict__ Ds,
                          const T* __restrict__ Dt, const T* __restrict__ LIFT,
                          T* __restrict__ out, const int* __restrict__ xmap, int nX) {
    constexpr int NF = 4 * NFP;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * K) return;
    const int s = (int)(tid / K);
    const int k = (int)(tid % K);
    const long long KN = (long long)K * NP;
    const long long K4 = (long long)K * 4;
    const T* Fs = F + (long long)s * 3 * KN;
    const T* Gs = G + (long long)s * 3 * KN;
    T acc[6][NP];
#pragma unroll
    for (int j = 0; j < 6; ++j)
#pragma unroll
        for (int i = 0; i < NP; ++i) acc[j][i] = T(0);

    for (int fc = 0; fc < 4; ++fc) {
        const long long fi = (long long)k * 4 + fc;           // per-face data
        const T n[3] = {nrm[fi], nrm[K4 + fi], nrm[2 * K4 + fi]};
        const T fs = fscale[fi];
        const T B0 = b0[fi], B1 = b1[fi], B2x = b2[fi], B2t = b2[K4 + fi], B3 = b3[fi];
        for (int m = 0; m < NFP; ++m) {
            const int sg = fc * NFP + m;
            const long long idx = (long long)k * NF + sg;
            const long long vm = vmapM[idx], vp = vmapP[idx];
            T fm[3], fp[3], gm[3], jg[3];
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                fm[c] = Fs[c * KN + vm];
                fp[c] = Fs[c * KN + vp];
                gm[c] = Gs[c * KN + vm];
                jg[c] = Gs[c * KN + vp] - gm[c];
            }
            T cj[3], cg[3];
            cross3(n, jg, cj);
            cross3(n, gm, cg);
            T dF[3];
#pragma unroll
            for (int c = 0; c < 3; ++c)
                dF[c] = B0 * (fp[c] - fm[c]) + B1 * cj[c] + (c == 0 ? B2x : B2t) * fm[c] + B3 * cg[c];
            if (X != nullptr) {
                const int xi = xmap[idx];
                if (xi >= 0) {
                    dF[1] += X[((long long)s * 2 + 0) * nX + xi];
                    dF[2] += X[((long long)s * 2 + 1) * nX + xi];
                }
            }
#pragma unroll
            for (int j = 0; j < 6; ++j) {
                const T fl = fs * n[kA[j]] * dF[kF[j]];
#pragma unroll
                for (int i = 0; i < NP; ++i) acc[j][i] += LIFT[i * NF + sg] * fl;
            }
        }
    }
    T f[3][NP];
#pragma unroll
    for (int c = 0; c < 3; ++c)
#pragma unroll
        for (int i = 0; i < NP; ++i) f[c][i] = Fs[c * KN + (long long)k * NP + i];
    T Rk[3][3];
#pragma unroll
    for (int a = 0; a < 3; ++a)
#pragma unroll
        for (int q = 0; q < 3; ++q) Rk[a][q] = R[(a * 3 + q) * (long long)K + k];
#pragma unroll
    for (int i = 0; i < NP; ++i) {
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            T dr = 0, ds = 0, dt = 0;
#pragma unroll
            for (int jj = 0; jj < NP; ++jj) {
                dr += Dr[i * NP + jj] * f[c][jj];
                ds += Ds[i * NP + jj] * f[c][jj];
                dt += Dt[i * NP + jj] * f[c][jj];
            }
#pragma unroll
            for (int j = 0; j < 6; ++j)
                if (kF[j] == c) {
                    const int a = kA[j];
                    acc[j][i] += Rk[a][0] * dr + Rk[a][1] * ds + Rk[a][2] * dt;
                }
        }
    }
    T* o = out + (long long)s * 6 * KN + (long long)k * NP;
#pragma unroll
    for (int j = 0; j < 6; ++j)
#pragma unroll
        for (int i = 0; i < NP; ++i) o[j * KN + i] = acc[j][i];
}

template <typename T, int NP, int NFP>
__global__ void terms_bwd1(int S, int K, const T* __restrict__ F, const T* __restrict__ G,
                           const T* __restrict__ gT, const T* __restrict__ R,
                           const T* __restrict__ nrm, const T* __restrict__ fscale,
                           const int* __restrict__ vmapM, const int* __restrict__ vmapP,
                           const T* __restrict__ b0, const T* __restrict__ b1,
                           const T* __restrict__ b2, const T* __restrict__ b3,
                           const T* __restrict__ Dr, const T* __restrict__ Ds,
                           const T* __restrict__ Dt, const T* __restrict__ LIFT,
                           T* __restrict__ gFvol, T* __restrict__ hbuf, T* __restrict__ gX,
                           T* __restrict__ gb0, T* __restrict__ gb1, T* __restrict__ gb2,
                           T* __restrict__ gb3, const int* __restrict__ xmap, int nX) {
    constexpr int NF = 4 * NFP;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * K) return;
    const int s = (int)(tid / K);
    const int k = (int)(tid % K);
    const long long KN = (long long)K * NP;
    const long long K4 = (long long)K * 4;
    const T* Fs = F + (long long)s * 3 * KN;
    const T* Gs = G + (long long)s * 3 * KN;
    T g[6][NP];
    const T* gs = gT + (long long)s * 6 * KN + (long long)k * NP;
#pragma unroll
    for (int j = 0; j < 6; ++j)
#pragma unroll
        for (int i = 0; i < NP; ++i) g[j][i] = gs[j * KN + i];
    T Rk[3][3];
#pragma unroll
    for (int a = 0; a < 3; ++a)
#pragma unroll
        for (int q = 0; q < 3; ++q) Rk[a][q] = R[(a * 3 + q) * (long long)K + k];
    // Volume transpose.
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        T w[3][NP];
#pragma unroll
        for (int i = 0; i < NP; ++i) {
            w[0][i] = w[1][i] = w[2][i] = T(0);
#pragma unroll
            for (int j = 0; j < 6; ++j)
                if (kF[j] == c) {
                    const int a = kA[j];
                    w[0][i] += Rk[a][0] * g[j][i];
                    w[1][i] += Rk[a][1] * g[j][i];
                    w[2][i] += Rk[a][2] * g[j][i];
                }
        }
        T* gv = gFvol + ((long long)s * 3 + c) * KN + (long long)k * NP;
#pragma unroll
        for (int jj = 0; jj < NP; ++jj) {
            T v = 0;
#pragma unroll
            for (int i = 0; i < NP; ++i)
                v += Dr[i * NP + jj] * w[0][i] + Ds[i * NP + jj] * w[1][i] + Dt[i * NP + jj] * w[2][i];
            gv[jj] = v;
        }
    }
    // Face transpose (coefficients per face, cotangents accumulated per face).
    for (int fc = 0; fc < 4; ++fc) {
        const long long fi = (long long)k * 4 + fc;
        const T n[3] = {nrm[fi], nrm[K4 + fi], nrm[2 * K4 + fi]};
        const T fs = fscale[fi];
        T sb0 = 0, sb1 = 0, sb3 = 0, sb2x = 0, sb2t = 0;
        for (int m = 0; m < NFP; ++m) {
            const int sg = fc * NFP + m;
            const long long idx = (long long)k * NF + sg;
            const long long vm = vmapM[idx], vp = vmapP[idx];
            T L[6];
#pragma unroll
            for (int j = 0; j < 6; ++j) {
                T v = 0;
#pragma unroll
                for (int i = 0; i < NP; ++i) v += LIFT[i * NF + sg] * g[j][i];
                L[j] = v;
            }
            T h[3] = {0, 0, 0};
#pragma unroll
            for (int j = 0; j < 6; ++j) h[kF[j]] += fs * n[kA[j]] * L[j];
            T fm[3], fp[3], gm[3], jg[3];
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                fm[c] = Fs[c * KN + vm];
                fp[c] = Fs[c * KN + vp];
                gm[c] = Gs[c * KN + vm];
                jg[c] = Gs[c * KN + vp] - gm[c];
            }
            T cj[3], cg[3];
            cross3(n, jg, cj);
            cross3(n, gm, cg);
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                // slot-major (coalesced across elements): ((s*3 + c)*NF + sg)*K + k
                hbuf[(((long long)s * 3 + c) * NF + sg) * K + k] = h[c];
                sb0 += h[c] * (fp[c] - fm[c]);
                sb1 += h[c] * cj[c];
                sb3 += h[c] * cg[c];
            }
            sb2x += h[0] * fm[0];
            sb2t += h[1] * fm[1] + h[2] * fm[2];
            if (gX != nullptr) {
                const int xi = xmap[idx];
                if (xi >= 0) {
                    gX[((long long)s * 2 + 0) * nX + xi] = h[1];
                    gX[((long long)s * 2 + 1) * nX + xi] = h[2];
                }
            }
        }
        // per-face coefficient cotangents, (S, [2,] 4, K)
        const long long o1 = ((long long)s * 4 + fc) * K + k;
        gb0[o1] = sb0;
        gb1[o1] = sb1;
        gb3[o1] = sb3;
        gb2[(((long long)s * 2 + 0) * 4 + fc) * K + k] = sb2x;
        gb2[(((long long)s * 2 + 1) * 4 + fc) * K + k] = sb2t;
    }
}

template <typename T, int NP, int NFP>
__global__ void terms_bwd2(int S, int K, const T* __restrict__ gFvol, const T* __restrict__ hbuf,
                           const T* __restrict__ nrm, const T* __restrict__ b0,
                           const T* __restrict__ b1, const T* __restrict__ b2,
                           const T* __restrict__ b3, const int* __restrict__ slotP,
                           const int* __restrict__ fmask, T* __restrict__ gF,
                           T* __restrict__ gG, const T* __restrict__ addF,
                           const T* __restrict__ addG) {
    // With h the lifted face cotangent and hn = h x n = -(n x h) (d/dv h.(n x v)):
    //   own slot:    gF- = (b2 - b0) h,  gG- = (b3 - b1) hn
    //   mirror slot: gF+ = b0' h',       gG+ = b1' hn'   (coefficients of the mirror face)
    constexpr int NF = 4 * NFP;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * K) return;
    const int s = (int)(tid / K);
    const int k = (int)(tid % K);
    const long long KN = (long long)K * NP;
    const long long K4 = (long long)K * 4;
    const T* hs = hbuf + (long long)s * 3 * NF * K;
    T af[3][NP], ag[3][NP];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const T* gv = gFvol + ((long long)s * 3 + c) * KN + (long long)k * NP;
#pragma unroll
        for (int i = 0; i < NP; ++i) {
            af[c][i] = gv[i];
            ag[c][i] = T(0);
        }
    }
    for (int fc = 0; fc < 4; ++fc) {
        const long long fi = (long long)k * 4 + fc;
        const T n[3] = {nrm[fi], nrm[K4 + fi], nrm[2 * K4 + fi]};
        const T B0 = b0[fi], B1 = b1[fi], B3 = b3[fi];
        const T B2[3] = {b2[fi], b2[K4 + fi], b2[K4 + fi]};
        const int q0 = slotP[(long long)k * NF + fc * NFP];
        const long long fi2 = (long long)(q0 / NF) * 4 + (q0 % NF) / NFP;   // mirror face
        const T n2[3] = {nrm[fi2], nrm[K4 + fi2], nrm[2 * K4 + fi2]};
        const T B0m = b0[fi2], B1m = b1[fi2];
        for (int m = 0; m < NFP; ++m) {
            const int sg = fc * NFP + m;
            const int q = slotP[(long long)k * NF + sg];
            const long long k2 = q / NF, sg2 = q % NF;
            T ho[3], hm[3];
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                ho[c] = hs[((long long)c * NF + sg) * K + k];
                hm[c] = hs[((long long)c * NF + sg2) * K + k2];
            }
            T hno[3], hnm[3];
            cross3(n, ho, hno);
            cross3(n2, hm, hnm);
            const int loc = fmask[sg];
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                af[c][loc] += (B2[c] - B0) * ho[c] + B0m * hm[c];
                ag[c][loc] -= (B3 - B1) * hno[c] + B1m * hnm[c];
            }
        }
    }
    // optional addends (cotangents from other uses of F and G, summed in this pass)
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const long long o = ((long long)s * 3 + c) * KN + (long long)k * NP;
#pragma unroll
        for (int i = 0; i < NP; ++i) {
            gF[o + i] = af[c][i] + (addF != nullptr ? addF[o + i] : T(0));
            gG[o + i] = ag[c][i] + (addG != nullptr ? addG[o + i] : T(0));
        }
    }
}

// Fused DG update after the curl terms T (S, 6, K, NP), curl_c = T_2c - T_2c+1:
//   H mode: H' = H - ch curl;            UPML: B' = a1 B - b1 curl, H' = hist H + new B' - old B
//   E mode: E' = ca E + cb curl;         UPML: D' = a1 D + b1 curl, E' = hist E + new D' - old D
// UPML elements: pml[k] = p >= 0, aux and factors (3, Kp, NP) (aux per shot).
template <typename T, int NP, bool HMODE>
__global__ void dgupd_fwd(int S, int K, int Kp, const T* __restrict__ T6, const T* __restrict__ Fo,
                          const T* __restrict__ aux, T* __restrict__ Fn, T* __restrict__ auxn,
                          const T* __restrict__ ch, const T* __restrict__ ca,
                          const T* __restrict__ cb, const int* __restrict__ pml,
                          const T* __restrict__ a1, const T* __restrict__ b1,
                          const T* __restrict__ hist, const T* __restrict__ nw,
                          const T* __restrict__ old) {
    // one thread per shot and node (coalesced across the nodes of consecutive elements)
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long KN = (long long)K * NP;
    if (tid >= (long long)S * KN) return;
    const int s = (int)(tid / KN);
    const long long node = tid % KN;
    const int k = (int)(node / NP), i = (int)(node % NP);
    const int p = pml != nullptr ? pml[k] : -1;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const long long id = ((long long)s * 3 + c) * KN + node;
        const long long it = (long long)s * 6 * KN + node;
        const T curl = T6[it + 2 * c * KN] - T6[it + (2 * c + 1) * KN];
        const T f = Fo[id];
        T r;
        if (p < 0) {
            r = HMODE ? f - ch[k] * curl : ca[k] * f + cb[k] * curl;
        } else {
            const long long u = ((long long)c * Kp + p) * NP + i;
            const long long ua = ((long long)s * 3 * Kp) * NP + u;
            const T ao = aux[ua];
            const T an = HMODE ? a1[u] * ao - b1[u] * curl : a1[u] * ao + b1[u] * curl;
            auxn[ua] = an;
            r = hist[u] * f + nw[u] * an - old[u] * ao;
        }
        Fn[id] = r;
    }
}

template <typename T, int NP, bool HMODE>
__global__ void dgupd_bwd(int S, int K, int Kp, const T* __restrict__ T6, const T* __restrict__ Fo,
                          const T* __restrict__ aux, const T* __restrict__ gFn,
                          const T* __restrict__ gauxn, T* __restrict__ gT6, T* __restrict__ gFo,
                          T* __restrict__ gaux, T* __restrict__ gca, T* __restrict__ gcb,
                          T* __restrict__ ghist, T* __restrict__ gnw, T* __restrict__ gold,
                          const T* __restrict__ ch, const T* __restrict__ ca,
                          const T* __restrict__ cb, const int* __restrict__ pml,
                          const T* __restrict__ a1, const T* __restrict__ b1,
                          const T* __restrict__ hist, const T* __restrict__ nw,
                          const T* __restrict__ old) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * K) return;
    const int s = (int)(tid / K), k = (int)(tid % K);
    const long long KN = (long long)K * NP;
    const int p = pml != nullptr ? pml[k] : -1;
    T sca = 0, scb = 0;
#pragma unroll
    for (int c = 0; c < 3; ++c)
#pragma unroll
        for (int i = 0; i < NP; ++i) {
            const long long id = ((long long)s * 3 + c) * KN + (long long)k * NP + i;
            const long long it = (long long)s * 6 * KN + (long long)k * NP + i;
            const T go = gFn[id];
            const T f = Fo[id];
            T gc;
            if (p < 0) {
                if (HMODE) {
                    gFo[id] = go;
                    gc = -ch[k] * go;
                } else {
                    const T curl = T6[it + 2 * c * KN] - T6[it + (2 * c + 1) * KN];
                    gFo[id] = ca[k] * go;
                    gc = cb[k] * go;
                    sca += go * f;
                    scb += go * curl;
                }
            } else {
                const long long u = ((long long)c * Kp + p) * NP + i;
                const long long ua = ((long long)s * 3 * Kp) * NP + u;
                const T ao = aux[ua];
                const T lA = gauxn[ua] + nw[u] * go;
                gFo[id] = hist[u] * go;
                gaux[ua] = a1[u] * lA - old[u] * go;
                gc = HMODE ? -b1[u] * lA : b1[u] * lA;
                if (!HMODE) {
                    const T curl = T6[it + 2 * c * KN] - T6[it + (2 * c + 1) * KN];
                    const T an = a1[u] * ao + b1[u] * curl;
                    ghist[ua] = go * f;
                    gnw[ua] = go * an;
                    gold[ua] = -go * ao;
                }
            }
            gT6[it + 2 * c * KN] = gc;
            gT6[it + (2 * c + 1) * KN] = -gc;
        }
    if (!HMODE) {
        gca[tid] = sca;
        gcb[tid] = scb;
    }
}

constexpr int kThreads = 128;

inline int blocks(long long n) { return (int)((n + kThreads - 1) / kThreads); }

template <typename T, int NP, int NFP>
int launch_fwd(int S, int K, const void* F, const void* G, const void* X, const void* R,
               const void* nrm, const void* fscale, const void* vmapM, const void* vmapP,
               const void* b0, const void* b1, const void* b2, const void* b3, const void* Dr,
               const void* Ds, const void* Dt, const void* LIFT, void* out, const void* xmap,
               int nX, cudaStream_t st) {
    terms_fwd<T, NP, NFP><<<blocks((long long)S * K), kThreads, 0, st>>>(
        S, K, (const T*)F, (const T*)G, (const T*)X, (const T*)R, (const T*)nrm,
        (const T*)fscale, (const int*)vmapM, (const int*)vmapP, (const T*)b0,
        (const T*)b1, (const T*)b2, (const T*)b3, (const T*)Dr, (const T*)Ds, (const T*)Dt,
        (const T*)LIFT, (T*)out, (const int*)xmap, nX);
    return (int)cudaGetLastError();
}

template <typename T, int NP, int NFP>
int launch_bwd(int S, int K, const void* F, const void* G, const void* gT, const void* R,
               const void* nrm, const void* fscale, const void* vmapM, const void* vmapP,
               const void* slotP, const void* fmask, const void* b0, const void* b1,
               const void* b2, const void* b3, const void* Dr, const void* Ds, const void* Dt,
               const void* LIFT, void* work, void* gF, void* gG, void* gX, void* gb0, void* gb1,
               void* gb2, void* gb3, const void* xmap, int nX, const void* addF,
               const void* addG, cudaStream_t st) {
    const long long KN = (long long)K * NP;
    T* gFvol = (T*)work;
    T* hbuf = gFvol + (long long)S * 3 * KN;
    terms_bwd1<T, NP, NFP><<<blocks((long long)S * K), kThreads, 0, st>>>(
        S, K, (const T*)F, (const T*)G, (const T*)gT, (const T*)R, (const T*)nrm,
        (const T*)fscale, (const int*)vmapM, (const int*)vmapP, (const T*)b0,
        (const T*)b1, (const T*)b2, (const T*)b3, (const T*)Dr, (const T*)Ds, (const T*)Dt,
        (const T*)LIFT, gFvol, hbuf, (T*)gX, (T*)gb0, (T*)gb1, (T*)gb2, (T*)gb3,
        (const int*)xmap, nX);
    terms_bwd2<T, NP, NFP><<<blocks((long long)S * K), kThreads, 0, st>>>(
        S, K, gFvol, hbuf, (const T*)nrm, (const T*)b0, (const T*)b1, (const T*)b2,
        (const T*)b3, (const int*)slotP, (const int*)fmask, (T*)gF, (T*)gG, (const T*)addF,
        (const T*)addG);
    return (int)cudaGetLastError();
}

}  // namespace

extern "C" {

// Workspace (elements of the field dtype) needed by dgtd3d_terms_bwd.
long long dgtd3d_bwd_workspace(int S, int K, int np, int nfp) {
    return (long long)S * 3 * K * np + (long long)S * 3 * K * 4 * nfp;
}

int dgtd3d_terms_fwd(int dtype, int np, int S, int K, const void* F, const void* G, const void* X,
                     const void* R, const void* nrm, const void* fscale, const void* vmapM,
                     const void* vmapP, const void* b0, const void* b1, const void* b2,
                     const void* b3, const void* Dr, const void* Ds, const void* Dt,
                     const void* LIFT, void* out, const void* xmap, int nX, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
#define FWD(T, NP, NFP) return launch_fwd<T, NP, NFP>(S, K, F, G, X, R, nrm, fscale, vmapM, vmapP, \
                                                       b0, b1, b2, b3, Dr, Ds, Dt, LIFT, out, xmap, \
                                                       nX, st)
    if (dtype == 0 && np == 4) FWD(float, 4, 3);
    if (dtype == 0 && np == 10) FWD(float, 10, 6);
    if (dtype == 1 && np == 4) FWD(double, 4, 3);
    if (dtype == 1 && np == 10) FWD(double, 10, 6);
#undef FWD
    return -1;
}

int dgtd3d_terms_bwd(int dtype, int np, int S, int K, const void* F, const void* G, const void* gT,
                     const void* R, const void* nrm, const void* fscale, const void* vmapM,
                     const void* vmapP, const void* slotP, const void* fmask, const void* b0,
                     const void* b1, const void* b2, const void* b3, const void* Dr,
                     const void* Ds, const void* Dt, const void* LIFT, void* work, void* gF,
                     void* gG, void* gX, void* gb0, void* gb1, void* gb2, void* gb3,
                     const void* xmap, int nX, const void* addF, const void* addG,
                     void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
#define BWD(T, NP, NFP) return launch_bwd<T, NP, NFP>(S, K, F, G, gT, R, nrm, fscale, vmapM, vmapP, \
                                                       slotP, fmask, b0, b1, b2, b3, Dr, Ds, Dt,   \
                                                       LIFT, work, gF, gG, gX, gb0, gb1, gb2, gb3, \
                                                       xmap, nX, addF, addG, st)
    if (dtype == 0 && np == 4) BWD(float, 4, 3);
    if (dtype == 0 && np == 10) BWD(float, 10, 6);
    if (dtype == 1 && np == 4) BWD(double, 4, 3);
    if (dtype == 1 && np == 10) BWD(double, 10, 6);
#undef BWD
    return -1;
}

// mode 0: H update, 1: E update. ptr: T6, Fo, aux, Fn, auxn, ch, ca, cb, pml, a1, b1, hist, nw, old
int dgtd3d_update_fwd(int dtype, int np, int mode, int S, int K, int Kp, const void* const* q,
                      void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
#define UF(T, NP, HM) do { dgupd_fwd<T, NP, HM><<<blocks((long long)S * K * NP), kThreads, 0, st>>>( \
        S, K, Kp, (const T*)q[0], (const T*)q[1], (const T*)q[2], (T*)q[3], (T*)q[4],         \
        (const T*)q[5], (const T*)q[6], (const T*)q[7], (const int*)q[8], (const T*)q[9],     \
        (const T*)q[10], (const T*)q[11], (const T*)q[12], (const T*)q[13]);                   \
        return (int)cudaGetLastError(); } while (0)
    if (dtype == 0 && np == 4) { if (mode == 0) UF(float, 4, true); UF(float, 4, false); }
    if (dtype == 0 && np == 10) { if (mode == 0) UF(float, 10, true); UF(float, 10, false); }
    if (dtype == 1 && np == 4) { if (mode == 0) UF(double, 4, true); UF(double, 4, false); }
    if (dtype == 1 && np == 10) { if (mode == 0) UF(double, 10, true); UF(double, 10, false); }
#undef UF
    return -1;
}

// ptr: T6, Fo, aux, gFn, gauxn, gT6, gFo, gaux, gca, gcb, ghist, gnw, gold,
//      ch, ca, cb, pml, a1, b1, hist, nw, old
int dgtd3d_update_bwd(int dtype, int np, int mode, int S, int K, int Kp, const void* const* q,
                      void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
#define UB(T, NP, HM) do { dgupd_bwd<T, NP, HM><<<blocks((long long)S * K), kThreads, 0, st>>>( \
        S, K, Kp, (const T*)q[0], (const T*)q[1], (const T*)q[2], (const T*)q[3],               \
        (const T*)q[4], (T*)q[5], (T*)q[6], (T*)q[7], (T*)q[8], (T*)q[9], (T*)q[10],            \
        (T*)q[11], (T*)q[12], (const T*)q[13], (const T*)q[14], (const T*)q[15],               \
        (const int*)q[16], (const T*)q[17], (const T*)q[18], (const T*)q[19], (const T*)q[20], \
        (const T*)q[21]);                                                                       \
        return (int)cudaGetLastError(); } while (0)
    if (dtype == 0 && np == 4) { if (mode == 0) UB(float, 4, true); UB(float, 4, false); }
    if (dtype == 0 && np == 10) { if (mode == 0) UB(float, 10, true); UB(float, 10, false); }
    if (dtype == 1 && np == 4) { if (mode == 0) UB(double, 4, true); UB(double, 4, false); }
    if (dtype == 1 && np == 10) { if (mode == 0) UB(double, 10, true); UB(double, 10, false); }
#undef UB
    return -1;
}

}  // extern "C"

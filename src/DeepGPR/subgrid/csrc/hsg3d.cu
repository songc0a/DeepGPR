// Fused coupling kernels of the Huygens subgridding (DeepGPR.subgrid.hsg) and
// their exact transposes. Batched over shots (fields ``(S, N)`` contiguous).
//
//   taps:  y[s, n] = sum_t w[n, t] x_{comp[n]}[s, idx[n, t]]          (precursors)
//   corr:  f_{c_j}[s, i_j] += coef_j (cb[i_j]) v_j(s)                  (surface corrections)
//          v_j = a V0[s, j] + b V1[s, j]       (mode 0: interpolated precursors)
//          v_j = src_{f_j}[s, k_j]             (mode 1: subgrid values at the OS)
//
// A ``corr`` launch must not contain two entries with the same target (the caller
// splits the entry list), so the forward pass needs no atomics. The transposes
// accumulate with atomics only where several entries share a node: the main-grid
// cotangents of the precursor taps, the subgrid cotangents of the OS values, and
// the ``cb`` cotangents.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

template <typename T>
struct P3 {
    T* p[3];
};

template <typename T>
__device__ inline void atomic_add(T* a, T v) {
    atomicAdd(a, v);
}

template <typename T>
__global__ void taps_fwd(int S, int n, int nt, long long N, const long long* __restrict__ idx,
                         const T* __restrict__ w, const int8_t* __restrict__ comp, P3<T> x,
                         T* __restrict__ y) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * n) return;
    const int s = (int)(tid / n);
    const int j = (int)(tid % n);
    const T* src = x.p[comp[j]] + (long long)s * N;
    T acc = 0;
    for (int t = 0; t < nt; ++t) acc += w[(long long)j * nt + t] * src[idx[(long long)j * nt + t]];
    y[tid] = acc;
}

template <typename T>
__global__ void taps_bwd(int S, int n, int nt, long long N, const long long* __restrict__ idx,
                         const T* __restrict__ w, const int8_t* __restrict__ comp, P3<T> gx,
                         const T* __restrict__ gy) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * n) return;
    const int s = (int)(tid / n);
    const int j = (int)(tid % n);
    const T g = gy[tid];
    if (g == T(0)) return;
    T* dst = gx.p[comp[j]] + (long long)s * N;
    for (int t = 0; t < nt; ++t) {
        const T wt = w[(long long)j * nt + t];
        if (wt != T(0)) atomic_add(dst + idx[(long long)j * nt + t], wt * g);
    }
}

template <typename T>
struct Corr {
    int S, m, mode;
    long long N, Nsrc, ldv;           // target field size, source field size, V row length
    const int8_t* comp;               // target component per entry
    const long long* idx;             // target node per entry
    const T* coef;
    const T* cb;                      // (N) or nullptr
    const int8_t* scomp;              // mode 1: source component per entry
    const long long* sidx;            // mode 1: source node per entry
    T a, b;                           // mode 0: v = a V0 + b V1
};

template <typename T>
__device__ inline T value(const Corr<T>& c, int s, int j, const T* V0, const T* V1, const P3<T>& src) {
    if (c.mode == 0) {
        T v = c.a * V0[(long long)s * c.ldv + j];
        if (V1 != nullptr) v += c.b * V1[(long long)s * c.ldv + j];
        return v;
    }
    return src.p[c.scomp[j]][(long long)s * c.Nsrc + c.sidx[j]];
}

template <typename T>
__global__ void corr_fwd(Corr<T> c, P3<T> f, const T* __restrict__ V0, const T* __restrict__ V1,
                         P3<T> src) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)c.S * c.m) return;
    const int s = (int)(tid / c.m);
    const int j = (int)(tid % c.m);
    const long long i = c.idx[j];
    T k = c.coef[j];
    if (c.cb != nullptr) k *= c.cb[i];
    f.p[c.comp[j]][(long long)s * c.N + i] += k * value(c, s, j, V0, V1, src);
}

// gV0/gV1 (mode 0, accumulated, unique per entry), gsrc (mode 1, atomics), gcb (atomics)
template <typename T>
__global__ void corr_bwd(Corr<T> c, P3<T> gf, const T* __restrict__ V0, const T* __restrict__ V1,
                         P3<T> src, T* __restrict__ gV0, T* __restrict__ gV1, P3<T> gsrc,
                         T* __restrict__ gcb) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)c.S * c.m) return;
    const int s = (int)(tid / c.m);
    const int j = (int)(tid % c.m);
    const long long i = c.idx[j];
    const T g = gf.p[c.comp[j]][(long long)s * c.N + i];
    if (g == T(0)) return;
    const T cbv = c.cb != nullptr ? c.cb[i] : T(1);
    const T k = c.coef[j] * cbv;
    if (c.mode == 0) {
        if (gV0 != nullptr) gV0[(long long)s * c.ldv + j] += c.a * k * g;
        if (gV1 != nullptr) gV1[(long long)s * c.ldv + j] += c.b * k * g;
    } else if (gsrc.p[c.scomp[j]] != nullptr) {
        atomic_add(gsrc.p[c.scomp[j]] + (long long)s * c.Nsrc + c.sidx[j], k * g);
    }
    if (gcb != nullptr && c.cb != nullptr)
        atomic_add(gcb + i, c.coef[j] * value(c, s, j, V0, V1, src) * g);
}

template <typename T>
P3<T> p3(void* const* p) {
    P3<T> r;
    for (int k = 0; k < 3; ++k) r.p[k] = static_cast<T*>(p[k]);
    return r;
}

inline unsigned blocks(long long n) { return (unsigned)((n + 255) / 256); }

template <typename T>
int run_taps(int bwd, const long long* gi, void* const* io, cudaStream_t st) {
    const int S = (int)gi[0], n = (int)gi[1], nt = (int)gi[2];
    const long long N = gi[3];
    if ((long long)S * n == 0) return 0;
    const long long* idx = static_cast<const long long*>(io[0]);
    const T* w = static_cast<const T*>(io[1]);
    const int8_t* comp = static_cast<const int8_t*>(io[2]);
    P3<T> x = p3<T>(io + 3);
    T* y = static_cast<T*>(io[6]);
    if (bwd) taps_bwd<T><<<blocks((long long)S * n), 256, 0, st>>>(S, n, nt, N, idx, w, comp, x, y);
    else taps_fwd<T><<<blocks((long long)S * n), 256, 0, st>>>(S, n, nt, N, idx, w, comp, x, y);
    return (int)cudaGetLastError();
}

// gi: S, m, mode, N, Nsrc, ldv; gd: a, b
// io: comp, idx, coef, cb, scomp, sidx, f[3], V0, V1, src[3]  (+ bwd: gV0, gV1, gsrc[3], gcb)
template <typename T>
int run_corr(int bwd, const long long* gi, const double* gd, void* const* io, cudaStream_t st) {
    Corr<T> c;
    c.S = (int)gi[0];
    c.m = (int)gi[1];
    c.mode = (int)gi[2];
    c.N = gi[3];
    c.Nsrc = gi[4];
    c.ldv = gi[5];
    c.a = T(gd[0]);
    c.b = T(gd[1]);
    if ((long long)c.S * c.m == 0) return 0;
    c.comp = static_cast<const int8_t*>(io[0]);
    c.idx = static_cast<const long long*>(io[1]);
    c.coef = static_cast<const T*>(io[2]);
    c.cb = static_cast<const T*>(io[3]);
    c.scomp = static_cast<const int8_t*>(io[4]);
    c.sidx = static_cast<const long long*>(io[5]);
    P3<T> f = p3<T>(io + 6);
    const T* V0 = static_cast<const T*>(io[9]);
    const T* V1 = static_cast<const T*>(io[10]);
    P3<T> src = p3<T>(io + 11);
    const long long n = (long long)c.S * c.m;
    if (bwd) {
        corr_bwd<T><<<blocks(n), 256, 0, st>>>(c, f, V0, V1, src, static_cast<T*>(io[14]),
                                               static_cast<T*>(io[15]), p3<T>(io + 16),
                                               static_cast<T*>(io[19]));
    } else {
        corr_fwd<T><<<blocks(n), 256, 0, st>>>(c, f, V0, V1, src);
    }
    return (int)cudaGetLastError();
}

}  // namespace

extern "C" {

int hsg3d_taps(int dtype, int bwd, const long long* gi, void* const* io, void* stream) {
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    return dtype == 1 ? run_taps<double>(bwd, gi, io, st) : run_taps<float>(bwd, gi, io, st);
}

int hsg3d_corr(int dtype, int bwd, const long long* gi, const double* gd, void* const* io,
               void* stream) {
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    return dtype == 1 ? run_corr<double>(bwd, gi, gd, io, st) : run_corr<float>(bwd, gi, gd, io, st);
}

}  // extern "C"

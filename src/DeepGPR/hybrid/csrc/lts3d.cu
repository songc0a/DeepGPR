// Fused elementwise updates of the local time stepping of the 3D hybrid
// (DeepGPR.hybrid.functional3d.lts_h_half_step) and their exact transposes.
//
// The recursion on the DG band evaluates the fused curl-terms kernel of dgtd3d.cu
// (T, six terms (S, 6, K, NP)) and then updates the recursion variables
// elementwise. Without fusion every update is several full-size PyTorch
// temporaries (about ten passes over (S, 3, K, NP) per sub-step), which costs as
// much as the operator itself. The two kernels here do one pass each:
//
//   z    kernel:  z_new  = z + a cb0[k] curl(Te)                       (E rate, cb0 = dt/eps)
//   eta  kernel:  eta_new = beta eta - b ch[k] curl(T)                  (H rate, ch = dt/mu)
//                 cavg_new = cavg + w curl(T)                           (weighted curl average)
//                 (first = 1: eta_new = -b ch curl(T), cavg_new = w curl(T))
//
// with curl(T)_c = T[2c] - T[2c+1]. The macro H update then is H - ch cavg (plus
// the UPML sub-steps of the PML elements, done in PyTorch on the PML subset).
//
// Transposes: g_z = g_znew (identity, not a kernel), g_Te[2c] = a cb0 g, g_Te[2c+1] =
// -a cb0 g, g_cb0[k] = a sum curl(Te) g (one thread per element, no atomics);
// g_eta = beta g_etanew, g_cavg = g_cavgnew (identity), g_T[2c] = w g_cavgnew - b ch
// g_etanew, g_T[2c+1] = -g_T[2c]. Layouts: (S, 3, K, NP) and (S, 6, K, NP) contiguous;
// cb0, ch: (K,) contiguous.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 256;
inline int blocks(long long n) { return (int)((n + kThreads - 1) / kThreads); }

template <typename T>
__global__ void z_fwd(int S, long long KN, int NP, const T* __restrict__ Te,
                      const T* __restrict__ z, const T* __restrict__ cb0, T a,
                      T* __restrict__ out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * KN) return;
    const int s = (int)(tid / KN);
    const long long node = tid % KN;
    const int k = (int)(node / NP);
    const T f = a * cb0[k];
    const long long it = (long long)s * 6 * KN + node;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const long long id = ((long long)s * 3 + c) * KN + node;
        out[id] = z[id] + f * (Te[it + 2 * c * KN] - Te[it + (2 * c + 1) * KN]);
    }
}

// One thread per element: writes g_Te for all shots / nodes of the element and
// sums the cb0 cotangent of the element (deterministic, no atomics).
template <typename T>
__global__ void z_bwd(int S, int K, int NP, const T* __restrict__ Te, const T* __restrict__ g,
                      const T* __restrict__ cb0, T a, T* __restrict__ gTe,
                      T* __restrict__ gcb0) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const long long KN = (long long)K * NP;
    const T f = a * cb0[k];
    T acc = 0;
    for (int s = 0; s < S; ++s) {
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < NP; ++i) {
                const long long node = (long long)k * NP + i;
                const long long id = ((long long)s * 3 + c) * KN + node;
                const long long it = (long long)s * 6 * KN + node;
                const T gv = g[id];
                const T curl = Te[it + 2 * c * KN] - Te[it + (2 * c + 1) * KN];
                acc += curl * gv;
                gTe[it + 2 * c * KN] = f * gv;
                gTe[it + (2 * c + 1) * KN] = -f * gv;
            }
        }
    }
    gcb0[k] = a * acc;
}

template <typename T>
__global__ void eta_fwd(int S, long long KN, int NP, const T* __restrict__ Tt,
                        const T* __restrict__ eta, const T* __restrict__ cavg,
                        const T* __restrict__ ch, T beta, T b, T w, int first,
                        T* __restrict__ eta_out, T* __restrict__ cavg_out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * KN) return;
    const int s = (int)(tid / KN);
    const long long node = tid % KN;
    const int k = (int)(node / NP);
    const T bc = b * ch[k];
    const long long it = (long long)s * 6 * KN + node;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const long long id = ((long long)s * 3 + c) * KN + node;
        const T curl = Tt[it + 2 * c * KN] - Tt[it + (2 * c + 1) * KN];
        if (first) {
            eta_out[id] = -bc * curl;
            cavg_out[id] = w * curl;
        } else {
            eta_out[id] = beta * eta[id] - bc * curl;
            cavg_out[id] = cavg[id] + w * curl;
        }
    }
}

// g_etanew may be null (unused eta output); g_cavgnew may be null.
template <typename T>
__global__ void eta_bwd(int S, long long KN, int NP, const T* __restrict__ g_eta_new,
                        const T* __restrict__ g_cavg_new, const T* __restrict__ ch, T beta, T b,
                        T w, T* __restrict__ gT, T* __restrict__ g_eta) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * KN) return;
    const int s = (int)(tid / KN);
    const long long node = tid % KN;
    const int k = (int)(node / NP);
    const T bc = b * ch[k];
    const long long it = (long long)s * 6 * KN + node;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const long long id = ((long long)s * 3 + c) * KN + node;
        const T ge = g_eta_new != nullptr ? g_eta_new[id] : (T)0;
        const T gc = g_cavg_new != nullptr ? g_cavg_new[id] : (T)0;
        const T v = w * gc - bc * ge;
        gT[it + 2 * c * KN] = v;
        gT[it + (2 * c + 1) * KN] = -v;
        if (g_eta != nullptr) g_eta[id] = beta * ge;
    }
}

// ---------------------------------------------------------------------------- 2D (TM)
// Terms T (S, 2, K, NP) = (D_x E, D_y E) or (D_x Hy, D_y Hx); fields (S, K, NP).
//   z   kernel: z_new = z + a cb0[k] (Te[0] - Te[1])
//   eta kernel: ex_new = beta ex - b ch[k] T[1], ey_new = beta ey + b ch[k] T[0],
//               Tavg_new = Tavg + w T   (first = 1: ex = -b ch T[1], ey = b ch T[0], Tavg = w T)
template <typename T>
__global__ void z2_fwd(int S, long long KN, int NP, const T* __restrict__ Te,
                       const T* __restrict__ z, const T* __restrict__ cb0, T a,
                       T* __restrict__ out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * KN) return;
    const int s = (int)(tid / KN);
    const long long node = tid % KN;
    const int k = (int)(node / NP);
    const long long it = (long long)s * 2 * KN + node;
    out[tid] = z[tid] + a * cb0[k] * (Te[it] - Te[it + KN]);
}

template <typename T>
__global__ void z2_bwd(int S, int K, int NP, const T* __restrict__ Te, const T* __restrict__ g,
                       const T* __restrict__ cb0, T a, T* __restrict__ gTe,
                       T* __restrict__ gcb0) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const long long KN = (long long)K * NP;
    const T f = a * cb0[k];
    T acc = 0;
    for (int s = 0; s < S; ++s) {
        for (int i = 0; i < NP; ++i) {
            const long long node = (long long)k * NP + i;
            const long long id = (long long)s * KN + node;
            const long long it = (long long)s * 2 * KN + node;
            const T gv = g[id];
            acc += (Te[it] - Te[it + KN]) * gv;
            gTe[it] = f * gv;
            gTe[it + KN] = -f * gv;
        }
    }
    gcb0[k] = a * acc;
}

template <typename T>
__global__ void eta2_fwd(int S, long long KN, int NP, const T* __restrict__ Tt,
                         const T* __restrict__ ex, const T* __restrict__ ey,
                         const T* __restrict__ tavg, const T* __restrict__ ch, T beta, T b, T w,
                         int first, T* __restrict__ ex_out, T* __restrict__ ey_out,
                         T* __restrict__ tavg_out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * KN) return;
    const int s = (int)(tid / KN);
    const long long node = tid % KN;
    const int k = (int)(node / NP);
    const T bc = b * ch[k];
    const long long it = (long long)s * 2 * KN + node;
    const T t0 = Tt[it], t1 = Tt[it + KN];
    if (first) {
        ex_out[tid] = -bc * t1;
        ey_out[tid] = bc * t0;
        tavg_out[it] = w * t0;
        tavg_out[it + KN] = w * t1;
    } else {
        ex_out[tid] = beta * ex[tid] - bc * t1;
        ey_out[tid] = beta * ey[tid] + bc * t0;
        tavg_out[it] = tavg[it] + w * t0;
        tavg_out[it + KN] = tavg[it + KN] + w * t1;
    }
}

template <typename T>
__global__ void eta2_bwd(int S, long long KN, int NP, const T* __restrict__ gex,
                         const T* __restrict__ gey, const T* __restrict__ gtavg,
                         const T* __restrict__ ch, T beta, T b, T* __restrict__ gT,
                         T* __restrict__ gex_out, T* __restrict__ gey_out, T w) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * KN) return;
    const int s = (int)(tid / KN);
    const long long node = tid % KN;
    const int k = (int)(node / NP);
    const T bc = b * ch[k];
    const long long it = (long long)s * 2 * KN + node;
    const T gx = gex != nullptr ? gex[tid] : (T)0;
    const T gy = gey != nullptr ? gey[tid] : (T)0;
    const T ga0 = gtavg != nullptr ? gtavg[it] : (T)0;
    const T ga1 = gtavg != nullptr ? gtavg[it + KN] : (T)0;
    gT[it] = w * ga0 + bc * gy;
    gT[it + KN] = w * ga1 - bc * gx;
    if (gex_out != nullptr) gex_out[tid] = beta * gx;
    if (gey_out != nullptr) gey_out[tid] = beta * gy;
}


// ---------------------------------------------------------------------------- Yee slabs
// The fine Yee planes of one interface (functional3d.LTSSlab) are slab tensors
// (S, n, NY, NZ) per component. Per sub-step the recursion needs the E rates
// q = curl_bwd(eta) (masked, scaled by dt/(p eps)) and the H rates r = curl_fwd(z)
// with the weighted average and the eta recursion; in PyTorch this is ~70
// launches of tiny kernels per slab and sub-step. Here: one kernel each.
//   slab_q: out = z + mask (a cb0) curl_bwd(f)     (zmode)   or   out = curl_bwd(f)
//   slab_r: r = curl_fwd(z); ravg_out = (first ? w r : ravg + w r);
//           eta_out = (first ? -b chr r : beta eta - b chr r)
// Differences: fwd f[i+1]-f[i] (0 in the last plane of the axis), bwd f[i]-f[i-1]
// (0 in the first plane), terms as functional3d.TERMS. Transposes gather the
// cotangent of the differences with the matching boundary conditions (no atomics).
struct SlabDims {
    int S, n[3];
    long long N, stride[3];
    double h[3];
};

// term tables (axis a_t, differentiated field f_t) as compile-time functions: in the
// unrolled term loops the slab coordinates, extents, strides, steps and field pointers
// are indexed with constants and stay in registers (a __constant__ table gave run-time
// indices and a local-memory stack frame)
__host__ __device__ constexpr int term_axis(int t) {
    return t == 0 ? 1 : t == 1 ? 2 : t == 2 ? 2 : t == 3 ? 0 : t == 4 ? 0 : 1;
}
__host__ __device__ constexpr int term_field(int t) { return 2 - t % 3; }

template <typename T>
struct C3 { const T* p[3]; };
template <typename T>
struct W3 { T* p[3]; };

__device__ inline void slab_coords(const SlabDims& g, long long node, int c[3]) {
    c[0] = (int)(node / g.stride[0]);
    const long long r = node % g.stride[0];
    c[1] = (int)(r / g.stride[1]);
    c[2] = (int)(r % g.stride[1]);
}

template <typename T, bool FWD>
__device__ inline void slab_curl(const SlabDims& g, const C3<T>& f, long long base, long long node,
                                 const int c[3], T out[3]) {
    out[0] = out[1] = out[2] = 0;
#pragma unroll
    for (int t = 0; t < 6; ++t) {
        const int a = term_axis(t);
        const T* ff = f.p[term_field(t)] + base;
        const T s = (t & 1) ? (T)-1 : (T)1;
        T d = 0;
        if (FWD) {
            if (c[a] < g.n[a] - 1) d = ff[node + g.stride[a]] - ff[node];
        } else {
            if (c[a] > 0) d = ff[node] - ff[node - g.stride[a]];
        }
        out[t >> 1] += s * d / (T)g.h[a];
    }
}

// cotangent v of the q output in the z mode: mask (a cb0) g, else g
template <typename T>
__device__ inline T fetch_v(const C3<T>& gq, const uint8_t* const* mask, const T* cb0, T a,
                            int zmode, int c, long long base, long long node) {
    const T gv = gq.p[c][base + node];
    if (!zmode) return gv;
    return mask[c][node] ? a * cb0[node] * gv : (T)0;
}

template <typename T, bool FWD>
__device__ inline void slab_curl_t(const SlabDims& g, const C3<T>& v, const uint8_t* const* mask,
                                   const T* cb0, T a, int zmode, long long base, long long node,
                                   const int c[3], T out[3]) {
    out[0] = out[1] = out[2] = 0;
#pragma unroll
    for (int t = 0; t < 6; ++t) {
        const int a_ = term_axis(t), cc = t >> 1;
        const T s = (t & 1) ? (T)-1 : (T)1;
        T d = 0;
        if (FWD) {
            if (c[a_] > 0) d += fetch_v(v, mask, cb0, a, zmode, cc, base, node - g.stride[a_]);
            if (c[a_] < g.n[a_] - 1) d -= fetch_v(v, mask, cb0, a, zmode, cc, base, node);
        } else {
            if (c[a_] > 0) d += fetch_v(v, mask, cb0, a, zmode, cc, base, node);
            if (c[a_] < g.n[a_] - 1) d -= fetch_v(v, mask, cb0, a, zmode, cc, base, node + g.stride[a_]);
        }
        out[term_field(t)] += s * d / (T)g.h[a_];
    }
}

template <typename T>
__global__ void slab_q_fwd(SlabDims g, C3<T> f, C3<T> z, const T* __restrict__ cb0,
                           const uint8_t* m0, const uint8_t* m1, const uint8_t* m2, T a,
                           int zmode, W3<T> out) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)g.S * g.N) return;
    const long long node = tid % g.N, base = tid - node;
    int c[3];
    slab_coords(g, node, c);
    T q[3];
    slab_curl<T, false>(g, f, base, node, c, q);
    const uint8_t* mask[3] = {m0, m1, m2};
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        if (zmode) {
            const T zv = z.p[k][tid];
            out.p[k][tid] = mask[k][node] ? zv + a * cb0[node] * q[k] : zv;
        } else {
            out.p[k][tid] = q[k];
        }
    }
}

// thread per node, loop over shots: g_f = curl_bwd^T(v), g_cb0 = sum_s sum_k mask a q g
template <typename T>
__global__ void slab_q_bwd(SlabDims g, C3<T> f, C3<T> gq, const T* __restrict__ cb0,
                           const uint8_t* m0, const uint8_t* m1, const uint8_t* m2, T a,
                           int flags, W3<T> gf, T* __restrict__ gcb0) {
    // flags: bit 0 z mode, bit 1 accumulate g_f into gf, bit 2 accumulate g_cb0 into gcb0
    const int zmode = flags & 1;
    const long long node = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= g.N) return;
    int c[3];
    slab_coords(g, node, c);
    const uint8_t* mask[3] = {m0, m1, m2};
    T acc = 0;
    for (int s = 0; s < g.S; ++s) {
        const long long base = (long long)s * g.N;
        T o[3];
        slab_curl_t<T, false>(g, gq, mask, cb0, a, zmode, base, node, c, o);
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            if (flags & 2)
                gf.p[k][base + node] += o[k];
            else
                gf.p[k][base + node] = o[k];
        }
        if (gcb0 != nullptr) {
            T q[3];
            slab_curl<T, false>(g, f, base, node, c, q);
#pragma unroll
            for (int k = 0; k < 3; ++k)
                if (mask[k][node]) acc += a * q[k] * gq.p[k][base + node];
        }
    }
    if (gcb0 != nullptr) {
        if (flags & 4)
            gcb0[node] += acc;
        else
            gcb0[node] = acc;
    }
}

// Last sub-step (corr given): corr = fine_h ? -chr (ravg_out - r0) : 0, the correction of the
// fine H planes (added to the plain Yee update by the caller).
template <typename T>
__global__ void slab_r_fwd(SlabDims g, C3<T> z, C3<T> eta, C3<T> ravg, C3<T> chr, T beta, T b, T w,
                           int first, W3<T> r_out, W3<T> eta_out, W3<T> ravg_out, C3<T> r0,
                           const uint8_t* mh0, const uint8_t* mh1, const uint8_t* mh2, W3<T> corr,
                           int acc) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)g.S * g.N) return;
    const long long node = tid % g.N, base = tid - node;
    int c[3];
    slab_coords(g, node, c);
    T r[3];
    slab_curl<T, true>(g, z, base, node, c, r);
    const uint8_t* mh[3] = {mh0, mh1, mh2};
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        if (r_out.p[k] != nullptr) r_out.p[k][tid] = r[k];
        const T ra = first ? w * r[k] : ravg.p[k][tid] + w * r[k];
        if (ravg_out.p[k] != nullptr) ravg_out.p[k][tid] = ra;
        if (eta_out.p[k] != nullptr) {
            const T bc = b * chr.p[k][node];
            eta_out.p[k][tid] = first ? -bc * r[k] : beta * eta.p[k][tid] - bc * r[k];
        }
        if (corr.p[k] != nullptr) {
            if (acc) {       // added directly to the fine H planes (corr: their slab views)
                if (mh[k][node]) corr.p[k][tid] += -chr.p[k][node] * (ra - r0.p[k][tid]);
            } else {
                corr.p[k][tid] = mh[k][node] ? -chr.p[k][node] * (ra - r0.p[k][tid]) : (T)0;
            }
        }
    }
}

// cotangent of r: v = g_r + w g_ravg_eff - b chr g_eta (each nullable), with
// g_ravg_eff = g_ravg - fine_h chr g_corr (the correction output of the last sub-step)
template <typename T>
struct RV {
    C3<T> gr, ge, ga, chr, gcorr;
    const uint8_t* mh[3];
    T b, w;
    __device__ inline T ga_eff(int c, long long base, long long node) const {
        T v = ga.p[c] != nullptr ? ga.p[c][base + node] : (T)0;
        if (gcorr.p[c] != nullptr && mh[c][node]) v -= chr.p[c][node] * gcorr.p[c][base + node];
        return v;
    }
    __device__ inline T at(int c, long long base, long long node) const {
        T v = w * ga_eff(c, base, node);
        if (gr.p[c] != nullptr) v += gr.p[c][base + node];
        if (ge.p[c] != nullptr) v -= b * chr.p[c][node] * ge.p[c][base + node];
        return v;
    }
};

template <typename T>
__global__ void slab_r_bwd(SlabDims g, RV<T> v, T beta, W3<T> gz, W3<T> ge_in, W3<T> ga_in,
                           W3<T> gr0, int acc) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)g.S * g.N) return;
    const long long node = tid % g.N, base = tid - node;
    int c[3];
    slab_coords(g, node, c);
    T o[3] = {0, 0, 0};
#pragma unroll
    for (int t = 0; t < 6; ++t) {
        const int a = term_axis(t), cc = t >> 1;
        const T s = (t & 1) ? (T)-1 : (T)1;
        T d = 0;
        if (c[a] > 0) d += v.at(cc, base, node - g.stride[a]);
        if (c[a] < g.n[a] - 1) d -= v.at(cc, base, node);
        o[term_field(t)] += s * d / (T)g.h[a];
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        if (acc)
            gz.p[k][tid] += o[k];
        else
            gz.p[k][tid] = o[k];
        if (ge_in.p[k] != nullptr) ge_in.p[k][tid] = beta * (v.ge.p[k] != nullptr ? v.ge.p[k][tid] : (T)0);
        if (ga_in.p[k] != nullptr) ga_in.p[k][tid] = v.ga_eff(k, base, node);
        if (gr0.p[k] != nullptr)
            gr0.p[k][tid] = (v.gcorr.p[k] != nullptr && v.mh[k][node])
                                ? v.chr.p[k][node] * v.gcorr.p[k][tid] : (T)0;
    }
}

// ---------------------------------------------------------------------------- UPML folds
// p Crank-Nicolson sub-steps of the UPML pair (aux, f) with a constant rate
// (functional3d._fold_upml): aux <- a1 aux + sign b1 rate, f <- hist f + new aux' - old aux.
// The transpose runs the recursion forward (trajectory in registers) and backward.
constexpr int kMaxP = 64;

template <typename T>
__device__ inline void fold_run(int p, T sign, T a1, T b1, T hist, T nw, T old, T rate, T& f, T& aux) {
    for (int j = 0; j < p; ++j) {
        const T an = a1 * aux + (sign * b1) * rate;
        f = hist * f + nw * an - old * aux;
        aux = an;
    }
}

template <typename T>
__device__ inline void fold_vjp(int p, T sign, T a1, T b1, T hist, T nw, T old, T f0, T aux0, T rate,
                                T gf, T gaux, bool want_fac, T& g_f0, T& g_aux0, T& g_rate, T& g_hist,
                                T& g_nw, T& g_old) {
    T fs[kMaxP + 1], as[kMaxP + 1];
    fs[0] = f0;
    as[0] = aux0;
    for (int j = 0; j < p; ++j) {
        as[j + 1] = a1 * as[j] + (sign * b1) * rate;
        fs[j + 1] = hist * fs[j] + nw * as[j + 1] - old * as[j];
    }
    T gfj = gf, gaj = gaux, gr = 0, gh = 0, gn = 0, go = 0;
    for (int j = p - 1; j >= 0; --j) {
        if (want_fac) {
            gh += gfj * fs[j];
            gn += gfj * as[j + 1];
            go -= gfj * as[j];
        }
        const T ga1 = gaj + nw * gfj;        // cotangent of as[j + 1]
        gr += (sign * b1) * ga1;
        gaj = a1 * ga1 - old * gfj;          // cotangent of as[j]
        gfj = hist * gfj;                    // cotangent of fs[j]
    }
    g_f0 = gfj;
    g_aux0 = gaj;
    g_rate = gr;
    g_hist = gh;
    g_nw = gn;
    g_old = go;
}

// Generic gather fold: entries e < n with (nullable: identity) index arrays into the
// flat per-shot arrays f (S, Nf), aux (S, Na), rate (S, Nr); factors per entry.
template <typename T>
struct FoldIn {
    const int *idx_f, *idx_a, *idx_r;
    const T *f, *aux, *rate, *a1, *b1, *hist, *nw, *old;
    long long Nf, Na, Nr;
};

// scatter = 0: results in the compact (S, n) arrays fval / auxn; scatter = 1: written
// directly into destination arrays shaped like f / aux at the entries' indices.
template <typename T>
__global__ void fold_fwd(int S, long long n, int p, T sign, FoldIn<T> in, T* __restrict__ fval,
                         T* __restrict__ auxn, int scatter) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const long long jf = in.idx_f ? in.idx_f[e] : e, ja = in.idx_a ? in.idx_a[e] : e,
                    jr = in.idx_r ? in.idx_r[e] : e;
    const T a1 = in.a1[e], b1 = in.b1[e], hist = in.hist[e], nw = in.nw[e], old = in.old[e];
    for (int s = 0; s < S; ++s) {
        T f = in.f[s * in.Nf + jf], aux = in.aux[s * in.Na + ja];
        fold_run(p, sign, a1, b1, hist, nw, old, in.rate[s * in.Nr + jr], f, aux);
        if (scatter) {
            fval[s * in.Nf + jf] = f;
            auxn[s * in.Na + ja] = aux;
        } else {
            fval[s * n + e] = f;
            auxn[s * n + e] = aux;
        }
    }
}

// g_f (S, Nf), g_aux (S, Na), g_rate (S, Nr) are written at the entries only (the
// caller zeroes them when the index arrays are given); factor cotangents (n,) nullable.
// mode bit 1 (compact): g_f / g_aux go to compact (S, n) arrays; bit 2 (gather): the
// output cotangents are read from arrays shaped like f / aux at the entries (gfval,
// gauxn: those arrays) and zeroed there (the folded entries replace the plain update).
template <typename T>
__global__ void fold_bwd(int S, long long n, int p, T sign, FoldIn<T> in, T* __restrict__ gfval,
                         T* __restrict__ gauxn, T* __restrict__ gf, T* __restrict__ gaux,
                         T* __restrict__ grate, T* __restrict__ ghist, T* __restrict__ gnw,
                         T* __restrict__ gold, int mode) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const long long jf = in.idx_f ? in.idx_f[e] : e, ja = in.idx_a ? in.idx_a[e] : e,
                    jr = in.idx_r ? in.idx_r[e] : e;
    const T a1 = in.a1[e], b1 = in.b1[e], hist = in.hist[e], nw = in.nw[e], old = in.old[e];
    const bool want = ghist != nullptr;
    const bool compact = (mode & 2) != 0, gather = (mode & 4) != 0;
    T sh = 0, sn = 0, so = 0;
    for (int s = 0; s < S; ++s) {
        T g_f, g_a, g_r, gh, gn, go;
        T go_f, go_a = 0;
        if (gather) {
            go_f = gfval[s * in.Nf + jf];
            gfval[s * in.Nf + jf] = 0;
            if (gauxn) {
                go_a = gauxn[s * in.Na + ja];
                gauxn[s * in.Na + ja] = 0;
            }
        } else {
            go_f = gfval[s * n + e];
            go_a = gauxn ? gauxn[s * n + e] : (T)0;
        }
        fold_vjp(p, sign, a1, b1, hist, nw, old, in.f[s * in.Nf + jf], in.aux[s * in.Na + ja],
                 in.rate[s * in.Nr + jr], go_f, go_a, want, g_f, g_a, g_r, gh, gn, go);
        if (compact) {
            gf[s * n + e] = g_f;
            gaux[s * n + e] = g_a;
        } else {
            gf[s * in.Nf + jf] = g_f;
            gaux[s * in.Na + ja] = g_a;
        }
        grate[s * in.Nr + jr] = g_r;
        sh += gh;
        sn += gn;
        so += go;
    }
    if (want) {
        ghist[e] = sh;
        gnw[e] = sn;
        gold[e] = so;
    }
}

template <typename T>
inline C3<T> c3(const void* const* q, int o) {
    C3<T> r;
    for (int k = 0; k < 3; ++k) r.p[k] = (const T*)q[o + k];
    return r;
}
template <typename T>
inline W3<T> w3(const void* const* q, int o) {
    W3<T> r;
    for (int k = 0; k < 3; ++k) r.p[k] = (T*)q[o + k];
    return r;
}

inline SlabDims slab_dims(int S, int n, int NY, int NZ, const double* h) {
    SlabDims g;
    g.S = S;
    g.n[0] = n;
    g.n[1] = NY;
    g.n[2] = NZ;
    g.stride[0] = (long long)NY * NZ;
    g.stride[1] = NZ;
    g.stride[2] = 1;
    g.N = (long long)n * NY * NZ;
    for (int a = 0; a < 3; ++a) g.h[a] = h[a];
    return g;
}

// ptr tables: q_fwd [f0..2, z0..2, cb0, m0..2, o0..2]; q_bwd [f0..2, g0..2, cb0, m0..2, gf0..2, gcb0]
template <typename T>
int run_slab_q(int bwd, const SlabDims& g, const void* const* q, double a, int zmode, cudaStream_t st) {
    const uint8_t *m0 = (const uint8_t*)q[7], *m1 = (const uint8_t*)q[8], *m2 = (const uint8_t*)q[9];
    if (!bwd)
        slab_q_fwd<T><<<blocks(g.S * g.N), kThreads, 0, st>>>(g, c3<T>(q, 0), c3<T>(q, 3), (const T*)q[6],
                                                              m0, m1, m2, (T)a, zmode & 1, w3<T>(q, 10));
    else
        slab_q_bwd<T><<<blocks(g.N), kThreads, 0, st>>>(g, c3<T>(q, 0), c3<T>(q, 3), (const T*)q[6], m0,
                                                        m1, m2, (T)a, zmode, w3<T>(q, 10), (T*)q[13]);
    return (int)cudaGetLastError();
}

// r_fwd [z0..2, eta0..2, ravg0..2, chr0..2, r_out0..2, eta_out0..2, ravg_out0..2, r0_0..2,
//        mh0..2, corr0..2]
// r_bwd [gr0..2, ge0..2, ga0..2, chr0..2, gz0..2, gein0..2, gcorr0..2, mh0..2, gain0..2, gr0_0..2]
template <typename T>
int run_slab_r(int bwd, const SlabDims& g, const void* const* q, double beta, double b, double w,
               int first, cudaStream_t st) {
    if (!bwd) {
        // first: bit 0 first sub-step, bit 1 accumulate (corr added to its destination)
        slab_r_fwd<T><<<blocks(g.S * g.N), kThreads, 0, st>>>(
            g, c3<T>(q, 0), c3<T>(q, 3), c3<T>(q, 6), c3<T>(q, 9), (T)beta, (T)b, (T)w, first & 1,
            w3<T>(q, 12), w3<T>(q, 15), w3<T>(q, 18), c3<T>(q, 21), (const uint8_t*)q[24],
            (const uint8_t*)q[25], (const uint8_t*)q[26], w3<T>(q, 27), (first & 2) ? 1 : 0);
    } else {
        RV<T> v;
        v.gr = c3<T>(q, 0);
        v.ge = c3<T>(q, 3);
        v.ga = c3<T>(q, 6);
        v.chr = c3<T>(q, 9);
        v.gcorr = c3<T>(q, 18);
        for (int k = 0; k < 3; ++k) v.mh[k] = (const uint8_t*)q[21 + k];
        v.b = (T)b;
        v.w = (T)w;
        slab_r_bwd<T><<<blocks(g.S * g.N), kThreads, 0, st>>>(g, v, (T)beta, w3<T>(q, 12), w3<T>(q, 15),
                                                              w3<T>(q, 24), w3<T>(q, 27),
                                                              (first & 2) ? 1 : 0);
    }
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------- DG macro update
// Macro-step update of the band with the p-fold UPML of the PML elements in one pass:
//   H mode: f' = f - ch[k] curl   (curl = cavg (S, 3, K, Np)),  PML: fold with sign -1
//   E mode: f' = ca[k] f + cb[k] curl (curl_c = T[2c] - T[2c+1] of the six terms), PML: sign +1
// (functional3d.lts_h_half_step / e_half_step with _dg_upml_fold). Transpose: one thread per
// element (sums of the ca/cb cotangents over nodes, components and shots; factor cotangents
// per PML entry summed over shots), deterministic.
template <typename T>
struct LtsUpd {
    const int* pml;                       // (K) PML slot of the element or -1; nullptr: none
    const T *rate, *f, *aux, *ch, *ca, *cb;
    const T *a1, *b1, *hist, *nw, *old;   // (3, Kp, Np)
};

template <typename T>
__device__ inline T upd_curl(int mode, const LtsUpd<T>& in, int S_, long long s, int K, int Np, int c,
                             long long k, int i) {
    const long long KN = (long long)K * Np;
    if (mode) {
        const long long t0 = (s * 6 + 2 * c) * KN + k * Np + i;
        return in.rate[t0] - in.rate[t0 + KN];
    }
    return in.rate[(s * 3 + c) * KN + k * Np + i];
}

template <typename T>
__global__ void ltsupd_fwd(int mode, int S, int K, int Np, int Kp, int p, LtsUpd<T> in,
                           T* __restrict__ fnew, T* __restrict__ auxn) {
    const long long KN = (long long)K * Np, N3 = 3 * KN;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * N3) return;
    const long long s = tid / N3, node = tid % N3;
    const int c = (int)(node / KN);
    const long long rem = node % KN, k = rem / Np;
    const int i = (int)(rem % Np);
    const T curl = upd_curl<T>(mode, in, S, s, K, Np, c, k, i);
    const int kp = in.pml != nullptr ? in.pml[k] : -1;
    const T fo = in.f[tid];
    if (kp < 0) {
        fnew[tid] = mode ? in.ca[k] * fo + in.cb[k] * curl : fo - in.ch[k] * curl;
        return;
    }
    const long long e = ((long long)c * Kp + kp) * Np + i, ea = s * 3LL * Kp * Np + e;
    T f = fo, aux = in.aux[ea];
    fold_run(p, mode ? (T)1 : (T)-1, in.a1[e], in.b1[e], in.hist[e], in.nw[e], in.old[e], curl, f, aux);
    fnew[tid] = f;
    auxn[ea] = aux;
}

template <typename T>
__global__ void ltsupd_bwd(int mode, int S, int K, int Np, int Kp, int p, LtsUpd<T> in,
                           const T* __restrict__ gfn, const T* __restrict__ gauxn,
                           T* __restrict__ grate, T* __restrict__ gf, T* __restrict__ gaux,
                           T* __restrict__ gca, T* __restrict__ gcb, T* __restrict__ ghist,
                           T* __restrict__ gnw, T* __restrict__ gold) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const long long KN = (long long)K * Np, N3 = 3 * KN, NA = 3LL * Kp * Np;
    const int kp = in.pml != nullptr ? in.pml[k] : -1;
    const T sign = mode ? (T)1 : (T)-1;
    const bool want = ghist != nullptr;
    T sca = 0, scb = 0;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < Np; ++i) {
            const long long node = (long long)c * KN + k * Np + i;
            const long long e = kp >= 0 ? ((long long)c * Kp + kp) * Np + i : -1;
            T sh = 0, sn = 0, so = 0;
            for (long long s = 0; s < S; ++s) {
                const long long id = s * N3 + node;
                const T g = gfn[id];
                const T curl = upd_curl<T>(mode, in, S, s, K, Np, c, k, i);
                T gfo, gr;
                if (kp < 0) {
                    if (mode) {
                        gfo = in.ca[k] * g;
                        gr = in.cb[k] * g;
                        sca += in.f[id] * g;
                        scb += curl * g;
                    } else {
                        gfo = g;
                        gr = -in.ch[k] * g;
                    }
                } else {
                    const long long ea = s * NA + e;
                    T ga, gh, gn, go;
                    fold_vjp(p, sign, in.a1[e], in.b1[e], in.hist[e], in.nw[e], in.old[e], in.f[id],
                             in.aux[ea], curl, g, gauxn != nullptr ? gauxn[ea] : (T)0, want, gfo, ga,
                             gr, gh, gn, go);
                    gaux[ea] = ga;
                    sh += gh;
                    sn += gn;
                    so += go;
                }
                gf[id] = gfo;
                if (mode) {
                    const long long t0 = (s * 6 + 2 * c) * KN + k * Np + i;
                    grate[t0] = gr;
                    grate[t0 + KN] = -gr;
                } else {
                    grate[id] = gr;
                }
            }
            if (want && kp >= 0) {
                ghist[e] = sh;
                gnw[e] = sn;
                gold[e] = so;
            }
        }
    }
    if (mode && gca != nullptr) {
        gca[k] = sca;
        gcb[k] = scb;
    }
}

// ltsupd: [pml, rate, f, aux, ch, ca, cb, a1, b1, hist, nw, old, fnew, auxn]
//   bwd:  [pml, rate, f, aux, ch, ca, cb, a1, b1, hist, nw, old, gfn, gauxn, grate, gf, gaux,
//          gca, gcb, ghist, gnw, gold]
template <typename T>
int run_ltsupd(int bwd, int mode, int S, int K, int Np, int Kp, int p, const void* const* q,
               cudaStream_t st) {
    LtsUpd<T> in;
    in.pml = (const int*)q[0];
    in.rate = (const T*)q[1];
    in.f = (const T*)q[2];
    in.aux = (const T*)q[3];
    in.ch = (const T*)q[4];
    in.ca = (const T*)q[5];
    in.cb = (const T*)q[6];
    in.a1 = (const T*)q[7];
    in.b1 = (const T*)q[8];
    in.hist = (const T*)q[9];
    in.nw = (const T*)q[10];
    in.old = (const T*)q[11];
    if (!bwd)
        ltsupd_fwd<T><<<blocks((long long)S * 3 * K * Np), kThreads, 0, st>>>(mode, S, K, Np, Kp, p, in,
                                                                             (T*)q[12], (T*)q[13]);
    else
        ltsupd_bwd<T><<<blocks(K), kThreads, 0, st>>>(mode, S, K, Np, Kp, p, in, (const T*)q[12],
                                                      (const T*)q[13], (T*)q[14], (T*)q[15], (T*)q[16],
                                                      (T*)q[17], (T*)q[18], (T*)q[19], (T*)q[20],
                                                      (T*)q[21]);
    return (int)cudaGetLastError();
}

// fold: [idx_f, idx_a, idx_r, f, aux, rate, a1, b1, hist, nw, old, fval, auxn] (+ bwd:
// [gfval, gauxn, gf, gaux, grate, ghist, gnw, gold] replacing fval/auxn at 11..)
template <typename T>
int run_fold(int bwd, int S, long long n, int p, double sign, long long Nf, long long Na, long long Nr,
             const void* const* q, cudaStream_t st) {
    FoldIn<T> in;
    in.idx_f = (const int*)q[0];
    in.idx_a = (const int*)q[1];
    in.idx_r = (const int*)q[2];
    in.f = (const T*)q[3];
    in.aux = (const T*)q[4];
    in.rate = (const T*)q[5];
    in.a1 = (const T*)q[6];
    in.b1 = (const T*)q[7];
    in.hist = (const T*)q[8];
    in.nw = (const T*)q[9];
    in.old = (const T*)q[10];
    in.Nf = Nf;
    in.Na = Na;
    in.Nr = Nr;
    // bwd: 0 forward (compact), 2 forward scattered into f / aux-shaped outputs;
    //      1 transpose, 1 | 2: compact g_f / g_aux, 1 | 4: gather-and-zero output cotangents
    if (!(bwd & 1))
        fold_fwd<T><<<blocks(n), kThreads, 0, st>>>(S, n, p, (T)sign, in, (T*)q[11], (T*)q[12],
                                                    (bwd & 2) ? 1 : 0);
    else
        fold_bwd<T><<<blocks(n), kThreads, 0, st>>>(S, n, p, (T)sign, in, (T*)q[11], (T*)q[12],
                                                    (T*)q[13], (T*)q[14], (T*)q[15], (T*)q[16],
                                                    (T*)q[17], (T*)q[18], bwd & 6);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------- 2D slabs (TM)
// Slab tensors (S, n, NY): z (Ez-like), eta = (eta_x Hx-like, eta_y Hy-like)
// (DeepGPR.hybrid.functional.lts_h_half_step):
//   d_e(z): dEy[i,j] = z[i,j+1] - z[i,j] (0 at j = NY-1), dEx[i,j] = z[i+1,j] - z[i,j] (0 at i = n-1)
//   curl_h(hx, hy) = (hy[i,j] - hy[i-1,j])/dx [i > 0] - (hx[i,j] - hx[i,j-1])/dy [j > 0]
//   slab_q2: out = fine_e ? z + a cb0 curl_h(eta) : z
//   slab_r2: (dEy, dEx) = d_e(z); ravg_out = ravg + w (dEy, dEx) [first: w (dEy, dEx)];
//            eta_out = (beta ex - b cHx dEy, beta ey + b cHy dEx) [first: (-b cHx dEy, b cHy dEx)];
//            last sub-step (r0): the corrections of the fine Hx / Hy rows and their CPML
//            auxiliaries, corr = (-(cHx + Ay) dy, -Fy dy, (cHy + Ax) dx, -Fx dx) with
//            dy = fine_hx (ravg_y_out - r0_y), dx = fine_hy (ravg_x_out - r0_x).
struct Slab2 {
    int S, n, NY;
    long long N;
    double dx, dy;
};

template <typename T>
__global__ void slab_q2_fwd(Slab2 g, const T* __restrict__ ex, const T* __restrict__ ey,
                            const T* __restrict__ z, const T* __restrict__ cb0,
                            const uint8_t* __restrict__ me, T a, T* __restrict__ out) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)g.S * g.N) return;
    const long long node = tid % g.N;
    const int i = (int)(node / g.NY), j = (int)(node % g.NY);
    const T q = (i > 0 ? ey[tid] - ey[tid - g.NY] : (T)0) / (T)g.dx
                - (j > 0 ? ex[tid] - ex[tid - 1] : (T)0) / (T)g.dy;
    out[tid] = me[node] ? z[tid] + a * cb0[node] * q : z[tid];
}

template <typename T>
__global__ void slab_q2_bwd(Slab2 g, const T* __restrict__ ex, const T* __restrict__ ey,
                            const T* __restrict__ gq, const T* __restrict__ cb0,
                            const uint8_t* __restrict__ me, T a, T* __restrict__ gex,
                            T* __restrict__ gey, T* __restrict__ gcb0, int acc) {
    // acc bit 1: accumulate g_ex / g_ey, bit 2: accumulate g_cb0
    const long long node = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= g.N) return;
    const int i = (int)(node / g.NY), j = (int)(node % g.NY);
    T acc_sum = 0;
    for (int s = 0; s < g.S; ++s) {
        const long long base = (long long)s * g.N, x = base + node;
        auto v = [&](long long y) { return me[y - base] ? a * cb0[y - base] * gq[y] : (T)0; };
        const T vy = ((i > 0 ? v(x) : (T)0) - (i < g.n - 1 ? v(x + g.NY) : (T)0)) / (T)g.dx;
        const T vx = -((j > 0 ? v(x) : (T)0) - (j < g.NY - 1 ? v(x + 1) : (T)0)) / (T)g.dy;
        if (acc & 2) {
            gey[x] += vy;
            gex[x] += vx;
        } else {
            gey[x] = vy;
            gex[x] = vx;
        }
        if (gcb0 != nullptr && me[node]) {
            const T q = (i > 0 ? ey[x] - ey[x - g.NY] : (T)0) / (T)g.dx
                        - (j > 0 ? ex[x] - ex[x - 1] : (T)0) / (T)g.dy;
            acc_sum += a * q * gq[x];
        }
    }
    if (gcb0 != nullptr) {
        if (acc & 4)
            gcb0[node] += acc_sum;
        else
            gcb0[node] = acc_sum;
    }
}

template <typename T>
struct Slab2Static {                 // per slab node (n, NY)
    const T *cHx, *cHy, *Ay, *Fy, *Ax, *Fx;
    const uint8_t *fhx, *fhy;
};

template <typename T>
__global__ void slab_r2_fwd(Slab2 g, Slab2Static<T> st, const T* __restrict__ z, const T* ex, const T* ey,
                            const T* ay, const T* ax, T beta, T b, T w, int first, T* ry_out, T* rx_out,
                            T* ex_out, T* ey_out, T* ay_out, T* ax_out, const T* r0y, const T* r0x,
                            T* c_hx, T* c_phy, T* c_hy, T* c_phx, int acc) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)g.S * g.N) return;
    const long long node = tid % g.N;
    const int i = (int)(node / g.NY), j = (int)(node % g.NY);
    const T dEy = j < g.NY - 1 ? z[tid + 1] - z[tid] : (T)0;
    const T dEx = i < g.n - 1 ? z[tid + g.NY] - z[tid] : (T)0;
    const T ray = first ? w * dEy : ay[tid] + w * dEy;
    const T rax = first ? w * dEx : ax[tid] + w * dEx;
    if (ry_out != nullptr) {
        ry_out[tid] = dEy;
        rx_out[tid] = dEx;
    }
    if (ay_out != nullptr) {
        ay_out[tid] = ray;
        ax_out[tid] = rax;
    }
    if (ex_out != nullptr) {
        const T bx = b * st.cHx[node], by = b * st.cHy[node];
        ex_out[tid] = first ? -bx * dEy : beta * ex[tid] - bx * dEy;
        ey_out[tid] = first ? by * dEx : beta * ey[tid] + by * dEx;
    }
    if (c_hx != nullptr) {
        const T dy_ = st.fhx[node] ? ray - r0y[tid] : (T)0;
        const T dx_ = st.fhy[node] ? rax - r0x[tid] : (T)0;
        if (acc) {           // added directly to the fine rows (c_*: their slab views)
            c_hx[tid] += -(st.cHx[node] + st.Ay[node]) * dy_;
            c_phy[tid] += -st.Fy[node] * dy_;
            c_hy[tid] += (st.cHy[node] + st.Ax[node]) * dx_;
            c_phx[tid] += -st.Fx[node] * dx_;
        } else {
            c_hx[tid] = -(st.cHx[node] + st.Ay[node]) * dy_;
            c_phy[tid] = -st.Fy[node] * dy_;
            c_hy[tid] = (st.cHy[node] + st.Ax[node]) * dx_;
            c_phx[tid] = -st.Fx[node] * dx_;
        }
    }
}

template <typename T>
struct R2V {                          // cotangent sources (nullable) of the r2 outputs
    const T *gry, *grx, *gexo, *geyo, *gay, *gax, *gchx, *gcphy, *gchy, *gcphx;
    Slab2Static<T> st;
    T b, w;
    __device__ inline T gay_eff(long long x, long long node) const {
        T v = gay != nullptr ? gay[x] : (T)0;
        if (gchx != nullptr && st.fhx[node])
            v += -(st.cHx[node] + st.Ay[node]) * gchx[x] - st.Fy[node] * gcphy[x];
        return v;
    }
    __device__ inline T gax_eff(long long x, long long node) const {
        T v = gax != nullptr ? gax[x] : (T)0;
        if (gchy != nullptr && st.fhy[node])
            v += (st.cHy[node] + st.Ax[node]) * gchy[x] - st.Fx[node] * gcphx[x];
        return v;
    }
    __device__ inline T vy(long long x, long long node) const {
        T v = w * gay_eff(x, node);
        if (gry != nullptr) v += gry[x];
        if (gexo != nullptr) v -= b * st.cHx[node] * gexo[x];
        return v;
    }
    __device__ inline T vx(long long x, long long node) const {
        T v = w * gax_eff(x, node);
        if (grx != nullptr) v += grx[x];
        if (geyo != nullptr) v += b * st.cHy[node] * geyo[x];
        return v;
    }
};

template <typename T>
__global__ void slab_r2_bwd(Slab2 g, R2V<T> v, T beta, T* gz, T* gexi, T* geyi, T* gayi, T* gaxi,
                            T* gr0y, T* gr0x, int acc) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)g.S * g.N) return;
    const long long node = tid % g.N, base = tid - node;
    const int i = (int)(node / g.NY), j = (int)(node % g.NY);
    T o = 0;
    if (j > 0) o += v.vy(tid - 1, node - 1);
    if (j < g.NY - 1) o -= v.vy(tid, node);
    if (i > 0) o += v.vx(tid - g.NY, node - g.NY);
    if (i < g.n - 1) o -= v.vx(tid, node);
    if (acc)
        gz[tid] += o;
    else
        gz[tid] = o;
    (void)base;
    if (gexi != nullptr) {
        gexi[tid] = beta * (v.gexo != nullptr ? v.gexo[tid] : (T)0);
        geyi[tid] = beta * (v.geyo != nullptr ? v.geyo[tid] : (T)0);
    }
    if (gayi != nullptr) {
        gayi[tid] = v.gay_eff(tid, node);
        gaxi[tid] = v.gax_eff(tid, node);
    }
    if (gr0y != nullptr) {
        const Slab2Static<T>& st = v.st;
        gr0y[tid] = (v.gchx != nullptr && st.fhx[node])
                        ? (st.cHx[node] + st.Ay[node]) * v.gchx[tid] + st.Fy[node] * v.gcphy[tid] : (T)0;
        gr0x[tid] = (v.gchy != nullptr && st.fhy[node])
                        ? -(st.cHy[node] + st.Ax[node]) * v.gchy[tid] + st.Fx[node] * v.gcphx[tid] : (T)0;
    }
}

inline Slab2 slab2_dims(int S, int n, int NY, const double* h) {
    Slab2 g;
    g.S = S;
    g.n = n;
    g.NY = NY;
    g.N = (long long)n * NY;
    g.dx = h[0];
    g.dy = h[1];
    return g;
}

// q2_fwd [ex, ey, z, cb0, me, out]; q2_bwd [ex, ey, g, cb0, me, gex, gey, gcb0]
template <typename T>
int run_slab_q2(int bwd, const Slab2& g, const void* const* q, double a, cudaStream_t st) {
    if (!bwd)
        slab_q2_fwd<T><<<blocks(g.S * g.N), kThreads, 0, st>>>(
            g, (const T*)q[0], (const T*)q[1], (const T*)q[2], (const T*)q[3], (const uint8_t*)q[4], (T)a,
            (T*)q[5]);
    else         // bwd bit 1: accumulate g_ex / g_ey, bit 2: accumulate g_cb0
        slab_q2_bwd<T><<<blocks(g.N), kThreads, 0, st>>>(
            g, (const T*)q[0], (const T*)q[1], (const T*)q[2], (const T*)q[3], (const uint8_t*)q[4], (T)a,
            (T*)q[5], (T*)q[6], (T*)q[7], bwd & 6);
    return (int)cudaGetLastError();
}

template <typename T>
inline Slab2Static<T> slab2_static(const void* const* q, int o) {
    Slab2Static<T> st;
    st.cHx = (const T*)q[o];
    st.cHy = (const T*)q[o + 1];
    st.Ay = (const T*)q[o + 2];
    st.Fy = (const T*)q[o + 3];
    st.Ax = (const T*)q[o + 4];
    st.Fx = (const T*)q[o + 5];
    st.fhx = (const uint8_t*)q[o + 6];
    st.fhy = (const uint8_t*)q[o + 7];
    return st;
}

// r2_fwd [z, ex, ey, ay, ax, (cHx, cHy, Ay, Fy, Ax, Fx, fhx, fhy), ry_out, rx_out, ex_out, ey_out,
//         ay_out, ax_out, r0y, r0x, c_hx, c_phy, c_hy, c_phx]
// r2_bwd [gry, grx, gexo, geyo, gay, gax, (statics 8), gchx, gcphy, gchy, gcphx, gz, gexi, geyi,
//         gayi, gaxi, gr0y, gr0x]
template <typename T>
int run_slab_r2(int bwd, const Slab2& g, const void* const* q, double beta, double b, double w, int first,
                cudaStream_t st) {
    if (!bwd) {
        slab_r2_fwd<T><<<blocks(g.S * g.N), kThreads, 0, st>>>(
            g, slab2_static<T>(q, 5), (const T*)q[0], (const T*)q[1], (const T*)q[2], (const T*)q[3],
            (const T*)q[4], (T)beta, (T)b, (T)w, first & 1, (T*)q[13], (T*)q[14], (T*)q[15], (T*)q[16],
            (T*)q[17], (T*)q[18], (const T*)q[19], (const T*)q[20], (T*)q[21], (T*)q[22], (T*)q[23],
            (T*)q[24], (first & 2) ? 1 : 0);
    } else {
        R2V<T> v;
        v.gry = (const T*)q[0];
        v.grx = (const T*)q[1];
        v.gexo = (const T*)q[2];
        v.geyo = (const T*)q[3];
        v.gay = (const T*)q[4];
        v.gax = (const T*)q[5];
        v.st = slab2_static<T>(q, 6);
        v.gchx = (const T*)q[14];
        v.gcphy = (const T*)q[15];
        v.gchy = (const T*)q[16];
        v.gcphx = (const T*)q[17];
        v.b = (T)b;
        v.w = (T)w;
        slab_r2_bwd<T><<<blocks(g.S * g.N), kThreads, 0, st>>>(g, v, (T)beta, (T*)q[18], (T*)q[19],
                                                               (T*)q[20], (T*)q[21], (T*)q[22], (T*)q[23],
                                                               (T*)q[24], (first & 2) ? 1 : 0);
    }
    return (int)cudaGetLastError();
}

}  // namespace

extern "C" {

int lts2d_z_fwd(int dtype, int S, int K, int NP, const void* Te, const void* z, const void* cb0,
                double a, void* out, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    const long long KN = (long long)K * NP;
    if (dtype == 0)
        z2_fwd<float><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const float*)Te, (const float*)z, (const float*)cb0, (float)a, (float*)out);
    else
        z2_fwd<double><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const double*)Te, (const double*)z, (const double*)cb0, a, (double*)out);
    return (int)cudaGetLastError();
}

int lts2d_z_bwd(int dtype, int S, int K, int NP, const void* Te, const void* g, const void* cb0,
                double a, void* gTe, void* gcb0, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == 0)
        z2_bwd<float><<<blocks(K), kThreads, 0, st>>>(S, K, NP, (const float*)Te, (const float*)g,
                                                      (const float*)cb0, (float)a, (float*)gTe,
                                                      (float*)gcb0);
    else
        z2_bwd<double><<<blocks(K), kThreads, 0, st>>>(S, K, NP, (const double*)Te, (const double*)g,
                                                       (const double*)cb0, a, (double*)gTe,
                                                       (double*)gcb0);
    return (int)cudaGetLastError();
}

int lts2d_eta_fwd(int dtype, int S, int K, int NP, const void* Tt, const void* ex, const void* ey,
                  const void* tavg, const void* ch, double beta, double b, double w, int first,
                  void* ex_out, void* ey_out, void* tavg_out, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    const long long KN = (long long)K * NP;
    if (dtype == 0)
        eta2_fwd<float><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const float*)Tt, (const float*)ex, (const float*)ey, (const float*)tavg,
            (const float*)ch, (float)beta, (float)b, (float)w, first, (float*)ex_out, (float*)ey_out,
            (float*)tavg_out);
    else
        eta2_fwd<double><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const double*)Tt, (const double*)ex, (const double*)ey,
            (const double*)tavg, (const double*)ch, beta, b, w, first, (double*)ex_out,
            (double*)ey_out, (double*)tavg_out);
    return (int)cudaGetLastError();
}

int lts2d_eta_bwd(int dtype, int S, int K, int NP, const void* gex, const void* gey,
                  const void* gtavg, const void* ch, double beta, double b, double w, void* gT,
                  void* gex_out, void* gey_out, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    const long long KN = (long long)K * NP;
    if (dtype == 0)
        eta2_bwd<float><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const float*)gex, (const float*)gey, (const float*)gtavg, (const float*)ch,
            (float)beta, (float)b, (float*)gT, (float*)gex_out, (float*)gey_out, (float)w);
    else
        eta2_bwd<double><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const double*)gex, (const double*)gey, (const double*)gtavg,
            (const double*)ch, beta, b, (double*)gT, (double*)gex_out, (double*)gey_out, w);
    return (int)cudaGetLastError();
}

int lts3d_z_fwd(int dtype, int S, int K, int NP, const void* Te, const void* z, const void* cb0,
                double a, void* out, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    const long long KN = (long long)K * NP;
    if (dtype == 0)
        z_fwd<float><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const float*)Te, (const float*)z, (const float*)cb0, (float)a, (float*)out);
    else
        z_fwd<double><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const double*)Te, (const double*)z, (const double*)cb0, a, (double*)out);
    return (int)cudaGetLastError();
}

int lts3d_z_bwd(int dtype, int S, int K, int NP, const void* Te, const void* g, const void* cb0,
                double a, void* gTe, void* gcb0, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == 0)
        z_bwd<float><<<blocks(K), kThreads, 0, st>>>(S, K, NP, (const float*)Te, (const float*)g,
                                                     (const float*)cb0, (float)a, (float*)gTe,
                                                     (float*)gcb0);
    else
        z_bwd<double><<<blocks(K), kThreads, 0, st>>>(S, K, NP, (const double*)Te, (const double*)g,
                                                      (const double*)cb0, a, (double*)gTe,
                                                      (double*)gcb0);
    return (int)cudaGetLastError();
}

int lts3d_eta_fwd(int dtype, int S, int K, int NP, const void* Tt, const void* eta,
                  const void* cavg, const void* ch, double beta, double b, double w, int first,
                  void* eta_out, void* cavg_out, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    const long long KN = (long long)K * NP;
    if (dtype == 0)
        eta_fwd<float><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const float*)Tt, (const float*)eta, (const float*)cavg, (const float*)ch,
            (float)beta, (float)b, (float)w, first, (float*)eta_out, (float*)cavg_out);
    else
        eta_fwd<double><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const double*)Tt, (const double*)eta, (const double*)cavg,
            (const double*)ch, beta, b, w, first, (double*)eta_out, (double*)cavg_out);
    return (int)cudaGetLastError();
}

int lts3d_eta_bwd(int dtype, int S, int K, int NP, const void* g_eta_new, const void* g_cavg_new,
                  const void* ch, double beta, double b, double w, void* gT, void* g_eta,
                  void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    const long long KN = (long long)K * NP;
    if (dtype == 0)
        eta_bwd<float><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const float*)g_eta_new, (const float*)g_cavg_new, (const float*)ch,
            (float)beta, (float)b, (float)w, (float*)gT, (float*)g_eta);
    else
        eta_bwd<double><<<blocks((long long)S * KN), kThreads, 0, st>>>(
            S, KN, NP, (const double*)g_eta_new, (const double*)g_cavg_new, (const double*)ch,
            beta, b, w, (double*)gT, (double*)g_eta);
    return (int)cudaGetLastError();
}

int lts3d_slab_q(int dtype, int bwd, int S, int n, int NY, int NZ, const double* h, const void** q,
                 double a, int zmode, void* stream) {
    const SlabDims g = slab_dims(S, n, NY, NZ, h);
    cudaStream_t st = (cudaStream_t)stream;
    return dtype == 0 ? run_slab_q<float>(bwd, g, q, a, zmode, st) : run_slab_q<double>(bwd, g, q, a, zmode, st);
}

int lts3d_slab_r(int dtype, int bwd, int S, int n, int NY, int NZ, const double* h, const void** q,
                 double beta, double b, double w, int first, void* stream) {
    const SlabDims g = slab_dims(S, n, NY, NZ, h);
    cudaStream_t st = (cudaStream_t)stream;
    return dtype == 0 ? run_slab_r<float>(bwd, g, q, beta, b, w, first, st)
                      : run_slab_r<double>(bwd, g, q, beta, b, w, first, st);
}

int lts3d_fold(int dtype, int bwd, int S, long long n, int p, double sign, long long Nf, long long Na,
               long long Nr, const void** q, void* stream) {
    if (p > kMaxP) return -2;
    cudaStream_t st = (cudaStream_t)stream;
    return dtype == 0 ? run_fold<float>(bwd, S, n, p, sign, Nf, Na, Nr, q, st)
                      : run_fold<double>(bwd, S, n, p, sign, Nf, Na, Nr, q, st);
}

int lts3d_update(int dtype, int bwd, int mode, int S, int K, int Np, int Kp, int p, const void** q,
                 void* stream) {
    if (p > kMaxP) return -2;
    cudaStream_t st = (cudaStream_t)stream;
    return dtype == 0 ? run_ltsupd<float>(bwd, mode, S, K, Np, Kp, p, q, st)
                      : run_ltsupd<double>(bwd, mode, S, K, Np, Kp, p, q, st);
}

int lts2d_slab_q(int dtype, int bwd, int S, int n, int NY, const double* h, const void** q, double a,
                 void* stream) {
    const Slab2 g = slab2_dims(S, n, NY, h);
    cudaStream_t st = (cudaStream_t)stream;
    return dtype == 0 ? run_slab_q2<float>(bwd, g, q, a, st) : run_slab_q2<double>(bwd, g, q, a, st);
}

int lts2d_slab_r(int dtype, int bwd, int S, int n, int NY, const double* h, const void** q, double beta,
                 double b, double w, int first, void* stream) {
    const Slab2 g = slab2_dims(S, n, NY, h);
    cudaStream_t st = (cudaStream_t)stream;
    return dtype == 0 ? run_slab_r2<float>(bwd, g, q, beta, b, w, first, st)
                      : run_slab_r2<double>(bwd, g, q, beta, b, w, first, st);
}

}  // extern "C"

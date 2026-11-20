// Fused "glue" of the 3D hybrid step and of its transpose: the DGTD-FDTD mortar
// coupling (projected Yee trace E*, interface flux input X_h, ghost planes, interface
// penalty rows), source injection and receiver sampling. Each operation replaces a
// chain of PyTorch gather / scatter / elementwise kernels by one or two launches with
// the same arithmetic (DeepGPR.hybrid.functional3d: interface_trace, interface_x,
// ghost planes in h_half_step, interface_penalty_rows, inject, sample; their
// transposes in vjp3d / vjp3d_lts).
//
// Sparse operators are stored in CSR form twice (by output and by input) so that
// every kernel gathers: a thread owns its output and no atomics are needed, except
// where several entries accumulate into one coefficient (face penalty gamma, source
// coefficients, source / receiver points sharing nodes).
//
// Layouts (S shots): Yee fields (S, NX, NY, NZ); DG fields (S, 3, K, Np); interface
// planes per coupling i at a row of a field (or of an LTS slab array); compact
// interface slots r in [0, nX) in coupling order; penalty rows (S, ncp, 2, NY, NZ).

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kT = 256;
inline int nblk(long long n) { return (int)((n + kT - 1) / kT); }

constexpr int MAXCP = 4;

template <typename T>
struct Planes {                 // per coupling: two plane arrays (lattice 0 / lattice 1)
    T* p[2][MAXCP];
    long long stride[MAXCP];    // per-shot stride of the arrays (elements)
    long long off_a[MAXCP];     // offset of plane a (e.g. interface or ghost row)
    long long off_b[MAXCP];     // offset of plane b (e.g. inner row)
};

template <typename T>
__device__ inline void atomic_add(T* a, T v) { atomicAdd(a, v); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ inline void atomic_add(double* a, double v) {
    unsigned long long* p = (unsigned long long*)a;
    unsigned long long old = *p, assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(v + __longlong_as_double(assumed)));
    } while (assumed != old);
}
#endif

// ------------------------------------------------------------------ trace E* (+ X_h)
// E*[s, c, r] = sum_e val_e plane_c(cp(r))[s, col_e]   (CSR rows c * nX + r)
// X_h[s, 0, r] = g_r n0_r E*_z,  X_h[s, 1, r] = -g_r n0_r E*_y   (g_r = g_if[face_r])
template <typename T>
__global__ void trace_fwd(int S, int nX, Planes<T> P, const int* __restrict__ rp,
                          const int* __restrict__ col, const T* __restrict__ val,
                          const int* __restrict__ cp_of, const int* __restrict__ face,
                          const T* __restrict__ n0, const T* __restrict__ gif,
                          T* __restrict__ Ex, T* __restrict__ Xh) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * nX) return;
    const int s = (int)(tid / nX), r = (int)(tid % nX);
    const int i = cp_of[r];
    const T* y = P.p[0][i] + s * P.stride[i] + P.off_a[i];
    const T* z = P.p[1][i] + s * P.stride[i] + P.off_a[i];
    T ey = 0, ez = 0;
    for (int e = rp[r]; e < rp[r + 1]; ++e) ey += val[e] * y[col[e]];
    for (int e = rp[nX + r]; e < rp[nX + r + 1]; ++e) ez += val[e] * z[col[e]];
    Ex[((long long)s * 2 + 0) * nX + r] = ey;
    Ex[((long long)s * 2 + 1) * nX + r] = ez;
    if (Xh != nullptr) {
        const T gn = gif[face[r]] * n0[r];
        Xh[((long long)s * 2 + 0) * nX + r] = gn * ez;
        Xh[((long long)s * 2 + 1) * nX + r] = -gn * ey;
    }
}

// transpose: plane_c(i)[s, j] += sum_e val_e gE[s, c, r_e]   (CSR rows (i * 2 + c) * NYZ + j)
// gE = gX (+ the X_h transpose: gE_y = -g n0 gXh_z, gE_z = g n0 gXh_y)
template <typename T>
__global__ void trace_bwd(int S, int nX, int ncp, int NYZ, Planes<T> P, const int* __restrict__ rp,
                          const int* __restrict__ col, const T* __restrict__ val,
                          const int* __restrict__ face, const T* __restrict__ n0,
                          const T* __restrict__ gif, const T* __restrict__ gX,
                          const T* __restrict__ gXh) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * ncp * 2 * NYZ) return;
    const int s = (int)(tid / ((long long)ncp * 2 * NYZ));
    const long long row = tid % ((long long)ncp * 2 * NYZ);
    const int i = (int)(row / (2 * NYZ)), c = (int)((row / NYZ) % 2), j = (int)(row % NYZ);
    const int b = rp[row], e1 = rp[row + 1];
    if (b == e1) return;
    T acc = 0;
    const T* gx = gX != nullptr ? gX + ((long long)s * 2 + c) * nX : nullptr;
    const T* gh = gXh != nullptr ? gXh + ((long long)s * 2 + (1 - c)) * nX : nullptr;
    for (int e = b; e < e1; ++e) {
        const int r = col[e];
        T g = gx != nullptr ? gx[r] : T(0);
        if (gh != nullptr) {
            const T gs = gif[face[r]] * n0[r];
            g += c == 0 ? -gs * gh[r] : gs * gh[r];
        }
        acc += val[e] * g;
    }
    P.p[c][i][s * P.stride[i] + P.off_a[i] + j] += acc;
}

// g_if cotangent of X_h: gacc[face_r] += n0_r sum_s (gXh_y E*_z - gXh_z E*_y)
template <typename T>
__global__ void xh_gif(int S, int nX, const int* __restrict__ face, const T* __restrict__ n0,
                       const T* __restrict__ gXh, const T* __restrict__ Ex, T* __restrict__ gacc) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nX) return;
    T a = 0;
    for (int s = 0; s < S; ++s) {
        const long long b = (long long)s * 2 * nX;
        a += gXh[b + r] * Ex[b + nX + r] - gXh[b + nX + r] * Ex[b + r];
    }
    atomic_add(&gacc[face[r]], a * n0[r]);
}

// ------------------------------------------------------------------ ghost planes
// lattice l of coupling i (l = 0: Hz on the Ey nodes from DG Hz; l = 1: Hy on the Ez
// nodes from DG Hy): v_j = sum_e val_e F[s, comp_l, node_e] (CSR rows (i * 2 + l) * NYZ + j),
// plane_l(i)[s, a, j] = 2 v_j - plane_l(i)[s, b, j]
template <typename T>
__global__ void ghost_fwd(int S, int ncp, int NYZ, long long KN, Planes<T> P,
                          const int* __restrict__ rp, const int* __restrict__ node,
                          const T* __restrict__ val, const T* __restrict__ F) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * ncp * 2 * NYZ) return;
    const int s = (int)(tid / ((long long)ncp * 2 * NYZ));
    const long long row = tid % ((long long)ncp * 2 * NYZ);
    const int i = (int)(row / (2 * NYZ)), l = (int)((row / NYZ) % 2), j = (int)(row % NYZ);
    const T* f = F + ((long long)s * 3 + (l == 0 ? 2 : 1)) * KN;
    T v = 0;
    for (int e = rp[row]; e < rp[row + 1]; ++e) v += val[e] * f[node[e]];
    T* t = P.p[l][i] + s * P.stride[i];
    t[P.off_a[i] + j] = T(2) * v - t[P.off_b[i] + j];
}

// transpose, DG part: lF[s, comp_l, node_r] += 2 sum_j val gg_j (gg: plane a of the cotangent)
// (CSR by slot: rows l * nX + r)
template <typename T>
__global__ void ghost_bwd_dg(int S, int nX, long long KN, Planes<T> P, const int* __restrict__ rp,
                             const int* __restrict__ col, const T* __restrict__ val,
                             const int* __restrict__ cp_of, const int* __restrict__ node_of,
                             T* __restrict__ lF) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * 2 * nX) return;
    const int s = (int)(tid / (2 * nX));
    const int row = (int)(tid % (2 * nX));
    const int l = row / nX, r = row % nX;
    const int i = cp_of[r];
    const T* gg = P.p[l][i] + s * P.stride[i] + P.off_a[i];
    T a = 0;
    for (int e = rp[row]; e < rp[row + 1]; ++e) a += val[e] * gg[col[e]];
    lF[((long long)s * 3 + (l == 0 ? 2 : 1)) * KN + node_of[r]] += T(2) * a;
}

// transpose, Yee part: plane b -= plane a; plane a = 0
template <typename T>
__global__ void ghost_bwd_yee(int S, int ncp, int NYZ, Planes<T> P) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * ncp * 2 * NYZ) return;
    const int s = (int)(tid / ((long long)ncp * 2 * NYZ));
    const long long row = tid % ((long long)ncp * 2 * NYZ);
    const int i = (int)(row / (2 * NYZ)), l = (int)((row / NYZ) % 2), j = (int)(row % NYZ);
    T* t = P.p[l][i] + s * P.stride[i];
    const T g = t[P.off_a[i] + j];
    t[P.off_b[i] + j] -= g;
    t[P.off_a[i] + j] = T(0);
}

// ------------------------------------------------------------------ interface penalty rows
// pen[s, i, l, j] = scale (sum_e val g_e E[s, l+1, node_e] - (sum_e val g_e) fe_{l+1}[s, row_i, j])
template <typename T>
__global__ void pen_fwd(int S, int ncp, int NYZ, long long KN, Planes<T> P,
                        const int* __restrict__ rp, const int* __restrict__ node,
                        const int* __restrict__ efc, const T* __restrict__ val,
                        const T* __restrict__ gif, const T* __restrict__ E, T scale,
                        T* __restrict__ pen) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * ncp * 2 * NYZ) return;
    const int s = (int)(tid / ((long long)ncp * 2 * NYZ));
    const long long row = tid % ((long long)ncp * 2 * NYZ);
    const int i = (int)(row / (2 * NYZ)), l = (int)((row / NYZ) % 2), j = (int)(row % NYZ);
    const T* e = E + ((long long)s * 3 + l + 1) * KN;
    T a = 0, b = 0;
    for (int q = rp[row]; q < rp[row + 1]; ++q) {
        const T vg = val[q] * gif[efc[q]];
        a += vg * e[node[q]];
        b += vg;
    }
    const T f = P.p[l][i][s * P.stride[i] + P.off_a[i] + j];
    pen[(((long long)s * ncp + i) * 2 + l) * NYZ + j] = scale * (a - b * f);
}

// transpose, slot part (CSR by slot, rows l * nX + r):
//   q = scale sum_j val gp_j;  lE[s, l+1, node_r] += g_r q;
//   gacc[face_r] += E[s, l+1, node_r] q - scale sum_j val gp_j fe_{l+1}[s, row, j]
template <typename T>
__global__ void pen_bwd_slot(int S, int nX, int ncp, int NYZ, long long KN, Planes<T> P,
                             const int* __restrict__ rp, const int* __restrict__ col,
                             const T* __restrict__ val, const int* __restrict__ cp_of,
                             const int* __restrict__ node_of, const int* __restrict__ face,
                             const T* __restrict__ gif, const T* __restrict__ E,
                             const T* __restrict__ gp1, const T* __restrict__ gp2, T scale,
                             T* __restrict__ lE, T* __restrict__ gacc) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * 2 * nX) return;
    const int s = (int)(tid / (2 * nX));
    const int row = (int)(tid % (2 * nX));
    const int l = row / nX, r = row % nX;
    const int i = cp_of[r];
    const long long go = (((long long)s * ncp + i) * 2 + l) * NYZ;
    const T* f = P.p[l][i] + s * P.stride[i] + P.off_a[i];
    T q = 0, rr = 0;
    for (int e = rp[row]; e < rp[row + 1]; ++e) {
        const int j = col[e];
        T g = gp1 != nullptr ? gp1[go + j] : T(0);
        if (gp2 != nullptr) g += gp2[go + j];
        q += val[e] * g;
        rr += val[e] * g * f[j];
    }
    q *= scale;
    rr *= scale;
    const long long id = ((long long)s * 3 + l + 1) * KN + node_of[r];
    lE[id] += gif[face[r]] * q;
    if (gacc != nullptr) atomic_add(&gacc[face[r]], E[id] * q - rr);
}

// transpose, Yee part (CSR by plane node): lfe_{l+1}[s, row_i, j] -= scale (sum_e val g_e) gp_j
template <typename T>
__global__ void pen_bwd_yee(int S, int ncp, int NYZ, Planes<T> P, const int* __restrict__ rp,
                            const int* __restrict__ efc, const T* __restrict__ val,
                            const T* __restrict__ gif, const T* __restrict__ gp1,
                            const T* __restrict__ gp2, T scale) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * ncp * 2 * NYZ) return;
    const int s = (int)(tid / ((long long)ncp * 2 * NYZ));
    const long long row = tid % ((long long)ncp * 2 * NYZ);
    const int i = (int)(row / (2 * NYZ)), l = (int)((row / NYZ) % 2), j = (int)(row % NYZ);
    T b = 0;
    for (int q = rp[row]; q < rp[row + 1]; ++q) b += val[q] * gif[efc[q]];
    const long long go = (((long long)s * ncp + i) * 2 + l) * NYZ + j;
    T g = gp1 != nullptr ? gp1[go] : T(0);
    if (gp2 != nullptr) g += gp2[go];
    P.p[l][i][s * P.stride[i] + P.off_a[i] + j] -= scale * b * g;
}

// ------------------------------------------------------------------ sources
// fe[s, node] -= cb[node] w amp[item]  (Yee entries);  E[s, comp, elem, :] -= cb_dg[elem] w[:] amp
template <typename T>
__global__ void inject_fd(int n, long long N, const int* __restrict__ shot,
                          const int* __restrict__ node, const int* __restrict__ item,
                          const T* __restrict__ w, const T* __restrict__ cb,
                          const T* __restrict__ amp, T* __restrict__ f) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    atomic_add(&f[shot[e] * N + node[e]], -cb[node[e]] * w[e] * amp[item[e]]);
}

template <typename T>
__global__ void inject_dg(int n, int Np, long long KN, int comp, int ncomp, const int* __restrict__ shot,
                          const int* __restrict__ elem, const int* __restrict__ item,
                          const T* __restrict__ w, const T* __restrict__ cb,
                          const T* __restrict__ amp, T* __restrict__ E) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)n * Np) return;
    const int e = (int)(tid / Np), p = (int)(tid % Np);
    atomic_add(&E[((long long)shot[e] * ncomp + comp) * KN + (long long)elem[e] * Np + p],
               -cb[elem[e]] * w[tid] * amp[item[e]]);
}

// transposes: gamp[item] -= lam cb w;  gcb[node] -= lam w amp[item]
template <typename T>
__global__ void inject_fd_bwd(int n, long long N, const int* __restrict__ shot,
                              const int* __restrict__ node, const int* __restrict__ item,
                              const T* __restrict__ w, const T* __restrict__ cb,
                              const T* __restrict__ amp, const T* __restrict__ lam,
                              T* __restrict__ gamp, T* __restrict__ gcb) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const T l = lam[shot[e] * N + node[e]];
    if (gamp != nullptr) atomic_add(&gamp[item[e]], -l * cb[node[e]] * w[e]);
    if (gcb != nullptr) atomic_add(&gcb[node[e]], -l * w[e] * amp[item[e]]);
}

template <typename T>
__global__ void inject_dg_bwd(int n, int Np, long long KN, int comp, int ncomp, const int* __restrict__ shot,
                              const int* __restrict__ elem, const int* __restrict__ item,
                              const T* __restrict__ w, const T* __restrict__ cb,
                              const T* __restrict__ amp, const T* __restrict__ lam,
                              T* __restrict__ gamp, T* __restrict__ gcb) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const T* l = lam + ((long long)shot[e] * ncomp + comp) * KN + (long long)elem[e] * Np;
    T a = 0;
    for (int p = 0; p < Np; ++p) a += l[p] * w[(long long)e * Np + p];
    if (gamp != nullptr) atomic_add(&gamp[item[e]], -a * cb[elem[e]]);
    if (gcb != nullptr) atomic_add(&gcb[elem[e]], -a * amp[item[e]]);
}

// ------------------------------------------------------------------ receivers
// out[o] = sum_{fd e in o} f[s, node] w + sum_{dg e in o} sum_p E[s, comp, elem, p] w_p
// (entries sorted by output o = shot * n + item; CSR offsets ofd / odg)
template <typename T>
__global__ void sample_fwd(int nout, int n, long long N, int Np, long long KN, int comp, int ncomp,
                           const int* __restrict__ ofd, const int* __restrict__ fnode,
                           const T* __restrict__ fw, const int* __restrict__ odg,
                           const int* __restrict__ delem, const T* __restrict__ dw,
                           const T* __restrict__ f, const T* __restrict__ E, T* __restrict__ out) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= nout) return;
    const int s = o / n;
    T a = 0;
    if (ofd != nullptr)
        for (int e = ofd[o]; e < ofd[o + 1]; ++e) a += f[s * N + fnode[e]] * fw[e];
    if (odg != nullptr)
        for (int e = odg[o]; e < odg[o + 1]; ++e) {
            const T* x = E + ((long long)s * ncomp + comp) * KN + (long long)delem[e] * Np;
            for (int p = 0; p < Np; ++p) a += x[p] * dw[(long long)e * Np + p];
        }
    out[o] = a;
}

// transpose (in place): lam_f[s, node] += g[o] w;  lam_E[s, comp, elem, p] += g[o] w_p
template <typename T>
__global__ void sample_fd_bwd(int n, int nrec, long long N, const int* __restrict__ shot,
                              const int* __restrict__ node, const int* __restrict__ item,
                              const T* __restrict__ w, const T* __restrict__ g, T* __restrict__ lam) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    atomic_add(&lam[shot[e] * N + node[e]], g[shot[e] * nrec + item[e]] * w[e]);
}

template <typename T>
__global__ void sample_dg_bwd(int n, int nrec, int Np, long long KN, int comp, int ncomp,
                              const int* __restrict__ shot, const int* __restrict__ elem,
                              const int* __restrict__ item, const T* __restrict__ w,
                              const T* __restrict__ g, T* __restrict__ lam) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)n * Np) return;
    const int e = (int)(tid / Np), p = (int)(tid % Np);
    atomic_add(&lam[((long long)shot[e] * ncomp + comp) * KN + (long long)elem[e] * Np + p],
               g[shot[e] * nrec + item[e]] * w[tid]);
}

// ------------------------------------------------------------------ DG combinations
// mode 0: out = a (x + y);   mode 1: out = x + a (y - z)   (n elements)
template <typename T>
__global__ void combine(long long n, int mode, T a, const T* __restrict__ x,
                        const T* __restrict__ y, const T* __restrict__ z, T* __restrict__ out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = mode == 0 ? a * (x[i] + y[i]) : x[i] + a * (y[i] - z[i]);
}

// ------------------------------------------------------------------ 2D (TM) coupling
// trace: X[s, r] = sum_e val_e Ez_i[s, row_i, col_e]     (CSR by slot r, coupling cp_of[r])
template <typename T>
__global__ void trace2_fwd(int S, int nX, Planes<T> P, const int* __restrict__ rp,
                           const int* __restrict__ col, const T* __restrict__ val,
                           const int* __restrict__ cp_of, T* __restrict__ X) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * nX) return;
    const int s = (int)(tid / nX), r = (int)(tid % nX);
    const int i = cp_of[r];
    const T* z = P.p[0][i] + s * P.stride[i] + P.off_a[i];
    T a = 0;
    for (int e = rp[r]; e < rp[r + 1]; ++e) a += val[e] * z[col[e]];
    X[tid] = a;
}

// transpose: Ez_i[s, row_i, j] += sum_e val_e gX[s, r_e]   (CSR by row node, rows i NY + j)
template <typename T>
__global__ void trace2_bwd(int S, int nX, int ncp, int NY, Planes<T> P, const int* __restrict__ rp,
                           const int* __restrict__ col, const T* __restrict__ val,
                           const T* __restrict__ gX) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * ncp * NY) return;
    const int s = (int)(tid / ((long long)ncp * NY));
    const int row = (int)(tid % ((long long)ncp * NY));
    const int i = row / NY, j = row % NY;
    if (rp[row] == rp[row + 1]) return;
    T a = 0;
    for (int e = rp[row]; e < rp[row + 1]; ++e) a += val[e] * gX[(long long)s * nX + col[e]];
    P.p[0][i][s * P.stride[i] + P.off_a[i] + j] += a;
}

// ghost rows: Hy_i[s, a, j] = 2 sum_e val_e F[s, node_e] - Hy_i[s, b, j]  (CSR by row node)
template <typename T>
__global__ void ghost2_fwd(int S, int ncp, int NY, long long KN, Planes<T> P,
                           const int* __restrict__ rp, const int* __restrict__ node,
                           const T* __restrict__ val, const T* __restrict__ F) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * ncp * NY) return;
    const int s = (int)(tid / ((long long)ncp * NY));
    const int row = (int)(tid % ((long long)ncp * NY));
    const int i = row / NY, j = row % NY;
    const T* f = F + (long long)s * KN;
    T v = 0;
    for (int e = rp[row]; e < rp[row + 1]; ++e) v += val[e] * f[node[e]];
    T* t = P.p[0][i] + s * P.stride[i];
    t[P.off_a[i] + j] = T(2) * v - t[P.off_b[i] + j];
}

// transpose: lF[s, node_r] += 2 sum_j val gg_j (CSR by slot); then plane b -= a, a = 0
template <typename T>
__global__ void ghost2_bwd_dg(int S, int nX, long long KN, Planes<T> P, const int* __restrict__ rp,
                              const int* __restrict__ col, const T* __restrict__ val,
                              const int* __restrict__ cp_of, const int* __restrict__ node_of,
                              T* __restrict__ lF) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * nX) return;
    const int s = (int)(tid / nX), r = (int)(tid % nX);
    const int i = cp_of[r];
    const T* gg = P.p[0][i] + s * P.stride[i] + P.off_a[i];
    T a = 0;
    for (int e = rp[r]; e < rp[r + 1]; ++e) a += val[e] * gg[col[e]];
    lF[(long long)s * KN + node_of[r]] += T(2) * a;
}

template <typename T>
__global__ void ghost2_bwd_yee(int S, int ncp, int NY, Planes<T> P) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * ncp * NY) return;
    const int s = (int)(tid / ((long long)ncp * NY));
    const int row = (int)(tid % ((long long)ncp * NY));
    const int i = row / NY, j = row % NY;
    T* t = P.p[0][i] + s * P.stride[i];
    const T g = t[P.off_a[i] + j];
    t[P.off_b[i] + j] -= g;
    t[P.off_a[i] + j] = T(0);
}

// scatter-add of compact per-entry values: f[s, idx_f[e]] += cf[s, e]; a[s, idx_a[e]] += ca[s, e]
template <typename T>
__global__ void scatter2(int S, int n, long long Nf, long long Na, const int* __restrict__ idx_f,
                         const int* __restrict__ idx_a, const T* __restrict__ cf,
                         const T* __restrict__ ca, T* __restrict__ f, T* __restrict__ a) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * n) return;
    const int s = (int)(tid / n), e = (int)(tid % n);
    if (f != nullptr) f[s * Nf + idx_f[e]] += cf[tid];
    if (a != nullptr) a[s * Na + idx_a[e]] += ca[tid];
}

template <typename T>
Planes<T> planes(const long long* gi, const void* const* ptr, int ncp) {
    // gi: ncp x (stride, off_a, off_b); ptr: ncp x (p0, p1)
    Planes<T> P;
    for (int i = 0; i < MAXCP; ++i) {
        P.p[0][i] = P.p[1][i] = nullptr;
        P.stride[i] = P.off_a[i] = P.off_b[i] = 0;
    }
    for (int i = 0; i < ncp; ++i) {
        P.p[0][i] = (T*)ptr[2 * i];
        P.p[1][i] = (T*)ptr[2 * i + 1];
        P.stride[i] = gi[3 * i];
        P.off_a[i] = gi[3 * i + 1];
        P.off_b[i] = gi[3 * i + 2];
    }
    return P;
}

#define C(t, k) ((const t*)p[k])
#define M(t, k) ((t*)p[k])

template <typename T>
int run(int op, const long long* gi, const void* const* p, const double* gd, cudaStream_t st) {
    // gi[0..7]: op parameters; gi[8..]: planes (ncp x 3); p[0..15]: op pointers; p[16..]: planes
    const int S = (int)gi[0];
    switch (op) {
    case 1: {   // trace_fwd: gi S, nX, ncp; p rp, col, val, cp_of, face, n0, gif, Ex, Xh
        const int nX = (int)gi[1], ncp = (int)gi[2];
        trace_fwd<T><<<nblk((long long)S * nX), kT, 0, st>>>(
            S, nX, planes<T>(gi + 8, p + 16, ncp), C(int, 0), C(int, 1), C(T, 2), C(int, 3),
            C(int, 4), C(T, 5), C(T, 6), M(T, 7), M(T, 8));
        break;
    }
    case 2: {   // trace_bwd: gi S, nX, ncp, NYZ; p rp, col, val, face, n0, gif, gX, gXh
        const int nX = (int)gi[1], ncp = (int)gi[2], NYZ = (int)gi[3];
        trace_bwd<T><<<nblk((long long)S * ncp * 2 * NYZ), kT, 0, st>>>(
            S, nX, ncp, NYZ, planes<T>(gi + 8, p + 16, ncp), C(int, 0), C(int, 1), C(T, 2),
            C(int, 3), C(T, 4), C(T, 5), C(T, 6), C(T, 7));
        break;
    }
    case 3: {   // xh_gif: gi S, nX; p face, n0, gXh, Ex, gacc
        const int nX = (int)gi[1];
        xh_gif<T><<<nblk(nX), kT, 0, st>>>(S, nX, C(int, 0), C(T, 1), C(T, 2), C(T, 3), M(T, 4));
        break;
    }
    case 4: {   // ghost_fwd: gi S, ncp, NYZ, KN; p rp, node, val, F
        const int ncp = (int)gi[1], NYZ = (int)gi[2];
        ghost_fwd<T><<<nblk((long long)S * ncp * 2 * NYZ), kT, 0, st>>>(
            S, ncp, NYZ, gi[3], planes<T>(gi + 8, p + 16, ncp), C(int, 0), C(int, 1), C(T, 2),
            C(T, 3));
        break;
    }
    case 5: {   // ghost_bwd: gi S, ncp, NYZ, KN, nX; p rp_t, col_t, val_t, cp_of, node_of, lF
        const int ncp = (int)gi[1], NYZ = (int)gi[2], nX = (int)gi[4];
        Planes<T> P = planes<T>(gi + 8, p + 16, ncp);
        ghost_bwd_dg<T><<<nblk((long long)S * 2 * nX), kT, 0, st>>>(
            S, nX, gi[3], P, C(int, 0), C(int, 1), C(T, 2), C(int, 3), C(int, 4), M(T, 5));
        ghost_bwd_yee<T><<<nblk((long long)S * ncp * 2 * NYZ), kT, 0, st>>>(S, ncp, NYZ, P);
        break;
    }
    case 6: {   // pen_fwd: gi S, ncp, NYZ, KN; p rp, node, efc, val, gif, E, pen; gd scale
        const int ncp = (int)gi[1], NYZ = (int)gi[2];
        pen_fwd<T><<<nblk((long long)S * ncp * 2 * NYZ), kT, 0, st>>>(
            S, ncp, NYZ, gi[3], planes<T>(gi + 8, p + 16, ncp), C(int, 0), C(int, 1), C(int, 2),
            C(T, 3), C(T, 4), C(T, 5), (T)gd[0], M(T, 6));
        break;
    }
    case 7: {   // pen_bwd: gi S, ncp, NYZ, KN, nX; p rp_t, col_t, val_t, cp_of, node_of, face,
                //   gif, E, gp1, gp2, lE, gacc, rp, efc, val; gd scale
                // planes: ncp x (fe_y, fe_z) then ncp x (lfe_y, lfe_z) (gi: 2 ncp x 3)
        const int ncp = (int)gi[1], NYZ = (int)gi[2], nX = (int)gi[4];
        pen_bwd_slot<T><<<nblk((long long)S * 2 * nX), kT, 0, st>>>(
            S, nX, ncp, NYZ, gi[3], planes<T>(gi + 8, p + 16, ncp), C(int, 0), C(int, 1), C(T, 2),
            C(int, 3), C(int, 4), C(int, 5), C(T, 6), C(T, 7), C(T, 8), C(T, 9), (T)gd[0],
            M(T, 10), M(T, 11));
        pen_bwd_yee<T><<<nblk((long long)S * ncp * 2 * NYZ), kT, 0, st>>>(
            S, ncp, NYZ, planes<T>(gi + 8 + 3 * ncp, p + 16 + 2 * ncp, ncp), C(int, 12),
            C(int, 13), C(T, 14), C(T, 6), C(T, 8), C(T, 9), (T)gd[0]);
        break;
    }
    case 8: {   // inject: gi nfd, N, ndg, Np, KN, comp; p fshot, fnode, fitem, fw, fcb, amp, f,
                //   dshot, delem, ditem, dw, dcb, E
        const int nfd = (int)gi[0], ndg = (int)gi[2], Np = (int)gi[3], comp = (int)gi[5];
        if (nfd > 0)
            inject_fd<T><<<nblk(nfd), kT, 0, st>>>(nfd, gi[1], C(int, 0), C(int, 1), C(int, 2),
                                                   C(T, 3), C(T, 4), C(T, 5), M(T, 6));
        if (ndg > 0)
            inject_dg<T><<<nblk((long long)ndg * Np), kT, 0, st>>>(
                ndg, Np, gi[4], comp, gi[6] ? (int)gi[6] : 3, C(int, 7), C(int, 8), C(int, 9),
                C(T, 10), C(T, 11), C(T, 5), M(T, 12));
        break;
    }
    case 9: {   // inject_bwd: same + p lam_f (13), lam_E (14), gamp (15), gcb_fd (6), gcb_dg (12)
        const int nfd = (int)gi[0], ndg = (int)gi[2], Np = (int)gi[3], comp = (int)gi[5];
        if (nfd > 0)
            inject_fd_bwd<T><<<nblk(nfd), kT, 0, st>>>(nfd, gi[1], C(int, 0), C(int, 1), C(int, 2),
                                                       C(T, 3), C(T, 4), C(T, 5), C(T, 13),
                                                       M(T, 15), M(T, 6));
        if (ndg > 0)
            inject_dg_bwd<T><<<nblk(ndg), kT, 0, st>>>(ndg, Np, gi[4], comp, gi[6] ? (int)gi[6] : 3,
                                                       C(int, 7), C(int, 8), C(int, 9), C(T, 10),
                                                       C(T, 11), C(T, 5), C(T, 14), M(T, 15), M(T, 12));
        break;
    }
    case 10: {  // sample_fwd: gi nout, n, N, Np, KN, comp; p ofd, fnode, fw, odg, delem, dw, f, E, out
        const int nout = (int)gi[0];
        sample_fwd<T><<<nblk(nout), kT, 0, st>>>(nout, (int)gi[1], gi[2], (int)gi[3], gi[4],
                                                 (int)gi[5], gi[6] ? (int)gi[6] : 3, C(int, 0),
                                                 C(int, 1), C(T, 2),
                                                 C(int, 3), C(int, 4), C(T, 5), C(T, 6), C(T, 7),
                                                 M(T, 8));
        break;
    }
    case 11: {  // sample_bwd: gi nfd, nrec, N, ndg, Np, KN, comp; p fshot, fnode, fitem, fw, g,
                //   lam_f, dshot, delem, ditem, dw, lam_E
        const int nfd = (int)gi[0], nrec = (int)gi[1], ndg = (int)gi[3], Np = (int)gi[4];
        if (nfd > 0)
            sample_fd_bwd<T><<<nblk(nfd), kT, 0, st>>>(nfd, nrec, gi[2], C(int, 0), C(int, 1),
                                                       C(int, 2), C(T, 3), C(T, 4), M(T, 5));
        if (ndg > 0)
            sample_dg_bwd<T><<<nblk((long long)ndg * Np), kT, 0, st>>>(
                ndg, nrec, Np, gi[5], (int)gi[6], gi[7] ? (int)gi[7] : 3, C(int, 6), C(int, 7),
                C(int, 8), C(T, 9), C(T, 4), M(T, 10));
        break;
    }
    case 12: {  // combine: gi n, mode; p x, y, z, out; gd a
        combine<T><<<nblk(gi[0]), kT, 0, st>>>(gi[0], (int)gi[1], (T)gd[0], C(T, 0), C(T, 1),
                                               C(T, 2), M(T, 3));
        break;
    }
    case 14: {  // trace2_fwd: gi S, nX, ncp; p rp, col, val, cp_of, X
        const int nX = (int)gi[1], ncp = (int)gi[2];
        trace2_fwd<T><<<nblk((long long)S * nX), kT, 0, st>>>(
            S, nX, planes<T>(gi + 8, p + 16, ncp), C(int, 0), C(int, 1), C(T, 2), C(int, 3), M(T, 4));
        break;
    }
    case 15: {  // trace2_bwd: gi S, nX, ncp, NY; p rp_t, col_t, val_t, gX
        const int nX = (int)gi[1], ncp = (int)gi[2], NY = (int)gi[3];
        trace2_bwd<T><<<nblk((long long)S * ncp * NY), kT, 0, st>>>(
            S, nX, ncp, NY, planes<T>(gi + 8, p + 16, ncp), C(int, 0), C(int, 1), C(T, 2), C(T, 3));
        break;
    }
    case 16: {  // ghost2_fwd: gi S, ncp, NY, KN; p rp, node, val, F
        const int ncp = (int)gi[1], NY = (int)gi[2];
        ghost2_fwd<T><<<nblk((long long)S * ncp * NY), kT, 0, st>>>(
            S, ncp, NY, gi[3], planes<T>(gi + 8, p + 16, ncp), C(int, 0), C(int, 1), C(T, 2), C(T, 3));
        break;
    }
    case 17: {  // ghost2_bwd: gi S, ncp, NY, KN, nX; p rp_h, col_h, val_h, cp_of, node_of, lF
        const int ncp = (int)gi[1], NY = (int)gi[2], nX = (int)gi[4];
        Planes<T> P = planes<T>(gi + 8, p + 16, ncp);
        ghost2_bwd_dg<T><<<nblk((long long)S * nX), kT, 0, st>>>(
            S, nX, gi[3], P, C(int, 0), C(int, 1), C(T, 2), C(int, 3), C(int, 4), M(T, 5));
        ghost2_bwd_yee<T><<<nblk((long long)S * ncp * NY), kT, 0, st>>>(S, ncp, NY, P);
        break;
    }
    case 13: {  // scatter2: gi S, n, Nf, Na; p idx_f, idx_a, cf, ca, f, a
        scatter2<T><<<nblk(gi[0] * gi[1]), kT, 0, st>>>((int)gi[0], (int)gi[1], gi[2], gi[3],
                                                       C(int, 0), C(int, 1), C(T, 2), C(T, 3),
                                                       M(T, 4), M(T, 5));
        break;
    }
    default:
        return -2;
    }
    return (int)cudaGetLastError();
}

#undef C
#undef M

}  // namespace

extern "C" {

int glue3d_call(int dtype, int op, const long long* gi, const void* const* p, const double* gd,
                void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == 0) return run<float>(op, gi, p, gd, st);
    if (dtype == 1) return run<double>(op, gi, p, gd, st);
    return -1;
}

}  // extern "C"

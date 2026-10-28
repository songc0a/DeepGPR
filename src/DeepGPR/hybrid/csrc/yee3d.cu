// Fused 3D Yee updates of the hybrid scheme and their exact transposes.
//
// Forward (one thread per shot and node), the same arithmetic as
// DeepGPR.hybrid.functional3d (``_yee_curl`` + ``_yee_upml`` + masks):
//
//   H update: rhs_c = sum_t sign_t s_t,  s_t = d_t/h_a [+ A d_t + B psi_t on CPML slabs]
//             d_t = f[i+1] - f[i] along the axis a_t of term t (0 in the last plane)
//             H' = H - chr rhs (active), UPML: B' = a1 B - b1 rhs, H' = hist H + new B' - old B
//   E update: d_t = f[i] - f[i-1] (0 in the first plane), rhs_c += pen on interface rows,
//             E' = ca E + cb rhs (active), UPML: D' = a1 D + b1 rhs, E' = hist E + new D' - old D
//   psi' = E psi - F d on every slab node (also inactive ones, as in PyTorch).
//
// Backward: pass 1 per node forms the cotangents of the old fields, auxiliaries
// and coefficients and the cotangent of every difference d_t (temporary array);
// pass 2 gathers the transposed differences onto the differentiated field. No
// atomics: results are deterministic.
//
// Terms t = (component c, axis a, field f, sign): (0,1,2,+) (0,2,1,-) (1,2,0,+)
// (1,0,2,-) (2,0,1,+) (2,1,0,-), as functional3d.TERMS.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __constant__ int tA[6] = {1, 2, 2, 0, 0, 1};
__device__ __constant__ int tF[6] = {2, 1, 0, 2, 1, 0};
__device__ __constant__ float tS[6] = {1.f, -1.f, 1.f, -1.f, 1.f, -1.f};

struct Grid {
    int S, n[3];
    long long N;                 // NX * NY * NZ
    long long stride[3];         // NY*NZ, NZ, 1
    double h[3];
};

template <typename T>
struct Prof {                    // CPML slab profile of one axis (side E or H)
    const int* pos;              // (n_axis) slab plane index or -1; nullptr: no CPML
    const T *A, *B, *E, *F;      // (n_slab)
    int nslab;
};

template <typename T>
struct Yee {
    Grid g;
    Prof<T> prof[3];
    const uint8_t* act[3];       // active mask of each component (N)
    const int* upos[3];          // compact UPML position per node or -1 (nullptr: none)
    int uoff[3];                 // offsets of the components in the compact arrays
    long long ntot_u;            // total compact length
    const T *a1, *b1;            // compact static UPML factors
    const T *hist, *nw, *old;    // compact UPML update factors (static for H, material for E)
};

template <typename T>
struct P6 {                      // six psi arrays, passed by value (no device pointer table)
    T* p[6];
};

__device__ inline void unflat(const Grid& g, long long node, int& i, int& j, int& k) {
    i = (int)(node / g.stride[0]);
    long long r = node % g.stride[0];
    j = (int)(r / g.stride[1]);
    k = (int)(r % g.stride[1]);
}

// flat index of the psi entry of term t at node (i, j, k) of shot s (slab along axis a)
__device__ inline long long psi_index(const Grid& g, int s, int a, int q, int i, int j, int k,
                                      int nslab) {
    int c[3] = {i, j, k};
    int d[3] = {g.n[0], g.n[1], g.n[2]};
    c[a] = q;
    d[a] = nslab;
    return (((long long)s * d[0] + c[0]) * d[1] + c[1]) * d[2] + c[2];
}

template <typename T, bool FWD_DIFF>
__device__ inline T diff(const T* f, const Grid& g, long long id, int coord, int a) {
    if (FWD_DIFF) return coord < g.n[a] - 1 ? f[id + g.stride[a]] - f[id] : T(0);
    return coord >= 1 ? f[id] - f[id - g.stride[a]] : T(0);
}

// ------------------------------------------------------------------ forward
template <typename T, bool HSIDE>
__global__ void yee_fwd(Yee<T> y, const T* __restrict__ src0, const T* __restrict__ src1,
                        const T* __restrict__ src2,           // differentiated fields
                        const T* __restrict__ u0, const T* __restrict__ u1,
                        const T* __restrict__ u2,             // updated fields (old)
                        P6<T> psi_in, P6<T> psi_out,
                        const T* __restrict__ aux_in, T* __restrict__ aux_out,
                        const T* __restrict__ ca, const T* __restrict__ cb,
                        const T* __restrict__ chr,
                        const T* __restrict__ pen, const int* __restrict__ rowmap, int nrows,
                        T* __restrict__ o0, T* __restrict__ o1, T* __restrict__ o2) {
    const Grid& g = y.g;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)g.S * g.N) return;
    const int s = (int)(tid / g.N);
    const long long node = tid % g.N;
    int i, j, k;
    unflat(g, node, i, j, k);
    const int co[3] = {i, j, k};
    const T* src[3] = {src0 + (long long)s * g.N, src1 + (long long)s * g.N, src2 + (long long)s * g.N};
    const T* uo[3] = {u0, u1, u2};
    T* out[3] = {o0, o1, o2};
    for (int c = 0; c < 3; ++c) {
        T rhs = 0;
        for (int tt = 0; tt < 2; ++tt) {
            const int t = 2 * c + tt, a = tA[t];
            const T d = diff<T, HSIDE>(src[tF[t]], g, node, co[a], a);
            T sv = d / T(g.h[a]);
            const Prof<T>& p = y.prof[a];
            if (p.pos != nullptr) {
                const int q = p.pos[co[a]];
                if (q >= 0) {
                    const long long pi = psi_index(g, s, a, q, i, j, k, p.nslab);
                    const T ps = psi_in.p[t][pi];
                    sv += p.A[q] * d + p.B[q] * ps;
                    psi_out.p[t][pi] = p.E[q] * ps - p.F[q] * d;
                }
            }
            rhs += T(tS[t]) * sv;
        }
        if (!HSIDE && pen != nullptr && c > 0 && rowmap[i] >= 0)
            rhs += pen[(((long long)s * nrows + rowmap[i]) * 2 + (c - 1)) * (g.stride[0]) + node % g.stride[0]];
        const long long id = (long long)s * g.N + node;
        const T f = uo[c][id];
        T res = f;
        if (y.act[c][node]) {
            const int pu = y.upos[c] != nullptr ? y.upos[c][node] : -1;
            if (pu >= 0) {
                const long long cu = y.uoff[c] + pu;
                const long long ai = (long long)s * y.ntot_u + cu;
                const T Ao = aux_in[ai];
                const T An = HSIDE ? y.a1[cu] * Ao - y.b1[cu] * rhs : y.a1[cu] * Ao + y.b1[cu] * rhs;
                aux_out[ai] = An;
                res = y.hist[cu] * f + y.nw[cu] * An - y.old[cu] * Ao;
            } else {
                res = HSIDE ? f - chr[node] * rhs : ca[node] * f + cb[node] * rhs;
            }
        }
        out[c][id] = res;
    }
}

// ------------------------------------------------------------------ backward
// pass 1: cotangents of the updated fields, auxiliaries, psi, coefficients and
// of the differences (lam_d, (6, S, N)).
template <typename T, bool HSIDE>
__global__ void yee_bwd1(Yee<T> y, const T* __restrict__ src0, const T* __restrict__ src1,
                         const T* __restrict__ src2, const T* __restrict__ u0,
                         const T* __restrict__ u1, const T* __restrict__ u2,
                         P6<T> psi_in, const T* __restrict__ aux_in,
                         const T* __restrict__ ca, const T* __restrict__ cb,
                         const T* __restrict__ chr, const T* __restrict__ pen,
                         const int* __restrict__ rowmap, int nrows,
                         const T* __restrict__ g0, const T* __restrict__ g1,
                         const T* __restrict__ g2,                 // cotangents of the outputs
                         P6<T> gpsi_out, const T* __restrict__ gaux_out,
                         T* __restrict__ gu0, T* __restrict__ gu1, T* __restrict__ gu2,
                         P6<T> gpsi_in, T* __restrict__ gaux_in,
                         T* __restrict__ lam_d, T* __restrict__ gca, T* __restrict__ gcb,
                         T* __restrict__ ghist, T* __restrict__ gnew, T* __restrict__ gold,
                         T* __restrict__ gpen) {
    const Grid& g = y.g;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)g.S * g.N) return;
    const int s = (int)(tid / g.N);
    const long long node = tid % g.N;
    int i, j, k;
    unflat(g, node, i, j, k);
    const int co[3] = {i, j, k};
    const long long id = (long long)s * g.N + node;
    const T* src[3] = {src0 + (long long)s * g.N, src1 + (long long)s * g.N, src2 + (long long)s * g.N};
    const T* uo[3] = {u0, u1, u2};
    const T* gout[3] = {g0, g1, g2};
    T* gu[3] = {gu0, gu1, gu2};
    T sca = 0, scb = 0;
    for (int c = 0; c < 3; ++c) {
        const T go = gout[c][id];
        const T f = uo[c][id];
        T lrhs = 0, gf = go;
        // recompute rhs (needed for the cb and UPML-aux gradients on the E side)
        T rhs = 0;
        if (!HSIDE) {
            for (int tt = 0; tt < 2; ++tt) {
                const int t = 2 * c + tt, a = tA[t];
                const T d = diff<T, false>(src[tF[t]], g, node, co[a], a);
                T sv = d / T(g.h[a]);
                const Prof<T>& p = y.prof[a];
                if (p.pos != nullptr) {
                    const int q = p.pos[co[a]];
                    if (q >= 0) sv += p.A[q] * d + p.B[q] * psi_in.p[t][psi_index(g, s, a, q, i, j, k, p.nslab)];
                }
                rhs += T(tS[t]) * sv;
            }
            if (pen != nullptr && c > 0 && rowmap[i] >= 0)
                rhs += pen[(((long long)s * nrows + rowmap[i]) * 2 + (c - 1)) * g.stride[0] + node % g.stride[0]];
        }
        if (y.act[c][node]) {
            const int pu = y.upos[c] != nullptr ? y.upos[c][node] : -1;
            if (pu >= 0) {
                const long long cu = y.uoff[c] + pu;
                const long long ai = (long long)s * y.ntot_u + cu;
                const T Ao = aux_in[ai];
                const T lA = gaux_out[ai] + y.nw[cu] * go;       // total cotangent of the new aux
                gf = y.hist[cu] * go;
                gaux_in[ai] = y.a1[cu] * lA - y.old[cu] * go;
                lrhs = HSIDE ? -y.b1[cu] * lA : y.b1[cu] * lA;
                if (!HSIDE) {
                    const T An = y.a1[cu] * Ao + y.b1[cu] * rhs;
                    ghist[ai] = go * f;
                    gnew[ai] = go * An;
                    gold[ai] = -go * Ao;
                }
            } else if (HSIDE) {
                lrhs = -chr[node] * go;
            } else {
                gf = ca[node] * go;
                lrhs = cb[node] * go;
                sca += go * f;
                scb += go * rhs;
            }
        }
        gu[c][id] = gf;
        if (!HSIDE && gpen != nullptr && c > 0 && rowmap[i] >= 0)
            gpen[(((long long)s * nrows + rowmap[i]) * 2 + (c - 1)) * g.stride[0] + node % g.stride[0]] = lrhs;
        for (int tt = 0; tt < 2; ++tt) {
            const int t = 2 * c + tt, a = tA[t];
            const T ls = T(tS[t]) * lrhs;
            T ld = ls / T(g.h[a]);
            const Prof<T>& p = y.prof[a];
            if (p.pos != nullptr) {
                const int q = p.pos[co[a]];
                if (q >= 0) {
                    const long long pi = psi_index(g, s, a, q, i, j, k, p.nslab);
                    const T gpo = gpsi_out.p[t][pi];
                    ld += p.A[q] * ls - p.F[q] * gpo;
                    gpsi_in.p[t][pi] = p.B[q] * ls + p.E[q] * gpo;
                }
            }
            lam_d[(long long)t * g.S * g.N + id] = ld;
        }
    }
    if (!HSIDE) {
        gca[id] = sca;
        gcb[id] = scb;
    }
}

// pass 2: gather the transposed differences onto the differentiated fields.
template <typename T, bool HSIDE>
__global__ void yee_bwd2(Grid g, const T* __restrict__ lam_d, T* __restrict__ gs0,
                         T* __restrict__ gs1, T* __restrict__ gs2) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)g.S * g.N) return;
    const long long node = tid % g.N;
    int i, j, k;
    unflat(g, node, i, j, k);
    const int co[3] = {i, j, k};
    T acc[3] = {0, 0, 0};
    for (int t = 0; t < 6; ++t) {
        const int a = tA[t];
        const T* L = lam_d + (long long)t * g.S * g.N;
        T v = 0;
        if (HSIDE) {            // d[i] = f[i+1] - f[i] for i <= n-2
            if (co[a] >= 1) v += L[tid - g.stride[a]];
            if (co[a] <= g.n[a] - 2) v -= L[tid];
        } else {                // d[i] = f[i] - f[i-1] for i >= 1
            if (co[a] >= 1) v += L[tid];
            if (co[a] + 1 <= g.n[a] - 1) v -= L[tid + g.stride[a]];
        }
        acc[tF[t]] += v;
    }
    gs0[tid] = acc[0];
    gs1[tid] = acc[1];
    gs2[tid] = acc[2];
}

constexpr int kThreads = 256;
inline int blocks(long long n) { return (int)((n + kThreads - 1) / kThreads); }

template <typename T>
Yee<T> make(const int* gi, const double* gd, const void* const* ptr) {
    // gi: S, NX, NY, NZ, nslab0..2 (of the side's profiles), uoff0..2, ntot_u, nrows
    Yee<T> y;
    y.g.S = gi[0];
    y.g.n[0] = gi[1];
    y.g.n[1] = gi[2];
    y.g.n[2] = gi[3];
    y.g.N = (long long)gi[1] * gi[2] * gi[3];
    y.g.stride[0] = (long long)gi[2] * gi[3];
    y.g.stride[1] = gi[3];
    y.g.stride[2] = 1;
    for (int a = 0; a < 3; ++a) y.g.h[a] = gd[a];
    // ptr layout: per axis (pos, A, B, E, F) x 3 = 15; act x3; upos x3; a1, b1, hist, new, old
    for (int a = 0; a < 3; ++a) {
        y.prof[a].pos = (const int*)ptr[5 * a];
        y.prof[a].A = (const T*)ptr[5 * a + 1];
        y.prof[a].B = (const T*)ptr[5 * a + 2];
        y.prof[a].E = (const T*)ptr[5 * a + 3];
        y.prof[a].F = (const T*)ptr[5 * a + 4];
        y.prof[a].nslab = gi[4 + a];
    }
    for (int c = 0; c < 3; ++c) {
        y.act[c] = (const uint8_t*)ptr[15 + c];
        y.upos[c] = (const int*)ptr[18 + c];
        y.uoff[c] = gi[7 + c];
    }
    y.ntot_u = gi[10];
    y.a1 = (const T*)ptr[21];
    y.b1 = (const T*)ptr[22];
    y.hist = (const T*)ptr[23];
    y.nw = (const T*)ptr[24];
    y.old = (const T*)ptr[25];
    return y;
}

template <typename T>
P6<T> pack(const void* const* io, int at) {
    P6<T> r;
    for (int t = 0; t < 6; ++t) r.p[t] = (T*)io[at + t];
    return r;
}

template <typename T, bool H>
int run_fwd(const int* gi, const double* gd, const void* const* sp, const void* const* io,
            cudaStream_t st) {
    Yee<T> y = make<T>(gi, gd, sp);
    // io: src0..2 (0-2), u0..2 (3-5), psi_in x6 (6-11), psi_out x6 (12-17), aux_in (18),
    //     aux_out (19), ca (20), cb (21), chr (22), pen (23), rowmap (24), o0..2 (25-27)
    yee_fwd<T, H><<<blocks((long long)y.g.S * y.g.N), kThreads, 0, st>>>(
        y, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3], (const T*)io[4],
        (const T*)io[5], pack<T>(io, 6), pack<T>(io, 12), (const T*)io[18], (T*)io[19],
        (const T*)io[20], (const T*)io[21], (const T*)io[22], (const T*)io[23], (const int*)io[24],
        gi[11], (T*)io[25], (T*)io[26], (T*)io[27]);
    return (int)cudaGetLastError();
}

template <typename T, bool H>
int run_bwd(const int* gi, const double* gd, const void* const* sp, const void* const* io,
            cudaStream_t st) {
    Yee<T> y = make<T>(gi, gd, sp);
    // io: src0..2 (0-2), u0..2 (3-5), psi_in x6 (6-11), aux_in (12), ca (13), cb (14), chr (15),
    //     pen (16), rowmap (17), g0..2 (18-20), gpsi_out x6 (21-26), gaux_out (27),
    //     gu0..2 (28-30), gpsi_in x6 (31-36), gaux_in (37), lam_d (38), gca (39), gcb (40),
    //     ghist (41), gnew (42), gold (43), gpen (44), gs0..2 (45-47)
    const long long n = (long long)y.g.S * y.g.N;
    yee_bwd1<T, H><<<blocks(n), kThreads, 0, st>>>(
        y, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3], (const T*)io[4],
        (const T*)io[5], pack<T>(io, 6), (const T*)io[12], (const T*)io[13], (const T*)io[14],
        (const T*)io[15], (const T*)io[16], (const int*)io[17], gi[11], (const T*)io[18],
        (const T*)io[19], (const T*)io[20], pack<T>(io, 21), (const T*)io[27], (T*)io[28],
        (T*)io[29], (T*)io[30], pack<T>(io, 31), (T*)io[37], (T*)io[38], (T*)io[39], (T*)io[40],
        (T*)io[41], (T*)io[42], (T*)io[43], (T*)io[44]);
    yee_bwd2<T, H><<<blocks(n), kThreads, 0, st>>>(y.g, (const T*)io[38], (T*)io[45],
                                                   (T*)io[46], (T*)io[47]);
    return (int)cudaGetLastError();
}

}  // namespace

extern "C" {

// side: 0 = H update (forward differences of E), 1 = E update (backward differences of H)
int yee3d_fwd(int dtype, int side, const int* gi, const double* gd, const void* const* sp,
              const void* const* io, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == 0) return side == 0 ? run_fwd<float, true>(gi, gd, sp, io, st)
                                     : run_fwd<float, false>(gi, gd, sp, io, st);
    if (dtype == 1) return side == 0 ? run_fwd<double, true>(gi, gd, sp, io, st)
                                     : run_fwd<double, false>(gi, gd, sp, io, st);
    return -1;
}

int yee3d_bwd(int dtype, int side, const int* gi, const double* gd, const void* const* sp,
              const void* const* io, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == 0) return side == 0 ? run_bwd<float, true>(gi, gd, sp, io, st)
                                     : run_bwd<float, false>(gi, gd, sp, io, st);
    if (dtype == 1) return side == 0 ? run_bwd<double, true>(gi, gd, sp, io, st)
                                     : run_bwd<double, false>(gi, gd, sp, io, st);
    return -1;
}

}  // extern "C"

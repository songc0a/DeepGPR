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
// Backward: one pass, one thread per node looping over the shots (see yee_bwd):
// the cotangents of the old fields, auxiliaries and coefficients at the node and
// the gather of the transposed differences, the difference cotangents of the
// neighbours re-evaluated in place of a temporary array. No atomics: results are
// deterministic.
//
// Terms t = (component c, axis a, field f, sign): (0,1,2,+) (0,2,1,-) (1,2,0,+)
// (1,0,2,-) (2,0,1,+) (2,1,0,-), as functional3d.TERMS.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// term tables (axis a_t, differentiated field f_t, sign) as compile-time functions: in
// the unrolled term loops every index into the small per-term arrays is a constant, so
// they stay in registers (a __constant__ table gave runtime indices and a local-memory
// stack frame)
__host__ __device__ constexpr int term_axis(int t) {
    return t == 0 ? 1 : t == 1 ? 2 : t == 2 ? 2 : t == 3 ? 0 : t == 4 ? 0 : 1;
}
__host__ __device__ constexpr int term_field(int t) { return 2 - t % 3; }
__host__ __device__ constexpr int term_sign(int t) { return (t & 1) ? -1 : 1; }

constexpr int kThreadsBwd = 128;  // transpose blocks (more resident warps at its register use)

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
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        T rhs = 0;
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
            const int t = 2 * c + tt, a = term_axis(t);
            const T d = diff<T, HSIDE>(src[term_field(t)], g, node, co[a], a);
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
            rhs += T(term_sign(t)) * sv;
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
// One pass, one thread per node looping over the shots. At its node the thread
// forms the cotangents of the old fields, auxiliaries, psi and coefficients; the
// transposed differences are gathered onto the differentiated fields by
// re-evaluating the difference cotangents of the three neighbours (node - e_a on
// the H side, node + e_a on the E side) instead of storing them: no temporary
// arrays, no atomics, deterministic. The per-node data (masks, coefficients, slab
// positions) is read once for all shots; per shot the plain loads are issued
// before any branch. Inputs read at neighbours (g, gpsi_out, gaux_out) must not
// share memory with the outputs. UPML: side-UPML nodes present (compile time).
//
// Difference cotangent of term t at a node, with lrhs the cotangent of the rhs of
// its component there (-chr g on active nodes of the H side, cb g on the E side,
// -/+ b1 (gaux_out + new g) on side-UPML nodes, 0 on inactive nodes):
//   ld_t = s_t lrhs / h_a [+ A s_t lrhs - F gpsi_out_t on CPML slab nodes].
//
// accf bit 0: accumulate the coefficient cotangents into their outputs, bit 1:
// accumulate the cotangents of the differentiated fields, bit 2: the coefficient
// cotangents are summed over the shots ((N,) and (n_upml,) instead of per shot).
template <typename T, bool HSIDE, bool UPML>
__global__ void __launch_bounds__(kThreadsBwd, (HSIDE && UPML) ? 8 : 1)
yee_bwd(Yee<T> y, const T* __restrict__ src0, const T* __restrict__ src1,
        const T* __restrict__ src2, const T* __restrict__ u0, const T* __restrict__ u1,
        const T* __restrict__ u2, P6<T> psi_in, const T* __restrict__ aux_in,
        const T* __restrict__ ca, const T* __restrict__ cb, const T* __restrict__ chr,
        const T* __restrict__ pen, const int* __restrict__ rowmap, int nrows,
        const T* __restrict__ g0, const T* __restrict__ g1, const T* __restrict__ g2,
        P6<T> gpsi_out, const T* __restrict__ gaux_out,
        T* __restrict__ gu0, T* __restrict__ gu1, T* __restrict__ gu2,
        P6<T> gpsi_in, T* __restrict__ gaux_in, T* __restrict__ gca, T* __restrict__ gcb,
        T* __restrict__ ghist, T* __restrict__ gnew, T* __restrict__ gold,
        T* __restrict__ gpen, T* __restrict__ gs0, T* __restrict__ gs1,
        T* __restrict__ gs2, int accf) {
    const Grid& g = y.g;
    const long long nl = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (nl >= g.N) return;
    const int node = (int)nl;
    const int st0 = (int)g.stride[0];
    const int i = node / st0, rem = node - i * st0;
    const int j = rem / g.n[2], k = rem - j * g.n[2];
    const int co[3] = {i, j, k};
    const T* gout[3] = {g0, g1, g2};
    const T* uo[3] = {u0, u1, u2};
    T* gu[3] = {gu0, gu1, gu2};
    T* gsrc[3] = {gs0, gs1, gs2};
    // ---- per-node data, read once for all shots
    bool act[3];
    int upo[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        act[c] = y.act[c][node] != 0;
        upo[c] = (UPML && act[c] && y.upos[c] != nullptr) ? y.upos[c][node] : -1;
    }
    const T kself = HSIDE ? -chr[node] : cb[node];
    const T caself = HSIDE ? T(0) : ca[node];
    bool hasn[3];
    int nb[3], qo[3], qn[3];     // neighbour (the node itself at the edge), slab positions
    int nbs[3];                  // E side: backward neighbour of the rhs differences
#pragma unroll
    for (int a = 0; a < 3; ++a) {
        hasn[a] = HSIDE ? co[a] >= 1 : co[a] + 1 <= g.n[a] - 1;
        nb[a] = hasn[a] ? (HSIDE ? node - (int)g.stride[a] : node + (int)g.stride[a]) : node;
        nbs[a] = co[a] >= 1 ? node - (int)g.stride[a] : node;
        const int* pos = y.prof[a].pos;
        qo[a] = pos != nullptr ? pos[co[a]] : -1;
        qn[a] = (pos != nullptr && hasn[a]) ? pos[HSIDE ? co[a] - 1 : co[a] + 1] : -1;
    }
    T kn[6];                     // -chr / cb of the neighbour of term t (0: inactive or none)
    int un[6];                   // its compact UPML position or -1
#pragma unroll
    for (int t = 0; t < 6; ++t) {
        const int a = term_axis(t), c = t >> 1;
        kn[t] = T(0);
        un[t] = -1;
        if (hasn[a] && y.act[c][nb[a]]) {
            kn[t] = HSIDE ? -chr[nb[a]] : cb[nb[a]];
            if (UPML && y.upos[c] != nullptr) un[t] = y.upos[c][nb[a]];
        }
    }
    const int row = (!HSIDE && pen != nullptr) ? rowmap[i] : -1;
    T sca_s = 0, scb_s = 0, sh[3] = {0, 0, 0}, sn[3] = {0, 0, 0}, so[3] = {0, 0, 0};
    for (int s = 0; s < g.S; ++s) {
        const long long off = (long long)s * g.N, id = off + node;
        // ---- plain loads of the shot first (no branches between them)
        T go[3], gn[6], gsv[3], f[3], hv[3], hb[6];
#pragma unroll
        for (int c = 0; c < 3; ++c) go[c] = gout[c][id];
#pragma unroll
        for (int t = 0; t < 6; ++t) gn[t] = gout[t >> 1][off + nb[term_axis(t)]];
#pragma unroll
        for (int q = 0; q < 3; ++q) gsv[q] = gsrc[q][id];   // used with accf & 2 only
        if (!HSIDE) {
            const T* sp[3] = {src0 + off, src1 + off, src2 + off};
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                f[c] = uo[c][id];
                hv[c] = sp[c][node];
            }
#pragma unroll
            for (int t = 0; t < 6; ++t) hb[t] = sp[term_field(t)][nbs[term_axis(t)]];
        }
        T ldo[6];
        T sca = 0, scb = 0;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            T lrhs = 0, gf = go[c], rhs = 0;
            if (!HSIDE) {        // rhs (for the cb and UPML-factor cotangents)
#pragma unroll
                for (int tt = 0; tt < 2; ++tt) {
                    const int t = 2 * c + tt, a = term_axis(t);
                    const T d = co[a] >= 1 ? hv[term_field(t)] - hb[t] : T(0);
                    T sv = d / T(g.h[a]);
                    if (qo[a] >= 0) {
                        const Prof<T>& p = y.prof[a];
                        sv += p.A[qo[a]] * d
                            + p.B[qo[a]] * psi_in.p[t][psi_index(g, s, a, qo[a], i, j, k, p.nslab)];
                    }
                    rhs += T(term_sign(t)) * sv;
                }
                if (row >= 0 && c > 0)
                    rhs += pen[(((long long)s * nrows + row) * 2 + (c - 1)) * st0 + rem];
            }
            if (act[c]) {
                if (UPML && upo[c] >= 0) {
                    const long long cu = y.uoff[c] + upo[c];
                    const long long ai = (long long)s * y.ntot_u + cu;
                    const T Ao = aux_in[ai];
                    const T lA = gaux_out[ai] + y.nw[cu] * go[c];
                    gf = y.hist[cu] * go[c];
                    gaux_in[ai] = y.a1[cu] * lA - y.old[cu] * go[c];
                    lrhs = HSIDE ? -y.b1[cu] * lA : y.b1[cu] * lA;
                    if (!HSIDE) {
                        const T An = y.a1[cu] * Ao + y.b1[cu] * rhs;
                        if (accf & 4) {
                            sh[c] += go[c] * f[c];
                            sn[c] += go[c] * An;
                            so[c] -= go[c] * Ao;
                        } else if (accf & 1) {
                            ghist[ai] += go[c] * f[c];
                            gnew[ai] += go[c] * An;
                            gold[ai] -= go[c] * Ao;
                        } else {
                            ghist[ai] = go[c] * f[c];
                            gnew[ai] = go[c] * An;
                            gold[ai] = -go[c] * Ao;
                        }
                    }
                } else if (HSIDE) {
                    lrhs = kself * go[c];
                } else {
                    gf = caself * go[c];
                    lrhs = kself * go[c];
                    sca += go[c] * f[c];
                    scb += go[c] * rhs;
                }
            }
            gu[c][id] = gf;
            if (row >= 0 && c > 0 && gpen != nullptr)
                gpen[(((long long)s * nrows + row) * 2 + (c - 1)) * st0 + rem] = lrhs;
#pragma unroll
            for (int tt = 0; tt < 2; ++tt) {
                const int t = 2 * c + tt, a = term_axis(t);
                const T ls = T(term_sign(t)) * lrhs;
                T ld = ls / T(g.h[a]);
                if (qo[a] >= 0) {
                    const Prof<T>& p = y.prof[a];
                    const long long pi = psi_index(g, s, a, qo[a], i, j, k, p.nslab);
                    const T gpo = gpsi_out.p[t][pi];
                    ld += p.A[qo[a]] * ls - p.F[qo[a]] * gpo;
                    gpsi_in.p[t][pi] = p.B[qo[a]] * ls + p.E[qo[a]] * gpo;
                }
                ldo[t] = ld;
            }
        }
        // ---- gather the transposed differences
        T acc[3] = {0, 0, 0};
#pragma unroll
        for (int t = 0; t < 6; ++t) {
            const int a = term_axis(t), c = t >> 1;
            T lr;                // rhs cotangent of the neighbour (0 without one)
            if (UPML && un[t] >= 0) {
                const long long cu = y.uoff[c] + un[t];
                const T lA = gaux_out[(long long)s * y.ntot_u + cu] + y.nw[cu] * gn[t];
                lr = HSIDE ? -y.b1[cu] * lA : y.b1[cu] * lA;
            } else {
                lr = kn[t] * gn[t];
            }
            const T ls = T(term_sign(t)) * lr;
            T ln = ls / T(g.h[a]);   // difference cotangent of term t at the neighbour
            if (qn[a] >= 0) {
                const Prof<T>& p = y.prof[a];
                ln += p.A[qn[a]] * ls
                    - p.F[qn[a]] * gpsi_out.p[t][psi_index(g, s, a, qn[a], i, j, k, p.nslab)];
            }
            T v = 0;
            if (HSIDE) {         // d[i] = f[i+1] - f[i] for i <= n-2
                if (co[a] >= 1) v += ln;
                if (co[a] <= g.n[a] - 2) v -= ldo[t];
            } else {             // d[i] = f[i] - f[i-1] for i >= 1
                if (co[a] >= 1) v += ldo[t];
                if (co[a] + 1 <= g.n[a] - 1) v -= ln;
            }
            acc[term_field(t)] += v;
        }
#pragma unroll
        for (int q = 0; q < 3; ++q) gsrc[q][id] = (accf & 2) ? gsv[q] + acc[q] : acc[q];
        if (!HSIDE) {
            if (accf & 4) {
                sca_s += sca;
                scb_s += scb;
            } else if (accf & 1) {
                gca[id] += sca;
                gcb[id] += scb;
            } else {
                gca[id] = sca;
                gcb[id] = scb;
            }
        }
    }
    if (!HSIDE && (accf & 4)) {
        if (accf & 1) {
            gca[node] += sca_s;
            gcb[node] += scb_s;
        } else {
            gca[node] = sca_s;
            gcb[node] = scb_s;
        }
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            if (!UPML || upo[c] < 0) continue;
            const long long cu = y.uoff[c] + upo[c];
            if (accf & 1) {
                ghist[cu] += sh[c];
                gnew[cu] += sn[c];
                gold[cu] += so[c];
            } else {
                ghist[cu] = sh[c];
                gnew[cu] = sn[c];
                gold[cu] = so[c];
            }
        }
    }
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
            cudaStream_t st, int accf = 0) {
    Yee<T> y = make<T>(gi, gd, sp);
    // io: src0..2 (0-2), u0..2 (3-5), psi_in x6 (6-11), aux_in (12), ca (13), cb (14), chr (15),
    //     pen (16), rowmap (17), g0..2 (18-20), gpsi_out x6 (21-26), gaux_out (27),
    //     gu0..2 (28-30), gpsi_in x6 (31-36), gaux_in (37), (38 unused), gca (39), gcb (40),
    //     ghist (41), gnew (42), gold (43), gpen (44), gs0..2 (45-47)
    const bool upml = y.ntot_u > 0;
    auto kern = upml ? yee_bwd<T, H, true> : yee_bwd<T, H, false>;
    kern<<<(int)((y.g.N + kThreadsBwd - 1) / kThreadsBwd), kThreadsBwd, 0, st>>>(
        y, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3], (const T*)io[4],
        (const T*)io[5], pack<T>(io, 6), (const T*)io[12], (const T*)io[13], (const T*)io[14],
        (const T*)io[15], (const T*)io[16], (const int*)io[17], gi[11], (const T*)io[18],
        (const T*)io[19], (const T*)io[20], pack<T>(io, 21), (const T*)io[27], (T*)io[28],
        (T*)io[29], (T*)io[30], pack<T>(io, 31), (T*)io[37], (T*)io[39], (T*)io[40],
        (T*)io[41], (T*)io[42], (T*)io[43], (T*)io[44], (T*)io[45], (T*)io[46], (T*)io[47],
        accf);
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

// as yee3d_bwd; accf bit 0: accumulate the coefficient cotangents (gca, gcb, UPML factors)
// into their outputs, bit 1: accumulate the cotangents of the differentiated fields (gs)
int yee3d_bwd_acc(int dtype, int side, int accf, const int* gi, const double* gd,
                  const void* const* sp, const void* const* io, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == 0) return side == 0 ? run_bwd<float, true>(gi, gd, sp, io, st, accf)
                                     : run_bwd<float, false>(gi, gd, sp, io, st, accf);
    if (dtype == 1) return side == 0 ? run_bwd<double, true>(gi, gd, sp, io, st, accf)
                                     : run_bwd<double, false>(gi, gd, sp, io, st, accf);
    return -1;
}

}  // extern "C"

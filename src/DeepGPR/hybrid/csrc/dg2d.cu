// Fused 2D (TM: Ez, Hx, Hy) nodal DGTD operators on triangles and their exact
// transposes, the same arithmetic as DeepGPR.hybrid.functional (dg_grad_e,
// dg_curl_h and the H/E updates with the stretched-coordinate CPML).
//
// Terms (one thread per shot and element), T (S, 2, K, NP):
//   mode 0 (H): T0 = D_x E, T1 = D_y E with the E-flux per face node
//     dE = cEp Ep + cEm Em + cHp Ht+ + cHm Ht- + m_if X,   Ht = Hy nx - Hx ny
//     cEp = mi/2 + mi a (Yp/(Ym+Yp) - 1/2),  cEm = -mi/2 - mp - mf - mi a (Yp/(Ym+Yp) - 1/2) - ma/2
//     cHp = mi a/(Ym+Yp),                     cHm = -mi a/(Ym+Yp) - ma/(2 Ym)
//   mode 1 (E): T0 = D_x Hy, T1 = D_y Hx with
//     dHx = c1 Hx+ + c0 Hx-, dHy = c1 Hy+ + c0 Hy-,  c1 = mi (1/2 + a (Zp/(Zm+Zp) - 1/2)), c0 = -c1 - ma/2
//     P = pEp E+ + pEm E-,  pEp = mi a/(Zm+Zp),  pEm = -pEp - mp a/Zm - ma/(2 Zm)
//     fx = nx dHy + nx^2 P,  fy = ny dHx - ny^2 P
//   D_a f = (r_a Dr + s_a Ds) f + LIFT(fscale n_a flux)  (n_a not applied in mode 1:
//   fx/fy already carry it).
// Transpose: pass 1 (per element) forms the volume transpose, the face-node
// cotangent of the flux and from it per slot the contributions to the own node
// (o*) and to the mirror node (p*) of E, Hx, Hy, plus the cotangents of the face
// coefficients (Ym, Yp or Zm, Zp) and of X; pass 2 (per element) gathers own and
// mirror (slotP) contributions onto the nodes. No atomics: deterministic.
//
// Layouts: fields (S, K, NP); per face (K, 3): nx, ny, fscale, mi, mp, ma, mf;
// per face node (K * NF, NF = 3 NFP): coefficients, vmapM/vmapP/slotP (int32),
// fmask (NF); X compact (S, nX) with xmap (K * NF) -> position or -1.
//
// Updates (one thread per shot and node for the forward, per element for the
// transpose, which sums the ca/cb cotangents of the element):
//   H: rhs_hx = -T1, rhs_hy = T0; CPML elements: rhs_hx = -(ray T1 + rby qhx),
//      rhs_hy = rax T0 + rbx qhy, qhx' = rey qhx - rfy T1, qhy' = rex qhy - rfx T0;
//      Hx' = Hx + ch rhs_hx, Hy' = Hy + ch rhs_hy
//   E: rhs = T0 - T1; CPML: rhs = (rax T0 + rbx qex) - (ray T1 + rby qey),
//      qex' = rex qex - rfx T0, qey' = rey qey - rfy T1; E' = ca E + cb rhs

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 128;
inline int blocks(long long n) { return (int)((n + kThreads - 1) / kThreads); }

template <typename T>
struct Geo2 {                       // static element / face data
    const T *rx, *ry, *sx, *sy;     // (K)
    const T *nx, *ny, *fs;          // (K, 3)
    const T *mi, *mp, *ma, *mf;     // (K, 3)
    const int *vmapM, *vmapP;       // (K * NF)
    const int *slotP, *fmask;       // (K * NF), (NF)
    const int* xmap;                // (K * NF) or nullptr
    const T *Dr, *Ds, *LIFT;        // (NP, NP), (NP, NP), (NP, NF)
    int K, nX;
    T alpha;
};

template <typename T, int NP, int NFP, int MODE>
__global__ void terms2_fwd(Geo2<T> g, int S, const T* __restrict__ E, const T* __restrict__ Hx,
                           const T* __restrict__ Hy, const T* __restrict__ X,
                           const T* __restrict__ cm, const T* __restrict__ cp,
                           T* __restrict__ out) {
    constexpr int NF = 3 * NFP;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * g.K) return;
    const int s = (int)(tid / g.K), k = (int)(tid % g.K);
    const long long KN = (long long)g.K * NP;
    const T* Es = E + (long long)s * KN;
    const T* Hxs = Hx + (long long)s * KN;
    const T* Hys = Hy + (long long)s * KN;
    const T a = g.alpha;
    T t0[NP], t1[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) t0[i] = t1[i] = T(0);
    for (int f = 0; f < 3; ++f) {
        const long long fi = (long long)k * 3 + f;
        const T nx = g.nx[fi], ny = g.ny[fi], fs = g.fs[fi];
        const T mi = g.mi[fi], mp = g.mp[fi], ma = g.ma[fi], mf = g.mf[fi];
        for (int m = 0; m < NFP; ++m) {
            const int sg = f * NFP + m;
            const long long idx = (long long)k * NF + sg;
            const int vm = g.vmapM[idx], vp = g.vmapP[idx];
            const T zm = cm[idx], zp = cp[idx];
            const T inv = T(1) / (zm + zp);
            T fx, fy;
            if (MODE == 0) {
                const T Em = Es[vm], Ep = Es[vp];
                const T Htm = Hys[vm] * nx - Hxs[vm] * ny, Htp = Hys[vp] * nx - Hxs[vp] * ny;
                T dE = mi * (T(0.5) * (Ep - Em)) - mp * Em - mf * Em
                     + mi * a * ((zp * (Ep - Em) + (Htp - Htm)) * inv - T(0.5) * (Ep - Em))
                     + ma * (T(-0.5) * Em - T(0.5) * Htm / zm);
                if (X != nullptr) {
                    const int xi = g.xmap[idx];
                    if (xi >= 0) dE += mf * X[(long long)s * g.nX + xi];
                }
                fx = fs * nx * dE;
                fy = fs * ny * dE;
            } else {
                const T w = zp * inv;
                const T c1 = mi * (T(0.5) + a * (w - T(0.5)));
                const T c0 = -c1 - T(0.5) * ma;
                const T dHx = c1 * Hxs[vp] + c0 * Hxs[vm];
                const T dHy = c1 * Hys[vp] + c0 * Hys[vm];
                const T pEp = mi * a * inv;
                const T pEm = -pEp - mp * a / zm - T(0.5) * ma / zm;
                const T P = pEp * Es[vp] + pEm * Es[vm];
                fx = fs * (nx * dHy + nx * nx * P);
                fy = fs * (ny * dHx - ny * ny * P);
            }
#pragma unroll
            for (int i = 0; i < NP; ++i) {
                t0[i] += g.LIFT[i * NF + sg] * fx;
                t1[i] += g.LIFT[i * NF + sg] * fy;
            }
        }
    }
    const T rx = g.rx[k], ry = g.ry[k], sx = g.sx[k], sy = g.sy[k];
    const T* v0 = MODE == 0 ? Es : Hys;     // field of T0
    const T* v1 = MODE == 0 ? Es : Hxs;     // field of T1
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        T dr0 = 0, ds0 = 0, dr1 = 0, ds1 = 0;
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            const T a0 = v0[(long long)k * NP + j], a1 = v1[(long long)k * NP + j];
            dr0 += g.Dr[i * NP + j] * a0;
            ds0 += g.Ds[i * NP + j] * a0;
            dr1 += g.Dr[i * NP + j] * a1;
            ds1 += g.Ds[i * NP + j] * a1;
        }
        t0[i] += rx * dr0 + sx * ds0;
        t1[i] += ry * dr1 + sy * ds1;
    }
    T* o = out + (long long)s * 2 * KN + (long long)k * NP;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        o[i] = t0[i];
        o[KN + i] = t1[i];
    }
}

// pass 1: work = [gvol (S, 3, K, NP) | slot values (S, 6, NF, K)], slot values
// v = 0..5: own E, own Hx, own Hy, mirror E, mirror Hx, mirror Hy.
template <typename T, int NP, int NFP, int MODE>
__global__ void terms2_bwd1(Geo2<T> g, int S, const T* __restrict__ E, const T* __restrict__ Hx,
                            const T* __restrict__ Hy, const T* __restrict__ gT,
                            const T* __restrict__ cm, const T* __restrict__ cp,
                            T* __restrict__ gvol, T* __restrict__ slot, T* __restrict__ gX,
                            T* __restrict__ gcm, T* __restrict__ gcp) {
    constexpr int NF = 3 * NFP;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * g.K) return;
    const int s = (int)(tid / g.K), k = (int)(tid % g.K);
    const long long KN = (long long)g.K * NP;
    const T* Es = E + (long long)s * KN;
    const T* Hxs = Hx + (long long)s * KN;
    const T* Hys = Hy + (long long)s * KN;
    const T a = g.alpha;
    T g0[NP], g1[NP];
    const T* gs = gT + (long long)s * 2 * KN + (long long)k * NP;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        g0[i] = gs[i];
        g1[i] = gs[KN + i];
    }
    // volume transpose: T0 = rx Dr v0 + sx Ds v0, T1 = ry Dr v1 + sy Ds v1
    const T rx = g.rx[k], ry = g.ry[k], sx = g.sx[k], sy = g.sy[k];
    T w0[NP], w1[NP];
#pragma unroll
    for (int j = 0; j < NP; ++j) {
        T a0 = 0, a1 = 0;
#pragma unroll
        for (int i = 0; i < NP; ++i) {
            a0 += g.Dr[i * NP + j] * (rx * g0[i]) + g.Ds[i * NP + j] * (sx * g0[i]);
            a1 += g.Dr[i * NP + j] * (ry * g1[i]) + g.Ds[i * NP + j] * (sy * g1[i]);
        }
        w0[j] = a0;
        w1[j] = a1;
    }
    T* gv = gvol + (long long)s * 3 * KN + (long long)k * NP;   // fields E, Hx, Hy
#pragma unroll
    for (int j = 0; j < NP; ++j) {
        if (MODE == 0) {
            gv[j] = w0[j] + w1[j];
            gv[KN + j] = T(0);
            gv[2 * KN + j] = T(0);
        } else {
            gv[j] = T(0);
            gv[KN + j] = w1[j];        // T1 = D_y Hx
            gv[2 * KN + j] = w0[j];    // T0 = D_x Hy
        }
    }
    const long long KF = (long long)g.K * NF;
    for (int f = 0; f < 3; ++f) {
        const long long fi = (long long)k * 3 + f;
        const T nx = g.nx[fi], ny = g.ny[fi], fs = g.fs[fi];
        const T mi = g.mi[fi], mp = g.mp[fi], ma = g.ma[fi], mf = g.mf[fi];
        for (int m = 0; m < NFP; ++m) {
            const int sg = f * NFP + m;
            const long long idx = (long long)k * NF + sg;
            T L0 = 0, L1 = 0;
#pragma unroll
            for (int i = 0; i < NP; ++i) {
                L0 += g.LIFT[i * NF + sg] * g0[i];
                L1 += g.LIFT[i * NF + sg] * g1[i];
            }
            const int vm = g.vmapM[idx], vp = g.vmapP[idx];
            const T zm = cm[idx], zp = cp[idx];
            const T inv = T(1) / (zm + zp);
            T oE, oHx, oHy, pE, pHx, pHy, gzm, gzp;
            if (MODE == 0) {
                const T q = fs * (nx * L0 + ny * L1);          // cotangent of dE
                const T yr = zp * inv;
                const T cEp = T(0.5) * mi + mi * a * (yr - T(0.5));
                const T cEm = -T(0.5) * mi - mp - mf - mi * a * (yr - T(0.5)) - T(0.5) * ma;
                const T cHp = mi * a * inv;
                const T cHm = -mi * a * inv - T(0.5) * ma / zm;
                oE = cEm * q;
                oHy = cHm * nx * q;
                oHx = -cHm * ny * q;
                pE = cEp * q;
                pHy = cHp * nx * q;
                pHx = -cHp * ny * q;
                if (gX != nullptr) {
                    const int xi = g.xmap[idx];
                    if (xi >= 0) gX[(long long)s * g.nX + xi] = mf * q;
                }
                const T Em = Es[vm], Ep = Es[vp];
                const T Htm = Hys[vm] * nx - Hxs[vm] * ny, Htp = Hys[vp] * nx - Hxs[vp] * ny;
                const T N = zp * (Ep - Em) + (Htp - Htm);
                gzm = q * (-mi * a * N * inv * inv + T(0.5) * ma * Htm / (zm * zm));
                gzp = q * (mi * a * ((Ep - Em) * inv - N * inv * inv));
            } else {
                const T gfx = fs * L0, gfy = fs * L1;
                const T gdHy = nx * gfx, gdHx = ny * gfy, gP = nx * nx * gfx - ny * ny * gfy;
                const T w = zp * inv;
                const T c1 = mi * (T(0.5) + a * (w - T(0.5)));
                const T c0 = -c1 - T(0.5) * ma;
                const T pEp = mi * a * inv;
                const T pEm = -pEp - mp * a / zm - T(0.5) * ma / zm;
                oHx = c0 * gdHx;
                oHy = c0 * gdHy;
                oE = pEm * gP;
                pHx = c1 * gdHx;
                pHy = c1 * gdHy;
                pE = pEp * gP;
                const T jump = gdHx * (Hxs[vp] - Hxs[vm]) + gdHy * (Hys[vp] - Hys[vm]);
                const T D2 = inv * inv;
                const T Em = Es[vm], Ep = Es[vp];
                gzm = jump * (-mi * a * zp * D2)
                    + gP * (Ep * (-mi * a * D2) + Em * (mi * a * D2 + mp * a / (zm * zm)
                                                       + T(0.5) * ma / (zm * zm)));
                gzp = jump * (mi * a * zm * D2) + gP * (Ep * (-mi * a * D2) + Em * (mi * a * D2));
            }
            const long long base = (long long)s * 6 * KF + (long long)sg * g.K + k;
            slot[base] = oE;
            slot[base + KF] = oHx;
            slot[base + 2 * KF] = oHy;
            slot[base + 3 * KF] = pE;
            slot[base + 4 * KF] = pHx;
            slot[base + 5 * KF] = pHy;
            gcm[(long long)s * KF + idx] = gzm;
            gcp[(long long)s * KF + idx] = gzp;
        }
    }
}

// pass 2: gather own and mirror slot contributions (+ volume part, + addends).
template <typename T, int NP, int NFP>
__global__ void terms2_bwd2(Geo2<T> g, int S, const T* __restrict__ gvol,
                            const T* __restrict__ slot, T* __restrict__ gE,
                            T* __restrict__ gHx, T* __restrict__ gHy,
                            const T* __restrict__ addE, const T* __restrict__ addHx,
                            const T* __restrict__ addHy) {
    constexpr int NF = 3 * NFP;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * g.K) return;
    const int s = (int)(tid / g.K), k = (int)(tid % g.K);
    const long long KN = (long long)g.K * NP;
    const long long KF = (long long)g.K * NF;
    T acc[3][NP];
    const T* gv = gvol + (long long)s * 3 * KN + (long long)k * NP;
#pragma unroll
    for (int c = 0; c < 3; ++c)
#pragma unroll
        for (int i = 0; i < NP; ++i) acc[c][i] = gv[c * KN + i];
    const T* sl = slot + (long long)s * 6 * KF;
    for (int sg = 0; sg < NF; ++sg) {
        const int q = g.slotP[(long long)k * NF + sg];
        const long long own = (long long)sg * g.K + k;
        const long long mir = (long long)(q % NF) * g.K + q / NF;
        const int loc = g.fmask[sg];
#pragma unroll
        for (int c = 0; c < 3; ++c) acc[c][loc] += sl[c * KF + own] + sl[(3 + c) * KF + mir];
    }
    const long long o = (long long)s * KN + (long long)k * NP;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        gE[o + i] = acc[0][i] + (addE != nullptr ? addE[o + i] : T(0));
        gHx[o + i] = acc[1][i] + (addHx != nullptr ? addHx[o + i] : T(0));
        gHy[o + i] = acc[2][i] + (addHy != nullptr ? addHy[o + i] : T(0));
    }
}

template <typename T>
struct Upd2 {                       // update data (CPML on Kp elements)
    const T* ch;                    // (K) dt/mu
    const int* pml;                 // (K) position or -1 (nullptr: none)
    const T *rax, *rbx, *rex, *rfx, *ray, *rby, *rey, *rfy;   // (Kp, NP)
    int K, Kp;
};

// mode 0: (Hx, Hy, qhx, qhy) -> new; mode 1: (E, qex, qey) -> new with ca, cb.
template <typename T, int NP, int MODE>
__global__ void upd2_fwd(Upd2<T> u, int S, const T* __restrict__ Tm, const T* __restrict__ f0,
                         const T* __restrict__ f1, const T* __restrict__ q0,
                         const T* __restrict__ q1, const T* __restrict__ ca,
                         const T* __restrict__ cb, T* __restrict__ n0, T* __restrict__ n1,
                         T* __restrict__ nq0, T* __restrict__ nq1) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long KN = (long long)u.K * NP;
    if (tid >= (long long)S * KN) return;
    const int s = (int)(tid / KN);
    const long long node = tid % KN;
    const int k = (int)(node / NP), i = (int)(node % NP);
    const T T0 = Tm[(long long)s * 2 * KN + node], T1 = Tm[(long long)s * 2 * KN + KN + node];
    const int p = u.pml != nullptr ? u.pml[k] : -1;
    if (MODE == 0) {
        T rhx = -T1, rhy = T0;
        if (p >= 0) {
            const long long v = (long long)p * NP + i;
            const long long qa = (long long)s * u.Kp * NP + v;
            const T qhx = q0[qa], qhy = q1[qa];
            rhx = -(u.ray[v] * T1 + u.rby[v] * qhx);
            rhy = u.rax[v] * T0 + u.rbx[v] * qhy;
            nq0[qa] = u.rey[v] * qhx - u.rfy[v] * T1;
            nq1[qa] = u.rex[v] * qhy - u.rfx[v] * T0;
        }
        const long long id = (long long)s * KN + node;
        n0[id] = f0[id] + u.ch[k] * rhx;
        n1[id] = f1[id] + u.ch[k] * rhy;
    } else {
        T rhs = T0 - T1;
        if (p >= 0) {
            const long long v = (long long)p * NP + i;
            const long long qa = (long long)s * u.Kp * NP + v;
            const T qex = q0[qa], qey = q1[qa];
            rhs = (u.rax[v] * T0 + u.rbx[v] * qex) - (u.ray[v] * T1 + u.rby[v] * qey);
            nq0[qa] = u.rex[v] * qex - u.rfx[v] * T0;
            nq1[qa] = u.rey[v] * qey - u.rfy[v] * T1;
        }
        const long long id = (long long)s * KN + node;
        n0[id] = ca[k] * f0[id] + cb[k] * rhs;
    }
}

template <typename T, int NP, int MODE>
__global__ void upd2_bwd(Upd2<T> u, int S, const T* __restrict__ Tm, const T* __restrict__ f0,
                         const T* __restrict__ q0, const T* __restrict__ q1,
                         const T* __restrict__ ca, const T* __restrict__ cb,
                         const T* __restrict__ g0, const T* __restrict__ g1,
                         const T* __restrict__ gq0, const T* __restrict__ gq1,
                         T* __restrict__ gT, T* __restrict__ gf0, T* __restrict__ gf1,
                         T* __restrict__ gqo0, T* __restrict__ gqo1, T* __restrict__ gca,
                         T* __restrict__ gcb) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)S * u.K) return;
    const int s = (int)(tid / u.K), k = (int)(tid % u.K);
    const long long KN = (long long)u.K * NP;
    const int p = u.pml != nullptr ? u.pml[k] : -1;
    T sca = 0, scb = 0;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        const long long node = (long long)k * NP + i;
        const long long id = (long long)s * KN + node;
        const long long it = (long long)s * 2 * KN + node;
        T gT0 = 0, gT1 = 0;
        if (MODE == 0) {
            const T gh0 = g0[id], gh1 = g1[id];       // cotangents of Hx', Hy'
            gf0[id] = gh0;
            gf1[id] = gh1;
            const T gr0 = u.ch[k] * gh0, gr1 = u.ch[k] * gh1;   // of rhs_hx, rhs_hy
            if (p < 0) {
                gT1 = -gr0;
                gT0 = gr1;
            } else {
                const long long v = (long long)p * NP + i;
                const long long qa = (long long)s * u.Kp * NP + v;
                const T gn0 = gq0[qa], gn1 = gq1[qa];           // of qhx', qhy'
                gT1 = -u.ray[v] * gr0 - u.rfy[v] * gn0;
                gT0 = u.rax[v] * gr1 - u.rfx[v] * gn1;
                gqo0[qa] = -u.rby[v] * gr0 + u.rey[v] * gn0;
                gqo1[qa] = u.rbx[v] * gr1 + u.rex[v] * gn1;
            }
        } else {
            const T ge = g0[id];
            const T f = f0[id];
            const T T0 = Tm[it], T1 = Tm[it + KN];
            gf0[id] = ca[k] * ge;
            const T gr = cb[k] * ge;
            T rhs = T0 - T1;
            if (p < 0) {
                gT0 = gr;
                gT1 = -gr;
            } else {
                const long long v = (long long)p * NP + i;
                const long long qa = (long long)s * u.Kp * NP + v;
                const T qex = q0[qa], qey = q1[qa];
                const T gn0 = gq0[qa], gn1 = gq1[qa];           // of qex', qey'
                rhs = (u.rax[v] * T0 + u.rbx[v] * qex) - (u.ray[v] * T1 + u.rby[v] * qey);
                gT0 = u.rax[v] * gr - u.rfx[v] * gn0;
                gT1 = -u.ray[v] * gr - u.rfy[v] * gn1;
                gqo0[qa] = u.rbx[v] * gr + u.rex[v] * gn0;
                gqo1[qa] = -u.rby[v] * gr + u.rey[v] * gn1;
            }
            sca += ge * f;
            scb += ge * rhs;
        }
        gT[it] = gT0;
        gT[it + KN] = gT1;
    }
    if (MODE == 1) {
        gca[tid] = sca;
        gcb[tid] = scb;
    }
}

template <typename T>
Geo2<T> make_geo(const int* gi, const double* gd, const void* const* gp) {
    // gi: K, nX ; gd: alpha ; gp: rx ry sx sy nx ny fs mi mp ma mf vmapM vmapP slotP fmask xmap Dr Ds LIFT
    Geo2<T> g;
    g.K = gi[0];
    g.nX = gi[1];
    g.alpha = (T)gd[0];
    g.rx = (const T*)gp[0];
    g.ry = (const T*)gp[1];
    g.sx = (const T*)gp[2];
    g.sy = (const T*)gp[3];
    g.nx = (const T*)gp[4];
    g.ny = (const T*)gp[5];
    g.fs = (const T*)gp[6];
    g.mi = (const T*)gp[7];
    g.mp = (const T*)gp[8];
    g.ma = (const T*)gp[9];
    g.mf = (const T*)gp[10];
    g.vmapM = (const int*)gp[11];
    g.vmapP = (const int*)gp[12];
    g.slotP = (const int*)gp[13];
    g.fmask = (const int*)gp[14];
    g.xmap = (const int*)gp[15];
    g.Dr = (const T*)gp[16];
    g.Ds = (const T*)gp[17];
    g.LIFT = (const T*)gp[18];
    return g;
}

template <typename T>
Upd2<T> make_upd(const int* gi, const void* const* up) {
    // gi: K, nX, Kp ; up: ch pml rax rbx rex rfx ray rby rey rfy
    Upd2<T> u;
    u.K = gi[0];
    u.Kp = gi[2];
    u.ch = (const T*)up[0];
    u.pml = (const int*)up[1];
    u.rax = (const T*)up[2];
    u.rbx = (const T*)up[3];
    u.rex = (const T*)up[4];
    u.rfx = (const T*)up[5];
    u.ray = (const T*)up[6];
    u.rby = (const T*)up[7];
    u.rey = (const T*)up[8];
    u.rfy = (const T*)up[9];
    return u;
}

template <typename T, int NP, int NFP>
int run_terms(int mode, int S, const int* gi, const double* gd, const void* const* gp,
              const void* const* io, cudaStream_t st) {
    // io: E Hx Hy X cm cp out
    Geo2<T> g = make_geo<T>(gi, gd, gp);
    const int nb = blocks((long long)S * g.K);
    if (mode == 0)
        terms2_fwd<T, NP, NFP, 0><<<nb, kThreads, 0, st>>>(
            g, S, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3],
            (const T*)io[4], (const T*)io[5], (T*)io[6]);
    else
        terms2_fwd<T, NP, NFP, 1><<<nb, kThreads, 0, st>>>(
            g, S, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3],
            (const T*)io[4], (const T*)io[5], (T*)io[6]);
    return (int)cudaGetLastError();
}

template <typename T, int NP, int NFP>
int run_terms_bwd(int mode, int S, const int* gi, const double* gd, const void* const* gp,
                  const void* const* io, cudaStream_t st) {
    // io: E Hx Hy gT cm cp work gX gcm gcp gE gHx gHy addE addHx addHy
    Geo2<T> g = make_geo<T>(gi, gd, gp);
    const int nb = blocks((long long)S * g.K);
    T* gvol = (T*)io[6];
    T* slot = gvol + (long long)S * 3 * g.K * NP;
    if (mode == 0)
        terms2_bwd1<T, NP, NFP, 0><<<nb, kThreads, 0, st>>>(
            g, S, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3],
            (const T*)io[4], (const T*)io[5], gvol, slot, (T*)io[7], (T*)io[8], (T*)io[9]);
    else
        terms2_bwd1<T, NP, NFP, 1><<<nb, kThreads, 0, st>>>(
            g, S, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3],
            (const T*)io[4], (const T*)io[5], gvol, slot, (T*)io[7], (T*)io[8], (T*)io[9]);
    terms2_bwd2<T, NP, NFP><<<nb, kThreads, 0, st>>>(
        g, S, gvol, slot, (T*)io[10], (T*)io[11], (T*)io[12], (const T*)io[13],
        (const T*)io[14], (const T*)io[15]);
    return (int)cudaGetLastError();
}

template <typename T, int NP>
int run_upd(int mode, int S, const int* gi, const void* const* up, const void* const* io,
            cudaStream_t st) {
    // io: T f0 f1 q0 q1 ca cb n0 n1 nq0 nq1
    Upd2<T> u = make_upd<T>(gi, up);
    const int nb = blocks((long long)S * u.K * NP);
    if (mode == 0)
        upd2_fwd<T, NP, 0><<<nb, kThreads, 0, st>>>(
            u, S, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3],
            (const T*)io[4], (const T*)io[5], (const T*)io[6], (T*)io[7], (T*)io[8], (T*)io[9],
            (T*)io[10]);
    else
        upd2_fwd<T, NP, 1><<<nb, kThreads, 0, st>>>(
            u, S, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3],
            (const T*)io[4], (const T*)io[5], (const T*)io[6], (T*)io[7], (T*)io[8], (T*)io[9],
            (T*)io[10]);
    return (int)cudaGetLastError();
}

template <typename T, int NP>
int run_upd_bwd(int mode, int S, const int* gi, const void* const* up, const void* const* io,
                cudaStream_t st) {
    // io: T f0 q0 q1 ca cb g0 g1 gq0 gq1 gT gf0 gf1 gqo0 gqo1 gca gcb
    Upd2<T> u = make_upd<T>(gi, up);
    const int nb = blocks((long long)S * u.K);
    if (mode == 0)
        upd2_bwd<T, NP, 0><<<nb, kThreads, 0, st>>>(
            u, S, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3],
            (const T*)io[4], (const T*)io[5], (const T*)io[6], (const T*)io[7], (const T*)io[8],
            (const T*)io[9], (T*)io[10], (T*)io[11], (T*)io[12], (T*)io[13], (T*)io[14],
            (T*)io[15], (T*)io[16]);
    else
        upd2_bwd<T, NP, 1><<<nb, kThreads, 0, st>>>(
            u, S, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3],
            (const T*)io[4], (const T*)io[5], (const T*)io[6], (const T*)io[7], (const T*)io[8],
            (const T*)io[9], (T*)io[10], (T*)io[11], (T*)io[12], (T*)io[13], (T*)io[14],
            (T*)io[15], (T*)io[16]);
    return (int)cudaGetLastError();
}

}  // namespace

extern "C" {

// Workspace (field dtype elements) of dg2d_terms_bwd: volume part + 6 slot values.
long long dg2d_bwd_workspace(int S, int K, int np, int nfp) {
    return (long long)S * 3 * K * np + 6LL * S * K * 3 * nfp;
}

#define DG2_DISPATCH(FN, ...)                                                                 \
    if (dtype == 0 && np == 3) return FN<float, 3, 2>(__VA_ARGS__);                           \
    if (dtype == 0 && np == 6) return FN<float, 6, 3>(__VA_ARGS__);                           \
    if (dtype == 0 && np == 10) return FN<float, 10, 4>(__VA_ARGS__);                         \
    if (dtype == 1 && np == 3) return FN<double, 3, 2>(__VA_ARGS__);                          \
    if (dtype == 1 && np == 6) return FN<double, 6, 3>(__VA_ARGS__);                          \
    if (dtype == 1 && np == 10) return FN<double, 10, 4>(__VA_ARGS__);                        \
    return -1;

int dg2d_terms_fwd(int dtype, int np, int mode, int S, const int* gi, const double* gd,
                   const void* const* gp, const void* const* io, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    DG2_DISPATCH(run_terms, mode, S, gi, gd, gp, io, st)
}

int dg2d_terms_bwd(int dtype, int np, int mode, int S, const int* gi, const double* gd,
                   const void* const* gp, const void* const* io, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    DG2_DISPATCH(run_terms_bwd, mode, S, gi, gd, gp, io, st)
}

#undef DG2_DISPATCH
#define UPD2_DISPATCH(FN, ...)                                                                \
    if (dtype == 0 && np == 3) return FN<float, 3>(__VA_ARGS__);                              \
    if (dtype == 0 && np == 6) return FN<float, 6>(__VA_ARGS__);                              \
    if (dtype == 0 && np == 10) return FN<float, 10>(__VA_ARGS__);                            \
    if (dtype == 1 && np == 3) return FN<double, 3>(__VA_ARGS__);                             \
    if (dtype == 1 && np == 6) return FN<double, 6>(__VA_ARGS__);                             \
    if (dtype == 1 && np == 10) return FN<double, 10>(__VA_ARGS__);                           \
    return -1;

int dg2d_update_fwd(int dtype, int np, int mode, int S, const int* gi, const void* const* up,
                    const void* const* io, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    UPD2_DISPATCH(run_upd, mode, S, gi, up, io, st)
}

int dg2d_update_bwd(int dtype, int np, int mode, int S, const int* gi, const void* const* up,
                    const void* const* io, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    UPD2_DISPATCH(run_upd_bwd, mode, S, gi, up, io, st)
}

#undef UPD2_DISPATCH

}  // extern "C"

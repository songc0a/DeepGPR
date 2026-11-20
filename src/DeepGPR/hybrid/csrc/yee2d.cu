// Fused 2D (TM) Yee updates of the hybrid scheme and their exact transposes,
// the same arithmetic as DeepGPR.hybrid.functional.step on full-grid arrays
// (inactive nodes have zero H coefficients and ca = 1, cb = 0; the CPML
// coefficient arrays vanish outside their faces):
//
//   H: dEy = Ez[i, j+1] - Ez[i, j] (0 in the last column), dEx = Ez[i+1, j] - Ez[i, j]
//      Hx' = Hx - cHx dEy - (Ahy dEy + Bhy phy),  phy' = Ehy phy - Fhy dEy
//      Hy' = Hy + cHy dEx + (Ahx dEx + Bhx phx),  phx' = Ehx phx - Fhx dEx
//   E: dHyx = Hy[i] - Hy[i-1] (0 in the first row), dHxy = Hx[j] - Hx[j-1]
//      rhs = dHyx/dx - dHxy/dy + (Aex dHyx + Bex pex) - (Aey dHxy + Bey pey)
//      pex' = Eex pex - Fex dHyx, pey' = Eey pey - Fey dHxy,  Ez' = ca Ez + cb rhs
//
// Forward and H transpose: one thread per shot and node. E transpose: the shots of
// a node are split over up to kEY threads (lanes), each reading the coefficients of
// the node and of its neighbours once; ca/cb cotangents per shot (S, NX, NY) or
// summed over the shots in registers and shared memory (NX, NY). The transposes
// evaluate the cotangents of the differences of the node and of its neighbour and
// gather them directly (no temporaries, no atomics). Inputs read at neighbours
// (gHx/gHy/gphy/gphx, gEz/gpex/gpey) must not share memory with the outputs.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 256;
inline int blocks(long long n) { return (int)((n + kThreads - 1) / kThreads); }

template <typename T>
struct Y2 {
    int S, NX, NY;
    T dx, dy;
    // H side: cHx cHy Ahy Bhy Ehy Fhy Ahx Bhx Ehx Fhx ; E side: Aex Bex Eex Fex Aey Bey Eey Fey
    const T *cHx, *cHy, *Ahy, *Bhy, *Ehy, *Fhy, *Ahx, *Bhx, *Ehx, *Fhx;
    const T *Aex, *Bex, *Eex, *Fex, *Aey, *Bey, *Eey, *Fey;
};

template <typename T>
__global__ void yee2_h_fwd(Y2<T> y, const T* __restrict__ Ez, const T* __restrict__ Hx,
                           const T* __restrict__ Hy, const T* __restrict__ phy,
                           const T* __restrict__ phx, T* __restrict__ oHx, T* __restrict__ oHy,
                           T* __restrict__ ophy, T* __restrict__ ophx) {
    const long long N = (long long)y.NX * y.NY;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)y.S * N) return;
    const long long node = tid % N;
    const int i = (int)(node / y.NY), j = (int)(node % y.NY);
    const T ez = Ez[tid];
    const T dEy = j < y.NY - 1 ? Ez[tid + 1] - ez : T(0);
    const T dEx = i < y.NX - 1 ? Ez[tid + y.NY] - ez : T(0);
    const T py = phy[tid], px = phx[tid];
    oHx[tid] = Hx[tid] - y.cHx[node] * dEy - (y.Ahy[node] * dEy + y.Bhy[node] * py);
    ophy[tid] = y.Ehy[node] * py - y.Fhy[node] * dEy;
    oHy[tid] = Hy[tid] + y.cHy[node] * dEx + (y.Ahx[node] * dEx + y.Bhx[node] * px);
    ophx[tid] = y.Ehx[node] * px - y.Fhx[node] * dEx;
}

// cotangents of dEy, dEx at (node) from the output cotangents there
template <typename T>
__device__ inline T gdEy(const Y2<T>& y, long long node, long long id, const T* gHx, const T* gphy) {
    return -(y.cHx[node] + y.Ahy[node]) * gHx[id] - y.Fhy[node] * gphy[id];
}
template <typename T>
__device__ inline T gdEx(const Y2<T>& y, long long node, long long id, const T* gHy, const T* gphx) {
    return (y.cHy[node] + y.Ahx[node]) * gHy[id] - y.Fhx[node] * gphx[id];
}

template <typename T>
__global__ void yee2_h_bwd(Y2<T> y, const T* __restrict__ gHx, const T* __restrict__ gHy,
                           const T* __restrict__ gphy, const T* __restrict__ gphx,
                           T* __restrict__ gEz, T* __restrict__ oHx, T* __restrict__ oHy,
                           T* __restrict__ ophy, T* __restrict__ ophx, int acc) {
    const long long N = (long long)y.NX * y.NY;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)y.S * N) return;
    const long long node = tid % N;
    const int i = (int)(node / y.NY), j = (int)(node % y.NY);
    oHx[tid] = gHx[tid];
    oHy[tid] = gHy[tid];
    ophy[tid] = -y.Bhy[node] * gHx[tid] + y.Ehy[node] * gphy[tid];
    ophx[tid] = y.Bhx[node] * gHy[tid] + y.Ehx[node] * gphx[tid];
    // Ez[i, j] enters dEy(i, j) with -1 (j <= NY-2), dEy(i, j-1) with +1 (j >= 1),
    // dEx(i, j) with -1 (i <= NX-2), dEx(i-1, j) with +1 (i >= 1)
    T g = 0;
    if (j <= y.NY - 2) g -= gdEy(y, node, tid, gHx, gphy);
    if (j >= 1) g += gdEy(y, node - 1, tid - 1, gHx, gphy);
    if (i <= y.NX - 2) g -= gdEx(y, node, tid, gHy, gphx);
    if (i >= 1) g += gdEx(y, node - y.NY, tid - y.NY, gHy, gphx);
    if (acc & 2)
        gEz[tid] += g;
    else
        gEz[tid] = g;
}

template <typename T>
__global__ void yee2_e_fwd(Y2<T> y, const T* __restrict__ Hx, const T* __restrict__ Hy,
                           const T* __restrict__ Ez, const T* __restrict__ pex,
                           const T* __restrict__ pey, const T* __restrict__ ca,
                           const T* __restrict__ cb, T* __restrict__ oEz, T* __restrict__ opex,
                           T* __restrict__ opey) {
    const long long N = (long long)y.NX * y.NY;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)y.S * N) return;
    const long long node = tid % N;
    const int i = (int)(node / y.NY), j = (int)(node % y.NY);
    const T dHyx = i >= 1 ? Hy[tid] - Hy[tid - y.NY] : T(0);
    const T dHxy = j >= 1 ? Hx[tid] - Hx[tid - 1] : T(0);
    const T px = pex[tid], py = pey[tid];
    const T rhs = dHyx / y.dx - dHxy / y.dy + (y.Aex[node] * dHyx + y.Bex[node] * px)
                - (y.Aey[node] * dHxy + y.Bey[node] * py);
    opex[tid] = y.Eex[node] * px - y.Fex[node] * dHyx;
    opey[tid] = y.Eey[node] * py - y.Fey[node] * dHxy;
    oEz[tid] = ca[node] * Ez[tid] + cb[node] * rhs;
}

// E transpose: blocks of kEX nodes x (up to kEY) shot lanes; lane y takes the shots
// y, y + kEY, ...; the summed ca/cb cotangents are reduced over the lanes in shared
// memory in a fixed order (deterministic).
constexpr int kEX = 32, kEY = 8;

template <typename T>
__global__ void yee2_e_bwd(Y2<T> y, const T* __restrict__ Hx, const T* __restrict__ Hy,
                           const T* __restrict__ Ez, const T* __restrict__ pex,
                           const T* __restrict__ pey, const T* __restrict__ ca,
                           const T* __restrict__ cb, const T* __restrict__ gEz,
                           const T* __restrict__ gpex, const T* __restrict__ gpey,
                           T* __restrict__ oHx, T* __restrict__ oHy, T* __restrict__ oEz,
                           T* __restrict__ opex, T* __restrict__ opey, T* __restrict__ gca,
                           T* __restrict__ gcb, int acc) {
    __shared__ T red[2][kEY][kEX];
    const long long N = (long long)y.NX * y.NY;
    const long long nl = (long long)blockIdx.x * kEX + threadIdx.x;
    const bool valid = nl < N;
    const int node = valid ? (int)nl : 0;
    const int lane = threadIdx.y, lanes = blockDim.y;
    const int i = node / y.NY, j = node - i * y.NY;
    // Hy[i, j] enters dHyx(i, j) with +1 (i >= 1) and dHyx(i+1, j) with -1 (i+1 <= NX-1);
    // Hx[i, j] enters dHxy(i, j) with +1 (j >= 1) and dHxy(i, j+1) with -1 (j+1 <= NY-1);
    // cotangents gdHyx = gr/dx + Aex gr - Fex gpex, gdHxy = -gr/dy - Aey gr - Fey gpey
    // with gr = cb gEz
    T sca = 0, scb = 0;
    if (valid) {
        const bool up = i >= 1, dn = i + 1 <= y.NX - 1, lf = j >= 1, rt = j + 1 <= y.NY - 1;
        const int nxp = dn ? node + y.NY : node, nyp = rt ? node + 1 : node;
        const int nxm = up ? node - y.NY : node, nym = lf ? node - 1 : node;
        const T can = ca[node], cbn = cb[node];
        const T Aex = y.Aex[node], Bex = y.Bex[node], Eex = y.Eex[node], Fex = y.Fex[node];
        const T Aey = y.Aey[node], Bey = y.Bey[node], Eey = y.Eey[node], Fey = y.Fey[node];
        const T cbx = cb[nxp], Aexx = y.Aex[nxp], Fexx = y.Fex[nxp];
        const T cby = cb[nyp], Aeyy = y.Aey[nyp], Feyy = y.Fey[nyp];
        for (int s = lane; s < y.S; s += lanes) {
            const long long off = (long long)s * N, tid = off + node;
            const T ge = gEz[tid], gpx = gpex[tid], gpy = gpey[tid];
            const T gex = gEz[off + nxp], gpxx = gpex[off + nxp];
            const T gey = gEz[off + nyp], gpyy = gpey[off + nyp];
            const T hy = Hy[tid], hym = Hy[off + nxm], hx = Hx[tid], hxm = Hx[off + nym];
            const T px = pex[tid], py = pey[tid], ez = Ez[tid];
            const T ohy = oHy[tid], ohx = oHx[tid];   // used with acc & 2 only
            const T gr = cbn * ge;
            const T dHyx = up ? hy - hym : T(0);
            const T dHxy = lf ? hx - hxm : T(0);
            const T rhs = dHyx / y.dx - dHxy / y.dy + (Aex * dHyx + Bex * px)
                        - (Aey * dHxy + Bey * py);
            oEz[tid] = can * ge;
            opex[tid] = Bex * gr + Eex * gpx;
            opey[tid] = -Bey * gr + Eey * gpy;
            if (acc & 8) {
                sca += ge * ez;
                scb += ge * rhs;
            } else if (acc & 4) {
                gca[tid] += ge * ez;
                gcb[tid] += ge * rhs;
            } else {
                gca[tid] = ge * ez;
                gcb[tid] = ge * rhs;
            }
            T gy = 0, gx = 0;
            if (up) gy += gr / y.dx + Aex * gr - Fex * gpx;
            if (dn) {
                const T g2 = cbx * gex;
                gy -= g2 / y.dx + Aexx * g2 - Fexx * gpxx;
            }
            if (lf) gx += -gr / y.dy - Aey * gr - Fey * gpy;
            if (rt) {
                const T g2 = cby * gey;
                gx -= -g2 / y.dy - Aeyy * g2 - Feyy * gpyy;
            }
            oHy[tid] = (acc & 2) ? ohy + gy : gy;
            oHx[tid] = (acc & 2) ? ohx + gx : gx;
        }
    }
    if (!(acc & 8)) return;                    // uniform over the launch
    red[0][lane][threadIdx.x] = sca;
    red[1][lane][threadIdx.x] = scb;
    __syncthreads();
    if (lane != 0 || !valid) return;
    T a = red[0][0][threadIdx.x], b = red[1][0][threadIdx.x];
    for (int l = 1; l < lanes; ++l) {
        a += red[0][l][threadIdx.x];
        b += red[1][l][threadIdx.x];
    }
    if (acc & 4) {
        gca[node] += a;
        gcb[node] += b;
    } else {
        gca[node] = a;
        gcb[node] = b;
    }
}

template <typename T>
Y2<T> make_y2(const int* gi, const double* gd, const void* const* cp) {
    Y2<T> y;
    y.S = gi[0];
    y.NX = gi[1];
    y.NY = gi[2];
    y.dx = (T)gd[0];
    y.dy = (T)gd[1];
    const T** f[18] = {&y.cHx, &y.cHy, &y.Ahy, &y.Bhy, &y.Ehy, &y.Fhy, &y.Ahx, &y.Bhx, &y.Ehx,
                       &y.Fhx, &y.Aex, &y.Bex, &y.Eex, &y.Fex, &y.Aey, &y.Bey, &y.Eey, &y.Fey};
    for (int k = 0; k < 18; ++k) *f[k] = (const T*)cp[k];
    return y;
}

template <typename T>
int run(int side, int dir, const int* gi, const double* gd, const void* const* cp,
        const void* const* io, cudaStream_t st) {
    // dir bit 0: transpose; bit 1: accumulate the cotangents of the differentiated fields
    // (gEz of the H transpose, gHx/gHy of the E transpose); bit 2: accumulate gca/gcb;
    // bit 3: gca/gcb summed over the shots (NX, NY)
    Y2<T> y = make_y2<T>(gi, gd, cp);
    const int acc = dir & 14;
    dir &= 1;
    const int nb = blocks((long long)y.S * y.NX * y.NY);
    if (side == 0 && dir == 0)        // io: Ez Hx Hy phy phx oHx oHy ophy ophx
        yee2_h_fwd<T><<<nb, kThreads, 0, st>>>(y, (const T*)io[0], (const T*)io[1],
                                                (const T*)io[2], (const T*)io[3], (const T*)io[4],
                                                (T*)io[5], (T*)io[6], (T*)io[7], (T*)io[8]);
    else if (side == 0)               // io: gHx gHy gphy gphx gEz oHx oHy ophy ophx
        yee2_h_bwd<T><<<nb, kThreads, 0, st>>>(y, (const T*)io[0], (const T*)io[1],
                                                (const T*)io[2], (const T*)io[3], (T*)io[4],
                                                (T*)io[5], (T*)io[6], (T*)io[7], (T*)io[8], acc);
    else if (dir == 0)                // io: Hx Hy Ez pex pey ca cb oEz opex opey
        yee2_e_fwd<T><<<nb, kThreads, 0, st>>>(y, (const T*)io[0], (const T*)io[1],
                                                (const T*)io[2], (const T*)io[3], (const T*)io[4],
                                                (const T*)io[5], (const T*)io[6], (T*)io[7],
                                                (T*)io[8], (T*)io[9]);
    else                              // io: Hx Hy Ez pex pey ca cb gEz gpex gpey oHx oHy oEz opex opey gca gcb
        yee2_e_bwd<T><<<(int)(((long long)y.NX * y.NY + kEX - 1) / kEX),
                        dim3(kEX, y.S < kEY ? y.S : kEY), 0, st>>>(
            y, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3],
            (const T*)io[4], (const T*)io[5], (const T*)io[6], (const T*)io[7], (const T*)io[8],
            (const T*)io[9], (T*)io[10], (T*)io[11], (T*)io[12], (T*)io[13], (T*)io[14],
            (T*)io[15], (T*)io[16], acc);
    return (int)cudaGetLastError();
}

}  // namespace

extern "C" {

// side: 0 = H update, 1 = E update; dir: 0 = forward, 1 = transpose.
int yee2d_run(int dtype, int side, int dir, const int* gi, const double* gd,
              const void* const* cp, const void* const* io, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == 0) return run<float>(side, dir, gi, gd, cp, io, st);
    if (dtype == 1) return run<double>(side, dir, gi, gd, cp, io, st);
    return -1;
}

}  // extern "C"

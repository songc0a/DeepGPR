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
// One thread per shot and node. The transposes evaluate the cotangents of the
// differences of the node and of its neighbour and gather them directly (no
// temporaries, no atomics); ca/cb cotangents per shot (S, NX, NY).

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
                           T* __restrict__ ophy, T* __restrict__ ophx) {
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

// cotangents of dHyx, dHxy at (node)
template <typename T>
__device__ inline T gdHyx(const Y2<T>& y, long long node, long long id, const T* cb,
                          const T* gEz, const T* gpex) {
    const T gr = cb[node] * gEz[id];
    return gr / y.dx + y.Aex[node] * gr - y.Fex[node] * gpex[id];
}
template <typename T>
__device__ inline T gdHxy(const Y2<T>& y, long long node, long long id, const T* cb,
                          const T* gEz, const T* gpey) {
    const T gr = cb[node] * gEz[id];
    return -gr / y.dy - y.Aey[node] * gr - y.Fey[node] * gpey[id];
}

template <typename T>
__global__ void yee2_e_bwd(Y2<T> y, const T* __restrict__ Hx, const T* __restrict__ Hy,
                           const T* __restrict__ Ez, const T* __restrict__ pex,
                           const T* __restrict__ pey, const T* __restrict__ ca,
                           const T* __restrict__ cb, const T* __restrict__ gEz,
                           const T* __restrict__ gpex, const T* __restrict__ gpey,
                           T* __restrict__ oHx, T* __restrict__ oHy, T* __restrict__ oEz,
                           T* __restrict__ opex, T* __restrict__ opey, T* __restrict__ gca,
                           T* __restrict__ gcb) {
    const long long N = (long long)y.NX * y.NY;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)y.S * N) return;
    const long long node = tid % N;
    const int i = (int)(node / y.NY), j = (int)(node % y.NY);
    const T ge = gEz[tid];
    const T gr = cb[node] * ge;
    const T dHyx = i >= 1 ? Hy[tid] - Hy[tid - y.NY] : T(0);
    const T dHxy = j >= 1 ? Hx[tid] - Hx[tid - 1] : T(0);
    const T px = pex[tid], py = pey[tid];
    const T rhs = dHyx / y.dx - dHxy / y.dy + (y.Aex[node] * dHyx + y.Bex[node] * px)
                - (y.Aey[node] * dHxy + y.Bey[node] * py);
    oEz[tid] = ca[node] * ge;
    opex[tid] = y.Bex[node] * gr + y.Eex[node] * gpex[tid];
    opey[tid] = -y.Bey[node] * gr + y.Eey[node] * gpey[tid];
    gca[tid] = ge * Ez[tid];
    gcb[tid] = ge * rhs;
    // Hy[i, j] enters dHyx(i, j) with +1 (i >= 1) and dHyx(i+1, j) with -1 (i+1 <= NX-1);
    // Hx[i, j] enters dHxy(i, j) with +1 (j >= 1) and dHxy(i, j+1) with -1 (j+1 <= NY-1)
    T gy = 0, gx = 0;
    if (i >= 1) gy += gdHyx(y, node, tid, cb, gEz, gpex);
    if (i + 1 <= y.NX - 1) gy -= gdHyx(y, node + y.NY, tid + y.NY, cb, gEz, gpex);
    if (j >= 1) gx += gdHxy(y, node, tid, cb, gEz, gpey);
    if (j + 1 <= y.NY - 1) gx -= gdHxy(y, node + 1, tid + 1, cb, gEz, gpey);
    oHy[tid] = gy;
    oHx[tid] = gx;
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
    Y2<T> y = make_y2<T>(gi, gd, cp);
    const int nb = blocks((long long)y.S * y.NX * y.NY);
    if (side == 0 && dir == 0)        // io: Ez Hx Hy phy phx oHx oHy ophy ophx
        yee2_h_fwd<T><<<nb, kThreads, 0, st>>>(y, (const T*)io[0], (const T*)io[1],
                                                (const T*)io[2], (const T*)io[3], (const T*)io[4],
                                                (T*)io[5], (T*)io[6], (T*)io[7], (T*)io[8]);
    else if (side == 0)               // io: gHx gHy gphy gphx gEz oHx oHy ophy ophx
        yee2_h_bwd<T><<<nb, kThreads, 0, st>>>(y, (const T*)io[0], (const T*)io[1],
                                                (const T*)io[2], (const T*)io[3], (T*)io[4],
                                                (T*)io[5], (T*)io[6], (T*)io[7], (T*)io[8]);
    else if (dir == 0)                // io: Hx Hy Ez pex pey ca cb oEz opex opey
        yee2_e_fwd<T><<<nb, kThreads, 0, st>>>(y, (const T*)io[0], (const T*)io[1],
                                                (const T*)io[2], (const T*)io[3], (const T*)io[4],
                                                (const T*)io[5], (const T*)io[6], (T*)io[7],
                                                (T*)io[8], (T*)io[9]);
    else                              // io: Hx Hy Ez pex pey ca cb gEz gpex gpey oHx oHy oEz opex opey gca gcb
        yee2_e_bwd<T><<<nb, kThreads, 0, st>>>(
            y, (const T*)io[0], (const T*)io[1], (const T*)io[2], (const T*)io[3],
            (const T*)io[4], (const T*)io[5], (const T*)io[6], (const T*)io[7], (const T*)io[8],
            (const T*)io[9], (T*)io[10], (T*)io[11], (T*)io[12], (T*)io[13], (T*)io[14],
            (T*)io[15], (T*)io[16]);
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

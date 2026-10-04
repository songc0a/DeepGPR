#ifndef DEEPGPR_NATIVE_API_H
#define DEEPGPR_NATIVE_API_H

#define DEEPGPR_ABI_VERSION 6

#if defined(_WIN32)
#if defined(DEEPGPR_BUILD)
#define DEEPGPR_API __declspec(dllexport)
#else
#define DEEPGPR_API __declspec(dllimport)
#endif
#else
#define DEEPGPR_API __attribute__((visibility("default")))
#endif

/*
 * Physical constants shared by the CPU and CUDA backends and mirrored by the
 * Python package (DeepGPR/config/constants.py; tests/test_native_abi.py checks
 * that both sides agree). CODATA 2018 values, SI units.
 */
#define DEEPGPR_EPSILON0 8.8541878128e-12   /* vacuum permittivity [F/m] */
#define DEEPGPR_MU0 1.25663706212e-06       /* vacuum permeability [H/m] */
#define DEEPGPR_EPSILON0_F 8.8541878128e-12f
#define DEEPGPR_MU0_F 1.25663706212e-06f

/*
 * Cells whose conductivity exceeds this value [S/m] are treated as perfect
 * electric conductors: their E update coefficients are zero and they receive
 * no material gradient.
 */
#define DEEPGPR_PEC_SIGMA_THRESHOLD 100.0
#define DEEPGPR_PEC_SIGMA_THRESHOLD_F 100.0f

/* Capacity of the per-thread error message buffer (bytes, incl. NUL). */
#define DEEPGPR_ERROR_BUFFER_SIZE 512

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Error reporting (additive to ABI 6).
 *
 * forward/backward clear the calling thread's error message on entry and set
 * it when they stop early (allocation failure, CUDA API or launch error).
 * deepgpr_last_error returns "" when the last call on this thread succeeded.
 */
DEEPGPR_API int deepgpr_supports_error_reporting(void);
DEEPGPR_API const char* deepgpr_last_error(void);
DEEPGPR_API void deepgpr_clear_last_error(void);

/* 1 when the adjoint result is independent of the number of threads. */
DEEPGPR_API int deepgpr_deterministic_adjoint(void);

DEEPGPR_API int deepgpr_abi_version(void);
DEEPGPR_API int deepgpr_supports_external_pml(void);
DEEPGPR_API int deepgpr_supports_int8_wavefield(void);
DEEPGPR_API int deepgpr_supports_conversion_backends(void);
DEEPGPR_API int deepgpr_supports_int8_reduction_backends(void);
/*
 * 1 when storage_type may carry WAVEFIELD_RHS_FROM_E (bit 4): with sampling
 * interval 1 only E^n is stored, R_saved receives the final E^nt frame, and
 * the adjoint rebuilds R^n = (E^(n+1) - ca E^n) / cb.
 */
DEEPGPR_API int deepgpr_supports_rhs_reconstruction(void);
DEEPGPR_API void deepgpr_test_wavefield_conversion(
    const float* input, void* encoded, float* decoded, long long count,
    int storage_kind, int conversion_backend);
DEEPGPR_API void set_fdtd_order(int order);

DEEPGPR_API void forward(
    const float* eps_r_pad, const float* sigma_pad, const float* mu_r_pad,
    void* E_saved, void* R_saved,
    float* Ex, float* Ey, float* Ez,
    float* Hx, float* Hy, float* Hz,
    float* ce_hist, float* ce_curl, float* ce_rhs,
    float* ch_hist, float* ch_curl, float* ch_rhs,
    float* x0_e_phi1, float* x0_e_phi2, float* x0_h_phi1, float* x0_h_phi2,
    float* xm_e_phi1, float* xm_e_phi2, float* xm_h_phi1, float* xm_h_phi2,
    float* y0_e_phi1, float* y0_e_phi2, float* y0_h_phi1, float* y0_h_phi2,
    float* ym_e_phi1, float* ym_e_phi2, float* ym_h_phi1, float* ym_h_phi2,
    float* z0_e_phi1, float* z0_e_phi2, float* z0_h_phi1, float* z0_h_phi2,
    float* zm_e_phi1, float* zm_e_phi2, float* zm_h_phi1, float* zm_h_phi2,
    int pml_x0, int pml_xm, int pml_y0, int pml_ym, int pml_z0, int pml_zm,
    const float* x0_e_coeff, const float* xm_e_coeff,
    const float* y0_e_coeff, const float* ym_e_coeff,
    const float* z0_e_coeff, const float* zm_e_coeff,
    const float* x0_h_coeff, const float* xm_h_coeff,
    const float* y0_h_coeff, const float* ym_h_coeff,
    const float* z0_h_coeff, const float* zm_h_coeff,
    float dt, int nt, int nshot, int nreceiver,
    float dx, float dy, float dz,
    const int* receiver_location, float* receiver_data, int receiver_component,
    int nx_fields, int ny_fields, int nz_fields, int nsource,
    const int* source_location, const float* source_waveform,
    int source_component, int sampling_interval, int fwi_mode, int storage_type,
    int save_model_history, int use_async_offload);

DEEPGPR_API void backward(
    const float* eps_r_pad, const float* sigma_pad, const float* mu_r_pad,
    const void* E_saved, const void* R_saved,
    float* lambda_ex, float* lambda_ey, float* lambda_ez,
    float* lambda_hx, float* lambda_hy, float* lambda_hz,
    float* ce_hist, float* ce_curl, float* ce_rhs,
    float* ch_hist, float* ch_curl, float* ch_rhs,
    float* lambda_x0_e_phi1, float* lambda_x0_e_phi2,
    float* lambda_x0_h_phi1, float* lambda_x0_h_phi2,
    float* lambda_xm_e_phi1, float* lambda_xm_e_phi2,
    float* lambda_xm_h_phi1, float* lambda_xm_h_phi2,
    float* lambda_y0_e_phi1, float* lambda_y0_e_phi2,
    float* lambda_y0_h_phi1, float* lambda_y0_h_phi2,
    float* lambda_ym_e_phi1, float* lambda_ym_e_phi2,
    float* lambda_ym_h_phi1, float* lambda_ym_h_phi2,
    float* lambda_z0_e_phi1, float* lambda_z0_e_phi2,
    float* lambda_z0_h_phi1, float* lambda_z0_h_phi2,
    float* lambda_zm_e_phi1, float* lambda_zm_e_phi2,
    float* lambda_zm_h_phi1, float* lambda_zm_h_phi2,
    int pml_x0, int pml_xm, int pml_y0, int pml_ym, int pml_z0, int pml_zm,
    const float* x0_e_coeff, const float* xm_e_coeff,
    const float* y0_e_coeff, const float* ym_e_coeff,
    const float* z0_e_coeff, const float* zm_e_coeff,
    const float* x0_h_coeff, const float* xm_h_coeff,
    const float* y0_h_coeff, const float* ym_h_coeff,
    const float* z0_h_coeff, const float* zm_h_coeff,
    float dt, int nt, int nshot, int nreceiver,
    float dx, float dy, float dz,
    int nx_fields, int ny_fields, int nz_fields,
    int ndata_source, const int* receiver_location, const float* data_grad,
    int receiver_component,
    int nsource, const int* source_location, int source_component,
    float* grad_source, int source_requires_grad,
    float* grad_eps_r, float* grad_sigma,
    int eps_r_requires_grad, int sigma_requires_grad,
    int sampling_interval, int fwi_mode, int storage_type, int use_async_offload);

#ifdef __cplusplus
}
#endif

#endif

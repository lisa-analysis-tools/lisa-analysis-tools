#include "galactic_response.hpp"
#include "gbt_global.h"

#include <cmath>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <string>
#include <stdio.h>

// ============================================================================
// GPU error checking
// ============================================================================

#ifdef __CUDACC__
#ifndef gpuErrchk
#define gpuErrchk(ans) { gpuAssert_gal((ans), __FILE__, __LINE__); }
inline void gpuAssert_gal(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert (galactic): %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif
#endif

// ============================================================================
// Thread block sizes
// ============================================================================

#define GAL_THREADS_SKY  256
#define GAL_THREADS_SPEC 256

// ============================================================================
// GalacticGridSetup — host implementation
// ============================================================================

static const double GL16_NODES[16] = {
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318,
    -0.7554044083550030, -0.6178762444026438, -0.4580167776572274,
    -0.2816035507792589, -0.0950125098376374,  0.0950125098376374,
     0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
     0.7554044083550030,  0.8656312023878318,  0.9445750230732326,
     0.9894009349916499
};

static const double GL16_WEIGHTS[16] = {
    0.0271524594117541, 0.0622535239386479, 0.0951585116824928,
    0.1246289712555339, 0.1495959888165767, 0.1691565193950025,
    0.1826034150449236, 0.1894506104550685, 0.1894506104550685,
    0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
    0.1246289712555339, 0.0951585116824928, 0.0622535239386479,
    0.0271524594117541
};

static constexpr double GGS_R_SUN = 8.12;
static constexpr double GGS_Z_SUN = 0.03;
static constexpr double GGS_D_MAX = 20.0;

static const double GGS_ROT[3][3] = {
    { -5.48755604e-02, -9.93821362e-01, -9.64768044e-02 },
    {  4.94109428e-01, -1.10990888e-01,  8.62285855e-01 },
    { -8.67666149e-01, -3.51679084e-04,  4.97147192e-01 }
};

static inline void ecl_to_gal_sky(double lam_ecl, double beta_ecl,
                                    double &lam_gal, double &beta_gal)
{
    double cb = cos(beta_ecl);
    double x  = cos(lam_ecl) * cb;
    double y  = sin(lam_ecl) * cb;
    double z  = sin(beta_ecl);

    double xr = GGS_ROT[0][0]*x + GGS_ROT[0][1]*y + GGS_ROT[0][2]*z;
    double yr = GGS_ROT[1][0]*x + GGS_ROT[1][1]*y + GGS_ROT[1][2]*z;
    double zr = GGS_ROT[2][0]*x + GGS_ROT[2][1]*y + GGS_ROT[2][2]*z;

    double phi = atan2(yr, xr);
    if (phi < 0.0) phi += 2.0 * M_PI;

    lam_gal  = phi;
    beta_gal = atan2(zr, sqrt(xr*xr + yr*yr));
}

void GalacticGridSetup::compute(int N_lambda_, int N_beta_,
                                double los_center_half_width, int los_center_sub)
{
    N_lambda = N_lambda_;
    N_beta   = N_beta_;
    N_sky    = N_lambda * N_beta;

    // Line-of-sight breakpoints: one 16-point Gauss-Legendre rule per piece.
    std::vector<double> edges;
    edges.push_back(0.0);
    if (los_center_half_width > 0.0) {
        if (los_center_sub < 1)
            throw std::invalid_argument("los_center_sub must be >= 1.");
        double a = fmax(0.0, GGS_R_SUN - los_center_half_width);
        double b = fmin(GGS_D_MAX, GGS_R_SUN + los_center_half_width);
        if (a > 0.0) edges.push_back(a);
        for (int k = 1; k < los_center_sub; k++)
            edges.push_back(a + (b - a) * k / los_center_sub);
        if (b < GGS_D_MAX) edges.push_back(b);
    }
    edges.push_back(GGS_D_MAX);
    int n_pieces = (int)edges.size() - 1;
    N_quad = 16 * n_pieces;

    std::vector<double> lams(N_lambda), betas(N_beta);
    for (int il = 0; il < N_lambda; il++)
        lams[il]  = 2.0 * M_PI * il / N_lambda;
    for (int ib = 0; ib < N_beta; ib++)
        betas[ib] = -M_PI/2.0 + M_PI * ib / (N_beta - 1);

    lam_ecl.resize(N_sky);
    beta_ecl.resize(N_sky);
    cos_beta_ecl.resize(N_sky);
    lam_gal.resize(N_sky);
    beta_gal.resize(N_sky);

    for (int il = 0; il < N_lambda; il++) {
        for (int ib = 0; ib < N_beta; ib++) {
            int idx           = il * N_beta + ib;
            lam_ecl[idx]      = lams[il];
            beta_ecl[idx]     = betas[ib];
            cos_beta_ecl[idx] = cos(betas[ib]);
            ecl_to_gal_sky(lams[il], betas[ib], lam_gal[idx], beta_gal[idx]);
        }
    }

    D_nodes.resize(N_quad);
    quad_weights.resize(N_quad);
    for (int p = 0; p < n_pieces; p++) {
        double lo = edges[p], hi = edges[p + 1];
        for (int k = 0; k < 16; k++) {
            D_nodes[16 * p + k]      = lo + 0.5 * (GL16_NODES[k] + 1.0) * (hi - lo);
            quad_weights[16 * p + k] = 0.5 * (hi - lo) * GL16_WEIGHTS[k];
        }
    }

    R_vals_quad.resize(N_quad * N_sky);
    z_vals_quad.resize(N_quad * N_sky);

    for (int q = 0; q < N_quad; q++) {
        double D = D_nodes[q];
        for (int i = 0; i < N_sky; i++) {
            double lg  = lam_gal[i];
            double bg  = beta_gal[i];
            double x = D * sin(lg) * cos(bg);
            double y = GGS_R_SUN - D * cos(lg) * cos(bg);
            double z = GGS_Z_SUN - D * sin(bg);
            R_vals_quad[q * N_sky + i] = sqrt(x*x + y*y);
            z_vals_quad[q * N_sky + i] = z;
        }
    }
}

void GalacticGridSetup::print_summary() const
{
    printf("GalacticGridSetup summary:\n");
    printf("  N_lambda=%d  N_beta=%d  N_sky=%d  N_quad=%d\n",
           N_lambda, N_beta, N_sky, N_quad);
    if (D_nodes.empty()) { printf("  (not yet computed)\n"); return; }
    printf("  D_nodes: [%.4f, %.4f] kpc\n", D_nodes.front(), D_nodes.back());
    double w_sum = 0.0; for (double w : quad_weights) w_sum += w;
    printf("  quad_weights sum = %.10f  (expect %.4f)\n", w_sum, GGS_D_MAX);
}

// ============================================================================
// Device helpers
// ============================================================================

static CUDA_DEVICE
double disk_density(double R, double z, double R_d, double z_d)
{
    return exp(-R / R_d) * exp(-fabs(z) / z_d);
}

static CUDA_DEVICE
double bulge_density(double R, double z, double r_b)
{
    return exp(-(R * R + z * z) / (r_b * r_b));
}

// Mass-normalized disk + bulge mixture; f_b = 0 falls back to the bare disk density
// (the normalization constant cancels in the sky-weight normalization).
static CUDA_DEVICE
double galaxy_density(double R, double z, double R_d, double z_d, double f_b, double r_b)
{
    if (f_b <= 0.0)
        return disk_density(R, z, R_d, z_d);
    double disk  = disk_density(R, z, R_d, z_d) / (4.0 * M_PI * R_d * R_d * z_d);
    double bulge = bulge_density(R, z, r_b) / (pow(M_PI, 1.5) * r_b * r_b * r_b);
    return (1.0 - f_b) * disk + f_b * bulge;
}

static CUDA_DEVICE
double gal_spectral_model(double f, double Amp, double alpha,
                           double f_1, double f_knee, double f_2,
                           double avg_d)
{
    double omega_f  = 2.0 * M_PI * f;
    double x        = omega_f * avg_d;
    double transfer = 4.0 * (x * sin(x)) * (x * sin(x))
                    * 4.0 * sin(2.0 * x) * sin(2.0 * x);
    double psd_gal  = Amp * exp(-pow(f / f_1, alpha)) * pow(f, -7.0 / 3.0)
                    * 0.5 * (1.0 + tanh(-(f - f_knee) / f_2));
    return psd_gal * transfer;
}


// ============================================================================
CUDA_DEVICE
double galaxy_XX(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double a = alpha0 + OMEGA_LISA * delta_t;
    double b = beta0, th = theta_N, ph = phi_N;
    double r = 0.0;
    r += (9363.0 / 8192.0);
    r += (-345.0 / 2048.0) * cos(2*th);
    r += (-111.0 / 8192.0) * cos(4*th);
    r += (-243.0 / 32768.0) * cos(4*b - 4*th - 4*ph);
    r += (243.0 / 8192.0) * cos(4*b - 2*th - 4*ph);
    r += (-729.0 / 16384.0) * cos(4*b - 4*ph);
    r += (243.0 / 8192.0) * cos(4*b + 2*th - 4*ph);
    r += (-243.0 / 32768.0) * cos(4*b + 4*th - 4*ph);
    r += SQRT3_GAL * (-81.0 / 4096.0) * sin(a - 4*b - 4*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 2048.0) * sin(a - 4*b - 2*th + 3*ph);
    r += SQRT3_GAL * (-81.0 / 2048.0) * sin(a - 4*b + 2*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 4096.0) * sin(a - 4*b + 4*th + 3*ph);
    r += SQRT3_GAL * (-15.0 / 2048.0) * sin(a - 4*th - ph);
    r += SQRT3_GAL * (327.0 / 1024.0) * sin(a - 2*th - ph);
    r += SQRT3_GAL * (-327.0 / 1024.0) * sin(a + 2*th - ph);
    r += SQRT3_GAL * (15.0 / 2048.0) * sin(a + 4*th - ph);
    r += (567.0 / 8192.0) * cos(2*a - 4*b - 4*th + 2*ph);
    r += (-81.0 / 2048.0) * cos(2*a - 4*b - 2*th + 2*ph);
    r += (-243.0 / 4096.0) * cos(2*a - 4*b + 2*ph);
    r += (-81.0 / 2048.0) * cos(2*a - 4*b + 2*th + 2*ph);
    r += (567.0 / 8192.0) * cos(2*a - 4*b + 4*th + 2*ph);
    r += (-27.0 / 4096.0) * cos(2*a - 4*th - 2*ph);
    r += (-243.0 / 1024.0) * cos(2*a - 2*th - 2*ph);
    r += (999.0 / 2048.0) * cos(2*a - 2*ph);
    r += (-243.0 / 1024.0) * cos(2*a + 2*th - 2*ph);
    r += (-27.0 / 4096.0) * cos(2*a + 4*th - 2*ph);
    r += SQRT3_GAL * (189.0 / 4096.0) * sin(3*a - 4*b - 4*th + ph);
    r += SQRT3_GAL * (27.0 / 2048.0) * sin(3*a - 4*b - 2*th + ph);
    r += SQRT3_GAL * (-27.0 / 2048.0) * sin(3*a - 4*b + 2*th + ph);
    r += SQRT3_GAL * (-189.0 / 4096.0) * sin(3*a - 4*b + 4*th + ph);
    r += SQRT3_GAL * (-9.0 / 2048.0) * sin(3*a - 4*th - 3*ph);
    r += SQRT3_GAL * (9.0 / 1024.0) * sin(3*a - 2*th - 3*ph);
    r += SQRT3_GAL * (-9.0 / 1024.0) * sin(3*a + 2*th - 3*ph);
    r += SQRT3_GAL * (9.0 / 2048.0) * sin(3*a + 4*th - 3*ph);
    r += (-945.0 / 16384.0) * cos(4*a - 4*b - 4*th);
    r += (-135.0 / 4096.0) * cos(4*a - 4*b - 2*th);
    r += (-243.0 / 8192.0) * cos(4*a - 4*b);
    r += (-135.0 / 4096.0) * cos(4*a - 4*b + 2*th);
    r += (-945.0 / 16384.0) * cos(4*a - 4*b + 4*th);
    r += (27.0 / 16384.0) * cos(4*a - 4*th - 4*ph);
    r += (-27.0 / 4096.0) * cos(4*a - 2*th - 4*ph);
    r += (81.0 / 8192.0) * cos(4*a - 4*ph);
    r += (-27.0 / 4096.0) * cos(4*a + 2*th - 4*ph);
    r += (27.0 / 16384.0) * cos(4*a + 4*th - 4*ph);
    r += SQRT3_GAL * (-63.0 / 4096.0) * sin(5*a - 4*b - 4*th - ph);
    r += SQRT3_GAL * (-9.0 / 2048.0) * sin(5*a - 4*b - 2*th - ph);
    r += SQRT3_GAL * (9.0 / 2048.0) * sin(5*a - 4*b + 2*th - ph);
    r += SQRT3_GAL * (63.0 / 4096.0) * sin(5*a - 4*b + 4*th - ph);
    r += (63.0 / 8192.0) * cos(6*a - 4*b - 4*th - 2*ph);
    r += (-9.0 / 2048.0) * cos(6*a - 4*b - 2*th - 2*ph);
    r += (-27.0 / 4096.0) * cos(6*a - 4*b - 2*ph);
    r += (-9.0 / 2048.0) * cos(6*a - 4*b + 2*th - 2*ph);
    r += (63.0 / 8192.0) * cos(6*a - 4*b + 4*th - 2*ph);
    r += SQRT3_GAL * (3.0 / 4096.0) * sin(7*a - 4*b - 4*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 2048.0) * sin(7*a - 4*b - 2*th - 3*ph);
    r += SQRT3_GAL * (3.0 / 2048.0) * sin(7*a - 4*b + 2*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 4096.0) * sin(7*a - 4*b + 4*th - 3*ph);
    r += (-3.0 / 32768.0) * cos(8*a - 4*b - 4*th - 4*ph);
    r += (3.0 / 8192.0) * cos(8*a - 4*b - 2*th - 4*ph);
    r += (-9.0 / 16384.0) * cos(8*a - 4*b - 4*ph);
    r += (3.0 / 8192.0) * cos(8*a - 4*b + 2*th - 4*ph);
    r += (-3.0 / 32768.0) * cos(8*a - 4*b + 4*th - 4*ph);
    return GAL_RESPONSE_K * r;
}

CUDA_DEVICE
double galaxy_XY(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double a = alpha0 + OMEGA_LISA * delta_t;
    double b = beta0, th = theta_N, ph = phi_N;
    double r = 0.0;
    r += (-9363.0 / 16384.0);
    r += (345.0 / 4096.0) * cos(2*th);
    r += (111.0 / 16384.0) * cos(4*th);
    r += (243.0 / 65536.0) * cos(4*b - 4*th - 4*ph);
    r += SQRT3_GAL * (-243.0 / 65536.0) * sin(4*b - 4*th - 4*ph);
    r += (-243.0 / 16384.0) * cos(4*b - 2*th - 4*ph);
    r += SQRT3_GAL * (243.0 / 16384.0) * sin(4*b - 2*th - 4*ph);
    r += (729.0 / 32768.0) * cos(4*b - 4*ph);
    r += SQRT3_GAL * (-729.0 / 32768.0) * sin(4*b - 4*ph);
    r += (-243.0 / 16384.0) * cos(4*b + 2*th - 4*ph);
    r += SQRT3_GAL * (243.0 / 16384.0) * sin(4*b + 2*th - 4*ph);
    r += (243.0 / 65536.0) * cos(4*b + 4*th - 4*ph);
    r += SQRT3_GAL * (-243.0 / 65536.0) * sin(4*b + 4*th - 4*ph);
    r += (-243.0 / 8192.0) * cos(a - 4*b - 4*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 8192.0) * sin(a - 4*b - 4*th + 3*ph);
    r += (243.0 / 4096.0) * cos(a - 4*b - 2*th + 3*ph);
    r += SQRT3_GAL * (-81.0 / 4096.0) * sin(a - 4*b - 2*th + 3*ph);
    r += (-243.0 / 4096.0) * cos(a - 4*b + 2*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 4096.0) * sin(a - 4*b + 2*th + 3*ph);
    r += (243.0 / 8192.0) * cos(a - 4*b + 4*th + 3*ph);
    r += SQRT3_GAL * (-81.0 / 8192.0) * sin(a - 4*b + 4*th + 3*ph);
    r += SQRT3_GAL * (15.0 / 4096.0) * sin(a - 4*th - ph);
    r += SQRT3_GAL * (-327.0 / 2048.0) * sin(a - 2*th - ph);
    r += SQRT3_GAL * (327.0 / 2048.0) * sin(a + 2*th - ph);
    r += SQRT3_GAL * (-15.0 / 4096.0) * sin(a + 4*th - ph);
    r += (-567.0 / 16384.0) * cos(2*a - 4*b - 4*th + 2*ph);
    r += SQRT3_GAL * (-567.0 / 16384.0) * sin(2*a - 4*b - 4*th + 2*ph);
    r += (81.0 / 4096.0) * cos(2*a - 4*b - 2*th + 2*ph);
    r += SQRT3_GAL * (81.0 / 4096.0) * sin(2*a - 4*b - 2*th + 2*ph);
    r += (243.0 / 8192.0) * cos(2*a - 4*b + 2*ph);
    r += SQRT3_GAL * (243.0 / 8192.0) * sin(2*a - 4*b + 2*ph);
    r += (81.0 / 4096.0) * cos(2*a - 4*b + 2*th + 2*ph);
    r += SQRT3_GAL * (81.0 / 4096.0) * sin(2*a - 4*b + 2*th + 2*ph);
    r += (-567.0 / 16384.0) * cos(2*a - 4*b + 4*th + 2*ph);
    r += SQRT3_GAL * (-567.0 / 16384.0) * sin(2*a - 4*b + 4*th + 2*ph);
    r += (27.0 / 8192.0) * cos(2*a - 4*th - 2*ph);
    r += (243.0 / 2048.0) * cos(2*a - 2*th - 2*ph);
    r += (-999.0 / 4096.0) * cos(2*a - 2*ph);
    r += (243.0 / 2048.0) * cos(2*a + 2*th - 2*ph);
    r += (27.0 / 8192.0) * cos(2*a + 4*th - 2*ph);
    r += (567.0 / 8192.0) * cos(3*a - 4*b - 4*th + ph);
    r += SQRT3_GAL * (-189.0 / 8192.0) * sin(3*a - 4*b - 4*th + ph);
    r += (81.0 / 4096.0) * cos(3*a - 4*b - 2*th + ph);
    r += SQRT3_GAL * (-27.0 / 4096.0) * sin(3*a - 4*b - 2*th + ph);
    r += (-81.0 / 4096.0) * cos(3*a - 4*b + 2*th + ph);
    r += SQRT3_GAL * (27.0 / 4096.0) * sin(3*a - 4*b + 2*th + ph);
    r += (-567.0 / 8192.0) * cos(3*a - 4*b + 4*th + ph);
    r += SQRT3_GAL * (189.0 / 8192.0) * sin(3*a - 4*b + 4*th + ph);
    r += SQRT3_GAL * (9.0 / 4096.0) * sin(3*a - 4*th - 3*ph);
    r += SQRT3_GAL * (-9.0 / 2048.0) * sin(3*a - 2*th - 3*ph);
    r += SQRT3_GAL * (9.0 / 2048.0) * sin(3*a + 2*th - 3*ph);
    r += SQRT3_GAL * (-9.0 / 4096.0) * sin(3*a + 4*th - 3*ph);
    r += (945.0 / 32768.0) * cos(4*a - 4*b - 4*th);
    r += SQRT3_GAL * (945.0 / 32768.0) * sin(4*a - 4*b - 4*th);
    r += (135.0 / 8192.0) * cos(4*a - 4*b - 2*th);
    r += SQRT3_GAL * (135.0 / 8192.0) * sin(4*a - 4*b - 2*th);
    r += (243.0 / 16384.0) * cos(4*a - 4*b);
    r += SQRT3_GAL * (243.0 / 16384.0) * sin(4*a - 4*b);
    r += (135.0 / 8192.0) * cos(4*a - 4*b + 2*th);
    r += SQRT3_GAL * (135.0 / 8192.0) * sin(4*a - 4*b + 2*th);
    r += (945.0 / 32768.0) * cos(4*a - 4*b + 4*th);
    r += SQRT3_GAL * (945.0 / 32768.0) * sin(4*a - 4*b + 4*th);
    r += (-27.0 / 32768.0) * cos(4*a - 4*th - 4*ph);
    r += (27.0 / 8192.0) * cos(4*a - 2*th - 4*ph);
    r += (-81.0 / 16384.0) * cos(4*a - 4*ph);
    r += (27.0 / 8192.0) * cos(4*a + 2*th - 4*ph);
    r += (-27.0 / 32768.0) * cos(4*a + 4*th - 4*ph);
    r += (-189.0 / 8192.0) * cos(5*a - 4*b - 4*th - ph);
    r += SQRT3_GAL * (63.0 / 8192.0) * sin(5*a - 4*b - 4*th - ph);
    r += (-27.0 / 4096.0) * cos(5*a - 4*b - 2*th - ph);
    r += SQRT3_GAL * (9.0 / 4096.0) * sin(5*a - 4*b - 2*th - ph);
    r += (27.0 / 4096.0) * cos(5*a - 4*b + 2*th - ph);
    r += SQRT3_GAL * (-9.0 / 4096.0) * sin(5*a - 4*b + 2*th - ph);
    r += (189.0 / 8192.0) * cos(5*a - 4*b + 4*th - ph);
    r += SQRT3_GAL * (-63.0 / 8192.0) * sin(5*a - 4*b + 4*th - ph);
    r += (-63.0 / 16384.0) * cos(6*a - 4*b - 4*th - 2*ph);
    r += SQRT3_GAL * (-63.0 / 16384.0) * sin(6*a - 4*b - 4*th - 2*ph);
    r += (9.0 / 4096.0) * cos(6*a - 4*b - 2*th - 2*ph);
    r += SQRT3_GAL * (9.0 / 4096.0) * sin(6*a - 4*b - 2*th - 2*ph);
    r += (27.0 / 8192.0) * cos(6*a - 4*b - 2*ph);
    r += SQRT3_GAL * (27.0 / 8192.0) * sin(6*a - 4*b - 2*ph);
    r += (9.0 / 4096.0) * cos(6*a - 4*b + 2*th - 2*ph);
    r += SQRT3_GAL * (9.0 / 4096.0) * sin(6*a - 4*b + 2*th - 2*ph);
    r += (-63.0 / 16384.0) * cos(6*a - 4*b + 4*th - 2*ph);
    r += SQRT3_GAL * (-63.0 / 16384.0) * sin(6*a - 4*b + 4*th - 2*ph);
    r += (9.0 / 8192.0) * cos(7*a - 4*b - 4*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 8192.0) * sin(7*a - 4*b - 4*th - 3*ph);
    r += (-9.0 / 4096.0) * cos(7*a - 4*b - 2*th - 3*ph);
    r += SQRT3_GAL * (3.0 / 4096.0) * sin(7*a - 4*b - 2*th - 3*ph);
    r += (9.0 / 4096.0) * cos(7*a - 4*b + 2*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 4096.0) * sin(7*a - 4*b + 2*th - 3*ph);
    r += (-9.0 / 8192.0) * cos(7*a - 4*b + 4*th - 3*ph);
    r += SQRT3_GAL * (3.0 / 8192.0) * sin(7*a - 4*b + 4*th - 3*ph);
    r += (3.0 / 65536.0) * cos(8*a - 4*b - 4*th - 4*ph);
    r += SQRT3_GAL * (3.0 / 65536.0) * sin(8*a - 4*b - 4*th - 4*ph);
    r += (-3.0 / 16384.0) * cos(8*a - 4*b - 2*th - 4*ph);
    r += SQRT3_GAL * (-3.0 / 16384.0) * sin(8*a - 4*b - 2*th - 4*ph);
    r += (9.0 / 32768.0) * cos(8*a - 4*b - 4*ph);
    r += SQRT3_GAL * (9.0 / 32768.0) * sin(8*a - 4*b - 4*ph);
    r += (-3.0 / 16384.0) * cos(8*a - 4*b + 2*th - 4*ph);
    r += SQRT3_GAL * (-3.0 / 16384.0) * sin(8*a - 4*b + 2*th - 4*ph);
    r += (3.0 / 65536.0) * cos(8*a - 4*b + 4*th - 4*ph);
    r += SQRT3_GAL * (3.0 / 65536.0) * sin(8*a - 4*b + 4*th - 4*ph);
    return GAL_RESPONSE_K * r;
}

CUDA_DEVICE
double galaxy_XZ(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double a = alpha0 + OMEGA_LISA * delta_t;
    double b = beta0, th = theta_N, ph = phi_N;
    double r = 0.0;
    r += (-9363.0 / 16384.0);
    r += (345.0 / 4096.0) * cos(2*th);
    r += (111.0 / 16384.0) * cos(4*th);
    r += (243.0 / 65536.0) * cos(4*b - 4*th - 4*ph);
    r += SQRT3_GAL * (243.0 / 65536.0) * sin(4*b - 4*th - 4*ph);
    r += (-243.0 / 16384.0) * cos(4*b - 2*th - 4*ph);
    r += SQRT3_GAL * (-243.0 / 16384.0) * sin(4*b - 2*th - 4*ph);
    r += (729.0 / 32768.0) * cos(4*b - 4*ph);
    r += SQRT3_GAL * (729.0 / 32768.0) * sin(4*b - 4*ph);
    r += (-243.0 / 16384.0) * cos(4*b + 2*th - 4*ph);
    r += SQRT3_GAL * (-243.0 / 16384.0) * sin(4*b + 2*th - 4*ph);
    r += (243.0 / 65536.0) * cos(4*b + 4*th - 4*ph);
    r += SQRT3_GAL * (243.0 / 65536.0) * sin(4*b + 4*th - 4*ph);
    r += (243.0 / 8192.0) * cos(a - 4*b - 4*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 8192.0) * sin(a - 4*b - 4*th + 3*ph);
    r += (-243.0 / 4096.0) * cos(a - 4*b - 2*th + 3*ph);
    r += SQRT3_GAL * (-81.0 / 4096.0) * sin(a - 4*b - 2*th + 3*ph);
    r += (243.0 / 4096.0) * cos(a - 4*b + 2*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 4096.0) * sin(a - 4*b + 2*th + 3*ph);
    r += (-243.0 / 8192.0) * cos(a - 4*b + 4*th + 3*ph);
    r += SQRT3_GAL * (-81.0 / 8192.0) * sin(a - 4*b + 4*th + 3*ph);
    r += SQRT3_GAL * (15.0 / 4096.0) * sin(a - 4*th - ph);
    r += SQRT3_GAL * (-327.0 / 2048.0) * sin(a - 2*th - ph);
    r += SQRT3_GAL * (327.0 / 2048.0) * sin(a + 2*th - ph);
    r += SQRT3_GAL * (-15.0 / 4096.0) * sin(a + 4*th - ph);
    r += (-567.0 / 16384.0) * cos(2*a - 4*b - 4*th + 2*ph);
    r += SQRT3_GAL * (567.0 / 16384.0) * sin(2*a - 4*b - 4*th + 2*ph);
    r += (81.0 / 4096.0) * cos(2*a - 4*b - 2*th + 2*ph);
    r += SQRT3_GAL * (-81.0 / 4096.0) * sin(2*a - 4*b - 2*th + 2*ph);
    r += (243.0 / 8192.0) * cos(2*a - 4*b + 2*ph);
    r += SQRT3_GAL * (-243.0 / 8192.0) * sin(2*a - 4*b + 2*ph);
    r += (81.0 / 4096.0) * cos(2*a - 4*b + 2*th + 2*ph);
    r += SQRT3_GAL * (-81.0 / 4096.0) * sin(2*a - 4*b + 2*th + 2*ph);
    r += (-567.0 / 16384.0) * cos(2*a - 4*b + 4*th + 2*ph);
    r += SQRT3_GAL * (567.0 / 16384.0) * sin(2*a - 4*b + 4*th + 2*ph);
    r += (27.0 / 8192.0) * cos(2*a - 4*th - 2*ph);
    r += (243.0 / 2048.0) * cos(2*a - 2*th - 2*ph);
    r += (-999.0 / 4096.0) * cos(2*a - 2*ph);
    r += (243.0 / 2048.0) * cos(2*a + 2*th - 2*ph);
    r += (27.0 / 8192.0) * cos(2*a + 4*th - 2*ph);
    r += (-567.0 / 8192.0) * cos(3*a - 4*b - 4*th + ph);
    r += SQRT3_GAL * (-189.0 / 8192.0) * sin(3*a - 4*b - 4*th + ph);
    r += (-81.0 / 4096.0) * cos(3*a - 4*b - 2*th + ph);
    r += SQRT3_GAL * (-27.0 / 4096.0) * sin(3*a - 4*b - 2*th + ph);
    r += (81.0 / 4096.0) * cos(3*a - 4*b + 2*th + ph);
    r += SQRT3_GAL * (27.0 / 4096.0) * sin(3*a - 4*b + 2*th + ph);
    r += (567.0 / 8192.0) * cos(3*a - 4*b + 4*th + ph);
    r += SQRT3_GAL * (189.0 / 8192.0) * sin(3*a - 4*b + 4*th + ph);
    r += SQRT3_GAL * (9.0 / 4096.0) * sin(3*a - 4*th - 3*ph);
    r += SQRT3_GAL * (-9.0 / 2048.0) * sin(3*a - 2*th - 3*ph);
    r += SQRT3_GAL * (9.0 / 2048.0) * sin(3*a + 2*th - 3*ph);
    r += SQRT3_GAL * (-9.0 / 4096.0) * sin(3*a + 4*th - 3*ph);
    r += (945.0 / 32768.0) * cos(4*a - 4*b - 4*th);
    r += SQRT3_GAL * (-945.0 / 32768.0) * sin(4*a - 4*b - 4*th);
    r += (135.0 / 8192.0) * cos(4*a - 4*b - 2*th);
    r += SQRT3_GAL * (-135.0 / 8192.0) * sin(4*a - 4*b - 2*th);
    r += (243.0 / 16384.0) * cos(4*a - 4*b);
    r += SQRT3_GAL * (-243.0 / 16384.0) * sin(4*a - 4*b);
    r += (135.0 / 8192.0) * cos(4*a - 4*b + 2*th);
    r += SQRT3_GAL * (-135.0 / 8192.0) * sin(4*a - 4*b + 2*th);
    r += (945.0 / 32768.0) * cos(4*a - 4*b + 4*th);
    r += SQRT3_GAL * (-945.0 / 32768.0) * sin(4*a - 4*b + 4*th);
    r += (-27.0 / 32768.0) * cos(4*a - 4*th - 4*ph);
    r += (27.0 / 8192.0) * cos(4*a - 2*th - 4*ph);
    r += (-81.0 / 16384.0) * cos(4*a - 4*ph);
    r += (27.0 / 8192.0) * cos(4*a + 2*th - 4*ph);
    r += (-27.0 / 32768.0) * cos(4*a + 4*th - 4*ph);
    r += (189.0 / 8192.0) * cos(5*a - 4*b - 4*th - ph);
    r += SQRT3_GAL * (63.0 / 8192.0) * sin(5*a - 4*b - 4*th - ph);
    r += (27.0 / 4096.0) * cos(5*a - 4*b - 2*th - ph);
    r += SQRT3_GAL * (9.0 / 4096.0) * sin(5*a - 4*b - 2*th - ph);
    r += (-27.0 / 4096.0) * cos(5*a - 4*b + 2*th - ph);
    r += SQRT3_GAL * (-9.0 / 4096.0) * sin(5*a - 4*b + 2*th - ph);
    r += (-189.0 / 8192.0) * cos(5*a - 4*b + 4*th - ph);
    r += SQRT3_GAL * (-63.0 / 8192.0) * sin(5*a - 4*b + 4*th - ph);
    r += (-63.0 / 16384.0) * cos(6*a - 4*b - 4*th - 2*ph);
    r += SQRT3_GAL * (63.0 / 16384.0) * sin(6*a - 4*b - 4*th - 2*ph);
    r += (9.0 / 4096.0) * cos(6*a - 4*b - 2*th - 2*ph);
    r += SQRT3_GAL * (-9.0 / 4096.0) * sin(6*a - 4*b - 2*th - 2*ph);
    r += (27.0 / 8192.0) * cos(6*a - 4*b - 2*ph);
    r += SQRT3_GAL * (-27.0 / 8192.0) * sin(6*a - 4*b - 2*ph);
    r += (9.0 / 4096.0) * cos(6*a - 4*b + 2*th - 2*ph);
    r += SQRT3_GAL * (-9.0 / 4096.0) * sin(6*a - 4*b + 2*th - 2*ph);
    r += (-63.0 / 16384.0) * cos(6*a - 4*b + 4*th - 2*ph);
    r += SQRT3_GAL * (63.0 / 16384.0) * sin(6*a - 4*b + 4*th - 2*ph);
    r += (-9.0 / 8192.0) * cos(7*a - 4*b - 4*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 8192.0) * sin(7*a - 4*b - 4*th - 3*ph);
    r += (9.0 / 4096.0) * cos(7*a - 4*b - 2*th - 3*ph);
    r += SQRT3_GAL * (3.0 / 4096.0) * sin(7*a - 4*b - 2*th - 3*ph);
    r += (-9.0 / 4096.0) * cos(7*a - 4*b + 2*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 4096.0) * sin(7*a - 4*b + 2*th - 3*ph);
    r += (9.0 / 8192.0) * cos(7*a - 4*b + 4*th - 3*ph);
    r += SQRT3_GAL * (3.0 / 8192.0) * sin(7*a - 4*b + 4*th - 3*ph);
    r += (3.0 / 65536.0) * cos(8*a - 4*b - 4*th - 4*ph);
    r += SQRT3_GAL * (-3.0 / 65536.0) * sin(8*a - 4*b - 4*th - 4*ph);
    r += (-3.0 / 16384.0) * cos(8*a - 4*b - 2*th - 4*ph);
    r += SQRT3_GAL * (3.0 / 16384.0) * sin(8*a - 4*b - 2*th - 4*ph);
    r += (9.0 / 32768.0) * cos(8*a - 4*b - 4*ph);
    r += SQRT3_GAL * (-9.0 / 32768.0) * sin(8*a - 4*b - 4*ph);
    r += (-3.0 / 16384.0) * cos(8*a - 4*b + 2*th - 4*ph);
    r += SQRT3_GAL * (3.0 / 16384.0) * sin(8*a - 4*b + 2*th - 4*ph);
    r += (3.0 / 65536.0) * cos(8*a - 4*b + 4*th - 4*ph);
    r += SQRT3_GAL * (-3.0 / 65536.0) * sin(8*a - 4*b + 4*th - 4*ph);
    return GAL_RESPONSE_K * r;
}

CUDA_DEVICE
double galaxy_YY(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double a = alpha0 + OMEGA_LISA * delta_t;
    double b = beta0, th = theta_N, ph = phi_N;
    double r = 0.0;
    r += (9363.0 / 8192.0);
    r += (-345.0 / 2048.0) * cos(2*th);
    r += (-111.0 / 8192.0) * cos(4*th);
    r += (243.0 / 65536.0) * cos(4*b - 4*th - 4*ph);
    r += SQRT3_GAL * (243.0 / 65536.0) * sin(4*b - 4*th - 4*ph);
    r += (-243.0 / 16384.0) * cos(4*b - 2*th - 4*ph);
    r += SQRT3_GAL * (-243.0 / 16384.0) * sin(4*b - 2*th - 4*ph);
    r += (729.0 / 32768.0) * cos(4*b - 4*ph);
    r += SQRT3_GAL * (729.0 / 32768.0) * sin(4*b - 4*ph);
    r += (-243.0 / 16384.0) * cos(4*b + 2*th - 4*ph);
    r += SQRT3_GAL * (-243.0 / 16384.0) * sin(4*b + 2*th - 4*ph);
    r += (243.0 / 65536.0) * cos(4*b + 4*th - 4*ph);
    r += SQRT3_GAL * (243.0 / 65536.0) * sin(4*b + 4*th - 4*ph);
    r += (243.0 / 8192.0) * cos(a - 4*b - 4*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 8192.0) * sin(a - 4*b - 4*th + 3*ph);
    r += (-243.0 / 4096.0) * cos(a - 4*b - 2*th + 3*ph);
    r += SQRT3_GAL * (-81.0 / 4096.0) * sin(a - 4*b - 2*th + 3*ph);
    r += (243.0 / 4096.0) * cos(a - 4*b + 2*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 4096.0) * sin(a - 4*b + 2*th + 3*ph);
    r += (-243.0 / 8192.0) * cos(a - 4*b + 4*th + 3*ph);
    r += SQRT3_GAL * (-81.0 / 8192.0) * sin(a - 4*b + 4*th + 3*ph);
    r += SQRT3_GAL * (-15.0 / 2048.0) * sin(a - 4*th - ph);
    r += SQRT3_GAL * (327.0 / 1024.0) * sin(a - 2*th - ph);
    r += SQRT3_GAL * (-327.0 / 1024.0) * sin(a + 2*th - ph);
    r += SQRT3_GAL * (15.0 / 2048.0) * sin(a + 4*th - ph);
    r += (-567.0 / 16384.0) * cos(2*a - 4*b - 4*th + 2*ph);
    r += SQRT3_GAL * (567.0 / 16384.0) * sin(2*a - 4*b - 4*th + 2*ph);
    r += (81.0 / 4096.0) * cos(2*a - 4*b - 2*th + 2*ph);
    r += SQRT3_GAL * (-81.0 / 4096.0) * sin(2*a - 4*b - 2*th + 2*ph);
    r += (243.0 / 8192.0) * cos(2*a - 4*b + 2*ph);
    r += SQRT3_GAL * (-243.0 / 8192.0) * sin(2*a - 4*b + 2*ph);
    r += (81.0 / 4096.0) * cos(2*a - 4*b + 2*th + 2*ph);
    r += SQRT3_GAL * (-81.0 / 4096.0) * sin(2*a - 4*b + 2*th + 2*ph);
    r += (-567.0 / 16384.0) * cos(2*a - 4*b + 4*th + 2*ph);
    r += SQRT3_GAL * (567.0 / 16384.0) * sin(2*a - 4*b + 4*th + 2*ph);
    r += (-27.0 / 4096.0) * cos(2*a - 4*th - 2*ph);
    r += (-243.0 / 1024.0) * cos(2*a - 2*th - 2*ph);
    r += (999.0 / 2048.0) * cos(2*a - 2*ph);
    r += (-243.0 / 1024.0) * cos(2*a + 2*th - 2*ph);
    r += (-27.0 / 4096.0) * cos(2*a + 4*th - 2*ph);
    r += (-567.0 / 8192.0) * cos(3*a - 4*b - 4*th + ph);
    r += SQRT3_GAL * (-189.0 / 8192.0) * sin(3*a - 4*b - 4*th + ph);
    r += (-81.0 / 4096.0) * cos(3*a - 4*b - 2*th + ph);
    r += SQRT3_GAL * (-27.0 / 4096.0) * sin(3*a - 4*b - 2*th + ph);
    r += (81.0 / 4096.0) * cos(3*a - 4*b + 2*th + ph);
    r += SQRT3_GAL * (27.0 / 4096.0) * sin(3*a - 4*b + 2*th + ph);
    r += (567.0 / 8192.0) * cos(3*a - 4*b + 4*th + ph);
    r += SQRT3_GAL * (189.0 / 8192.0) * sin(3*a - 4*b + 4*th + ph);
    r += SQRT3_GAL * (-9.0 / 2048.0) * sin(3*a - 4*th - 3*ph);
    r += SQRT3_GAL * (9.0 / 1024.0) * sin(3*a - 2*th - 3*ph);
    r += SQRT3_GAL * (-9.0 / 1024.0) * sin(3*a + 2*th - 3*ph);
    r += SQRT3_GAL * (9.0 / 2048.0) * sin(3*a + 4*th - 3*ph);
    r += (945.0 / 32768.0) * cos(4*a - 4*b - 4*th);
    r += SQRT3_GAL * (-945.0 / 32768.0) * sin(4*a - 4*b - 4*th);
    r += (135.0 / 8192.0) * cos(4*a - 4*b - 2*th);
    r += SQRT3_GAL * (-135.0 / 8192.0) * sin(4*a - 4*b - 2*th);
    r += (243.0 / 16384.0) * cos(4*a - 4*b);
    r += SQRT3_GAL * (-243.0 / 16384.0) * sin(4*a - 4*b);
    r += (135.0 / 8192.0) * cos(4*a - 4*b + 2*th);
    r += SQRT3_GAL * (-135.0 / 8192.0) * sin(4*a - 4*b + 2*th);
    r += (945.0 / 32768.0) * cos(4*a - 4*b + 4*th);
    r += SQRT3_GAL * (-945.0 / 32768.0) * sin(4*a - 4*b + 4*th);
    r += (27.0 / 16384.0) * cos(4*a - 4*th - 4*ph);
    r += (-27.0 / 4096.0) * cos(4*a - 2*th - 4*ph);
    r += (81.0 / 8192.0) * cos(4*a - 4*ph);
    r += (-27.0 / 4096.0) * cos(4*a + 2*th - 4*ph);
    r += (27.0 / 16384.0) * cos(4*a + 4*th - 4*ph);
    r += (189.0 / 8192.0) * cos(5*a - 4*b - 4*th - ph);
    r += SQRT3_GAL * (63.0 / 8192.0) * sin(5*a - 4*b - 4*th - ph);
    r += (27.0 / 4096.0) * cos(5*a - 4*b - 2*th - ph);
    r += SQRT3_GAL * (9.0 / 4096.0) * sin(5*a - 4*b - 2*th - ph);
    r += (-27.0 / 4096.0) * cos(5*a - 4*b + 2*th - ph);
    r += SQRT3_GAL * (-9.0 / 4096.0) * sin(5*a - 4*b + 2*th - ph);
    r += (-189.0 / 8192.0) * cos(5*a - 4*b + 4*th - ph);
    r += SQRT3_GAL * (-63.0 / 8192.0) * sin(5*a - 4*b + 4*th - ph);
    r += (-63.0 / 16384.0) * cos(6*a - 4*b - 4*th - 2*ph);
    r += SQRT3_GAL * (63.0 / 16384.0) * sin(6*a - 4*b - 4*th - 2*ph);
    r += (9.0 / 4096.0) * cos(6*a - 4*b - 2*th - 2*ph);
    r += SQRT3_GAL * (-9.0 / 4096.0) * sin(6*a - 4*b - 2*th - 2*ph);
    r += (27.0 / 8192.0) * cos(6*a - 4*b - 2*ph);
    r += SQRT3_GAL * (-27.0 / 8192.0) * sin(6*a - 4*b - 2*ph);
    r += (9.0 / 4096.0) * cos(6*a - 4*b + 2*th - 2*ph);
    r += SQRT3_GAL * (-9.0 / 4096.0) * sin(6*a - 4*b + 2*th - 2*ph);
    r += (-63.0 / 16384.0) * cos(6*a - 4*b + 4*th - 2*ph);
    r += SQRT3_GAL * (63.0 / 16384.0) * sin(6*a - 4*b + 4*th - 2*ph);
    r += (-9.0 / 8192.0) * cos(7*a - 4*b - 4*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 8192.0) * sin(7*a - 4*b - 4*th - 3*ph);
    r += (9.0 / 4096.0) * cos(7*a - 4*b - 2*th - 3*ph);
    r += SQRT3_GAL * (3.0 / 4096.0) * sin(7*a - 4*b - 2*th - 3*ph);
    r += (-9.0 / 4096.0) * cos(7*a - 4*b + 2*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 4096.0) * sin(7*a - 4*b + 2*th - 3*ph);
    r += (9.0 / 8192.0) * cos(7*a - 4*b + 4*th - 3*ph);
    r += SQRT3_GAL * (3.0 / 8192.0) * sin(7*a - 4*b + 4*th - 3*ph);
    r += (3.0 / 65536.0) * cos(8*a - 4*b - 4*th - 4*ph);
    r += SQRT3_GAL * (-3.0 / 65536.0) * sin(8*a - 4*b - 4*th - 4*ph);
    r += (-3.0 / 16384.0) * cos(8*a - 4*b - 2*th - 4*ph);
    r += SQRT3_GAL * (3.0 / 16384.0) * sin(8*a - 4*b - 2*th - 4*ph);
    r += (9.0 / 32768.0) * cos(8*a - 4*b - 4*ph);
    r += SQRT3_GAL * (-9.0 / 32768.0) * sin(8*a - 4*b - 4*ph);
    r += (-3.0 / 16384.0) * cos(8*a - 4*b + 2*th - 4*ph);
    r += SQRT3_GAL * (3.0 / 16384.0) * sin(8*a - 4*b + 2*th - 4*ph);
    r += (3.0 / 65536.0) * cos(8*a - 4*b + 4*th - 4*ph);
    r += SQRT3_GAL * (-3.0 / 65536.0) * sin(8*a - 4*b + 4*th - 4*ph);
    return GAL_RESPONSE_K * r;
}

CUDA_DEVICE
double galaxy_YZ(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double a = alpha0 + OMEGA_LISA * delta_t;
    double b = beta0, th = theta_N, ph = phi_N;
    double r = 0.0;
    r += (-9363.0 / 16384.0);
    r += (345.0 / 4096.0) * cos(2*th);
    r += (111.0 / 16384.0) * cos(4*th);
    r += (-243.0 / 32768.0) * cos(4*b - 4*th - 4*ph);
    r += (243.0 / 8192.0) * cos(4*b - 2*th - 4*ph);
    r += (-729.0 / 16384.0) * cos(4*b - 4*ph);
    r += (243.0 / 8192.0) * cos(4*b + 2*th - 4*ph);
    r += (-243.0 / 32768.0) * cos(4*b + 4*th - 4*ph);
    r += SQRT3_GAL * (-81.0 / 4096.0) * sin(a - 4*b - 4*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 2048.0) * sin(a - 4*b - 2*th + 3*ph);
    r += SQRT3_GAL * (-81.0 / 2048.0) * sin(a - 4*b + 2*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 4096.0) * sin(a - 4*b + 4*th + 3*ph);
    r += SQRT3_GAL * (15.0 / 4096.0) * sin(a - 4*th - ph);
    r += SQRT3_GAL * (-327.0 / 2048.0) * sin(a - 2*th - ph);
    r += SQRT3_GAL * (327.0 / 2048.0) * sin(a + 2*th - ph);
    r += SQRT3_GAL * (-15.0 / 4096.0) * sin(a + 4*th - ph);
    r += (567.0 / 8192.0) * cos(2*a - 4*b - 4*th + 2*ph);
    r += (-81.0 / 2048.0) * cos(2*a - 4*b - 2*th + 2*ph);
    r += (-243.0 / 4096.0) * cos(2*a - 4*b + 2*ph);
    r += (-81.0 / 2048.0) * cos(2*a - 4*b + 2*th + 2*ph);
    r += (567.0 / 8192.0) * cos(2*a - 4*b + 4*th + 2*ph);
    r += (27.0 / 8192.0) * cos(2*a - 4*th - 2*ph);
    r += (243.0 / 2048.0) * cos(2*a - 2*th - 2*ph);
    r += (-999.0 / 4096.0) * cos(2*a - 2*ph);
    r += (243.0 / 2048.0) * cos(2*a + 2*th - 2*ph);
    r += (27.0 / 8192.0) * cos(2*a + 4*th - 2*ph);
    r += SQRT3_GAL * (189.0 / 4096.0) * sin(3*a - 4*b - 4*th + ph);
    r += SQRT3_GAL * (27.0 / 2048.0) * sin(3*a - 4*b - 2*th + ph);
    r += SQRT3_GAL * (-27.0 / 2048.0) * sin(3*a - 4*b + 2*th + ph);
    r += SQRT3_GAL * (-189.0 / 4096.0) * sin(3*a - 4*b + 4*th + ph);
    r += SQRT3_GAL * (9.0 / 4096.0) * sin(3*a - 4*th - 3*ph);
    r += SQRT3_GAL * (-9.0 / 2048.0) * sin(3*a - 2*th - 3*ph);
    r += SQRT3_GAL * (9.0 / 2048.0) * sin(3*a + 2*th - 3*ph);
    r += SQRT3_GAL * (-9.0 / 4096.0) * sin(3*a + 4*th - 3*ph);
    r += (-945.0 / 16384.0) * cos(4*a - 4*b - 4*th);
    r += (-135.0 / 4096.0) * cos(4*a - 4*b - 2*th);
    r += (-243.0 / 8192.0) * cos(4*a - 4*b);
    r += (-135.0 / 4096.0) * cos(4*a - 4*b + 2*th);
    r += (-945.0 / 16384.0) * cos(4*a - 4*b + 4*th);
    r += (-27.0 / 32768.0) * cos(4*a - 4*th - 4*ph);
    r += (27.0 / 8192.0) * cos(4*a - 2*th - 4*ph);
    r += (-81.0 / 16384.0) * cos(4*a - 4*ph);
    r += (27.0 / 8192.0) * cos(4*a + 2*th - 4*ph);
    r += (-27.0 / 32768.0) * cos(4*a + 4*th - 4*ph);
    r += SQRT3_GAL * (-63.0 / 4096.0) * sin(5*a - 4*b - 4*th - ph);
    r += SQRT3_GAL * (-9.0 / 2048.0) * sin(5*a - 4*b - 2*th - ph);
    r += SQRT3_GAL * (9.0 / 2048.0) * sin(5*a - 4*b + 2*th - ph);
    r += SQRT3_GAL * (63.0 / 4096.0) * sin(5*a - 4*b + 4*th - ph);
    r += (63.0 / 8192.0) * cos(6*a - 4*b - 4*th - 2*ph);
    r += (-9.0 / 2048.0) * cos(6*a - 4*b - 2*th - 2*ph);
    r += (-27.0 / 4096.0) * cos(6*a - 4*b - 2*ph);
    r += (-9.0 / 2048.0) * cos(6*a - 4*b + 2*th - 2*ph);
    r += (63.0 / 8192.0) * cos(6*a - 4*b + 4*th - 2*ph);
    r += SQRT3_GAL * (3.0 / 4096.0) * sin(7*a - 4*b - 4*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 2048.0) * sin(7*a - 4*b - 2*th - 3*ph);
    r += SQRT3_GAL * (3.0 / 2048.0) * sin(7*a - 4*b + 2*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 4096.0) * sin(7*a - 4*b + 4*th - 3*ph);
    r += (-3.0 / 32768.0) * cos(8*a - 4*b - 4*th - 4*ph);
    r += (3.0 / 8192.0) * cos(8*a - 4*b - 2*th - 4*ph);
    r += (-9.0 / 16384.0) * cos(8*a - 4*b - 4*ph);
    r += (3.0 / 8192.0) * cos(8*a - 4*b + 2*th - 4*ph);
    r += (-3.0 / 32768.0) * cos(8*a - 4*b + 4*th - 4*ph);
    return GAL_RESPONSE_K * r;
}

CUDA_DEVICE
double galaxy_ZZ(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double a = alpha0 + OMEGA_LISA * delta_t;
    double b = beta0, th = theta_N, ph = phi_N;
    double r = 0.0;
    r += (9363.0 / 8192.0);
    r += (-345.0 / 2048.0) * cos(2*th);
    r += (-111.0 / 8192.0) * cos(4*th);
    r += (243.0 / 65536.0) * cos(4*b - 4*th - 4*ph);
    r += SQRT3_GAL * (-243.0 / 65536.0) * sin(4*b - 4*th - 4*ph);
    r += (-243.0 / 16384.0) * cos(4*b - 2*th - 4*ph);
    r += SQRT3_GAL * (243.0 / 16384.0) * sin(4*b - 2*th - 4*ph);
    r += (729.0 / 32768.0) * cos(4*b - 4*ph);
    r += SQRT3_GAL * (-729.0 / 32768.0) * sin(4*b - 4*ph);
    r += (-243.0 / 16384.0) * cos(4*b + 2*th - 4*ph);
    r += SQRT3_GAL * (243.0 / 16384.0) * sin(4*b + 2*th - 4*ph);
    r += (243.0 / 65536.0) * cos(4*b + 4*th - 4*ph);
    r += SQRT3_GAL * (-243.0 / 65536.0) * sin(4*b + 4*th - 4*ph);
    r += (-243.0 / 8192.0) * cos(a - 4*b - 4*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 8192.0) * sin(a - 4*b - 4*th + 3*ph);
    r += (243.0 / 4096.0) * cos(a - 4*b - 2*th + 3*ph);
    r += SQRT3_GAL * (-81.0 / 4096.0) * sin(a - 4*b - 2*th + 3*ph);
    r += (-243.0 / 4096.0) * cos(a - 4*b + 2*th + 3*ph);
    r += SQRT3_GAL * (81.0 / 4096.0) * sin(a - 4*b + 2*th + 3*ph);
    r += (243.0 / 8192.0) * cos(a - 4*b + 4*th + 3*ph);
    r += SQRT3_GAL * (-81.0 / 8192.0) * sin(a - 4*b + 4*th + 3*ph);
    r += SQRT3_GAL * (-15.0 / 2048.0) * sin(a - 4*th - ph);
    r += SQRT3_GAL * (327.0 / 1024.0) * sin(a - 2*th - ph);
    r += SQRT3_GAL * (-327.0 / 1024.0) * sin(a + 2*th - ph);
    r += SQRT3_GAL * (15.0 / 2048.0) * sin(a + 4*th - ph);
    r += (-567.0 / 16384.0) * cos(2*a - 4*b - 4*th + 2*ph);
    r += SQRT3_GAL * (-567.0 / 16384.0) * sin(2*a - 4*b - 4*th + 2*ph);
    r += (81.0 / 4096.0) * cos(2*a - 4*b - 2*th + 2*ph);
    r += SQRT3_GAL * (81.0 / 4096.0) * sin(2*a - 4*b - 2*th + 2*ph);
    r += (243.0 / 8192.0) * cos(2*a - 4*b + 2*ph);
    r += SQRT3_GAL * (243.0 / 8192.0) * sin(2*a - 4*b + 2*ph);
    r += (81.0 / 4096.0) * cos(2*a - 4*b + 2*th + 2*ph);
    r += SQRT3_GAL * (81.0 / 4096.0) * sin(2*a - 4*b + 2*th + 2*ph);
    r += (-567.0 / 16384.0) * cos(2*a - 4*b + 4*th + 2*ph);
    r += SQRT3_GAL * (-567.0 / 16384.0) * sin(2*a - 4*b + 4*th + 2*ph);
    r += (-27.0 / 4096.0) * cos(2*a - 4*th - 2*ph);
    r += (-243.0 / 1024.0) * cos(2*a - 2*th - 2*ph);
    r += (999.0 / 2048.0) * cos(2*a - 2*ph);
    r += (-243.0 / 1024.0) * cos(2*a + 2*th - 2*ph);
    r += (-27.0 / 4096.0) * cos(2*a + 4*th - 2*ph);
    r += (567.0 / 8192.0) * cos(3*a - 4*b - 4*th + ph);
    r += SQRT3_GAL * (-189.0 / 8192.0) * sin(3*a - 4*b - 4*th + ph);
    r += (81.0 / 4096.0) * cos(3*a - 4*b - 2*th + ph);
    r += SQRT3_GAL * (-27.0 / 4096.0) * sin(3*a - 4*b - 2*th + ph);
    r += (-81.0 / 4096.0) * cos(3*a - 4*b + 2*th + ph);
    r += SQRT3_GAL * (27.0 / 4096.0) * sin(3*a - 4*b + 2*th + ph);
    r += (-567.0 / 8192.0) * cos(3*a - 4*b + 4*th + ph);
    r += SQRT3_GAL * (189.0 / 8192.0) * sin(3*a - 4*b + 4*th + ph);
    r += SQRT3_GAL * (-9.0 / 2048.0) * sin(3*a - 4*th - 3*ph);
    r += SQRT3_GAL * (9.0 / 1024.0) * sin(3*a - 2*th - 3*ph);
    r += SQRT3_GAL * (-9.0 / 1024.0) * sin(3*a + 2*th - 3*ph);
    r += SQRT3_GAL * (9.0 / 2048.0) * sin(3*a + 4*th - 3*ph);
    r += (945.0 / 32768.0) * cos(4*a - 4*b - 4*th);
    r += SQRT3_GAL * (945.0 / 32768.0) * sin(4*a - 4*b - 4*th);
    r += (135.0 / 8192.0) * cos(4*a - 4*b - 2*th);
    r += SQRT3_GAL * (135.0 / 8192.0) * sin(4*a - 4*b - 2*th);
    r += (243.0 / 16384.0) * cos(4*a - 4*b);
    r += SQRT3_GAL * (243.0 / 16384.0) * sin(4*a - 4*b);
    r += (135.0 / 8192.0) * cos(4*a - 4*b + 2*th);
    r += SQRT3_GAL * (135.0 / 8192.0) * sin(4*a - 4*b + 2*th);
    r += (945.0 / 32768.0) * cos(4*a - 4*b + 4*th);
    r += SQRT3_GAL * (945.0 / 32768.0) * sin(4*a - 4*b + 4*th);
    r += (27.0 / 16384.0) * cos(4*a - 4*th - 4*ph);
    r += (-27.0 / 4096.0) * cos(4*a - 2*th - 4*ph);
    r += (81.0 / 8192.0) * cos(4*a - 4*ph);
    r += (-27.0 / 4096.0) * cos(4*a + 2*th - 4*ph);
    r += (27.0 / 16384.0) * cos(4*a + 4*th - 4*ph);
    r += (-189.0 / 8192.0) * cos(5*a - 4*b - 4*th - ph);
    r += SQRT3_GAL * (63.0 / 8192.0) * sin(5*a - 4*b - 4*th - ph);
    r += (-27.0 / 4096.0) * cos(5*a - 4*b - 2*th - ph);
    r += SQRT3_GAL * (9.0 / 4096.0) * sin(5*a - 4*b - 2*th - ph);
    r += (27.0 / 4096.0) * cos(5*a - 4*b + 2*th - ph);
    r += SQRT3_GAL * (-9.0 / 4096.0) * sin(5*a - 4*b + 2*th - ph);
    r += (189.0 / 8192.0) * cos(5*a - 4*b + 4*th - ph);
    r += SQRT3_GAL * (-63.0 / 8192.0) * sin(5*a - 4*b + 4*th - ph);
    r += (-63.0 / 16384.0) * cos(6*a - 4*b - 4*th - 2*ph);
    r += SQRT3_GAL * (-63.0 / 16384.0) * sin(6*a - 4*b - 4*th - 2*ph);
    r += (9.0 / 4096.0) * cos(6*a - 4*b - 2*th - 2*ph);
    r += SQRT3_GAL * (9.0 / 4096.0) * sin(6*a - 4*b - 2*th - 2*ph);
    r += (27.0 / 8192.0) * cos(6*a - 4*b - 2*ph);
    r += SQRT3_GAL * (27.0 / 8192.0) * sin(6*a - 4*b - 2*ph);
    r += (9.0 / 4096.0) * cos(6*a - 4*b + 2*th - 2*ph);
    r += SQRT3_GAL * (9.0 / 4096.0) * sin(6*a - 4*b + 2*th - 2*ph);
    r += (-63.0 / 16384.0) * cos(6*a - 4*b + 4*th - 2*ph);
    r += SQRT3_GAL * (-63.0 / 16384.0) * sin(6*a - 4*b + 4*th - 2*ph);
    r += (9.0 / 8192.0) * cos(7*a - 4*b - 4*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 8192.0) * sin(7*a - 4*b - 4*th - 3*ph);
    r += (-9.0 / 4096.0) * cos(7*a - 4*b - 2*th - 3*ph);
    r += SQRT3_GAL * (3.0 / 4096.0) * sin(7*a - 4*b - 2*th - 3*ph);
    r += (9.0 / 4096.0) * cos(7*a - 4*b + 2*th - 3*ph);
    r += SQRT3_GAL * (-3.0 / 4096.0) * sin(7*a - 4*b + 2*th - 3*ph);
    r += (-9.0 / 8192.0) * cos(7*a - 4*b + 4*th - 3*ph);
    r += SQRT3_GAL * (3.0 / 8192.0) * sin(7*a - 4*b + 4*th - 3*ph);
    r += (3.0 / 65536.0) * cos(8*a - 4*b - 4*th - 4*ph);
    r += SQRT3_GAL * (3.0 / 65536.0) * sin(8*a - 4*b - 4*th - 4*ph);
    r += (-3.0 / 16384.0) * cos(8*a - 4*b - 2*th - 4*ph);
    r += SQRT3_GAL * (-3.0 / 16384.0) * sin(8*a - 4*b - 2*th - 4*ph);
    r += (9.0 / 32768.0) * cos(8*a - 4*b - 4*ph);
    r += SQRT3_GAL * (9.0 / 32768.0) * sin(8*a - 4*b - 4*ph);
    r += (-3.0 / 16384.0) * cos(8*a - 4*b + 2*th - 4*ph);
    r += SQRT3_GAL * (-3.0 / 16384.0) * sin(8*a - 4*b + 2*th - 4*ph);
    r += (3.0 / 65536.0) * cos(8*a - 4*b + 4*th - 4*ph);
    r += SQRT3_GAL * (3.0 / 65536.0) * sin(8*a - 4*b + 4*th - 4*ph);
    return GAL_RESPONSE_K * r;
}

// ============================================================================
// Kernel 1: compute normalized sky weights
// ============================================================================

CUDA_KERNEL
void compute_column_densities_kernel(
    double *col_density_out,
    const double *R_vals_quad,
    const double *z_vals_quad,
    const double *quad_weights,
    const double *cos_beta_ecl,
    double R_d, double z_d,
    double f_b, double r_b,
    int N_quad, int N_sky)
{
#ifdef __CUDACC__
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
#else
    int i = 0, stride = 1;
#endif
    for (; i < N_sky; i += stride)
    {
        double col = 0.0;
        for (int q = 0; q < N_quad; q++)
            col += quad_weights[q] * galaxy_density(
                       R_vals_quad[q * N_sky + i],
                       z_vals_quad[q * N_sky + i],
                       R_d, z_d, f_b, r_b);
        col_density_out[i] = col * cos_beta_ecl[i];
    }
}

CUDA_KERNEL
void normalize_weights_kernel(double *weights, double inv_total, int N_sky)
{
#ifdef __CUDACC__
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
#else
    int i = 0, stride = 1;
#endif
    for (; i < N_sky; i += stride)
        weights[i] *= inv_total;
}

// ============================================================================
// Detector-tensor response from the orbit arms (low-frequency limit)
//
// For a source in direction n (ecliptic lam, beta) and orthonormal p, q ⊥ n:
//   A+(u) = (u.p)^2 - (u.q)^2,  Ax(u) = 2 (u.p)(u.q)
//   F_X = A(u12) - A(u13),  F_Y = A(u23) - A(u12),  F_Z = A(u13) - A(u23)
//   R_IJ = K * (F+_I F+_J + Fx_I Fx_J)   (polarization/inclination average, same K as the closed forms)
// Output order k: XX, XY, XZ, YY, YZ, ZZ.
// ============================================================================

static CUDA_DEVICE
void orbit_response(double lam, double beta, const double *arms_t, double *out)
{
    double cb = cos(beta);
    double n0 = cb * cos(lam), n1 = cb * sin(lam), n2 = sin(beta);
    // p = n x z / |n x z| (east), q = n x p (north); fixed basis at the poles
    double p0, p1, p2;
    if (cb > 1e-12) { p0 = n1 / cb; p1 = -n0 / cb; p2 = 0.0; }
    else            { p0 = 1.0;     p1 = 0.0;      p2 = 0.0; }
    double q0 = n1 * p2 - n2 * p1;
    double q1 = n2 * p0 - n0 * p2;
    double q2 = n0 * p1 - n1 * p0;

    double Ap[3], Ax[3];  // per arm: u12, u13, u23
    for (int a = 0; a < 3; a++) {
        const double *u = arms_t + 3 * a;
        double up = u[0] * p0 + u[1] * p1 + u[2] * p2;
        double uq = u[0] * q0 + u[1] * q1 + u[2] * q2;
        Ap[a] = up * up - uq * uq;
        Ax[a] = 2.0 * up * uq;
    }
    double FpX = Ap[0] - Ap[1], FxX = Ax[0] - Ax[1];
    double FpY = Ap[2] - Ap[0], FxY = Ax[2] - Ax[0];
    double FpZ = Ap[1] - Ap[2], FxZ = Ax[1] - Ax[2];

    out[0] = GAL_RESPONSE_K * (FpX * FpX + FxX * FxX);
    out[1] = GAL_RESPONSE_K * (FpX * FpY + FxX * FxY);
    out[2] = GAL_RESPONSE_K * (FpX * FpZ + FxX * FxZ);
    out[3] = GAL_RESPONSE_K * (FpY * FpY + FxY * FxY);
    out[4] = GAL_RESPONSE_K * (FpY * FpZ + FxY * FxZ);
    out[5] = GAL_RESPONSE_K * (FpZ * FpZ + FxZ * FxZ);
}

CUDA_KERNEL
void sky_average_orbit_response_kernel(
    double *R_avg_out,
    const double *weights,
    const double *lam_ecl,
    const double *beta_ecl,
    const double *arms,
    int N_times, int N_sky)
{
#ifdef __CUDACC__
    int t_idx = blockIdx.y;
    if (t_idx >= N_times) return;
    const double *arms_t = arms + t_idx * 9;

    __shared__ double s[6][GAL_THREADS_SKY];
    int tid = threadIdx.x;

    double acc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double r[6];
    for (int i = blockIdx.x * blockDim.x + tid; i < N_sky; i += gridDim.x * blockDim.x)
    {
        orbit_response(lam_ecl[i], beta_ecl[i], arms_t, r);
        double w = weights[i];
        for (int k = 0; k < 6; k++) acc[k] += w * r[k];
    }

    for (int k = 0; k < 6; k++) s[k][tid] = acc[k];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            for (int k = 0; k < 6; k++)
                s[k][tid] += s[k][tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        for (int k = 0; k < 6; k++)
            atomicAdd(&R_avg_out[t_idx * 6 + k], s[k][0]);
#else
    double r[6];
    for (int t_idx = 0; t_idx < N_times; t_idx++)
    {
        const double *arms_t = arms + t_idx * 9;
        double acc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < N_sky; i++)
        {
            orbit_response(lam_ecl[i], beta_ecl[i], arms_t, r);
            for (int k = 0; k < 6; k++) acc[k] += weights[i] * r[k];
        }
        for (int k = 0; k < 6; k++)
            R_avg_out[t_idx * 6 + k] = acc[k];
    }
#endif
}

// ============================================================================
// Kernel 2: sky-averaged response R_avg[t, k]
// Output layout: R_avg[t_idx * 6 + k], k in {XX,XY,XZ,YY,YZ,ZZ}
// Uses atomicAdd across sky-pixel blocks for a given time.
// ============================================================================

CUDA_KERNEL
void sky_average_response_kernel(
    double *R_avg_out,
    const double *weights,
    const double *lam_ecl,
    const double *beta_ecl,
    const double *times,
    double alpha0, double beta0, double t0,
    int N_times, int N_sky)
{
#ifdef __CUDACC__
    int t_idx = blockIdx.y;
    if (t_idx >= N_times) return;

    double delta_t = times[t_idx] - t0;

    __shared__ double s[6][GAL_THREADS_SKY];
    int tid = threadIdx.x;

    double acc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (int i = blockIdx.x * blockDim.x + tid; i < N_sky; i += gridDim.x * blockDim.x)
    {
        double lam  = lam_ecl[i];
        double th   = 0.5 * M_PI - beta_ecl[i];   // polar angle from the ecliptic pole
        double w    = weights[i];

        acc[0] += w * galaxy_XX(alpha0, beta0, th, lam, delta_t);
        acc[1] += w * galaxy_XY(alpha0, beta0, th, lam, delta_t);
        acc[2] += w * galaxy_XZ(alpha0, beta0, th, lam, delta_t);
        acc[3] += w * galaxy_YY(alpha0, beta0, th, lam, delta_t);
        acc[4] += w * galaxy_YZ(alpha0, beta0, th, lam, delta_t);
        acc[5] += w * galaxy_ZZ(alpha0, beta0, th, lam, delta_t);
    }

    for (int k = 0; k < 6; k++) s[k][tid] = acc[k];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            for (int k = 0; k < 6; k++)
                s[k][tid] += s[k][tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        for (int k = 0; k < 6; k++)
            atomicAdd(&R_avg_out[t_idx * 6 + k], s[k][0]);

#else
    for (int t_idx = 0; t_idx < N_times; t_idx++)
    {
        double delta_t   = times[t_idx] - t0;
        double acc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < N_sky; i++)
        {
            double lam  = lam_ecl[i];
            double th   = 0.5 * M_PI - beta_ecl[i];   // polar angle from the ecliptic pole
            double w    = weights[i];
            acc[0] += w * galaxy_XX(alpha0, beta0, th, lam, delta_t);
            acc[1] += w * galaxy_XY(alpha0, beta0, th, lam, delta_t);
            acc[2] += w * galaxy_XZ(alpha0, beta0, th, lam, delta_t);
            acc[3] += w * galaxy_YY(alpha0, beta0, th, lam, delta_t);
            acc[4] += w * galaxy_YZ(alpha0, beta0, th, lam, delta_t);
            acc[5] += w * galaxy_ZZ(alpha0, beta0, th, lam, delta_t);
        }
        for (int k = 0; k < 6; k++)
            R_avg_out[t_idx * 6 + k] = acc[k];
    }
#endif
}

// ============================================================================
// Kernel 3: spectral broadcast
// R_gal_arr[(t * N_freqs + f) * 6 + k] = R_avg[t * 6 + k] * S_gal(f; params)
// Used only for diagnostic / get_noise_covariance_arr — NOT on hot path.
// ============================================================================

CUDA_KERNEL
void apply_spectral_model_kernel(
    double *R_gal_arr,
    const double *R_avg,
    const double *freqs,
    double Amp, double alpha, double f_1, double f_knee, double f_2,
    double avg_d,
    int N_times, int N_freqs)
{
#ifdef __CUDACC__
    int f_idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int t_idx    = blockIdx.y * blockDim.y + threadIdx.y;
    int f_stride = gridDim.x * blockDim.x;
    int t_stride = gridDim.y * blockDim.y;
#else
    int f_idx = 0, t_idx = 0, f_stride = 1, t_stride = 1;
#endif

    for (int t = t_idx; t < N_times; t += t_stride)
    for (int f = f_idx; f < N_freqs; f += f_stride)
    {
        double freq = freqs[f];
        if (freq <= 0.0) freq = freqs[1];

        double S = gal_spectral_model(freq, Amp, alpha, f_1, f_knee, f_2, avg_d);

        int out_base = (t * N_freqs + f) * 6;
        int avg_base =  t * 6;

        for (int k = 0; k < 6; k++)
            R_gal_arr[out_base + k] = R_avg[avg_base + k] * S;
    }
}

// ============================================================================
// GalacticGrid method implementations
// ============================================================================

void GalacticGrid::allocate_and_setup(
    const double *h_R_vals_quad,
    const double *h_z_vals_quad,
    const double *h_quad_weights,
    const double *h_cos_beta_ecl,
    const double *h_lam_ecl,
    const double *h_beta_ecl,
    int N_quad_in, int N_sky_in,
    double alpha0_in, double beta0_in,
    double t0_in,
    int N_times_in, int N_freqs_in)
{
    N_quad        = N_quad_in;
    N_sky         = N_sky_in;
    alpha0        = alpha0_in;
    beta0         = beta0_in;
    t0            = t0_in;
    N_times_alloc = N_times_in;
    N_freqs_alloc = N_freqs_in;
    initialized   = false;

#ifdef __CUDACC__
    gpuErrchk(cudaMalloc(&R_vals_quad,  N_quad * N_sky * sizeof(double)));
    gpuErrchk(cudaMalloc(&z_vals_quad,  N_quad * N_sky * sizeof(double)));
    gpuErrchk(cudaMalloc(&quad_weights, N_quad          * sizeof(double)));
    gpuErrchk(cudaMalloc(&cos_beta_ecl, N_sky           * sizeof(double)));
    gpuErrchk(cudaMalloc(&lam_ecl,      N_sky           * sizeof(double)));
    gpuErrchk(cudaMalloc(&beta_ecl,     N_sky           * sizeof(double)));
    gpuErrchk(cudaMalloc(&weights,      N_sky           * sizeof(double)));
    gpuErrchk(cudaMalloc(&R_avg,        N_times_in * 6  * sizeof(double)));
    gpuErrchk(cudaMalloc(&R_gal_arr,    N_times_in * N_freqs_in * 6 * sizeof(double)));

    gpuErrchk(cudaMemcpy(R_vals_quad,  h_R_vals_quad,  N_quad * N_sky * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(z_vals_quad,  h_z_vals_quad,  N_quad * N_sky * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(quad_weights, h_quad_weights, N_quad          * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cos_beta_ecl, h_cos_beta_ecl, N_sky           * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(lam_ecl,      h_lam_ecl,      N_sky           * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(beta_ecl,     h_beta_ecl,     N_sky           * sizeof(double), cudaMemcpyHostToDevice));
#else
    R_vals_quad  = const_cast<double*>(h_R_vals_quad);
    z_vals_quad  = const_cast<double*>(h_z_vals_quad);
    quad_weights = const_cast<double*>(h_quad_weights);
    cos_beta_ecl = const_cast<double*>(h_cos_beta_ecl);
    lam_ecl      = const_cast<double*>(h_lam_ecl);
    beta_ecl     = const_cast<double*>(h_beta_ecl);
    weights      = new double[N_sky];
    R_avg        = new double[N_times_in * 6];
    R_gal_arr    = new double[N_times_in * N_freqs_in * 6];
#endif

    initialized = true;
}

void GalacticGrid::free_gpu()
{
    if (!initialized) return;
#ifdef __CUDACC__
    gpuErrchk(cudaFree(R_vals_quad));
    gpuErrchk(cudaFree(z_vals_quad));
    gpuErrchk(cudaFree(quad_weights));
    gpuErrchk(cudaFree(cos_beta_ecl));
    gpuErrchk(cudaFree(lam_ecl));
    gpuErrchk(cudaFree(beta_ecl));
    gpuErrchk(cudaFree(weights));
    gpuErrchk(cudaFree(R_avg));
    gpuErrchk(cudaFree(R_gal_arr));
    if (arms != nullptr) gpuErrchk(cudaFree(arms));
#else
    delete[] weights;
    delete[] R_avg;
    delete[] R_gal_arr;
    delete[] arms;
#endif
    arms = nullptr;
    use_orbit_response = false;
    initialized = false;
}

void GalacticGrid::set_detector_arms(const double *arms_in, int N_times_in)
{
    size_t n = (size_t)N_times_in * 9;
#ifdef __CUDACC__
    if (arms != nullptr) gpuErrchk(cudaFree(arms));
    gpuErrchk(cudaMalloc(&arms, n * sizeof(double)));
    gpuErrchk(cudaMemcpy(arms, arms_in, n * sizeof(double), cudaMemcpyDefault));
#else
    delete[] arms;
    arms = new double[n];
    std::memcpy(arms, arms_in, n * sizeof(double));
#endif
    N_arms_times = N_times_in;
    use_orbit_response = true;
}

void GalacticGrid::compute_sky_weights(double R_d, double z_d, double f_b, double r_b)
{
#ifdef __CUDACC__
    int n_blocks = (N_sky + GAL_THREADS_SKY - 1) / GAL_THREADS_SKY;
    compute_column_densities_kernel<<<n_blocks, GAL_THREADS_SKY>>>(
        weights, R_vals_quad, z_vals_quad, quad_weights, cos_beta_ecl,
        R_d, z_d, f_b, r_b, N_quad, N_sky);
    gpuErrchk(cudaGetLastError());

    double *h_weights = new double[N_sky];
    gpuErrchk(cudaMemcpy(h_weights, weights, N_sky * sizeof(double), cudaMemcpyDeviceToHost));
    double total = 0.0;
    for (int i = 0; i < N_sky; i++) total += h_weights[i];
    delete[] h_weights;

    double inv_total = (total > 0.0) ? 1.0 / total : 0.0;
    normalize_weights_kernel<<<n_blocks, GAL_THREADS_SKY>>>(weights, inv_total, N_sky);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    compute_column_densities_kernel(
        weights, R_vals_quad, z_vals_quad, quad_weights, cos_beta_ecl,
        R_d, z_d, f_b, r_b, N_quad, N_sky);
    double total = 0.0;
    for (int i = 0; i < N_sky; i++) total += weights[i];
    double inv_total = (total > 0.0) ? 1.0 / total : 0.0;
    normalize_weights_kernel(weights, inv_total, N_sky);
#endif
}

void GalacticGrid::compute_sky_average(const double *d_times, int N_times)
{
    if (use_orbit_response && N_arms_times != N_times)
        throw std::invalid_argument("detector arms were set for " + std::to_string(N_arms_times) +
                                    " times, initialize() got " + std::to_string(N_times));
#ifdef __CUDACC__
    gpuErrchk(cudaMemset(R_avg, 0, N_times * 6 * sizeof(double)));
    int n_blocks_sky = (N_sky + GAL_THREADS_SKY - 1) / GAL_THREADS_SKY;
    dim3 grid(n_blocks_sky, N_times);
    if (use_orbit_response)
        sky_average_orbit_response_kernel<<<grid, GAL_THREADS_SKY>>>(
            R_avg, weights, lam_ecl, beta_ecl, arms, N_times, N_sky);
    else
        sky_average_response_kernel<<<grid, GAL_THREADS_SKY>>>(
            R_avg, weights, lam_ecl, beta_ecl, d_times,
            alpha0, beta0, t0, N_times, N_sky);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    for (int i = 0; i < N_times * 6; i++) R_avg[i] = 0.0;
    if (use_orbit_response)
        sky_average_orbit_response_kernel(
            R_avg, weights, lam_ecl, beta_ecl, arms, N_times, N_sky);
    else
        sky_average_response_kernel(
            R_avg, weights, lam_ecl, beta_ecl, d_times,
            alpha0, beta0, t0, N_times, N_sky);
#endif
}

void GalacticGrid::initialize(double R_d, double z_d,
                               const double *d_times, int N_times,
                               double f_b, double r_b)
{
    if (f_b < 0.0 || f_b > 1.0)
        throw std::invalid_argument("f_b must be in [0, 1], got " + std::to_string(f_b));
    if (f_b > 0.0 && r_b <= 0.0)
        throw std::invalid_argument("r_b must be > 0 when f_b > 0, got " + std::to_string(r_b));
    compute_sky_weights(R_d, z_d, f_b, r_b);
    compute_sky_average(d_times, N_times);
}

void GalacticGrid::compute_gal_covariance(
    const double *d_freqs, int N_freqs, int N_times,
    double Amp, double alpha,
    double f_1, double f_knee, double f_2,
    double avg_d)
{
#ifdef __CUDACC__
    dim3 block(16, 8);
    dim3 grid(
        (N_freqs + block.x - 1) / block.x,
        (N_times + block.y - 1) / block.y
    );
    apply_spectral_model_kernel<<<grid, block>>>(
        R_gal_arr, R_avg, d_freqs,
        Amp, alpha, f_1, f_knee, f_2,
        avg_d, N_times, N_freqs);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    apply_spectral_model_kernel(
        R_gal_arr, R_avg, d_freqs,
        Amp, alpha, f_1, f_knee, f_2,
        avg_d, N_times, N_freqs);
#endif
}

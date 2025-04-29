#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include "Lambert_esa.h"
#include "array_math.h"
/**
 * Normalize input vector for delta-v model.
 * @param x_not_standardized  Input vector (size = 18) containing raw features:
 *                            [L0, e, cos(ω), sin(ω), Δv0_mag, Δvt_mag,
 *                             cos(Δv0_az), sin(Δv0_az), cos(Δv0_el), sin(Δv0_el),
 *                             cos(Δvt_az), sin(Δvt_az), cos(Δvt_el), sin(Δvt_el),
 *                             TOF, T_max/mass, Isp, μ]
 * @param V0                  Output reference velocity scale (computed as L0/T0).
 * @return std::vector<double> Standardized feature vector (size 16) normalized for delta-v model,
 *                            with elements in order:
 *                            [e,
 *                             cos(ω), sin(ω),
 *                             normalized Δv0 magnitude,
 *                             normalized Δvt magnitude,
 *                             cos(Δv0 azimuth), sin(Δv0 azimuth),
 *                             cos(Δv0 elevation), sin(Δv0 elevation),
 *                             cos(Δvt azimuth), sin(Δvt azimuth),
 *                             cos(Δvt elevation), sin(Δvt elevation),
 *                             normalized time-of-flight,
 *                             normalized thrust acceleration,
 *                             normalized specific impulse]
 */
inline std::vector<double> nn_input_2_normlization_dv(const std::vector<double>& x_not_standardized, double& V0)
{
    // Gravitational parameter mu
    double mu = x_not_standardized[17];
    // Reference length scale L0
    double L0 = x_not_standardized[0];
    // Compute time and velocity scales
    double T0 = std::sqrt(L0 * L0 * L0 / mu);
    V0 = L0 / T0;
    double A0 = L0 / (T0 * T0);

    // Build standardized feature vector (16 elements)
    std::vector<double> x_standardized(16);
    x_standardized[0] = x_not_standardized[1];            // Eccentricity e
    x_standardized[1] = x_not_standardized[2];            // cos(argument of periapsis)
    x_standardized[2] = x_not_standardized[3];            // sin(argument of periapsis)
    x_standardized[3] = x_not_standardized[4] / V0;       // dv0 magnitude normalized
    x_standardized[4] = x_not_standardized[5] / V0;       // dvt magnitude normalized
    x_standardized[5] = x_not_standardized[6];            // cos(dv0 azimuth)
    x_standardized[6] = x_not_standardized[7];            // sin(dv0 azimuth)
    x_standardized[7] = x_not_standardized[8];            // cos(dv0 elevation)
    x_standardized[8] = x_not_standardized[9];            // sin(dv0 elevation)
    x_standardized[9] = x_not_standardized[10];           // cos(dvt azimuth)
    x_standardized[10] = x_not_standardized[11];           // sin(dvt azimuth)
    x_standardized[11] = x_not_standardized[12];           // cos(dvt elevation)
    x_standardized[12] = x_not_standardized[13];           // sin(dvt elevation)
    x_standardized[13] = x_not_standardized[14] / T0;      // time-of-flight normalized
    x_standardized[14] = x_not_standardized[15] / A0;      // Acceleration from inital mass
    x_standardized[15] = x_not_standardized[16] / V0;      // Specific impulse normalized

    return x_standardized;
}

/**
 * Normalize input vector for time-minimization model.
 * @param x_not_standardized  Input vector (size = 18) containing raw features:
 *                            [L0, e, cos(ω), sin(ω), Δv0_mag, Δvt_mag,
 *                             cos(Δv0_az), sin(Δv0_az), cos(Δv0_el), sin(Δv0_el),
 *                             cos(Δvt_az), sin(Δvt_az), cos(Δvt_el), sin(Δvt_el),
 *                             TOF, T_max/mass, Isp, μ]
 * @param T0                  Output reference time scale (computed as sqrt(L0^3/μ)).
 * @return std::vector<double> Standardized feature vector (size 16) normalized for time-min model,
 *                            with elements in order:
 *                            [e,
 *                             cos(ω), sin(ω),
 *                             normalized Δv0 magnitude,
 *                             normalized Δvt magnitude,
 *                             cos(Δv0 azimuth), sin(Δv0 azimuth),
 *                             cos(Δv0 elevation), sin(Δv0 elevation),
 *                             cos(Δvt azimuth), sin(Δvt azimuth),
 *                             cos(Δvt elevation), sin(Δvt elevation),
 *                             normalized time-of-flight,
 *                             normalized thrust acceleration,
 *                             normalized specific impulse]
 */
inline std::vector<double> nn_input_2_normlization_tmin(const std::vector<double>& x_not_standardized, double& T0)
{
    // Gravitational parameter mu
    double mu = x_not_standardized[17];
    // Reference length scale L0
    double L0 = x_not_standardized[0];
    // Compute time and velocity scales
    T0 = std::sqrt(L0 * L0 * L0 / mu);
    double V0 = L0 / T0;
    double A0 = L0 / (T0 * T0);

    // Build standardized feature vector (16 elements)
    std::vector<double> x_standardized(16);
    x_standardized[0] = x_not_standardized[1];            // Eccentricity e
    x_standardized[1] = x_not_standardized[2];            // cos(argument of periapsis)
    x_standardized[2] = x_not_standardized[3];            // sin(argument of periapsis)
    x_standardized[3] = x_not_standardized[4] / V0;       // dv0 magnitude normalized
    x_standardized[4] = x_not_standardized[5] / V0;       // dvt magnitude normalized
    x_standardized[5] = x_not_standardized[6];            // cos(dv0 azimuth)
    x_standardized[6] = x_not_standardized[7];            // sin(dv0 azimuth)
    x_standardized[7] = x_not_standardized[8];            // cos(dv0 elevation)
    x_standardized[8] = x_not_standardized[9];            // sin(dv0 elevation)
    x_standardized[9] = x_not_standardized[10];           // cos(dvt azimuth)
    x_standardized[10] = x_not_standardized[11];           // sin(dvt azimuth)
    x_standardized[11] = x_not_standardized[12];           // cos(dvt elevation)
    x_standardized[12] = x_not_standardized[13];           // sin(dvt elevation)
    x_standardized[13] = x_not_standardized[14] / T0;      // time-of-flight normalized
    x_standardized[14] = x_not_standardized[15] / A0;      // Acceleration from inital mass
    x_standardized[15] = x_not_standardized[16] / V0;      // Specific impulse normalized

    return x_standardized;
}

// Compute true anomaly f in [0, 2π) from state vector rv and mu
inline double f_true(const double* rv, double mu) {
    double rx = rv[0], ry = rv[1], rz = rv[2];
    double vx = rv[3], vy = rv[4], vz = rv[5];

    // Specific angular momentum h = r × v
    double hx = ry * vz - rz * vy;
    double hy = rz * vx - rx * vz;
    double hz = rx * vy - ry * vx;

    // Constants
    double r = std::sqrt(rx * rx + ry * ry + rz * rz);
    double inv_r = 1.0 / r;
    double inv_mu = 1.0 / mu;

    // v × h
    double vxh_x = vy * hz - vz * hy;
    double vxh_y = vz * hx - vx * hz;
    double vxh_z = vx * hy - vy * hx;

    // Eccentricity vector e = (v×h)/mu - r/|r|
    double ex = vxh_x * inv_mu - rx * inv_r;
    double ey = vxh_y * inv_mu - ry * inv_r;
    double ez = vxh_z * inv_mu - rz * inv_r;

    // Compute argument of true anomaly
    double e = std::sqrt(ex * ex + ey * ey + ez * ez);
    double cosf = (ex * rx + ey * ry + ez * rz) / (e * r);
    cosf = std::clamp(cosf, -1.0, 1.0);
    double f = std::acos(cosf);

    // Adjust quadrant based on radial velocity
    double rv_dot = rx * vx + ry * vy + rz * vz;
    return (rv_dot < 0.0) ? 2 * 3.14159265358979323846 - f : f;
}

/**
 * Map raw inputs to feature vector using rotation and Lambert solver.
 * @param x_not_standardized Input vector (size = 17) containing:
 *                            [rx0, ry0, rz0, vx0, vy0, vz0,
 *                             rxt, ryt, rzt, vxt, vyt, vzt,
 *                             dt, mass, T_max, Isp, mu].
 * @return std::vector<double> Output feature vector (size 18):
 *                            [a, e, cos(f), sin(f), Δv0_mag, Δvt_mag,
 *                             cos(Δv0_az), sin(Δv0_az), cos(Δv0_el), sin(Δv0_el),
 *                             cos(Δvt_az), sin(Δvt_az), cos(Δvt_el), sin(Δvt_el),
 *                             dt, T_max/mass, Isp, mu].
 */
inline std::vector<double> nn_input_1_rotate_lambert(const std::vector<double>& x_not_standardized) {
    // Unpack state and parameters
    double rv0[6], rvt[6];
    std::memcpy(rv0, x_not_standardized.data(), 6 * sizeof(double));
    std::memcpy(rvt, x_not_standardized.data() + 6, 6 * sizeof(double));
    double dt = x_not_standardized[12];
    double mass = x_not_standardized[13];
    double T_max = x_not_standardized[14];
    double Isp = x_not_standardized[15];
    double mu = x_not_standardized[16];

    // Rotate into normalized frame
    double rv0NU[6], rvtNU[6];
    rv_rotate(rv0, rvt, rv0NU, rvtNU);

    // Compute average orbital normal
    double h0[3], h1[3], avg[3];
    array_cross(h0, rv0, rv0 + 3);
    array_multi(h0, h0, 1.0 / array_norm2(h0, 3), 3);
    array_cross(h1, rvt, rvt + 3);
    array_multi(h1, h1, 1.0 / array_norm2(h1, 3), 3);
    avg[0] = -(h0[0] + h1[0]);
    avg[1] = -(h0[1] + h1[1]);
    avg[2] = -(h0[2] + h1[2]);
    array_multi(avg, avg, 1.0 / array_norm2(avg, 3), 3);

    // Solve Lambert problem
    double dv0[3], dvt[3], a, e;
    int flag = 0;
    lambert(dv0, dvt, a, e, rv0NU, rvtNU, dt, avg, flag, mu, 0, 0, 0);
    if (flag < 1) std::cerr << "Lambert solver failed" << std::endl;

    double rv_dep[6] = { rv0NU[0], rv0NU[1], rv0NU[2], dv0[0], dv0[1], dv0[2] };

    // Compute delta-v vectors
    array_minus(dv0, rv0NU + 3, dv0, 3);
    array_minus(dvt, rvtNU + 3, dvt, 3);

    // True anomaly at departure
    double f = f_true(rv_dep, mu);
    double cosf = std::cos(f);
    double sinf = std::sin(f);

    // Convert delta-v to spherical
    auto cartesianToSpherical = [](const double* vec, double* out) {
        double x = vec[0], y = vec[1], z = vec[2];
        double r = std::sqrt(x * x + y * y + z * z);
        out[0] = r;
        out[1] = std::atan2(y, x);
        out[2] = (r == 0.0) ? 0.0 : std::acos(z / r);
        };
    double dv0_rad[3], dvt_rad[3];
    cartesianToSpherical(dv0, dv0_rad);
    cartesianToSpherical(dvt, dvt_rad);

    // Assemble result vector (18 elements)
    std::vector<double> row_result(18);
    row_result[0] = a;
    row_result[1] = e;
    row_result[2] = cosf;
    row_result[3] = sinf;
    row_result[4] = array_norm2(dv0, 3);
    row_result[5] = array_norm2(dvt, 3);
    row_result[6] = std::cos(dv0_rad[1]);
    row_result[7] = std::sin(dv0_rad[1]);
    row_result[8] = std::cos(dv0_rad[2]);
    row_result[9] = std::sin(dv0_rad[2]);
    row_result[10] = std::cos(dvt_rad[1]);
    row_result[11] = std::sin(dvt_rad[1]);
    row_result[12] = std::cos(dvt_rad[2]);
    row_result[13] = std::sin(dvt_rad[2]);
    row_result[14] = dt;
    row_result[15] = T_max/ mass; 
    row_result[16] = Isp;
    row_result[17] = mu;

    return row_result;
}
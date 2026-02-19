#include "ScaledBesselI1.h"
using std::vector;

ScaledBesselI1::ScaledBesselI1(double xmax_, int N_): xmax(xmax_), N(N_)
{
    xs.resize(N);
    Fs.resize(N);

    dx = xmax / (N - 1);

    for (int i = 0; i < N; i++) {
        double x = i * dx;
        xs[i] = x;

        // scaled Bessel: I1(x) * exp(-x)
        double I1 = boost::math::cyl_bessel_i(1, x);
        Fs[i] = I1 * std::exp(-x);
    }
}

//double ScaledBesselI1::eval(double x) const
//{
//    // Region 1: negative or tiny x → clamp to first point
//    if (x <= 0.0)
//        return Fs[0];
//
//    // Region 2: outside the interpolation domain → use asymptotic
//    if (x >= xmax)
//        return asymptotic_I1_scaled(x);
//
//    // Region 3: inside interpolation domain
//    double fidx = x / dx;
//    int idx = int(fidx);
//
//    if (idx >= N - 1)
//        idx = N - 2;
//
//    double t = fidx - idx;
//
//    return (1.0 - t) * Fs[idx] + t * Fs[idx + 1];
//}

//double ScaledBesselI1::asymptotic_I1_scaled(double x) const
//{
//    // nu = 1
//    double invx = 1.0 / x;
//    double sqrtTerm = 1.0 / std::sqrt(2.0 * M_PI * x);
//
//    // series for scaled Iν(x) at large x:
//    double term1 = -(3.0) * invx / 8.0;                  // (4ν² - 1) with ν=1 => 3
//    double term2 = +(3.0 * (-5.0)) * invx * invx / 128;  // (4ν² -1)(4ν² -9) /128
//    double term3 = -(3.0 * (-5.0) * (-21.0)) * invx*invx*invx / 3072;
//
//    return sqrtTerm * (1.0 + term1 + term2 + term3);
//}

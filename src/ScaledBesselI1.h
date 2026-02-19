#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <boost/math/special_functions/bessel.hpp>

class ScaledBesselI1 {
public:
    ScaledBesselI1(double xmax, int N);

    // returns I1(x) * exp(-x)
    inline double eval(double x) const{
        // Region 1: negative or tiny x → clamp to first point
        if (x <= 0.0)
            return Fs[0];

        // Region 2: outside the interpolation domain → use asymptotic
        if (x >= xmax)
            return asymptotic_I1_scaled(x);

        // Region 3: inside interpolation domain
        double fidx = x / dx;
        int idx = int(fidx);

        if (idx >= N - 1)
            idx = N - 2;

        double t = fidx - idx;

        return (1.0 - t) * Fs[idx] + t * Fs[idx + 1];
    };

private:
    double xmax;
    int N;
    double dx;

    std::vector<double> xs;
    std::vector<double> Fs;

    inline double asymptotic_I1_scaled(double x) const{
            // nu = 1
        double invx = 1.0 / x;
        double sqrtTerm = 1.0 / std::sqrt(2.0 * M_PI * x);

        // series for scaled Iν(x) at large x:
        double term1 = -(3.0) * invx / 8.0;                  // (4ν² - 1) with ν=1 => 3
        double term2 = +(3.0 * (-5.0)) * invx * invx / 128;  // (4ν² -1)(4ν² -9) /128
        double term3 = -(3.0 * (-5.0) * (-21.0)) * invx*invx*invx / 3072;

        return sqrtTerm * (1.0 + term1 + term2 + term3);
    };
};


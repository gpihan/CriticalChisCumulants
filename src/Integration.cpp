#include "Integration.h"
#include <iostream>
#include <boost/math/quadrature/gauss.hpp>
#include <vector>
#include <gsl/gsl_integration.h>
#include <stdexcept>


// constructor definition
//NumericalIntegration::NumericalIntegration() : {}
//
NumericalIntegration::NumericalIntegration() {}

void NumericalIntegration::GetGaussLegendreCT32(std::vector<double>& x, std::vector<double>& w) {
    constexpr int N = 32;
    auto gl = boost::math::quadrature::gauss<double, N>();
    
    auto points = gl.abscissa();
    auto weights = gl.weights(); 
    
    //x.assign(points.begin(), points.end());
    //w.assign(weights.begin(), weights.end());

    x.resize(N);
    w.resize(N);

    if(N % 2 ==0){
        for (int i = 0; i < N / 2; ++i) {
            x[i] = -points[N/2 - 1 - i];
            x[N/2 + i] = points[i];     
            w[i] = weights[N/2 - 1 - i];
            w[N/2 + i] = weights[i];    
        }
    }
    else{
        for (int i = 1; i < N / 2 + 1; ++i) {
            x[i-1] = -points[N/2 + 1 - i];
            x[N/2 + i] = points[i];     
            w[i-1] = weights[N/2 + 1 - i];
            w[N/2 + i] = weights[i];    
        }
        x[N/2] = 0.0;
        w[N/2] = weights[0];
    }
}



void NumericalIntegration::GetGaussLegendreRT(int n, std::vector<double>& x, std::vector<double>& w) {
    gsl_integration_glfixed_table* table = gsl_integration_glfixed_table_alloc(n);
    if (!table) {
        throw std::runtime_error("Failed to allocate GSL Gauss-Legendre table.");
    }

    x.resize(n);
    w.resize(n);

    double a = -1.0;
    double b = 1.0;

    for (size_t i = 0; i < n; ++i) {
        double xi, wi;
        gsl_integration_glfixed_point(a, b, i, &xi, &wi, table);
        x[i] = xi; 
        w[i] = wi;
    }

    gsl_integration_glfixed_table_free(table);
}


void NumericalIntegration::GetGaussLaguerreCT32(std::vector<double>& x, std::vector<double>& w){
    size_t n = sizeof(coefficients_xlag32) / sizeof(coefficients_xlag32[0]);
    x.resize(n);
    w.resize(n);

    for (size_t i = 0; i < n; ++i) {
        x[i] = coefficients_xlag32[i]; 
        w[i] = coefficients_wlag32[i];
    }
}

void NumericalIntegration::GetGaussLaguerreCT64(std::vector<double>& x, std::vector<double>& w){
    size_t n = sizeof(coefficients_xlag64) / sizeof(coefficients_xlag64[0]);
    x.resize(n);
    w.resize(n);

    for (size_t i = 0; i < n; ++i) {
        x[i] = coefficients_xlag64[i]; 
        w[i] = coefficients_wlag64[i];
    }
}

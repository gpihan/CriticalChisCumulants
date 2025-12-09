#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <boost/math/special_functions/bessel.hpp>

#include <omp.h>
#include <atomic>

#include "Integration.h"

using namespace std;


typedef struct acceptance{
    double ymin;
    double ymax;
    double pTmin;
    double pTmax;
} Acceptance;

typedef struct surfaceElement {
    double x[4];            // position in (tau, x, y, eta)
    double sinh_eta_s;      // caching the sinh and cosh of eta_s
    double cosh_eta_s;
    double s[4];            // hypersurface vector in (tau, x, y, eta)
    double u[4];            // flow velocity in (tau, x, y, eta)
    double W[4][4];         // W^{\mu\nu}
    double q[4];            // baryon diffusion current
    double pi_b;            // bulk pressure
    double rho_B;           // net baryon density

    double epsilon_f;
    double T_f;
    double mu_B;
    double mu_S;
    double mu_C;
    double eps_plus_p_over_T_FO;  // (energy_density+pressure)/temperature
} SurfaceElement;

#include <vector>
#include <cstddef>

vector<vector<SurfaceElement>> splitVector(vector<SurfaceElement>& vec, int N) {
    vector<vector<SurfaceElement>> chunks;
    if (N <= 0) return chunks;

    size_t total = vec.size();
    size_t base_size = total / N;
    size_t remainder = total % N;

    chunks.reserve(N);

    size_t start = 0;
    for (int i = 0; i < N; ++i) {
        // First 'remainder' chunks get one extra element
        size_t chunk_size = base_size + (i < remainder ? 1 : 0);

        if (chunk_size == 0) break;  // more chunks than elements

        size_t end = start + chunk_size;
        chunks.emplace_back(vec.begin() + start, vec.begin() + end);
        start = end;
    }

    return chunks;
}

inline double I0_fast(double x)
{
    double ax = fabs(x);
    if (ax < 3.75) {
        double t = ax / 3.75;
        t *= t;
        return 1.0 + t*(3.5156229 + t*(3.0899424 + t*(1.2067492
             + t*(0.2659732 + t*(0.0360768 + t*0.0045813)))));
    } else {
        double t = 3.75/ax;
        return (exp(ax)/sqrt(ax)) *
            (0.39894228 + t*(0.01328592 + t*(0.00225319 + t*(-0.00157565
            + t*(0.00916281 + t*(-0.02057706 + t*(0.02635537
            + t*(-0.01647633 + t*0.00392377))))))));
    }
}

inline double I1_fast(double x)
{
    double ax = fabs(x);
    double y, ans;
    if (ax < 3.75) {
        double t = ax / 3.75;
        t *= t;
        ans = ax*(0.5 + t*(0.87890594 + t*(0.51498869 + t*(0.15084934
                + t*(0.02658733 + t*(0.00301532 + t*0.00032411))))));
    } else {
        double t = 3.75/ax;
        ans = 0.2282967 + t*(-0.2895312 + t*(0.1787654 - t*(0.4200590
              + t*(0.6140116 - t*(0.6470344 + t*(0.3708892
              - t*0.09347564))))));
        ans = ans * exp(ax) / sqrt(ax);
    }
    return (x < 0.0 ? -ans : ans);
}

inline double I0_scaled_fast(double x)
{
    if (x < 3.75)
        return I0_fast(x) * exp(-x);
    else {
        double ax = x;
        double t = 3.75/ax;
        return (1.0/sqrt(ax)) *
            (0.39894228 + t*(0.01328592 + t*(0.00225319 + t*(-0.00157565
            + t*(0.00916281 + t*(-0.02057706 + t*(0.02635537
            + t*(-0.01647633 + t*0.00392377))))))));
    }
}


inline double I1_scaled_fast(double x)
{
    if (x < 3.75)
        return I1_fast(x) * exp(-x);
    else {
        double ax = x;
        double t = 3.75/ax;

        double poly = 0.2282967 + t*(-0.2895312 + t*(0.1787654 - t*(0.4200590
                     + t*(0.6140116 - t*(0.6470344 + t*(0.3708892
                     - t*0.09347564))))));

        return poly / sqrt(ax);
    }
}

vector<double> MilneToCartesian(const vector<double>& Milne, const SurfaceElement& Surf){
    // This function translates a 4-vector expressed in Milne coordinates (tau, x, y, eta) in 
    // a 4-vector expressed in the cartesian coordinates. 
    // Note that the input Milne vector is expected to have the tau factor in the 
    // eta direction already included. 
    vector<double> Cart(Milne.size());
    Cart[0] = Surf.cosh_eta_s * Milne[0] + Surf.sinh_eta_s * Milne[3];
    Cart[1] = Milne[1];
    Cart[2] = Milne[2];
    Cart[3] = Surf.sinh_eta_s * Milne[0] + Surf.cosh_eta_s * Milne[3]; 
    return Cart;
}

vector<double> MilneToCartesianSigmaCov(const vector<double>& Milne, const SurfaceElement& Surf){
    vector<double> Cart(4);
    double tau = Surf.x[0];
    double ch  = Surf.cosh_eta_s;
    double sh  = Surf.sinh_eta_s;

    double dsT  = Milne[0];   // dσ^τ
    double dsX  = Milne[1];   // dσ^x
    double dsY  = Milne[2];   // dσ^y
    double dsEt = Milne[3];   // τ dσ^η

    Cart[0] =  tau * dsT * ch - dsEt * sh;         // dσ^t
    Cart[1] =  tau * dsX;                          // dσ^x
    Cart[2] =  tau * dsY;                          // dσ^y
    Cart[3] =  -(tau * dsT * sh - dsEt * ch);         // dσ^z
                                        
    return Cart;
}

inline double dotCov(const vector<double>& a, const vector<double>& b, double coeff=1.0) {
    // Function to compute Minkowski inner product with (+,-,-,-) metric
    // when input contra-cov 4-vectors are represented by std vectors.
    return coeff * (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]);
}

inline double dotMinkowski(vector<double> a, vector<double> b, double tau=1.0, double coeff=1.0) {
    // Function to compute Minkowski inner product with (+,-,-,-) metric
    // when input contra-contra 4-vectors are represented by std vectors.
    return coeff * (a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]);
}
inline double dotMinkowski(const double a[4], const double b[4], double tau=1.0, double coeff=1.0) {
    // Function to compute Minkowski inner product with (+,-,-,-) metric
    // when input contra contra 4-vectors are represented by arrays.
    return coeff * (a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]);
}
inline double dotContraDens(const vector<double>& a, const vector<double>& b, SurfaceElement Surf){
    // This function computes the product between a contravariante vector and a density vector
    // like s^mu.
    double tau = Surf.x[0];
    return tau * (a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - (a[3]*b[3])/tau);
}

vector<SurfaceElement> selectPositiveVolumeCells(const std::vector<SurfaceElement>& allCells, int N = -1) {
    // This function selects contributing hydro cells e.g. the ones with 4-velocity of flow 
    // u^mu in the direction of the cell outward 4-vector sigma^mu
    int NMAX;
    if(N == -1){
        NMAX = allCells.size();
    }
    else{
        NMAX = N;
    }
    vector<SurfaceElement> selected;
    SurfaceElement cell;
    double tau;
    selected.reserve(allCells.size());
    for (int i = 0; i < NMAX; i++) {
        cell = allCells[i];
        tau = cell.x[0];
        // Compute 4-volume element = s^μ u_μ
        // Here, the product is not the regular Minkowsky product as s^mu is not a regular 4-vector, its a density vector
        // meaning is S * n where S is a number that changes with the frame (the volume of the considered cell = Jacobian) and 
        // n is a regular 4-vector. The results, dsigma^mu is not Lorentz invariant.  
        double dV4 = tau * (cell.u[0]*cell.s[0] + cell.u[1]*cell.s[1] + cell.u[2]*cell.s[2] + cell.u[3]*(cell.s[3]/tau));
        if (dV4 > 0.0) {
            selected.push_back(cell);
        }
    }
    return selected;
}

int get_number_of_lines_of_binary_surface_file(string filename) {
    // This function computes the total number of lines in the surface binary file.
    std::ifstream surface_file(filename.c_str(), std::ios::binary);
    int count = 0;
    float temp = 0.;
    while(surface_file) {
        surface_file.read((char*) &temp, sizeof(float));
        count++;
    }
    int counted_line = count/34;/////THIS NUMBER IS MAYBE WRONG
    surface_file.close();
    return(counted_line);
}

vector<SurfaceElement> ReadFreezeOutSurface(string Surfpath) {
    // This function reads the surface file and fill the surface elements properties. 
    cout << "reading freeze-out surface" << endl;
    
    double hbarc = 0.1973;

    vector<SurfaceElement> surface;

    bool boost_invariant = false;
    ostringstream surfdat_stream;
    surfdat_stream << Surfpath;

    // new counting, mac compatible ...
    int NCells = get_number_of_lines_of_binary_surface_file(surfdat_stream.str());

    cout << "NCells = " << NCells << endl;
    flush(cout);

    ifstream surfdat;
    surfdat.open(surfdat_stream.str().c_str(), std::ios::binary);

    int i = 0;
    while (i < NCells) {
        SurfaceElement temp_cell;
        float array[34];
        for (int ii = 0; ii < 34; ii++) {
            float temp = 0.;
            surfdat.read((char*)&temp, sizeof(float));
            array[ii] = temp;
        }
        temp_cell.x[0] = array[0];
        temp_cell.x[1] = array[1];
        temp_cell.x[2] = array[2];
        temp_cell.x[3] = array[3];
        if (boost_invariant) {
            temp_cell.x[3] = 0.0;
        }

        // MUSIC convention
        temp_cell.s[0] = array[4];
        temp_cell.s[1] = array[5];
        temp_cell.s[2] = array[6];
        temp_cell.s[3] = array[7];

        temp_cell.u[0] = array[8];
        temp_cell.u[1] = array[9];
        temp_cell.u[2] = array[10];
        temp_cell.u[3] = array[11];

        temp_cell.epsilon_f            = array[12] * hbarc;
        temp_cell.T_f                  = array[13] * hbarc;
        temp_cell.mu_B                 = array[14] * hbarc;
        temp_cell.mu_S                 = array[15] * hbarc;
        temp_cell.mu_C                 = array[16] * hbarc;
        temp_cell.eps_plus_p_over_T_FO = array[17] * hbarc;

        temp_cell.W[0][0] = array[18] * hbarc;
        temp_cell.W[0][1] = array[19] * hbarc;
        temp_cell.W[0][2] = array[20] * hbarc;
        temp_cell.W[0][3] = array[21] * hbarc;
        temp_cell.W[1][1] = array[22] * hbarc;
        temp_cell.W[1][2] = array[23] * hbarc;
        temp_cell.W[1][3] = array[24] * hbarc;
        temp_cell.W[2][2] = array[25] * hbarc;
        temp_cell.W[2][3] = array[26] * hbarc;
        temp_cell.W[3][3] = array[27] * hbarc;

        temp_cell.pi_b  = array[28] * hbarc;
        temp_cell.rho_B = array[29]; // 1/fm3

        temp_cell.q[0] = array[30];
        temp_cell.q[1] = array[31];
        temp_cell.q[2] = array[32];
        temp_cell.q[3] = array[33];

        temp_cell.sinh_eta_s = sinh(temp_cell.x[3]);
        temp_cell.cosh_eta_s = cosh(temp_cell.x[3]);

        if (temp_cell.epsilon_f < 0)  {
            cout << "epsilon_f < 0.!" << endl;
            exit(1);
        }
        if (temp_cell.T_f < 0) {
            cout << "T_f < 0.!" << endl;
            exit(1);
        }
        surface.push_back(temp_cell);
        i++;
    }
    surfdat.close();
    return surface;
}



// Full space Laguerre+Legendre integral ---------------------------------------------
// The integrals will not contain the exp(mu_i/T) factors as they can be factorized when computing the 
// total yield of particle i
//double alphaBar(int mu, double T, double mu_i, vector<double>& u){
//    // This function computes the \bar{\alpha} function for its use 
//    // in the Laguerre integration.
//    double uT; // Transverse 4-velocity of flow
//    // Here, the definition does not contain fugacity factor.
//    //double prefactor = 2 * M_PI * exp(mu_i/T) * T * T; 
//    double prefactor = 2 * M_PI * T * T; 
//    if(mu == 0){
//        return prefactor; 
//    }
//    else if(mu == 1){
//        uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
//        return prefactor * u[1] / uT; 
//        //return 0.0; For Test function 
//    }
//    else if(mu == 2){
//        uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
//        return prefactor * u[2] / uT;
//        //return 0.0; For Test function 
//    }
//    else if(mu == 3){
//        return prefactor;
//
//    }
//    else{
//        cout << "index in alpha bar must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}

//double alphaBar(int mu, double prefactor, double u1ouT, double u2ouT){
//    // This function computes the \bar{\alpha} function for its use 
//    // in the Laguerre integration.
//    //double uT; // Transverse 4-velocity of flow
//    // Here, the definition does not contain fugacity factor.
//    //double prefactor = 2 * M_PI * exp(mu_i/T) * T * T; 
//    //double prefactor = 2 * M_PI * T * T; 
//    if(mu == 0){
//        return prefactor; 
//    }
//    else if(mu == 1){
//        return prefactor * u1ouT; 
//    }
//    else if(mu == 2){
//        return prefactor * u2ouT;
//    }
//    else if(mu == 3){
//        return prefactor;
//    }
//    else{
//        cout << "index in alpha bar must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}

/// REDO EFFCIENCY
inline array<double,4> alphaBar_all(double prefactor, double u1ouT, double u2ouT){
    return {
        prefactor,          // μ = 0
        prefactor * u1ouT,  // μ = 1
        prefactor * u2ouT,  // μ = 2
        prefactor           // μ = 3
    };
}

//double gbar(int mu, double y, double m, double T, vector<double>& u){ 
//    double Gamma = cosh(y) * u[0] - sinh(y) * u[3];
//    double prefactor = exp(-m/T * Gamma)/(Gamma * Gamma);
//
//    if(mu == 0){
//        return prefactor * cosh(y); 
//    }
//    else if(mu == 1){
//        return prefactor; 
//    }
//    else if(mu == 2){
//        return prefactor;
//    }
//    else if(mu == 3){
//        return prefactor * sinh(y);
//    }
//    else{
//        cout << "index in g bar must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}

//double gbar(int mu, double y, double prefactor){ 
//    if(mu == 0){
//        return prefactor * cosh(y); 
//    }
//    else if(mu == 1){
//        return prefactor; 
//    }
//    else if(mu == 2){
//        return prefactor;
//    }
//    else if(mu == 3){
//        return prefactor * sinh(y);
//    }
//    else{
//        cout << "index in g bar must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}

inline array<double,4> gbar_all(double y, double prefactor)
{
    double cy = cosh(y);
    double sy = sinh(y);

    return {
        prefactor * cy,   // μ = 0
        prefactor,        // μ = 1
        prefactor,        // μ = 2
        prefactor * sy    // μ = 3
    };
}


double GetpT(double treshold, double mT, double m){
    // This function checks if the transverse mass is large compared to the mass
    // it avoids computing unecessary square roots.
    if(mT/m < treshold){
        return sqrt(mT * mT - m * m);
    }
    else{
        return mT;
    }
}

double Modified_Bessel_scaled(int NU, double x){
    // This function returns the scaled modified bessel function of the second kind: I(x, nu) e^{-x}.
    // As the Boost Modified bessel functions overflows for x around 700, we impose a cut. 
    // If x is larger than XLIM, the function returns the asymptotic expansion of the scaled bessel function
    // at order x^(-7/2). 
    double XLIM = 100.0;
    double nu = (double) NU;
    if(x > XLIM){
        // If input larger than XLIM (= max 700) return the asymptotic expansion at order 3 = do not overflow
        return 1/sqrt(2 * M_PI * x) * (1 -   (4 * nu - 1)/(8 * x) +  (4 * nu * nu -1) * (4 * nu * nu - 9)/(128 * x * x) - (4 * nu * nu -1) * (4 * nu * nu - 9) * (4 * nu * nu - 25)/(3072 * x * x * x));
    }
    else{
        return boost::math::cyl_bessel_i(NU, x) * exp(-x); 
    }
}



//double fbar_scaled(int mu, double x, double y, double m, double T, vector<double>& u, double threshold){
//    // This function coomputes the \bar{f} functions in the Laguerre integration. 
//    // Note that we use the scaled Bessel functions. 
//    double Gamma = cosh(y) * u[0] - sinh(y) * u[3];
//    double prefactor = x + m/T*Gamma;
//
//    double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
//
//    double mT = T/Gamma * x + m;
//    //double pT = GetpT(threshold, mT, m);
//    double pT = sqrt(mT * mT - m * m);
//
//    if(mu == 0){
//        return prefactor * mT * Modified_Bessel_scaled(0, pT/T * uT); 
//    }
//    else if(mu == 1){
//        return prefactor * pT * Modified_Bessel_scaled(1, pT/T * uT); 
//    }
//    else if(mu == 2){
//        return prefactor * pT * Modified_Bessel_scaled(1, pT/T * uT); 
//    }
//    else if(mu == 3){
//        return prefactor * mT * Modified_Bessel_scaled(0, pT/T * uT); 
//    }
//    else{
//        cout << "index in f bar must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}

//double fbar_scaled(int mu, double x, double y, double m, double T, vector<double>& u, double threshold){

//double fbar_scaled(int mu, double pT, double mT, double prefactor, double pToTuT){
//    // This function coomputes the \bar{f} functions in the Laguerre integration. 
//    // Note that we use the scaled Bessel functions. 
//    //double Gamma = cosh(y) * u[0] - sinh(y) * u[3];
//    //double prefactor = x + m/T*Gamma;
//
//    //double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
//
//    //double mT = T/Gamma * x + m;
//    //double pT = GetpT(threshold, mT, m);
//    //double pT = sqrt(mT * mT - m * m);
//    
//    // prefactor = x + m/T * Gamma
//    // pToTuT = pT/T * ||uT||
//    
//    if(mu == 0){
//        return prefactor * mT * I0_scaled_fast(pToTuT); 
//        //Modified_Bessel_scaled(0, pToTuT); 
//    }
//    else if(mu == 1){
//        return prefactor * pT * I1_scaled_fast(pToTuT);
//            //Modified_Bessel_scaled(1, pToTuT); 
//    }
//    else if(mu == 2){
//        return prefactor * pT * I1_scaled_fast(pToTuT);
//            //Modified_Bessel_scaled(1, pToTuT); 
//    }
//    else if(mu == 3){
//        return prefactor * mT * I0_scaled_fast(pToTuT);
//            //Modified_Bessel_scaled(0, pToTuT); 
//    }
//    else{
//        cout << "index in f bar must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}

inline array<double,4> fbar_scaled_all(
        double prefactor,   // x + m/T * Gamma
        double pT,
        double mT,
        double pToTuT       // pT/T * uT
)
{
    // ----- 1. Compute both scaled Bessel functions once -----
    double I0 = I0_scaled_fast(pToTuT);
    double I1 = I1_scaled_fast(pToTuT);

    // ----- 2. Precompute shared multiplicative factors -----
    double pref_mT = prefactor * mT;
    double pref_pT = prefactor * pT;

    // ----- 3. Return μ=0..3 in a single compact, branchless form -----
    return {
        pref_mT * I0,   // μ = 0
        pref_pT * I1,   // μ = 1
        pref_pT * I1,   // μ = 2
        pref_mT * I0    // μ = 3
    };
}

//bool checkdsigmamuXimu(double x, double y, double m, double T, double mu_i, vector<double>& u, vector<double>& dsigma, double threshold){
//    // This function checks if the calculated 4-momentum is in the same direction as the outward cell 4-vector dsigma^mu
//    // It corresponds to the actual positive contributions to the particle production. In principle, negative contributions 
//    // corresponds to particles flowing back to the fluid. 
//    // It is different from the condition u^mu dsigma_mu > 0 (hydro flow towards the detectors) as u^mu and p^mu are fairly 
//    // independent. 
//    // This function ensures that the ratio of number of particles inside and outside acceptance actually makes sense. 
//    // Note that the use of scaled modified Bessel functions in fbar_scaled is not an issue as the norm is an 
//    // an exponential function e(pT/T ||uT||) > 0 that is the same for all Xk0 here, it does not change the 
//    // sign check. 
//
//    double x0 = alphaBar(0, T, mu_i, u) * gbar(0, y, m, T, u) * fbar_scaled(0, x, y, m, T, u, threshold) * dsigma[0];
//    double x1 = alphaBar(1, T, mu_i, u) * gbar(1, y, m, T, u) * fbar_scaled(1, x, y, m, T, u, threshold) * dsigma[1];
//    double x2 = alphaBar(2, T, mu_i, u) * gbar(2, y, m, T, u) * fbar_scaled(2, x, y, m, T, u, threshold) * dsigma[2];
//    double x3 = alphaBar(3, T, mu_i, u) * gbar(3, y, m, T, u) * fbar_scaled(3, x, y, m, T, u, threshold) * dsigma[3];
//    return (x0+x1+x2+x3>0);
//
//    //vector<double> Xi;
//    //for(int i = 0; i<4;i++){
//    //    Xi.push_back(alphaBar(i, T, mu_i, u) * gbar(i, y, m, T, u) * fbar_scaled(i, x, y, m, T, u, threshold));
//    //}
//    //// Note: we use the dotCov function here because after translation into Cartesian coordinates, dsigma is still covariant.
//    //return (dotCov(Xi, dsigma)>0);
//}


bool checkdsigmamuXimuFULL(double abar0, double abar1, double abar2, double abar3, 
        double gbar0, double gbar1, double gbar2, double gbar3, 
        double fbar0, double fbar1, double fbar2, double fbar3, 
        const vector<double>& dsigma){
    // This function checks if the calculated 4-momentum is in the same direction as the outward cell 4-vector dsigma^mu
    // It corresponds to the actual positive contributions to the particle production. In principle, negative contributions 
    // corresponds to particles flowing back to the fluid. 
    // It is different from the condition u^mu dsigma_mu > 0 (hydro flow towards the detectors) as u^mu and p^mu are fairly 
    // independent. 
    // This function ensures that the ratio of number of particles inside and outside acceptance actually makes sense. 
    // Note that the use of scaled modified Bessel functions in fbar_scaled is not an issue as the norm is an 
    // an exponential function e(pT/T ||uT||) > 0 that is the same for all Xk0 here, it does not change the 
    // sign check. 

    double x0 = abar0 * gbar0 * fbar0 * dsigma[0];
    double x1 = abar1 * gbar1 * fbar1 * dsigma[1];
    double x2 = abar2 * gbar2 * fbar2 * dsigma[2];
    double x3 = abar3 * gbar3 * fbar3 * dsigma[3];
    return (x0+x1+x2+x3>0);

    //vector<double> Xi;
    //for(int i = 0; i<4;i++){
    //    Xi.push_back(alphaBar(i, T, mu_i, u) * gbar(i, y, m, T, u) * fbar_scaled(i, x, y, m, T, u, threshold));
    //}
    //// Note: we use the dotCov function here because after translation into Cartesian coordinates, dsigma is still covariant.
    //return (dotCov(Xi, dsigma)>0);
}

//double getScaledW(double x, double y, double T, double m, double WLag, vector<double>& u, double threshold){
//    // This function computes the numerically regularized Laguerre weight e^x WLag(x).
//    // The value of Laguerre weight is more than exponentially decreasing and can reach values very close to 0.
//    // So close that that the machine cannot handle it anymore. 
//    // This trick allows to conserved reasonnable values for the Laguerre weight. 
//    // It is made possible by the use of the scaled modified Bessel functions. This exponential 
//    // factor that we see here is the one coming for there. 
//    // If the log evaluation: x + log(WLag(x)) < -32 (2 * machine double precision), return 0.
//    // If the log evaluation is large than 50, there is an issue. The code shows that there is an issue 
//    // and set the output value to 0. 
//    // If the exponant value e(x+log(WLag(x)) > than 700, the Laguerre weights is so suppressed that the result 
//    // is identically 0. 
//    
//    double Gamma = cosh(y) * u[0] - sinh(y) * u[3];
//    double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
//    double mT = T/Gamma * x + m;
//    //double pT = GetpT(threshold, mT, m);
//    double pT = sqrt(mT *mT - m * m);
//
//    double expon = pT / T * uT;
//    double check = log(WLag) + pT / T * uT; 
//
//    if(check > 50){
//        cout << "Very large value of w * e^(pT / T uT), exponant value : " << check << endl;
//        return 0.0;
//    } 
//    else if(check < -40){ // exp(-37) ~ 1e-16.
//        return 0.0;
//    }
//    else{
//        if(expon > 700){ // Machine limit
//            // Since -40 < log(WLag) + expon < 50, if expon > 700 it means log(WLag) < -750.....
//            return 0.0;
//        }
//        else{
//            return WLag * exp(expon);
//        }
//    }
//}

//double getScaledW(double x, double y, double T, double m, double WLag, vector<double>& u, double threshold){
//double getScaledW(double pToTuT, double exppToTuT, double WLag){
//    // This function computes the numerically regularized Laguerre weight e^x WLag(x).
//    // The value of Laguerre weight is more than exponentially decreasing and can reach values very close to 0.
//    // So close that that the machine cannot handle it anymore. 
//    // This trick allows to conserved reasonnable values for the Laguerre weight. 
//    // It is made possible by the use of the scaled modified Bessel functions. This exponential 
//    // factor that we see here is the one coming for there. 
//    // If the log evaluation: x + log(WLag(x)) < -32 (2 * machine double precision), return 0.
//    // If the log evaluation is large than 50, there is an issue. The code shows that there is an issue 
//    // and set the output value to 0. 
//    // If the exponant value e(x+log(WLag(x)) > than 700, the Laguerre weights is so suppressed that the result 
//    // is identically 0. 
//    
//    double check = log(WLag) + pToTuT; 
//
//    if(check > 50){
//        cout << "Very large value of w * e^(pT / T uT), exponant value : " << check << endl;
//        return 0.0;
//    } 
//    else if(check < -40){ // exp(-37) ~ 1e-16.
//        return 0.0;
//    }
//    else{
//        if( pToTuT > 700){ // Machine limit
//            // Since -40 < log(WLag) + expon < 50, if expon > 700 it means log(WLag) < -750.....
//            return 0.0;
//        }
//        else{
//            return WLag * exppToTuT;
//        }
//    }
//}

double getScaledW_all(double pToTuT, double exppToTuT, double WLag, double logWLag){
    // This function computes the numerically regularized Laguerre weight e^x WLag(x).
    // The value of Laguerre weight is more than exponentially decreasing and can reach values very close to 0.
    // So close that that the machine cannot handle it anymore. 
    // This trick allows to conserved reasonnable values for the Laguerre weight. 
    // It is made possible by the use of the scaled modified Bessel functions. This exponential 
    // factor that we see here is the one coming for there. 
    // If the log evaluation: x + log(WLag(x)) < -32 (2 * machine double precision), return 0.
    // If the log evaluation is large than 50, there is an issue. The code shows that there is an issue 
    // and set the output value to 0. 
    // If the exponant value e(x+log(WLag(x)) > than 700, the Laguerre weights is so suppressed that the result 
    // is identically 0. 
    
    if (pToTuT > 700.0)
        return 0.0;

    // Compute combined exponent (only once)
    const double check = logWLag + pToTuT;

    // Safe range check: -40 < check < 50
    if (check < -40.0 || check > 50.0)
        return 0.0;

    // Normal case
    return WLag * exppToTuT;
}

//double IntegralFull(double YM, const vector<double>& Omega, const vector<double>& Y, int Nleg, const vector<double>& X, const vector<double>& W, int Nlag, 
//        int mu, double m, double T, 
//        vector<double>& u, vector<double>& dsigma, double threshold, 
//        double prefacta, double uToT, double u1ouT, double u2ouT){
//
//    // This function computes the Laguerre+Legendre quadrature to coompute integrals in the full space. 
//    double s = 0.0;
//    double x, y, scaledW;
//    double gb, fb, LegO;
//    double Gamma, prefactgbar, tmp2, tmp3, prefactfbar;
//    double mT, pT;
//    double pToTuT;
//    double exppToTuT;
//    double gbar0, gbar1, gbar2, gbar3;
//
//    // precompute alpha bar function for efficiency
//    double abarmu = alphaBar(mu, prefacta, u1ouT, u2ouT);
//    double abar0 = alphaBar(0, prefacta, u1ouT, u2ouT);
//    double abar1 = alphaBar(1, prefacta, u1ouT, u2ouT);
//    double abar2 = alphaBar(2, prefacta, u1ouT, u2ouT);
//    double abar3 = alphaBar(3, prefacta, u1ouT, u2ouT);
//
//    for(int i = 0; i<Nleg; i++){
//        y = Y[i] * YM; // current rapidity, Legendre are calculated in [-1,1], times YM = [-YM, YM]
//
//        // Precompute rapidity dependent functions for efficiency.
//        Gamma = cosh(y) * u[0] - sinh(y) * u[3];
//        prefactgbar = exp(-m/T * Gamma) / (Gamma * Gamma);
//
//        tmp2 = m/T * Gamma;
//        tmp3 = T/Gamma;
//
//        //gb = gbar(mu, y, m, T, u);
//        gb = gbar(mu, y, prefactgbar);
//        gbar0 = gbar(0, y, prefactgbar);
//        gbar1 = gbar(1, y, prefactgbar);
//        gbar2 = gbar(2, y, prefactgbar);
//        gbar3 = gbar(3, y, prefactgbar);
//
//        for(int j = 0; j<Nlag; j++){
//            x = X[j];
//
//            //double Gamma = cosh(y) * u[0] - sinh(y) * u[3];
//            // x + m/T Gamma
//            //double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
//
//            // precompute fbar pre factor, mT, pT, pT/T ||uT||, exp(pT/T ||uT||)
//            // fbar prefactor = x + m/T * Gamma
//            prefactfbar = x + tmp2;
//            // mT = T/Gamma * x + m
//            mT = tmp3 * x + m;
//            pT = GetpT(threshold, mT, m);
//            pToTuT = pT * uToT;
//            exppToTuT = exp(pToTuT);
//
//
//            // Check if the cell will contribute positively to the integrals.
//            //if(checkdsigmamuXimu(x, y, m, T, mu_i, u, dsigma, threshold)){
//            //    // Compute scaled Laguerre weight for numerical regularization. 
//            //    scaledW = getScaledW(x, y, T, m, W[j], u, threshold);
//            //    fb = fbar_scaled(mu, x, y, m, T, u, threshold);
//            //    s += Omega[i] * gb * scaledW * fb;
//            if(checkdsigmamuXimu(pT, mT, prefactfbar, pToTuT, gbar0, gbar1, gbar2, gbar3, abar0, abar1, abar2, abar3, dsigma)){
//                // Compute scaled Laguerre weight for numerical regularization. 
//                s += Omega[i] * gb * getScaledW(pToTuT, exppToTuT, W[j]) * fbar_scaled(mu, pT, mT, prefactfbar, pToTuT);
//            }
//        }
//    }
//    //return alphaBar(mu, T, mu_i, u) * YM * s;
//    return abarmu * YM * s;
//}

array<double,4> IntegralFull_all(
        double YM,
        const vector<double>& Omega, const vector<double>& Y, int Nleg,
        const vector<double>& X, const vector<double>& W, const vector<double>& logW, int Nlag,
        double m, double T, const vector<double>& u, const vector<double>& dsigma,
        double threshold,
        double prefacta, double u1ouT, double u2ouT, double uToT)
{
    const auto abar = alphaBar_all(prefacta, u1ouT, u2ouT);

    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    double y, Gamma, x, pT, mT;
    double prefactfbar, pToTuT, exppToTuT;

    for (int i = 0; i < Nleg; i++)
    {
        y = Y[i] * YM;
        Gamma = cosh(y) * u[0] - sinh(y) * u[3];
        const double tmp2 = (m/T) * Gamma;  // m/T Γ
        const double prefactgbar = exp(-tmp2) / (Gamma * Gamma);
        const auto gb = gbar_all(y, prefactgbar);
        const double Oi = Omega[i];
        const double tmp3 = T / Gamma;      // mT = tmp3*x + m

        for (int j = 0; j < Nlag; j++)
        {
            x = X[j];
            // prefactor_f̄ = x + m/T Γ
            prefactfbar = x + tmp2;
            // mT = T/Γ * x + m
            mT = tmp3 * x + m;
            pT = GetpT(threshold, mT, m);
            // pT/T ||uT||
            pToTuT   = pT * uToT;
            exppToTuT = exp(pToTuT);
            // Compute 4× f̄(μ)
            const auto fb = fbar_scaled_all(prefactfbar, pT, mT, pToTuT);
            // dσμ p^μ positivity check
            if (checkdsigmamuXimuFULL(
                    abar[0], abar[1], abar[2], abar[3],
                    gb[0], gb[1], gb[2], gb[3],
                    fb[0], fb[1], fb[2], fb[3],
                    dsigma))
            {
                // Scaled Laguerre weight (pre-logged)
                const double scaledW =
                    getScaledW_all(pToTuT, exppToTuT, W[j], logW[j]);

                if (scaledW == 0.0)
                    continue;

                s0 += Oi * gb[0] * fb[0] * scaledW;
                s1 += Oi * gb[1] * fb[1] * scaledW;
                s2 += Oi * gb[2] * fb[2] * scaledW;
                s3 += Oi * gb[3] * fb[3] * scaledW;
            }
        }
    }

    // Final contraction ᾱ(μ) × ∫
    return {
        abar[0] * YM * s0,
        abar[1] * YM * s1,
        abar[2] * YM * s2,
        abar[3] * YM * s3
    };
}



//double deltaNcellFull(double YM, const vector<double>& Omega, const vector<double>& Y, int Nleg, const vector<double>& X, const vector<double>& W, int Nlag, double m, double T, double mu_i, vector<double>& u, vector<double>& dsigma, double threshold, double g=1.0){
//    // This function computes the final contributions from the cells in the full acceptance. 
//
//    double Nmu0 = IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, 0, m, T, mu_i, u, dsigma, threshold) * dsigma[0];
//    double Nmu1 = IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, 1, m, T, mu_i, u, dsigma, threshold) * dsigma[1];
//    double Nmu2 = IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, 2, m, T, mu_i, u, dsigma, threshold) * dsigma[2];
//    double Nmu3 = IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, 3, m, T, mu_i, u, dsigma, threshold) * dsigma[3];
//
//    return (Nmu0 + Nmu1 + Nmu2 + Nmu3) * g/(8 * M_PI * M_PI * M_PI); // g degeneracy factor
//    //vector<double> Nmu;
//    //for(int i = 0; i<4; i++){
//    //    Nmu.push_back(IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, i, m, T, mu_i, u, dsigma, threshold));
//    //}
//    //return dotCov(Nmu, dsigma, g/(8 * M_PI * M_PI * M_PI)); // g degeneracy factor
//}
//double deltaNcellFull(double YM, const vector<double>& Omega, const vector<double>& Y, int Nleg, const vector<double>& X, const vector<double>& W, int Nlag, double m, double T, vector<double>& u, vector<double>& dsigma, double threshold, double g=1.0){
//    // This function computes the final contributions from the cells in the full acceptance. 
//    
//    double uT = sqrt(u[1] * u[1] + u[2] * u[2]);
//    double uToT = uT/T; 
//    double u1ouT, u2ouT;
//    if(uT < 1e-14){
//        u1ouT = 0.0;
//        u2ouT = 0.0;
//    }else{
//        u1ouT = u[1]/uT;
//        u2ouT = u[2]/uT;
//    }
//    double prefacta = 2 * M_PI * T * T; 
//
//    double Nmu0 = IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, 0, m, T, u, dsigma, threshold, prefacta, uToT, u1ouT, u2ouT) * dsigma[0];
//    double Nmu1 = IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, 1, m, T, u, dsigma, threshold, prefacta, uToT, u1ouT, u2ouT) * dsigma[1];
//    double Nmu2 = IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, 2, m, T, u, dsigma, threshold, prefacta, uToT, u1ouT, u2ouT) * dsigma[2];
//    double Nmu3 = IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, 3, m, T, u, dsigma, threshold, prefacta, uToT, u1ouT, u2ouT) * dsigma[3];
//
//    return (Nmu0 + Nmu1 + Nmu2 + Nmu3) * g/(8 * M_PI * M_PI * M_PI); // g degeneracy factor
//    //vector<double> Nmu;
//    //for(int i = 0; i<4; i++){
//    //    Nmu.push_back(IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, i, m, T, mu_i, u, dsigma, threshold));
//    //}
//    //return dotCov(Nmu, dsigma, g/(8 * M_PI * M_PI * M_PI)); // g degeneracy factor
//}

double deltaNcellFull(
        double YM,
        const vector<double>& Omega,  // Legendre weights
        const vector<double>& Y,      // Legendre nodes
        int Nleg,
        const vector<double>& X,      // Laguerre nodes
        const vector<double>& W,      // Laguerre weights
        int Nlag,
        double m, double T,
        const vector<double>& u,
        const vector<double>& dsigma,
        double threshold,
        double g)
{
    double uT = std::sqrt(u[1]*u[1] + u[2]*u[2]);
    double uToT = uT / T;

    double u1ouT = (uT > 1e-14 ? u[1] / uT : 0.0); // if uT ~ 0 u1/uT are not defined, make it 0.
    double u2ouT = (uT > 1e-14 ? u[2] / uT : 0.0); // if uT ~ 0 u1/uT are not defined, make it 0.

    // prefactor = 2π T^2   (no fugacity -- computed at the end)
    double prefacta = 2.0 * M_PI * T * T;

    // Conpute useful log of Laguerre weitghts to check and compute scaled weights.
    vector<double> logW(Nlag);
    for (int j = 0; j < Nlag; j++)
        logW[j] = std::log(W[j]);


    array<double,4> Nmu = IntegralFull_all(
        YM,
        Omega, Y, Nleg,
        X, W, logW, Nlag,
        m, T,
        u, dsigma,
        threshold,
        prefacta, u1ouT, u2ouT, uToT
    );

    // ---------------------------------------------------
    // 4. Contract with dsigma_μ
    // ---------------------------------------------------
    double contracted =
          Nmu[0] * dsigma[0]
        + Nmu[1] * dsigma[1]
        + Nmu[2] * dsigma[2]
        + Nmu[3] * dsigma[3];

    // ---------------------------------------------------
    // 5. Return final contribution
    // ---------------------------------------------------
    return contracted * g / (8.0 * M_PI * M_PI * M_PI);
}


// Acceptance space Legendre-Legendre integral ---------------------------------------------
// -----------------------------------------------------------------------------------------
//double alphaTilde(int mu, double pTm, double pTM, double T, double mu_i, vector<double>& u){
//    // This function computes the \tilde{\alpha} function appearing in the Legendre quadrature 
//    // to compute the particle yield in the acceptance. 
//    // Note: following the full space integral, not fugacity factor here also.
//    //double prefactor = M_PI * exp(mu_i/T) * (pTM - pTm); 
//    double prefactor = M_PI * (pTM - pTm); 
//    if(mu == 0){
//        return prefactor; 
//    }
//    else if(mu == 1){
//        double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
//        return prefactor * u[1] / uT; 
//    }
//    else if(mu == 2){
//        double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
//        return prefactor * u[2] / uT;
//    }
//    else if(mu == 3){
//        return prefactor;
//    }
//    else{
//        cout << "index in tilde bar must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}

//double alphaTilde(int mu, double prefactor, double u1ouT, double u2ouT){
//    // This function computes the \tilde{\alpha} function appearing in the Legendre quadrature 
//    // to compute the particle yield in the acceptance. 
//    // Note: following the full space integral, not fugacity factor here also.
//    //double prefactor = M_PI * exp(mu_i/T) * (pTM - pTm); 
//    if(mu == 0){
//        return prefactor; 
//    }
//    else if(mu == 1){
//        return prefactor * u1ouT; 
//    }
//    else if(mu == 2){
//        return prefactor * u2ouT;
//    }
//    else if(mu == 3){
//        return prefactor;
//    }
//    else{
//        cout << "index in tilde bar must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}

//double gtilde(int mu, double y){ 
//    // This function computes the \tilde{g} function appearing in the Legendre quadrature 
//    // to compute the particle yield in the acceptance. 
//    if(mu == 0){
//        return cosh(y); 
//    }
//    else if(mu == 1){
//        return 1.0; 
//    }
//    else if(mu == 2){
//        return 1.0;
//    }
//    else if(mu == 3){
//        return sinh(y);
//    }
//    else{
//        cout << "index in g tilde must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}

double GetmT(double treshold, double pT, double m){
    // This function checks the if pT is larger compared to the mass.
    // To avoid computing unecessary square roots for efficiency.
    if(pT/m < treshold){return sqrt(pT * pT + m * m);}
    else{return pT;}
}

//double ftilde(int mu, double pTm, double pTM, double x, double y, double m, double T, vector<double>& u, double threshold){
//    // This functions computes the \tilde{f} functions appearing in the Legendre quadrature. 
//    // Note the contrary to the Full space Laguerre quadrature, the modified Bessel functions are not scaled here.
//    // This is unecessary for finite size acceptance, these functions do not overflow. 
//    double Gamma = cosh(y) * u[0] - sinh(y) * u[3];
//    double pT = 0.5 *((pTM - pTm) * x + (pTM + pTm));
//    double mT = GetmT(threshold, pT, m);
//
//    double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
//
//    double prefactor = pT * exp(-mT/T * Gamma);
//
//    if(mu == 0){
//        return prefactor * mT * boost::math::cyl_bessel_i(0, pT/T * uT); 
//    }
//    else if(mu == 1){
//        return prefactor * pT * boost::math::cyl_bessel_i(1, pT/T * uT); 
//    }
//    else if(mu == 2){
//        return prefactor * pT * boost::math::cyl_bessel_i(1, pT/T * uT); 
//    }
//    else if(mu == 3){
//        return prefactor * mT * boost::math::cyl_bessel_i(0, pT/T * uT); 
//
//    }
//    else{
//        cout << "index in f bar must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}

//double ftilde(int mu, double pTm, double pTM, double x, double y, double m, double T, vector<double>& u, double threshold){
//    // This functions computes the \tilde{f} functions appearing in the Legendre quadrature. 
//    // Note the contrary to the Full space Laguerre quadrature, the modified Bessel functions are not scaled here.
//    // This is unecessary for finite size acceptance, these functions do not overflow. 
//    double Gamma = cosh(y) * u[0] - sinh(y) * u[3];
//    double pT = 0.5 *((pTM - pTm) * x + (pTM + pTm));
//    double mT = GetmT(threshold, pT, m);
//
//    double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
//
//    double prefactor = pT * exp(-mT/T * Gamma);
//
//    if(mu == 0){
//        return prefactor * mT * boost::math::cyl_bessel_i(0, pT/T * uT); 
//    }
//    else if(mu == 1){
//        return prefactor * pT * boost::math::cyl_bessel_i(1, pT/T * uT); 
//    }
//    else if(mu == 2){
//        return prefactor * pT * boost::math::cyl_bessel_i(1, pT/T * uT); 
//    }
//    else if(mu == 3){
//        return prefactor * mT * boost::math::cyl_bessel_i(0, pT/T * uT); 
//
//    }
//    else{
//        cout << "index in f bar must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}
//double ftilde(int mu, double prefactor, double pT, double mT, double pToTuT){
//    // This functions computes the \tilde{f} functions appearing in the Legendre quadrature. 
//    // Note the contrary to the Full space Laguerre quadrature, the modified Bessel functions are not scaled here.
//    // This is unecessary for finite size acceptance, these functions do not overflow. 
//
//    if(mu == 0){
//        return prefactor * mT * I0_fast(pToTuT);
//            //boost::math::cyl_bessel_i(0, pT/T * uT); 
//    }
//    else if(mu == 1){
//        return prefactor * pT * I1_fast(pToTuT); 
//            //boost::math::cyl_bessel_i(1, pT/T * uT); 
//    }
//    else if(mu == 2){
//        return prefactor * pT * I1_fast(pToTuT); 
//            //boost::math::cyl_bessel_i(1, pT/T * uT); 
//    }
//    else if(mu == 3){
//        return prefactor * mT *  I0_fast(pToTuT);
//            //boost::math::cyl_bessel_i(0, pT/T * uT); 
//
//    }
//    else{
//        cout << "index in f bar must be between 0 and 3" << endl;
//        return 0.0;
//    }
//}

bool checkdsigmamuXimuLEG(double at0, double at1, double at2, double at3, double gt0, double gt1, double gt2, double gt3, double ft0, double ft1, double ft2, double ft3, const vector<double>& dsigma){
    // This function is equivalent to checkdsigmamuXimu but is adapted for the use with the Legendre x Legendre integrals


    //double x0 = alphaTilde(0, prefactatidle, u1ouT, u2ouT) * gtilde(0, y) * ftilde(0, prefactftilde, pT, mT, pToTuT) * dsigma[0];
    //double x1 = alphaTilde(1, prefactatidle, u1ouT, u2ouT) * gtilde(1, y) * ftilde(1, prefactftilde, pT, mT, pToTuT) * dsigma[1];
    //double x2 = alphaTilde(2, prefactatidle, u1ouT, u2ouT) * gtilde(2, y) * ftilde(2, prefactftilde, pT, mT, pToTuT) * dsigma[2];
    //double x3 = alphaTilde(3, prefactatidle, u1ouT, u2ouT) * gtilde(3, y) * ftilde(3, prefactftilde, pT, mT, pToTuT) * dsigma[3];

    double x0 = at0 * gt0 * ft0 * dsigma[0];
    double x1 = at1 * gt1 * ft1 * dsigma[1];
    double x2 = at2 * gt2 * ft2 * dsigma[2];
    double x3 = at3 * gt3 * ft3 * dsigma[3];

    return (x0+x1+x2+x3>0);

    //vector<double> Xi;
    //for(int i = 0; i<4;i++){
    //    Xi.push_back(alphaTilde(i, pTm, pTM, T, mu_i, u) * gtilde(i, y) * ftilde(i, pTm, pTM, x, y, m, T, u, threshold));
    //}
    //return (dotCov(Xi, dsigma)>0);
}

inline array<double,4> gtilde_all(double y)
{
    double cy = cosh(y);
    double sy = sinh(y);

    return { cy, 1.0, 1.0, sy };
}

inline array<double,4> ftilde_all(
        double prefactftilde,   // pT * exp(-mT/T * Gamma)
        double pT,
        double mT,
        double pToTuT           // pT/T * uT
)
{
    // Compute BOTH scaled Bessels once.
    // pToTuT >= 0 always → safe for Bessel I.
    double I0 = I0_fast(pToTuT);
    double I1 = I1_fast(pToTuT);

    // Prefactor applied only once
    double pref_mT = prefactftilde * mT;
    double pref_pT = prefactftilde * pT;

    // Branchless return
    return {
        pref_mT * I0,
        pref_pT * I1,
        pref_pT * I1,
        pref_mT * I0 
    };
}
inline array<double,4> alphaTilde_all(
        double prefactor,
        double u1ouT,
        double u2ouT)
{
    return {
        prefactor,          // μ = 0
        prefactor * u1ouT,  // μ = 1
        prefactor * u2ouT,  // μ = 2
        prefactor           // μ = 3
    };
}

//array<double,4> IntegralAcc(double YM, double pTm, double pTM, const vector<double>& Omega, const vector<double>& Y, int Nleg, const vector<double>& X, const vector<double>& Om, 
//        double m, double T, const vector<double>& dsigma, const vector<double>& u, double threshold,
//        double prefactatilde, double u1ouT, double u2ouT, double uT){
//    // This function computes the Legendre quadratures.
//
//    double at0 = alphaTilde(0, prefactatilde, u1ouT, u2ouT);
//    double at1 = alphaTilde(1, prefactatilde, u1ouT, u2ouT);
//    double at2 = alphaTilde(2, prefactatilde, u1ouT, u2ouT);
//    double at3 = alphaTilde(3, prefactatilde, u1ouT, u2ouT);
//
//
//    double s0 = 0.0; 
//    double s1 = 0.0;  
//    double s2 = 0.0;
//    double s3 = 0.0;
//    double x, y;
//    double pT, mT, prefactftilde, pToTuT, Gamma;
//    double ft0, ft1, ft2, ft3;
//    double gt0, gt1, gt2, gt3;
//
//    for(int i = 0; i<Nleg; i++){
//        y = Y[i] * YM; // current rapidity, Legendre are calculated in [-1,1], times YM = [-YM, YM]
//        gt0 = gtilde(0, y);
//        gt1 = gtilde(1, y);
//        gt2 = gtilde(2, y);
//        gt3 = gtilde(3, y);
//        Gamma = cosh(y) * u[0] - sinh(y) * u[3];
//        for(int j = 0; j<Nleg; j++){
//            x = X[j];
//
//            pT = 0.5 *((pTM - pTm) * x + (pTM + pTm));
//            mT = GetmT(threshold, pT, m);
//            prefactftilde = pT * exp(-mT/T * Gamma);
//            pToTuT = pT/T*uT;
//
//            ft0 = ftilde(0, prefactftilde, pT, mT, pToTuT);
//            ft1 = ftilde(1, prefactftilde, pT, mT, pToTuT);
//            ft2 = ftilde(2, prefactftilde, pT, mT, pToTuT);
//            ft3 = ftilde(3, prefactftilde, pT, mT, pToTuT);
//
//            if(checkdsigmamuXimuLEG(at0, at1, at2, at3, gt0, gt1, gt2, gt3, ft0, ft1, ft2, ft3, dsigma)){
//                        //y, prefactatilde, u1ouT, u2ouT, prefactftilde, pT, mT, pToTuT, dsigma)){
//                s0 += Omega[i] * gt0 * Omega[j] * ft0;
//                s1 += Omega[i] * gt1 * Omega[j] * ft1;
//                s2 += Omega[i] * gt2 * Omega[j] * ft2;
//                s3 += Omega[i] * gt3 * Omega[j] * ft3;
//            }
//        }
//    }
//    array<double,4> out;
//    out[0] = at0 * YM * s0;
//    out[1] = at1 * YM * s1;
//    out[2] = at2 * YM * s2;
//    out[3] = at3 * YM * s3;
//
//    return out;
//    //return alphaTilde(mu, pTm, pTM, T, mu_i, u) * YM * s;
//}

array<double,4> IntegralAcc(
        double YM, double pTm, double pTM,
        const vector<double>& Omega, const vector<double>& Y, int Nleg,
        const vector<double>& X, const vector<double>& Om, 
        double m,double T,
        const vector<double>& dsigma, const vector<double>& u,
        double threshold,
        double prefactatilde, double u1ouT, double u2ouT, double uT)
{
    const auto at = alphaTilde_all(prefactatilde, u1ouT, u2ouT);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    double y, Gamma, x, pT, mT, prefactftilde, pToTuT;
    for (int i = 0; i < Nleg; i++)
    {
        y = Y[i] * YM;
        Gamma = cosh(y) * u[0] - sinh(y) * u[3];
        const auto gt = gtilde_all(y);
        const double Oi = Omega[i];

        for (int j = 0; j < Nleg; j++)
        {
            x = X[j];
            pT = 0.5 * ((pTM - pTm)*x + (pTM + pTm));
            mT = GetmT(threshold, pT, m);
            prefactftilde = pT * exp(-mT/T * Gamma);
            pToTuT = (pT/T) * uT;
            const auto ft = ftilde_all(prefactftilde, pT, mT, pToTuT);
            if (checkdsigmamuXimuLEG(at[0], at[1], at[2], at[3],
                                     gt[0], gt[1], gt[2], gt[3],
                                     ft[0], ft[1], ft[2], ft[3],
                                     dsigma))
            {
                double Oj = Omega[j];
                s0 += Oi * gt[0] * Oj * ft[0];
                s1 += Oi * gt[1] * Oj * ft[1];
                s2 += Oi * gt[2] * Oj * ft[2];
                s3 += Oi * gt[3] * Oj * ft[3];
            }
        }
    }
    return {
        at[0] * YM * s0,
        at[1] * YM * s1,
        at[2] * YM * s2,
        at[3] * YM * s3
    };
}

//double deltaNcellAcc(double YM, double pTm, double pTM, const vector<double>& Omega, const vector<double>& Y, int Nleg, const vector<double>& X, const vector<double>& Om, double m, double T, double mu_i, vector<double>& u, vector<double>& dsigma, double threshold, double g=1.0){
//    // This function computes the total particle yield in acceptance.
//
//    double Nmu0 = IntegralAcc(YM, pTm, pTM, Omega, Y, Nleg, X, Om, 0, m, T, mu_i, dsigma, u, threshold) * dsigma[0];
//    double Nmu1 = IntegralAcc(YM, pTm, pTM, Omega, Y, Nleg, X, Om, 1, m, T, mu_i, dsigma, u, threshold) * dsigma[1];
//    double Nmu2 = IntegralAcc(YM, pTm, pTM, Omega, Y, Nleg, X, Om, 2, m, T, mu_i, dsigma, u, threshold) * dsigma[2];
//    double Nmu3 = IntegralAcc(YM, pTm, pTM, Omega, Y, Nleg, X, Om, 3, m, T, mu_i, dsigma, u, threshold) * dsigma[3];
//
//    return (Nmu0 + Nmu1 + Nmu2 + Nmu3)*g/(8 * M_PI * M_PI * M_PI);
//    // Need covariant dsigma in cartesian coordinates.
//    
//    //vector<double> N;
//    //for(int i = 0; i < 4; i++){
//    //   N.push_back(IntegralAcc(YM, pTm, pTM, Omega, Y, Nleg, X, Om, i, m, T, mu_i, dsigma, u, threshold)); 
//    //}
//    //// Need covariant dsigma in cartesian coordinates.
//    //return dotCov(dsigma, N, g/(8 * M_PI * M_PI * M_PI));
//}

double deltaNcellAcc(double YM,double pTm, double pTM,
        const vector<double>& Omega, const vector<double>& Y, int Nleg,
        const vector<double>& X, const vector<double>& Om, 
        double m, double T, const vector<double>& u, const vector<double>& dsigma,
        double threshold, double g = 1.0)
{
    double prefactatilde = M_PI * (pTM - pTm);
    double uT = sqrt(u[1]*u[1] + u[2]*u[2]);
    double u1ouT = (uT > 1e-14 ? u[1] / uT : 0.0);
    double u2ouT = (uT > 1e-14 ? u[2] / uT : 0.0);

    array<double,4> Nmu = IntegralAcc(YM, pTm, pTM,
            Omega, Y, Nleg, X, Om,
            m, T, dsigma, u, threshold,
            prefactatilde, u1ouT, u2ouT, uT);

    double contracted =
        Nmu[0] * dsigma[0] +
        Nmu[1] * dsigma[1] +
        Nmu[2] * dsigma[2] +
        Nmu[3] * dsigma[3];

    return contracted * g / (8.0 * M_PI * M_PI * M_PI);
}

double analytic_number_density(double m, double T, double mu, int g = 2)
{
    const double hbarc = 0.1973269804; // GeV*fm
    // m/T
    double z = m / T;
    // Bessel K2
    double K2 = boost::math::cyl_bessel_k(2, z);  // C++17/20
    // density in GeV^3:
    double n_GeV3 = g / (2.0 * M_PI * M_PI) * m*m * T * K2 * exp(mu / T);
    // convert to fm^-3:
    double n_fm3 = n_GeV3 / std::pow(hbarc, 3);
    return n_fm3;
}

double Test(Acceptance Full, SurfaceElement Surf){
    // For this test function to work, one needs to set mu = 1, mu = 2 to 0 in function
    // alphabar. Otherwise having 0 for u^mu leads to nan.
    double threshold = 1000.0;
    double m = 0.938;
    double hbarc = 0.1973269804; // GeV*fm

    double T = Surf.T_f; 
    double mu_i = Surf.mu_B;
    vector<double> u = {1.0, 0, 0, 0};

    // dsigma should be covariant
    // here it's in the rest frame so it's fine.
    vector<double> dsigma = {1.0, 0.0, 0.0, 0.0};

    NumericalIntegration NumInt;
    vector<double> XLeg, WLeg, XLag, WLag;

    NumInt.GetGaussLegendreCT32(XLeg, WLeg);    
    NumInt.GetGaussLaguerreCT32(XLag, WLag);    

    int Nleg = XLeg.size();
    int Nlag = XLag.size();

    double YM = Full.ymax;

    double Analytic = analytic_number_density(m, T, mu_i);
    //cout << "Num Int : " << deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, mu_i, u, dsigma, threshold, 2) * exp(mu_i / T)  / (hbarc * hbarc * hbarc)  << "Analytic Int : " << Analytic << " T : " << T << " muB " << mu_i << endl; 
    cout << "Num Int : " << deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, u, dsigma, threshold, 2) * exp(mu_i / T)  / (hbarc * hbarc * hbarc)  << "Analytic Int : " << Analytic << " T : " << T << " muB " << mu_i << endl; 
    return 1.0;
}

double TestWithTransverseFlow(Acceptance Full, SurfaceElement Surf)
{
    double m = 0.938;
    double hbarc = 0.1973269804; // GeV*fm

    double T = Surf.T_f; 
    double mu_i = Surf.mu_B;
    double vT = 0.5;  // choose some transverse velocity
    double gammaT = 1.0 / std::sqrt(1.0 - vT*vT);

    // Already in cartesian coordinates.
    // fluid 4-velocity in lab frame
    std::vector<double> u = {gammaT, gammaT*vT, 0.0, 0.0};

    // comoving time-like hypersurface: dsigma^mu ∝ u^mu
    double dV = 1.0;  // fm^3
    // get dsigma covariant.
    std::vector<double> dsigma = {gammaT*dV, -gammaT*vT*dV, 0.0, 0.0};

    NumericalIntegration NumInt;
    std::vector<double> XLeg, WLeg, XLag, WLag;
    NumInt.GetGaussLegendreCT32(XLeg, WLeg);
    NumInt.GetGaussLaguerreCT32(XLag, WLag);

    int Nleg = XLeg.size();
    int Nlag = XLag.size();

    double YM = Full.ymax;  // for a *true* test, make sure this mimics full y range

    double threshold = 1000.0;

    //double num_int = deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, mu_i, u, dsigma, threshold, 2) * exp(mu_i/T) / (hbarc * hbarc * hbarc);
    double num_int = deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, u, dsigma, threshold, 2) * exp(mu_i/T) / (hbarc * hbarc * hbarc);

    double analytic = analytic_number_density(m, T, mu_i);  // n(T, mu)*dV

    std::cout << "vT = " << vT
              << "  Num Int : " << num_int 
              << "  Analytic : " << analytic
              << "  ratio = " << num_int/analytic
              << std::endl;

    return num_int;
}

//vector<double> GetProbabilityProtons(Acceptance Acc, Acceptance Full, SurfaceElement Surf){
//    // This function computes the probability p = NpAcc/NpFull representing the probability 
//    // of net proton to end up in the acceptance for one hydro cell.
//    // It also returns the Net proton yields in acceptance and in the full space (for checks).
//    vector<double> p;
//    p.resize(9);
//
//    const double hbarc = 0.1973269804; // GeV*fm
//    double threshold = 1000.0;
//    double m = 0.938;
//    double gp = 2;
//
//    double T = Surf.T_f; 
//    double mu_p = Surf.mu_B + Surf.mu_C;
//    vector<double> u = {Surf.u[0], Surf.u[1], Surf.u[2], Surf.u[3]};
//
//    vector<double> dsigma = {Surf.s[0], Surf.s[1], Surf.s[2], Surf.s[3]};
//
//    vector<double> uC = MilneToCartesian(u, Surf);
//    vector<double> dsigmaC = MilneToCartesianSigmaCov(dsigma, Surf);
//
//
//    NumericalIntegration NumInt;
//    vector<double> XLeg, WLeg, XLag, WLag;
//
//    NumInt.GetGaussLegendreCT32(XLeg, WLeg);    
//    NumInt.GetGaussLaguerreCT32(XLag, WLag);    
//
//    int Nleg = XLeg.size();
//    int Nlag = XLag.size();
//
//    double YM = Full.ymax;
//    double YMAcc = Acc.ymax;
//
//    double pTm = Acc.pTmin;
//    double pTM = Acc.pTmax;
//
//    double NpFull;
//    double NpAcc;
//
//    NpFull = deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, mu_p, uC, dsigmaC, threshold, gp);
//    NpAcc = deltaNcellAcc(YMAcc, pTm, pTM, WLeg, XLeg, Nleg, XLeg, WLeg, m, T, mu_p, uC, dsigmaC, threshold, gp);
//
//    // alpha
//    p[0] = NpAcc/NpFull;
//    // net proton Acc
//    p[1] = 2 * NpAcc * sinh(mu_p/T) / (hbarc * hbarc * hbarc);
//    // net proton Full
//    p[2] = 2 * NpFull * sinh(mu_p/T) / (hbarc * hbarc * hbarc);
//    // sum proton Acc
//    p[3] = 2 * NpAcc * cosh(mu_p/T) / (hbarc * hbarc * hbarc);
//    // sum proton Full
//    p[4] = 2 * NpFull * cosh(mu_p/T) / (hbarc * hbarc * hbarc);
//    // protons Acc
//    p[5] = (p[1] + p[3])/2;
//    // antiprotons Acc
//    p[6] = (p[3] - p[1])/2;
//    // protons Full
//    p[7] = (p[2] + p[4])/2;
//    // antiprotons Full
//    p[8] = (p[4] - p[2])/2;
//
//    return p;
//}



bool GetProbabilityProtons(Acceptance Acc, Acceptance Full, SurfaceElement Surf, double* p, 
        const vector<double>& XLeg,
        const vector<double>& WLeg,
        const vector<double>& XLag,
        const vector<double>& WLag,
        int NLeg, int NLag){
    // This function computes the probability p = NpAcc/NpFull representing the probability 
    // of net proton to end up in the acceptance for one hydro cell.
    // It also returns the Net proton yields in acceptance and in the full space (for checks).

    // Put this out
    double hbarc = 0.1973269804; // GeV*fm
    double threshold = 1000.0;
    double m = 0.938;
    double gp = 2;

    // Compute useful cell infos here
    double T = Surf.T_f; 
    double mu_p = Surf.mu_B + Surf.mu_C;
    vector<double> u = {Surf.u[0], Surf.u[1], Surf.u[2], Surf.u[3]};
    vector<double> dsigma = {Surf.s[0], Surf.s[1], Surf.s[2], Surf.s[3]};
    vector<double> uC = MilneToCartesian(u, Surf);
    vector<double> dsigmaC = MilneToCartesianSigmaCov(dsigma, Surf);

    double YM = Full.ymax;
    double YMAcc = Acc.ymax;

    double pTm = Acc.pTmin;
    double pTM = Acc.pTmax;

    double NpFull;
    double NpAcc;

    //NpFull = deltaNcellFull(YM, WLeg, XLeg, NLeg, XLag, WLag, NLag, m, T, mu_p, uC, dsigmaC, threshold, gp);
    NpFull = deltaNcellFull(YM, WLeg, XLeg, NLeg, XLag, WLag, NLag, m, T, uC, dsigmaC, threshold, gp);
    //NpAcc = deltaNcellAcc(YMAcc, pTm, pTM, WLeg, XLeg, NLeg, XLeg, WLeg, m, T, mu_p, uC, dsigmaC, threshold, gp);
    NpAcc = deltaNcellAcc(YMAcc, pTm, pTM, WLeg, XLeg, NLeg, XLeg, WLeg, m, T, uC, dsigmaC, threshold, gp);

    // alpha
    p[0] = NpAcc/NpFull;
    // net proton Acc
    p[1] = 2 * NpAcc * sinh(mu_p/T) / (hbarc * hbarc * hbarc);
    // net proton Full
    p[2] = 2 * NpFull * sinh(mu_p/T) / (hbarc * hbarc * hbarc);
    // sum proton Acc
    p[3] = 2 * NpAcc * cosh(mu_p/T) / (hbarc * hbarc * hbarc);
    // sum proton Full
    p[4] = 2 * NpFull * cosh(mu_p/T) / (hbarc * hbarc * hbarc);
    // protons Acc
    p[5] = (p[1] + p[3])/2;
    // antiprotons Acc
    p[6] = (p[3] - p[1])/2;
    // protons Full
    p[7] = (p[2] + p[4])/2;
    // antiprotons Full
    p[8] = (p[4] - p[2])/2;


    // Manage negative contributions, avoid cells if they have negative probabilities.
    if(p[0] < 0.0){
        return false;
    }
    else if(p[0] > 1.0){
        return false; 
    }
    else{
        return true;
    }
}

//double GetFullNB(Acceptance Full, SurfaceElement Surf){
//    // This function computes the probability p = NAcc/NFull representing the probability 
//    // of ending up in the acceptance for one hydro cell.
//    // It also returns the yields in acceptance and in the full space (for checks).
//    double threshold = 1000.0;
//    double m = 0.938;
//
//    double T = Surf.T_f; 
//    double mu_i = Surf.mu_B;
//    vector<double> u = {Surf.u[0], Surf.u[1], Surf.u[2], Surf.u[3]};
//
//    vector<double> dsigma = {Surf.s[0], Surf.s[1], Surf.s[2], Surf.s[3]};
//
//    //vector<double> uC = MilneToCartesian(u, Surf);
//    //vector<double> dsigmaC = MilneToCartesian(dsigma, Surf);
//
//    vector<double> uC = MilneToCartesian(u, Surf);
//    vector<double> dsigmaC = MilneToCartesianSigmaCov(dsigma, Surf);
//
//    NumericalIntegration NumInt;
//    vector<double> XLeg, WLeg, XLag, WLag;
//
//    NumInt.GetGaussLegendreCT32(XLeg, WLeg);    
//    NumInt.GetGaussLaguerreCT32(XLag, WLag);    
//
//    int Nleg = XLeg.size();
//    int Nlag = XLag.size();
//
//    double YM = Full.ymax;
//
//    double NpFull;
//    double gp = 2;
//    double gDelt = 4;
//
//    NpFull = deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, mu_i, uC, dsigmaC, threshold, gp);
//    double mu_p = Surf.mu_B + Surf.mu_C;
//    double mu_n = Surf.mu_B;
//
//
//    double NpFullDelta = deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, 1.232, T, mu_i, uC, dsigmaC, threshold, gDelt);
//    double mudpp = Surf.mu_B + 2 * Surf.mu_C;
//    double mudp = Surf.mu_B + Surf.mu_C;
//    double mud0 = Surf.mu_B;
//    double mudm = Surf.mu_B - Surf.mu_C;
//
//
//
//
//    return 2 * NpFull * (sinh(mu_p/T) + sinh(mu_n/T))
//        + 2 * NpFullDelta * (sinh(mudpp/T) + sinh(mudp/T) + sinh(mud0/T) + sinh(mudm/T));
//    //return NpFull * exp(mu_n/T);
//
//}
double GetFullNB(Acceptance Full, SurfaceElement Surf){
    // This function computes the probability p = NAcc/NFull representing the probability 
    // of ending up in the acceptance for one hydro cell.
    // It also returns the yields in acceptance and in the full space (for checks).
    double threshold = 100.0;
    double m = 0.938;

    double T = Surf.T_f; 
    double mu_i = Surf.mu_B;
    vector<double> u = {Surf.u[0], Surf.u[1], Surf.u[2], Surf.u[3]};

    vector<double> dsigma = {Surf.s[0], Surf.s[1], Surf.s[2], Surf.s[3]};

    vector<double> uC = MilneToCartesian(u, Surf);
    vector<double> dsigmaC = MilneToCartesianSigmaCov(dsigma, Surf);

    NumericalIntegration NumInt;
    vector<double> XLeg, WLeg, XLag, WLag;

    NumInt.GetGaussLegendreCT32(XLeg, WLeg);    
    NumInt.GetGaussLaguerreCT32(XLag, WLag);    

    int Nleg = XLeg.size();
    int Nlag = XLag.size();

    double YM = Full.ymax;

    double NpFull;
    double gp = 2;


    NpFull = deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, uC, dsigmaC, threshold, gp);
    double mu_p = Surf.mu_B + Surf.mu_C;
    double mu_n = Surf.mu_B;


    return 2 * NpFull * (sinh(mu_p/T) + sinh(mu_n/T));

}
double GetAccNB(Acceptance Acc, SurfaceElement Surf){
    // This function computes the probability p = NAcc/NFull representing the probability 
    // of ending up in the acceptance for one hydro cell.
    // It also returns the yields in acceptance and in the full space (for checks).
    double threshold = 100.0;
    double m = 0.938;

    double T = Surf.T_f; 
    double mu_i = Surf.mu_B;
    vector<double> u = {Surf.u[0], Surf.u[1], Surf.u[2], Surf.u[3]};

    vector<double> dsigma = {Surf.s[0], Surf.s[1], Surf.s[2], Surf.s[3]};

    const vector<double> uC = MilneToCartesian(u, Surf);
    const vector<double> dsigmaC = MilneToCartesianSigmaCov(dsigma, Surf);

    NumericalIntegration NumInt;
    vector<double> XLeg, WLeg;

    NumInt.GetGaussLegendreCT32(XLeg, WLeg);    

    int NLeg = XLeg.size();

    double YMAcc = Acc.ymax;
    double pTm = Acc.pTmin;
    double pTM = Acc.pTmax;

    double gp = 2;


    double NpAcc = deltaNcellAcc(YMAcc, pTm, pTM, WLeg, XLeg, NLeg, XLeg, WLeg, m, T, uC, dsigmaC, threshold, gp);

    double mu_p = Surf.mu_B + Surf.mu_C;
    double mu_n = Surf.mu_B;


    return 2 * NpAcc * (sinh(mu_p/T) + sinh(mu_n/T));

}

int main(int argc, char *argv[]) {

    // Read user imputs.
    string energy = argv[1];
    string centrality = argv[2];

    // Read Surface file
    const string SurfacePath = "surfacesFiles/"; 
    //const string fname = "AuAu"+energy+"/hydro_results_C"+centrality+"/surface_eps_0.26.dat";
    const string fname = "PbPb"+energy+"/hydro_results_C"+centrality+"/surface_eps_0.2.dat";

    vector<SurfaceElement> surface;

    //cout << "Read Surface : " << endl;
    surface = ReadFreezeOutSurface(SurfacePath+fname); 

    int Nmax = -1;
    // Remove negative 4-volume cells.
    vector<SurfaceElement> selectedCells = selectPositiveVolumeCells(surface, Nmax);

    //Define acceptance and Full space ranges.
    Acceptance Acc;
    //Acc.pTmin = 0.4; Acc.pTmax = 2.0;
    //Acc.ymin = -0.5; Acc.ymax = 0.5;
    // Checked that this, gives the same as the full space
    Acc.pTmin = 0.4; Acc.pTmax = 2.0;
    Acc.ymin = -0.5; Acc.ymax = 0.5;

    Acceptance FullSpace;
    FullSpace.pTmin = 0.0; FullSpace.pTmax = 100; // in Laguerre this is always [0, +infinity[
    FullSpace.ymin = -4.0; FullSpace.ymax = 4.0;// |y|= 4 is enough  (and sweet spot actually)

    vector<double> pvec;
    bool OutGet;

    // Setup parallelization
    size_t n_cells = selectedCells.size();
    size_t n_double_per_cell = 9;
    size_t cells_per_thread = n_cells / omp_get_max_threads() + 1;

    cout << "Total number of cells " << n_cells << endl;
    cout << "Start running on " <<  omp_get_max_threads() << " threads" << endl;
    cout << "\n" << endl;

    // Prepare integration
    NumericalIntegration NumInt;
    vector<double> XLeg, WLeg, XLag, WLag;

    NumInt.GetGaussLegendreCT32(XLeg, WLeg);    
    NumInt.GetGaussLaguerreCT32(XLag, WLag);    

    int NLeg = XLeg.size();
    int NLag = XLag.size();


    /// HERE
    vector<double> OUT;
    vector<bool> OUT_CHECK;
    OUT_CHECK.resize(n_cells);
    OUT.resize(n_cells * n_double_per_cell);
    atomic<size_t> counter(0);

    //size_t DebugCell = 1411080;
    #pragma omp parallel
    {
        size_t local_count = 0;
        #pragma omp for schedule(static)
        for (size_t k = 0; k < selectedCells.size(); ++k) {
        //for (size_t k = DebugCell-100000; k < DebugCell+100000; ++k) {
            bool OutGet = GetProbabilityProtons(Acc, FullSpace, selectedCells[k], &OUT[k * n_double_per_cell], XLeg, WLeg, XLag, WLag, NLeg, NLag);
            OUT_CHECK[k] = OutGet; 
            ++local_count;

            if (local_count % 1000 == 0) {
                counter += 1000;
                if (omp_get_thread_num() == 0)
                    cout << "\rProgress: " << (100.0 * counter / selectedCells.size()) << "%   " << std::flush;
            }
        }
    }

    // Write in a file.
    double pvec0, pvec1, pvec2;
    double pvec3, pvec4, pvec5;
    double pvec6, pvec7, pvec8;
    double tau, x, y, eta;
    ofstream outFile("p_values_"+energy+"_"+centrality+".txt");

    outFile << "# alpha netpAcc netpFull sumpAcc sumpFull protonAcc antiprotonAcc protonFull antiprotonFull" << "\n"; 
    for(int j = 0; j < OUT_CHECK.size(); j++){
        pvec0 = OUT[j*n_double_per_cell];
        pvec1 = OUT[j*n_double_per_cell+1];
        pvec2 = OUT[j*n_double_per_cell+2];

        pvec3 = OUT[j*n_double_per_cell+3];
        pvec4 = OUT[j*n_double_per_cell+4];
        pvec5 = OUT[j*n_double_per_cell+5];

        pvec6 = OUT[j*n_double_per_cell+6];
        pvec7 = OUT[j*n_double_per_cell+7];
        pvec8 = OUT[j*n_double_per_cell+8];

        tau = selectedCells[j].x[0];
        x = selectedCells[j].x[1];
        y = selectedCells[j].x[2];
        eta = selectedCells[j].x[3];

        OutGet = OUT_CHECK[j];
        outFile << pvec0 << "  " << pvec1 << "  " << pvec2 << " " << pvec3 << " " << pvec4 << " " 
            << pvec5 << " " << pvec6 << " " << pvec7 << " " << pvec8 << " " << OutGet << " " << tau << " " << x << " " << y << " " << eta << "\n"; 
    }
    outFile.close();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Test compared to analytical formula, validation of the integration process.
    //int DebugCell = 29865;
    //double Int = Test(FullSpace, selectedCells[DebugCell]);
    //double IntFlow = TestWithTransverseFlow(FullSpace, selectedCells[DebugCell]);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////


    //vector<double> p;
    //int DebugCell = 1411080;
    ////int i = 1411080-10000;
    ////p = GetProbabilityProtons(Acc, FullSpace, selectedCells[i]);
    ////for(auto& e:p){cout << e << " || ";}
    ////cout << selectedCells[i].T_f << " " << selectedCells[i].mu_B << " " << selectedCells[i].mu_C << endl;

    //for(int i = DebugCell-10000; i < DebugCell+10000; i++){
    //p = GetProbabilityProtons(Acc, FullSpace, selectedCells[i]);
    //cout << DebugCell + i << " ----------------------------------------- " << endl;
    ////for(auto& e:p){cout << e << " || ";}
    ////cout << selectedCells[i].T_f << " " << selectedCells[i].mu_B << " " << selectedCells[i].mu_C << endl;
    //}

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Test separate Acc, Full
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //SurfaceElement cell;
    //double eta;
    //double Int1;
    //double Int2;
    //double s1 = 0.0;;
    //double s2 = 0.0;;
    //for(int i = 0; i<selectedCells.size(); i++){
    //    cell = selectedCells[i]; 
    //    Int1 = GetFullNB(FullSpace, selectedCells[i]);
    //    Int2 = GetAccNB(Acc, selectedCells[i]);
    //    //if(i%1000==0){cout << "cell done : "<< ((double) i)/((double) selectedCells.size()) * 100.0 << " | sum Full : " << s1 << " | Nblocal Full : " << Int1/(0.1973 * 0.1973 * 0.1973) << " | sum Acc : " << s2 << " | Nblocal Acc : " << Int2/(0.1973 * 0.1973 * 0.1973) << endl;}
    //    //if(i%1000==0){cout << "cell done : "<< ((double) i)/((double) selectedCells.size()) * 100.0 << " | sum Full : " << s1 << " | sum Acc : " << s2 << endl;}
    //    //if(i%1000==0){cout  << i << " " << s1 << " " << s2 << endl;}
    //    if(i%100==0){cout  << i << " " << s2/s1 << endl;}
    //    s1+= Int1/(0.1973 * 0.1973 * 0.1973);
    //    s2+= Int2/(0.1973 * 0.1973 * 0.1973);
    //}
    //cout << "Total sum 1 : " << s1 << endl;
    //cout << "Total sum 2 : " << s2 << endl;
    return 0;
}

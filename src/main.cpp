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
double alphaBar(int mu, double T, double mu_i, vector<double>& u){
    // This function computes the \bar{\alpha} function for its use 
    // in the Laguerre integration.
    double uT; // Transverse 4-velocity of flow
    // Here, the definition does not contain fugacity factor.
    //double prefactor = 2 * M_PI * exp(mu_i/T) * T * T; 
    double prefactor = 2 * M_PI * T * T; 
    if(mu == 0){
        return prefactor; 
    }
    else if(mu == 1){
        uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
        return prefactor * u[1] / uT; 
        //return 0.0; For Test function 
    }
    else if(mu == 2){
        uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
        return prefactor * u[2] / uT;
        //return 0.0; For Test function 
    }
    else if(mu == 3){
        return prefactor;

    }
    else{
        cout << "index in alpha bar must be between 0 and 3" << endl;
        return 0.0;
    }
}

double gbar(int mu, double y, double m, double T, vector<double>& u){ 
    double Gamma = cosh(y) * u[0] - sinh(y) * u[3];
    double prefactor = exp(-m/T * Gamma)/(Gamma * Gamma);

    if(mu == 0){
        return prefactor * cosh(y); 
    }
    else if(mu == 1){
        return prefactor; 
    }
    else if(mu == 2){
        return prefactor;
    }
    else if(mu == 3){
        return prefactor * sinh(y);
    }
    else{
        cout << "index in g bar must be between 0 and 3" << endl;
        return 0.0;
    }
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

double fbar_scaled(int mu, double x, double y, double m, double T, vector<double>& u, double threshold){
    // This function coomputes the \bar{f} functions in the Laguerre integration. 
    // Note that we use the scaled Bessel functions. 
    double Gamma = cosh(y) * u[0] - sinh(y) * u[3];
    double prefactor = x + m/T*Gamma;

    double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 

    double mT = T/Gamma * x + m;
    //double pT = GetpT(threshold, mT, m);
    double pT = sqrt(mT * mT - m * m);

    if(mu == 0){
        return prefactor * mT * Modified_Bessel_scaled(0, pT/T * uT); 
    }
    else if(mu == 1){
        return prefactor * pT * Modified_Bessel_scaled(1, pT/T * uT); 
    }
    else if(mu == 2){
        return prefactor * pT * Modified_Bessel_scaled(1, pT/T * uT); 
    }
    else if(mu == 3){
        return prefactor * mT * Modified_Bessel_scaled(0, pT/T * uT); 
    }
    else{
        cout << "index in f bar must be between 0 and 3" << endl;
        return 0.0;
    }
}

bool checkdsigmamuXimu(double x, double y, double m, double T, double mu_i, vector<double>& u, vector<double>& dsigma, double threshold){
    // This function checks if the calculated 4-momentum is in the same direction as the outward cell 4-vector dsigma^mu
    // It corresponds to the actual positive contributions to the particle production. In principle, negative contributions 
    // corresponds to particles flowing back to the fluid. 
    // It is different from the condition u^mu dsigma_mu > 0 (hydro flow towards the detectors) as u^mu and p^mu are fairly 
    // independent. 
    // This function ensures that the ratio of number of particles inside and outside acceptance actually makes sense. 
    // Note that the use of scaled modified Bessel functions in fbar_scaled is not an issue as the norm is an 
    // an exponential function e(pT/T ||uT||) > 0 that is the same for all Xk0 here, it does not change the 
    // sign check. 

    vector<double> Xi;
    for(int i = 0; i<4;i++){
        Xi.push_back(alphaBar(i, T, mu_i, u) * gbar(i, y, m, T, u) * fbar_scaled(i, x, y, m, T, u, threshold));
    }
    // Note: we use the dotCov function here because after translation into Cartesian coordinates, dsigma is still covariant.
    return (dotCov(Xi, dsigma)>0);
}

double getScaledW(double x, double y, double T, double m, double WLag, vector<double>& u, double threshold){
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
    
    double Gamma = cosh(y) * u[0] - sinh(y) * u[3];
    double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
    double mT = T/Gamma * x + m;
    //double pT = GetpT(threshold, mT, m);
    double pT = sqrt(mT *mT - m * m);

    double expon = pT / T * uT;
    double check = log(WLag) + pT / T * uT; 

    if(check > 50){
        cout << "Very large value of w * e^(pT / T uT), exponant value : " << check << endl;
        return 0.0;
    } 
    else if(check < -40){ // exp(-37) ~ 1e-16.
        return 0.0;
    }
    else{
        if(expon > 700){ // Machine limit
            // Since -40 < log(WLag) + expon < 50, if expon > 700 it means log(WLag) < -750.....
            return 0.0;
        }
        else{
            return WLag * exp(expon);
        }
    }
}

double IntegralFull(double YM, vector<double>& Omega, vector<double>& Y, int Nleg, vector<double>& X, vector<double>& W, int Nlag, int mu, double m, double T, double mu_i, vector<double>& u, vector<double>& dsigma, double threshold){
    // This function computes the Laguerre+Legendre quadrature to coompute integrals in the full space. 
    double s = 0.0;
    double x, y, scaledW;
    double gb, fb, LegO;
    for(int i = 0; i<Nleg; i++){
        y = Y[i] * YM; // current rapidity, Legendre are calculated in [-1,1], times YM = [-YM, YM]
        for(int j = 0; j<Nlag; j++){
            x = X[j];
            // Check if the cell will contribute positively to the integrals.
            if(checkdsigmamuXimu(x, y, m, T, mu_i, u, dsigma, threshold)){
                // Compute scaled Laguerre weight for numerical regularization. 
                scaledW = getScaledW(x, y, T, m, W[j], u, threshold);
                gb = gbar(mu, y, m, T, u);
                fb = fbar_scaled(mu, x, y, m, T, u, threshold);
                s += Omega[i] * gb * scaledW * fb;
            }
        }
    }
    return alphaBar(mu, T, mu_i, u) * YM * s;
}

double deltaNcellFull(double YM, vector<double>& Omega, vector<double>& Y, int Nleg, vector<double>& X, vector<double>& W, int Nlag, double m, double T, double mu_i, vector<double>& u, vector<double>& dsigma, double threshold, double g=1.0){
    // This function computes the final contributions from the cells in the full acceptance. 
    vector<double> Nmu;
    for(int i = 0; i<4; i++){
        Nmu.push_back(IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, i, m, T, mu_i, u, dsigma, threshold));
    }
    return dotCov(Nmu, dsigma, g/(8 * M_PI * M_PI * M_PI)); // g degeneracy factor
}

// Acceptance space Legendre-Legendre integral ---------------------------------------------
// -----------------------------------------------------------------------------------------
double alphaTilde(int mu, double pTm, double pTM, double T, double mu_i, vector<double>& u){
    // This function computes the \tilde{\alpha} function appearing in the Legendre quadrature 
    // to compute the particle yield in the acceptance. 
    // Note: following the full space integral, not fugacity factor here also.
    //double prefactor = M_PI * exp(mu_i/T) * (pTM - pTm); 
    double prefactor = M_PI * (pTM - pTm); 
    if(mu == 0){
        return prefactor; 
    }
    else if(mu == 1){
        double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
        return prefactor * u[1] / uT; 
    }
    else if(mu == 2){
        double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
        return prefactor * u[2] / uT;
    }
    else if(mu == 3){
        return prefactor;
    }
    else{
        cout << "index in tilde bar must be between 0 and 3" << endl;
        return 0.0;
    }
}

double gtilde(int mu, double y){ 
    // This function computes the \tilde{g} function appearing in the Legendre quadrature 
    // to compute the particle yield in the acceptance. 
    double prefactor = 1.0;
    if(mu == 0){
        return prefactor * cosh(y); 
    }
    else if(mu == 1){
        return prefactor; 
    }
    else if(mu == 2){
        return prefactor;
    }
    else if(mu == 3){
        return prefactor * sinh(y);
    }
    else{
        cout << "index in g tilde must be between 0 and 3" << endl;
        return 0.0;
    }
}

double GetmT(double treshold, double pT, double m){
    // This function checks the if pT is larger compared to the mass.
    // To avoid computing unecessary square roots for efficiency.
    if(pT/m < treshold){return sqrt(pT * pT + m * m);}
    else{return pT;}
}

double ftilde(int mu, double pTm, double pTM, double x, double y, double m, double T, vector<double>& u, double threshold){
    // This functions computes the \tilde{f} functions appearing in the Legendre quadrature. 
    // Note the contrary to the Full space Laguerre quadrature, the modified Bessel functions are not scaled here.
    // This is unecessary for finite size acceptance, these functions do not overflow. 
    double Gamma = cosh(y) * u[0] - sinh(y) * u[3];
    double pT = 0.5 *((pTM - pTm) * x + (pTM + pTm));
    double mT = GetmT(threshold, pT, m);

    double uT = sqrt(u[1] * u[1] + u[2] * u[2]); 

    double prefactor = pT * exp(-mT/T * Gamma);

    if(mu == 0){
        return prefactor * mT * boost::math::cyl_bessel_i(0, pT/T * uT); 
    }
    else if(mu == 1){
        return prefactor * pT * boost::math::cyl_bessel_i(1, pT/T * uT); 
    }
    else if(mu == 2){
        return prefactor * pT * boost::math::cyl_bessel_i(1, pT/T * uT); 
    }
    else if(mu == 3){
        return prefactor * mT * boost::math::cyl_bessel_i(0, pT/T * uT); 

    }
    else{
        cout << "index in f bar must be between 0 and 3" << endl;
        return 0.0;
    }
}

bool checkdsigmamuXimuLEG(double x, double y, double m, double pTm, double pTM, double T, double mu_i, vector<double>& u, vector<double>& dsigma, double threshold){
    // This function is equivalent to checkdsigmamuXimu but is adapted for the use with the Legendre x Legendre integrals
    vector<double> Xi;
    for(int i = 0; i<4;i++){
        Xi.push_back(alphaTilde(i, pTm, pTM, T, mu_i, u) * gtilde(i, y) * ftilde(i, pTm, pTM, x, y, m, T, u, threshold));
    }
    return (dotCov(Xi, dsigma)>0);
}

double IntegralAcc(double YM, double pTm, double pTM, vector<double>& Omega, vector<double>& Y, int Nleg, vector<double>& X, vector<double>& Om, int mu, double m, double T, double mu_i, vector<double>& dsigma, vector<double>& u, double threshold){
    // This function computes the Legendre quadratures.
    double s = 0.0;
    double x, y;
    for(int i = 0; i<Nleg; i++){
        y = Y[i] * YM; // current rapidity, Legendre are calculated in [-1,1], times YM = [-YM, YM]
        for(int j = 0; j<Nleg; j++){
            x = X[j];
            if(checkdsigmamuXimuLEG(x, y, m, pTm, pTM, T, mu_i, u, dsigma, threshold)){
                s += Omega[i] * gtilde(mu, y) * Omega[j] * ftilde(mu, pTm, pTM, x, y, m, T, u, threshold) ;
            }
        }
    }
    return alphaTilde(mu, pTm, pTM, T, mu_i, u) * YM * s;
}

double deltaNcellAcc(double YM, double pTm, double pTM, vector<double>& Omega, vector<double>& Y, int Nleg, vector<double>& X, vector<double>& Om, double m, double T, double mu_i, vector<double>& u, vector<double>& dsigma, double threshold, double g=1.0){
    // This function computes the total particle yield in acceptance.
    double s = 0.0;
    vector<double> N;
    for(int i = 0; i < 4; i++){
       N.push_back(IntegralAcc(YM, pTm, pTM, Omega, Y, Nleg, X, Om, i, m, T, mu_i, dsigma, u, threshold)); 
    }
    // Need covariant dsigma in cartesian coordinates.
    return dotCov(dsigma, N, g/(8 * M_PI * M_PI * M_PI));
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
    cout << "Num Int : " << deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, mu_i, u, dsigma, threshold, 2) * exp(mu_i / T)  / (hbarc * hbarc * hbarc)  << "Analytic Int : " << Analytic << " T : " << T << " muB " << mu_i << endl; 
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

    double num_int = deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, mu_i, u, dsigma, threshold, 2) * exp(mu_i/T) / (hbarc * hbarc * hbarc);

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



bool GetProbabilityProtons(Acceptance Acc, Acceptance Full, SurfaceElement Surf, double* p){
    // This function computes the probability p = NpAcc/NpFull representing the probability 
    // of net proton to end up in the acceptance for one hydro cell.
    // It also returns the Net proton yields in acceptance and in the full space (for checks).
    double hbarc = 0.1973269804; // GeV*fm
    double threshold = 1000.0;
    double m = 0.938;
    double gp = 2;

    double T = Surf.T_f; 
    double mu_p = Surf.mu_B + Surf.mu_C;
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
    double YMAcc = Acc.ymax;

    double pTm = Acc.pTmin;
    double pTM = Acc.pTmax;

    double NpFull;
    double NpAcc;

    NpFull = deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, mu_p, uC, dsigmaC, threshold, gp);
    NpAcc = deltaNcellAcc(YMAcc, pTm, pTM, WLeg, XLeg, Nleg, XLeg, WLeg, m, T, mu_p, uC, dsigmaC, threshold, gp);

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

double GetFullNB(Acceptance Full, SurfaceElement Surf){
    // This function computes the probability p = NAcc/NFull representing the probability 
    // of ending up in the acceptance for one hydro cell.
    // It also returns the yields in acceptance and in the full space (for checks).
    double threshold = 1000.0;
    double m = 0.938;

    double T = Surf.T_f; 
    double mu_i = Surf.mu_B;
    vector<double> u = {Surf.u[0], Surf.u[1], Surf.u[2], Surf.u[3]};

    vector<double> dsigma = {Surf.s[0], Surf.s[1], Surf.s[2], Surf.s[3]};

    //vector<double> uC = MilneToCartesian(u, Surf);
    //vector<double> dsigmaC = MilneToCartesian(dsigma, Surf);

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
    double gDelt = 4;

    NpFull = deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, mu_i, uC, dsigmaC, threshold, gp);
    double mu_p = Surf.mu_B + Surf.mu_C;
    double mu_n = Surf.mu_B;


    double NpFullDelta = deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, 1.232, T, mu_i, uC, dsigmaC, threshold, gDelt);
    double mudpp = Surf.mu_B + 2 * Surf.mu_C;
    double mudp = Surf.mu_B + Surf.mu_C;
    double mud0 = Surf.mu_B;
    double mudm = Surf.mu_B - Surf.mu_C;




    return 2 * NpFull * (sinh(mu_p/T) + sinh(mu_n/T))
        + 2 * NpFullDelta * (sinh(mudpp/T) + sinh(mudp/T) + sinh(mud0/T) + sinh(mudm/T));
    //return NpFull * exp(mu_n/T);

}

int main(int argc, char *argv[]) {

    // Read user imputs.
    string energy = argv[1];
    string centrality = argv[2];

    // Read Surface file
    const string SurfacePath = "surfacesFiles/"; 
    const string fname = "AuAu"+energy+"/hydro_results_C"+centrality+"/surface_eps_0.26.dat";

    vector<SurfaceElement> surface;

    //cout << "Read Surface : " << endl;
    surface = ReadFreezeOutSurface(SurfacePath+fname); 

    int Nmax = -1;
    // Remove negative 4-volume cells.
    vector<SurfaceElement> selectedCells = selectPositiveVolumeCells(surface, Nmax);
    cout << "yolo1" << endl;

    //Define acceptance and Full space ranges.
    Acceptance Acc;
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



    vector<double> OUT;
    vector<bool> OUT_CHECK;
    OUT_CHECK.resize(n_cells);
    OUT.resize(n_cells * n_double_per_cell);
    atomic<size_t> counter(0);

    #pragma omp parallel
    {
        size_t local_count = 0;
        #pragma omp for schedule(static)
        for (size_t k = 0; k < selectedCells.size(); ++k) {
            bool OutGet = GetProbabilityProtons(Acc, FullSpace, selectedCells[k], &OUT[k * n_double_per_cell]);
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
    ofstream outFile("p_values.txt");

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

    //OutGet = GetProbability(Acc, FullSpace, selectedCells[DebugCell], &OUT[DebugCell * n_double_per_cell]);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Test compared to analytical formula, validation of the integration process.
    //int DebugCell = 29865;
    //double Int = Test(FullSpace, selectedCells[DebugCell]);
    //double IntFlow = TestWithTransverseFlow(FullSpace, selectedCells[DebugCell]);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////


    //int DebugCell = 29865;
    //vector<double> p;
    //p = GetProbabilityProtons(Acc, FullSpace, selectedCells[DebugCell]);
    //for(auto& e:p){cout << e << " || ";}
    //cout << endl;
    
    //double Int;
    //double s1 = 0.0;
    //double s2 = 0.0;
    //double dV, dV2;
    //double etamin = 100.0;
    //double etamax = -100.0;
    //SurfaceElement cell;


    ////for(auto &cell:selectedCells){
    //for(int i = 0; i<selectedCells.size(); i++){
    //    cell = selectedCells[i]; 
    //    eta = cell.x[3];

    //    //vector<double> u = {cell.u[0], cell.u[1], cell.u[2], cell.u[3]};
    //    //vector<double> dsigma = {cell.s[0], cell.s[1], cell.s[2], cell.s[3]};

    //    //vector<double> uC = MilneToCartesian(u, cell);
    //    //vector<double> dsigmaC = MilneToCartesianSigmaCov(dsigma, cell);
    //    ////dV = dotMinkowski(uC, dsigmaC);
    //    //double tau = cell.x[0]; 
    //    //dV = tau * (u[0]*dsigma[0] + u[1]*dsigma[1] + u[2]*dsigma[2] + u[3]*(dsigma[3]/tau));
    //    //dV2 =  uC[0]*dsigmaC[0] + uC[1]*dsigmaC[1] + uC[2]*dsigmaC[2] + uC[3]*dsigmaC[3];
    //    //if(dV > 0){
    //    //    s1 +=  dV * cell.rho_B;
    //    //    s2 +=  dV2 * cell.rho_B;
    //    //}

    //    ///////////////// Correct version from the surface!!!!!
    //    //dV = tau * (u[0]*dsigma[0] + u[1]*dsigma[1] + u[2]*dsigma[2] + u[3]*(dsigma[3]/tau));
    //    //if(dV > 0){
    //    //    s1 += dotMinkowski(uC, dsigmaC) * cell.rho_B;
    //    //    // here ---------------------------------------
    //    //    s2 += tau * (u[0]*dsigma[0] + u[1]*dsigma[1] + u[2]*dsigma[2] + u[3]*(dsigma[3]/tau)) * cell.rho_B;
    //    //}
    //    


    //    //cout << dotMinkowski(cell.u, cell.s) * cell.rho_B * cell.x[0] << endl;
    //    //cout << dotMinkowski(dsigmaC, dsigmaC) << " " << dotMinkowskiMilne(dsigma, dsigma, cell) << " " << dotMinkowski(dsigma, dsigma)  << " tau = " << cell.x[0]<< endl;
    //    //cout << "Milne : " << dotMinkowskiMilne(u, dsigma, cell) << " Cartesian: " << dotMinkowski(uC, dsigmaC)  << " tau = " << cell.x[0]<< endl;
    //    //cout << "Milne : " << dotMinkowskiMilne(u, dsigma, cell) << " Cartesian: " << dotMinkowski(uC, dsigmaC)  << " tau = " << cell.x[0]<< endl;
    //    //s1 += dotMinkowski(cell.u, cell.s, cell.x[0]) * cell.rho_B;
    //    //s1 += dotMinkowskiMilne(u, dsigma, cell) * cell.rho_B;
    //    //s += dotMinkowski(uC, dsigmaC) * cell.rho_B;
    //    Int = GetFullNB(FullSpace, selectedCells[i]);
    //    if(i%100==0){cout << "cell done : "<< ((double) i)/((double) selectedCells.size()) * 100.0 << " | sum : " << s1 << " | Nblocal : " << Int/(0.1973 * 0.1973 * 0.1973) << endl;}
    //    //if(i%100==0){cout << "cell done : "<< ((double) i)/((double) selectedCells.size()) * 100.0 << " | sum : " << s1 << endl;}
    //    s1+= Int/(0.1973 * 0.1973 * 0.1973);
    //}
    //for(int i = 0; i < etas.size(); i++){
    ////    cout << etas[i] << " " << dNB_deta[i] << endl;  
    ////}
    //cout << "Total sum 1 : " << s1 << endl;
    //cout << "Total sum 2 : " << s2 << endl;

    //double pvec0, pvec1, pvec2;
    //pvec0 = OUT[DebugCell*n_double_per_cell];
    //pvec1 = OUT[DebugCell*n_double_per_cell+1];
    //pvec2 = OUT[DebugCell*n_double_per_cell+2];
    //cout << "Acc : " << pvec1 << " Full : " << pvec2 << endl;

    //OUT.resize(n_cells * n_double_per_cell);
    //OUT_CHECK.resize(n_cells);

    //atomic<size_t> counter(0);

    //#pragma omp parallel
    //{
    //    size_t local_count = 0;
    //    #pragma omp for schedule(static)
    //    for (size_t k = 0; k < selectedCells.size(); ++k) {
    //        bool OutGet = GetProbability(Acc, FullSpace, selectedCells[k], &OUT[k * n_double_per_cell]);
    //        OUT_CHECK[k] = OutGet; 
    //        ++local_count;

    //        if (local_count % 1000 == 0) {
    //            counter += 1000;
    //            if (omp_get_thread_num() == 0)
    //                cout << "\rProgress: " << (100.0 * counter / selectedCells.size()) << "%   " << std::flush;
    //        }
    //    }
    //}

    //// Write in a file.
    //double pvec0, pvec1, pvec2;
    //double tau, x, y, eta;
    //ofstream outFile("p_values.txt");
    //for(int j = 0; j < OUT_CHECK.size(); j++){
    //    pvec0 = OUT[j*n_double_per_cell];
    //    pvec1 = OUT[j*n_double_per_cell+1];
    //    pvec2 = OUT[j*n_double_per_cell+2];

    //    tau = selectedCells[j].x[0];
    //    x = selectedCells[j].x[1];
    //    y = selectedCells[j].x[2];
    //    eta = selectedCells[j].x[3];

    //    OutGet = OUT_CHECK[j];
    //    //outFile << pvec[0] << "  " << pvec[1] << "  " << pvec[2] << " " << pvec[3] << " " << OutGet << "\n"; 
    //    outFile << pvec0 << "  " << pvec1 << "  " << pvec2 << " " << OutGet << " " << tau << " " << x << " " << y << " " << eta << "\n"; 
    //}
    //outFile.close();
    return 0;
}

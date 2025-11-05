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

#ifdef USE_OPENMP
#include <omp.h>
#endif

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

vector<double> MilneToCartesian(vector<double>& Milne, SurfaceElement Surf){
    // This function translates a 4-vector expressed in Milne coordinates (tau, x, y, eta) in 
    // a 4-vector expressed in the cartesian coordinates. 
    // Note that the input Milne vector is expected to have the tau factor in the 
    // eta direction already included. 
    vector<double> Cart;
    Cart.resize(Milne.size());
    Cart[0] = Surf.cosh_eta_s * Milne[0] + Surf.sinh_eta_s * Milne[3];
    Cart[1] = Milne[1];
    Cart[2] = Milne[2];
    Cart[3] = Surf.sinh_eta_s * Milne[0] + Surf.cosh_eta_s * Milne[3]; 
    return Cart;
}

inline double dotMinkowski(vector<double> a, vector<double> b, double coeff=1.0) {
    // Function to compute Minkowski inner product with (+,-,-,-) metric
    // when input 4-vectors are represented by vectors.
    return coeff * (a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]);
}
inline double dotMinkowski(const double a[4], const double b[4], double coeff=1.0) {
    // Function to compute Minkowski inner product with (+,-,-,-) metric
    // when input 4-vectors are represented by arrays.
    return coeff * (a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]);
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
    selected.reserve(allCells.size());
    for (int i = 0; i < NMAX; i++) {
        cell = allCells[i];
        // Compute 4-volume element = s^μ u_μ
        double dV4 = dotMinkowski(cell.s, cell.u);
        if (dV4 > 0.0) {
            selected.push_back(cell);
        }
    }
    return selected;
}

int get_number_of_lines_of_binary_surface_file(string filename) {
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
    //cout << "reading freeze-out surface" << endl;

    vector<SurfaceElement> surface;

    bool boost_invariant = false;
    ostringstream surfdat_stream;
    surfdat_stream << Surfpath;

    // new counting, mac compatible ...
    int NCells = get_number_of_lines_of_binary_surface_file(surfdat_stream.str());

    //cout << "NCells = " << NCells << endl;
    //flush(cout);

    ifstream surfdat;
    surfdat.open(surfdat_stream.str().c_str(), std::ios::binary);

    // Now allocate memory: array of surfaceElements with length NCells
    //surface = (SurfaceElement *) malloc((NCells)*sizeof(SurfaceElement));
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

        temp_cell.s[0] = array[4];
        temp_cell.s[1] = array[5];
        temp_cell.s[2] = array[6];
        temp_cell.s[3] = array[7];

        temp_cell.u[0] = array[8];
        temp_cell.u[1] = array[9];
        temp_cell.u[2] = array[10];
        temp_cell.u[3] = array[11];

        temp_cell.epsilon_f            = array[12];
        temp_cell.T_f                  = array[13];
        temp_cell.mu_B                 = array[14];
        temp_cell.mu_S                 = array[15];
        temp_cell.mu_C                 = array[16];
        temp_cell.eps_plus_p_over_T_FO = array[17];

        temp_cell.W[0][0] = array[18];
        temp_cell.W[0][1] = array[19];
        temp_cell.W[0][2] = array[20];
        temp_cell.W[0][3] = array[21];
        temp_cell.W[1][1] = array[22];
        temp_cell.W[1][2] = array[23];
        temp_cell.W[1][3] = array[24];
        temp_cell.W[2][2] = array[25];
        temp_cell.W[2][3] = array[26];
        temp_cell.W[3][3] = array[27];

        temp_cell.pi_b  = array[28];
        temp_cell.rho_B = array[29];

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
// --------------------------------------------------------------------------
double alphaBar(int mu, double T, double mu_i, vector<double>& u){
    // This function computes the \bar{\alpha} function for its use 
    // in the Laguerre integration.
    double uT; // Transverse 4-velocity of flow
    double prefactor = 2 * M_PI * exp(mu_i) * T * T; 
    if(mu == 0){
        return prefactor; 
    }
    else if(mu == 1){
        uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
        return prefactor * u[1] / uT; 
    }
    else if(mu == 2){
        uT = sqrt(u[1] * u[1] + u[2] * u[2]); 
        return prefactor * u[2] / uT;
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
    double pT = GetpT(threshold, mT, m);

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
    //double Xi0 = alphabar(0, T, mu_i, u) * gbar(0, y, m, T, u) * fbar_scaled(0, x, y, m, T, u, threshold) * dsigma[0];
    //double Xi1 = alphaBar(1, T, mu_i, u) * gbar(1, y, m, T, u) * fbar_scaled(1, x, y, m, T, u, threshold) * dsigma[1];
    //double Xi2 = alphaBar(2, T, mu_i, u) * gbar(2, y, m, T, u) * fbar_scaled(2, x, y, m, T, u, threshold) * dsigma[2];
    //double Xi3 = alphaBar(3, T, mu_i, u) * gbar(3, y, m, T, u) * fbar_scaled(3, x, y, m, T, u, threshold) * dsigma[3];

    vector<double> Xi;
    for(int i = 0; i<4;i++){
        Xi.push_back(alphaBar(i, T, mu_i, u) * gbar(i, y, m, T, u) * fbar_scaled(i, x, y, m, T, u, threshold));
    }
    return (dotMinkowski(Xi, dsigma)>0);
    //return (Xi0 - Xi1 - Xi2 - Xi3 > 0);
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
    double pT = GetpT(threshold, mT, m);

    double expon = pT / T * uT;
    double check = log(WLag) + pT / T * uT; 

    if(check > 50){
        cout << "Very large value of w * e^(pT / T uT), exponant value : " << check << endl;
        return 0.0;
    } 
    else if(check < -32){
        return 0.0;
    }
    else{
        if(expon > 700){ // Machine limit
            // Since log(WLag) + expon < 50, if expon > 700 it means log(WLag) < -750.....
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
    for(int i = 0; i<Nleg; i++){
        y = Y[i] * YM; // current rapidity, Legendre are calculated in [-1,1], times YM = [-YM, YM]
        for(int j = 0; j<Nlag; j++){
            x = X[j];
            // Check if the cell will contribute positively to the integrals.
            if(checkdsigmamuXimu(x, y, m, T, mu_i, u, dsigma, threshold)){
                // Compute scaled Laguerre weight for numerical regularization. 
                scaledW = getScaledW(x, y, T, m, W[j], u, W[j]);
                s += Omega[i] * gbar(mu, y, m, T, u) * scaledW * fbar_scaled(mu, x, y, m, T, u, threshold);
            }
        }
    }
    return alphaBar(mu, T, mu_i, u) * YM * s;
}

double deltaNcellFull(double YM, vector<double>& Omega, vector<double>& Y, int Nleg, vector<double>& X, vector<double>& W, int Nlag, double m, double T, double mu_i, vector<double>& u, vector<double>& dsigma, double threshold){
    // This function computes the final contributions from the cells in the full acceptance. 
    vector<double> Nmu;
    for(int i = 0; i<4; i++){
        Nmu.push_back(IntegralFull(YM, Omega, Y, Nleg, X, W, Nlag, i, m, T, mu_i, u, dsigma, threshold));
    }
    return dotMinkowski(Nmu, dsigma, 1/(8 * M_PI * M_PI * M_PI)); 
}

// Acceptance space Legendre-Legendre integral ---------------------------------------------
// -----------------------------------------------------------------------------------------

double alphaTilde(int mu, double pTm, double pTM, double T, double mu_i, vector<double>& u){
    // This function computes the \tilde{\alpha} function appearing in the Legendre quadrature 
    // to compute the particle yield in the acceptance. 
    double prefactor = M_PI * exp(mu_i) * (pTM - pTm); 
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

    double uT = sqrt(u[1] * u[1] + u[1] * u[1]); 

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
    //double Xi0 = alphaTilde(0, pTm, pTM, T, mu_i, u) * gtilde(0, y) * ftilde(0, pTm, pTM, x, y, m, T, u, threshold) * dsigma[0];
    //double Xi1 = alphaTilde(1, pTm, pTM, T, mu_i, u) * gtilde(1, y) * ftilde(1, pTm, pTM, x, y, m, T, u, threshold) * dsigma[1];
    //double Xi2 = alphaTilde(2, pTm, pTM, T, mu_i, u) * gtilde(2, y) * ftilde(2, pTm, pTM, x, y, m, T, u, threshold) * dsigma[2];
    //double Xi3 = alphaTilde(3, pTm, pTM, T, mu_i, u) * gtilde(3, y) * ftilde(3, pTm, pTM, x, y, m, T, u, threshold) * dsigma[3];
    //return (Xi0 - Xi1 - Xi2 - Xi3 > 0);

    vector<double> Xi;
    for(int i = 0; i<4;i++){
        Xi.push_back(alphaTilde(i, pTm, pTM, T, mu_i, u) * gtilde(i, y) * ftilde(i, pTm, pTM, x, y, m, T, u, threshold));
    }
    return (dotMinkowski(Xi, dsigma)>0);
}

//double pTIntegralAcc(double pTm, double pTM, vector<double>& X, vector<double>& Omega, int Nleg, int mu, double y, double m, double T, vector<double>& u, double threshold){
//    double s = 0.0;
//    for(int i = 0; i<Nleg; i++){
//        s += Omega[i] * ftilde(mu, pTm, pTM, X[i], y, m, T, u, threshold);
//    }
//    return s;
//}
double IntegralAcc(double YM, double pTm, double pTM, vector<double>& Omega, vector<double>& Y, int Nleg, vector<double>& X, vector<double>& Om, int mu, double m, double T, double mu_i, vector<double>& dsigma, vector<double>& u, double threshold){
    // This function computes the Legendre quadratures.
    double s = 0.0;
    double x, y;
    for(int i = 0; i<Nleg; i++){
        y = Y[i] * YM; // current rapidity, Legendre are calculated in [-1,1], times YM = [-YM, YM]
        for(int j = 0; j<Nleg; j++){
            x = X[j];
            //s += Omega[i] * gtilde(mu, y) * pTIntegralAcc(pTm, pTM, X, Om, Nleg, mu, y, m, T, u, threshold);
            if(checkdsigmamuXimuLEG(x, y, m, pTm, pTM, T, mu_i, u, dsigma, threshold)){
                s += Omega[i] * gtilde(mu, y) * Omega[j] * ftilde(mu, pTm, pTM, x, y, m, T, u, threshold) ;
            }
        }
    }
    return alphaTilde(mu, pTm, pTM, T, mu_i, u) * YM * s;
}

double deltaNcellAcc(double YM, double pTm, double pTM, vector<double>& Omega, vector<double>& Y, int Nleg, vector<double>& X, vector<double>& Om, double m, double T, double mu_i, vector<double>& u, vector<double>& dsigma, double threshold){
    //double N0 = IntegralAcc(YM, pTm, pTM, Omega, Y, Nleg, X, Om, 0, m, T, mu_i, u, threshold);
    //double N1 = IntegralAcc(YM, pTm, pTM, Omega, Y, Nleg, X, Om, 1, m, T, mu_i, u, threshold);
    //double N2 = IntegralAcc(YM, pTm, pTM, Omega, Y, Nleg, X, Om, 2, m, T, mu_i, u, threshold);
    //double N3 = IntegralAcc(YM, pTm, pTM, Omega, Y, Nleg, X, Om, 3, m, T, mu_i, u, threshold);
    //return (dsigma[0] * N0 - dsigma[1] * N1 - dsigma[2] * N2 - dsigma[3] * N3)/(8 * M_PI * M_PI * M_PI);

    // This function computes the total particle yield in acceptance.
    double s = 0.0;
    vector<double> N;
    for(int i = 0; i < 4; i++){
       N.push_back(IntegralAcc(YM, pTm, pTM, Omega, Y, Nleg, X, Om, i, m, T, mu_i, dsigma, u, threshold)); 
    }
    return dotMinkowski(N, dsigma, 1/(8 * M_PI * M_PI * M_PI));
}


bool GetProbability(Acceptance Acc, Acceptance Full, SurfaceElement Surf, vector<double>& p){
    // This function computes the probability p = NAcc/NFull representing the probability 
    // of ending up in the acceptance for one hydro cell.
    // It also returns the yields in acceptance and in the full space (for checks).
    double threshold = 1000.0;
    double m = 0.938;

    double T = Surf.T_f; 
    double mu_i = Surf.mu_B;
    vector<double> u = {Surf.u[0], Surf.u[1], Surf.u[2], Surf.u[3]};

    vector<double> dsigma = {Surf.s[0], Surf.s[1], Surf.s[2], Surf.s[3]};

    vector<double> uC = MilneToCartesian(u, Surf);
    vector<double> dsigmaC = MilneToCartesian(dsigma, Surf);


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

    NpFull = deltaNcellFull(YM, WLeg, XLeg, Nleg, XLag, WLag, Nlag, m, T, mu_i, uC, dsigmaC, threshold);
    NpAcc = deltaNcellAcc(YMAcc, pTm, pTM, WLeg, XLeg, Nleg, XLeg, WLeg, m, T, mu_i, uC, dsigmaC, threshold);
    p.resize(3);
    p[0] = NpAcc/NpFull;
    p[1] = NpAcc;
    p[2] = NpFull;

    // Manage negative contributions, avoid cells if they have negative probabilities.
    if(NpFull < 0.0){
        return false;
    }
    else if(NpAcc < 0.0){
        return false; 
    }
    else{
        return true;
    }
}

int main(int argc, char *argv[]) {

    // Read user imputs.
    string energy = argv[1];
    string centrality = argv[2];

    // Read Surface file
    const string SurfacePath = "surfacesFiles/"; 
    const string fname = "AuAu"+energy+"/hydro_results_C"+centrality+"/surface_eps_0.26.dat";

    vector<SurfaceElement> surface;
    surface = ReadFreezeOutSurface(SurfacePath+fname); 

    //int Nmax = -1;
    // Remove negative 4-volume cells.
    vector<SurfaceElement> selectedCells = selectPositiveVolumeCells(surface);

    // Define acceptance and Full space ranges.
    Acceptance Acc;
    Acc.pTmin = 0.4; Acc.pTmax = 2.0;
    Acc.ymin = -0.5; Acc.ymax = 0.5;

    Acceptance FullSpace;
    FullSpace.pTmin = 0.4; FullSpace.pTmax = 100000; // in Laguerre this is always +infinity
    FullSpace.ymin = -5.0; FullSpace.ymax = 5.0;// |y|= 5 is enough 


    vector<double> pvec;
    bool OutGet;

    //double uT, u0, u3;
    //double y = 0.0;

    // Setup parallelization
    vector<vector<double>> OUT;
    vector<bool> OUT_CHECK;

    int Nchunks = 1000;  

    vector<vector<SurfaceElement>> selectedCellsChunks = splitVector(selectedCells, Nchunks);
    // Compute probability in each cells
    double j = 0.0;
    for(auto& chunk:selectedCellsChunks){
        cout << j/((double) selectedCellsChunks.size()) * 100 << endl; 

        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for(int i = 0; i< chunk.size(); i++){
            OUT_CHECK.push_back(GetProbability(Acc, FullSpace, chunk[i], pvec));
            OUT.push_back(pvec);
        }
        j+=1.0;
    }

    // Write in a file.
    ofstream outFile("p_values.txt");
    for(int j = 0; j < OUT.size(); j++){
        pvec = OUT[j];
        OutGet = OUT_CHECK[j];
        outFile << pvec[0] << "  " << pvec[1] << "  " << pvec[2] << " " << pvec[3] << " " << OutGet << "\n"; 
    }
    outFile.close();
    return 0;
}

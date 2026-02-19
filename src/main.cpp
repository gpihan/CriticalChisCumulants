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
#include <cstddef>

#include <omp.h>
#include <atomic>

#include "Integration.h"
#include "ScaledBesselI1.h"

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


struct chis{
    double T, muB, chi1, chi2, chi3, chi4;
};

vector<chis> ReadTable(string fname){
    vector<chis> table;
    ifstream fin(fname);
    
    double T, muB, tmp1, tmp2, tmp3, tmp4, tmp5, chi1, chi2, chi3, chi4; 
    while (fin >> T >> muB >> tmp1 >> chi1 >> tmp3 >> tmp4 >> tmp5 >> chi2 >> chi3 >> chi4) {
        //table.push_back({T, muB, chi2 * T * T, chi3 * T, chi4}); // Normalization of chis are chi2/T^2, chi3/T, chi4
        table.push_back({T, muB, chi1 * T * T * T /(0.1973 * 0.1973 * 0.1973), chi2, chi3, chi4}); // Normalization of chis are chi2/T^2, chi3/T, chi4
    }
    return table;
} 

struct Interpolator2D {
    vector<double> xs;   // unique sorted x
    vector<double> ys;   // unique sorted y
    vector<double> F;    // flattened grid values

    int nx, ny;

    // bilinear evaluate
    inline double eval(double x, double y) const;
};

inline double Interpolator2D::eval(double x, double y) const {
    // locate index with binary_search or manual bisection
    int ix = upper_bound(xs.begin(), xs.end(), x) - xs.begin() - 1;
    int iy = upper_bound(ys.begin(), ys.end(), y) - ys.begin() - 1;

    // Clamp
    ix = max(0, min(ix, nx-2));
    iy = max(0, min(iy, ny-2));

    double x0 = xs[ix],   x1 = xs[ix+1];
    double y0 = ys[iy],   y1 = ys[iy+1];

    double f00 = F[ix*ny + iy];
    double f01 = F[ix*ny + iy+1];
    double f10 = F[(ix+1)*ny + iy];
    double f11 = F[(ix+1)*ny + iy+1];

    double tx = (x - x0) / (x1 - x0);
    double ty = (y - y0) / (y1 - y0);

    return (1-tx)*(1-ty)*f00 +
           (1-tx)*ty*f01   +
           tx*(1-ty)*f10   +
           tx*ty*f11;
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

inline array<double,4> alphaBar_all(double prefactor, double u1ouT, double u2ouT){
    return {
        prefactor,          // μ = 0
        prefactor * u1ouT,  // μ = 1
        prefactor * u2ouT,  // μ = 2
        prefactor           // μ = 3
    };
}

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

inline array<double,4> fbar_scaled_all(
        double prefactor,   // x + m/T * Gamma
        double pT,
        double mT,
        double pToTuT,       // pT/T * uT
        const ScaledBesselI1& ScaledB1
)
{
    // ----- 1. Compute both scaled Bessel functions once -----
    double I0 = I0_scaled_fast(pToTuT);
    //double I1 = I1_scaled_fast(pToTuT);
    //double I1 = Modified_Bessel_scaled(1, pToTuT); // The I0 parametrisation is correct but the I1 is wrong.
    //                                               // prefer using the original I1 from boost
    // Interpolation
    double I1 = ScaledB1.eval(pToTuT);

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
}

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

array<double,4> IntegralFull_all(
        double YM,
        const vector<double>& Omega, const vector<double>& Y, int Nleg,
        const vector<double>& X, const vector<double>& W, const vector<double>& logW, int Nlag,
        double m, double T, const vector<double>& u, const vector<double>& dsigma,
        double threshold,
        double prefacta, double u1ouT, double u2ouT, double uToT, const ScaledBesselI1& ScaledB1)
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
            const auto fb = fbar_scaled_all(prefactfbar, pT, mT, pToTuT, ScaledB1);
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
        double g, const ScaledBesselI1& ScaledB1)
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
        prefacta, u1ouT, u2ouT, uToT, ScaledB1
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

double GetmT(double treshold, double pT, double m){
    // This function checks the if pT is larger compared to the mass.
    // To avoid computing unecessary square roots for efficiency.
    if(pT/m < treshold){return sqrt(pT * pT + m * m);}
    else{return pT;}
}

bool checkdsigmamuXimuLEG(double at0, double at1, double at2, double at3, double gt0, double gt1, double gt2, double gt3, double ft0, double ft1, double ft2, double ft3, const vector<double>& dsigma){
    // This function is equivalent to checkdsigmamuXimu but is adapted for the use with the Legendre x Legendre integrals
    double x0 = at0 * gt0 * ft0 * dsigma[0];
    double x1 = at1 * gt1 * ft1 * dsigma[1];
    double x2 = at2 * gt2 * ft2 * dsigma[2];
    double x3 = at3 * gt3 * ft3 * dsigma[3];
    return (x0+x1+x2+x3>0);
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
        double pToTuT,           // pT/T * uT
        const ScaledBesselI1& ScaledB1
)
{
    // Compute BOTH scaled Bessels once.
    double I0 = I0_fast(pToTuT);
    // Interpolation
    double I1 = ScaledB1.eval(pToTuT) * exp(pToTuT);


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

array<double,4> IntegralAcc(
        double YM, double pTm, double pTM,
        const vector<double>& Omega, const vector<double>& Y, int Nleg,
        const vector<double>& X, const vector<double>& Om, 
        double m,double T,
        const vector<double>& dsigma, const vector<double>& u,
        double threshold,
        double prefactatilde, double u1ouT, double u2ouT, double uT, const ScaledBesselI1& ScaledB1)
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
            const auto ft = ftilde_all(prefactftilde, pT, mT, pToTuT, ScaledB1);
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

double deltaNcellAcc(double YM,double pTm, double pTM,
        const vector<double>& Omega, const vector<double>& Y, int Nleg,
        const vector<double>& X, const vector<double>& Om, 
        double m, double T, const vector<double>& u, const vector<double>& dsigma,
        double threshold, double g, const ScaledBesselI1& ScaledB1)
{
    double prefactatilde = M_PI * (pTM - pTm);
    double uT = sqrt(u[1]*u[1] + u[2]*u[2]);
    double u1ouT = (uT > 1e-14 ? u[1] / uT : 0.0);
    double u2ouT = (uT > 1e-14 ? u[2] / uT : 0.0);

    array<double,4> Nmu = IntegralAcc(YM, pTm, pTM,
            Omega, Y, Nleg, X, Om,
            m, T, dsigma, u, threshold,
            prefactatilde, u1ouT, u2ouT, uT, ScaledB1);

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
    return 1.0;
}

bool GetProbabilityProtons(Acceptance Acc, Acceptance Full, SurfaceElement Surf, double* p, 
        const vector<double>& XLeg,
        const vector<double>& WLeg,
        const vector<double>& XLag,
        const vector<double>& WLag,
        int NLeg, int NLag, const Interpolator2D& chi1, const Interpolator2D& chi2, const Interpolator2D& chi3, const Interpolator2D& chi4, const ScaledBesselI1& ScaledB1){
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
    double muB = Surf.mu_B;
    vector<double> u = {Surf.u[0], Surf.u[1], Surf.u[2], Surf.u[3]};
    vector<double> dsigma = {Surf.s[0], Surf.s[1], Surf.s[2], Surf.s[3]};
    vector<double> uC = MilneToCartesian(u, Surf);
    vector<double> dsigmaC = MilneToCartesianSigmaCov(dsigma, Surf);

    double dV = uC[0] * dsigmaC[0] + uC[1] * dsigmaC[1] + uC[2] * dsigmaC[2] + uC[3] * dsigmaC[3]; 

    double currentchi1 = chi1.eval(T, muB);
    double currentchi2 = chi2.eval(T, muB);
    double currentchi3 = chi3.eval(T, muB);
    double currentchi4 = chi4.eval(T, muB);

    double YM = Full.ymax;
    double YMAcc = Acc.ymax;

    double pTm = Acc.pTmin;
    double pTM = Acc.pTmax;

    double NpFull = 1.0;
    double NpAcc = 1.0;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // By passing alpha calculation for efficieny of tests ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //NpFull = deltaNcellFull(YM, WLeg, XLeg, NLeg, XLag, WLag, NLag, m, T, uC, dsigmaC, threshold, gp, ScaledB1);
    //NpAcc = deltaNcellAcc(YMAcc, pTm, pTM, WLeg, XLeg, NLeg, XLeg, WLeg, m, T, uC, dsigmaC, threshold, gp, ScaledB1);

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
    //chi2 * dV
    p[9] = currentchi2 * dV * T * T * T;
    //chi3 * dV
    p[10] = currentchi3 * dV * T * T * T;
    //chi4 * dV
    p[11] = currentchi4 * dV * T * T * T;
    // cell volume
    p[12] = dV;
    // chi1 * dV
    p[13] = Surf.rho_B * dV;
    // chi1 Table * dV
    p[14] = currentchi1 * dV;


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

bool GetSTOREDDATA(Acceptance Acc, Acceptance Full, SurfaceElement Surf, double* p, 
        const vector<double>& XLeg,
        const vector<double>& WLeg,
        const vector<double>& XLag,
        const vector<double>& WLag,
        int NLeg, int NLag, const ScaledBesselI1& ScaledB1){
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
    double muB = Surf.mu_B;
    double muQ = Surf.mu_C;
    double muS = Surf.mu_S;
    vector<double> u = {Surf.u[0], Surf.u[1], Surf.u[2], Surf.u[3]};
    vector<double> dsigma = {Surf.s[0], Surf.s[1], Surf.s[2], Surf.s[3]};
    vector<double> uC = MilneToCartesian(u, Surf);
    vector<double> dsigmaC = MilneToCartesianSigmaCov(dsigma, Surf);

    double dV = uC[0] * dsigmaC[0] + uC[1] * dsigmaC[1] + uC[2] * dsigmaC[2] + uC[3] * dsigmaC[3]; 

    double YM = Full.ymax;
    double YMAcc = Acc.ymax;

    double pTm = Acc.pTmin;
    double pTM = Acc.pTmax;

    double NpFull = 1.0;
    double NpAcc = 1.0;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // By passing alpha calculation for efficieny of tests ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    NpFull = deltaNcellFull(YM, WLeg, XLeg, NLeg, XLag, WLag, NLag, m, T, uC, dsigmaC, threshold, gp, ScaledB1);
    NpAcc = deltaNcellAcc(YMAcc, pTm, pTM, WLeg, XLeg, NLeg, XLeg, WLeg, m, T, uC, dsigmaC, threshold, gp, ScaledB1);

    // T
    p[0] = T;
    // muB
    p[1] = muB;
    // muQ
    p[2] = muQ;
    // muS
    p[3] = muS;
    // alpha
    p[4] = NpAcc/NpFull;
    // nB surface
    p[5] = Surf.rho_B;
    // Cell Volume
    p[6] = dV;

    // Manage negative contributions, avoid cells if they have negative probabilities.
    if(p[4] < 0.0){
        return false;
    }
    else if(p[4] > 1.0){
        return false; 
    }
    else{
        return true;
    }
}

void extract_unique_xy_structured(const vector<chis>& data, int nx, int ny,
                                  vector<double>& xs, vector<double>& ys)
{
    xs.resize(nx);
    ys.resize(ny);

    // xs comes from the first element of each block of ny rows
    for (int i = 0; i < nx; i++)
        xs[i] = data[i * ny].T;

    // ys comes from the first row (assuming order as described)
    for (int j = 0; j < ny; j++)
        ys[j] = data[j].muB;
}

vector<double> ExtractChi1(vector<chis> Chis){
    vector<double> chi;
    for(auto& c:Chis){
        chi.push_back(c.chi1);
    }
    return chi;
}

vector<double> ExtractChi2(vector<chis> Chis){
    vector<double> chi;
    for(auto& c:Chis){
        chi.push_back(c.chi2);
    }
    return chi;
}
vector<double> ExtractChi3(vector<chis> Chis){
    vector<double> chi;
    for(auto& c:Chis){
        chi.push_back(c.chi3);
    }
    return chi;
}
vector<double> ExtractChi4(vector<chis> Chis){
    vector<double> chi;
    for(auto& c:Chis){
        chi.push_back(c.chi4);
    }
    return chi;
}

int main(int argc, char *argv[]) {

    // Read user imputs.
    string energy = argv[1];
    string centrality = argv[2];

    // Read Surface file
    const string SurfacePath = "surfacesFiles/"; 
    const string fname = "AuAu"+energy+"/hydro_results_C"+centrality+"/surface_eps_0.26.dat";
    //const string fname = "PbPb"+energy+"/hydro_results_C"+centrality+"/surface_eps_0.2.dat";

    // Setup chis interpolations
    // Bessel I1 interpolation
    double pToTuTThreshold = 50.0; // after that, asymptotic expansion has rel err 1e-7. 
    int Nbess = 6000;
    static const ScaledBesselI1 I1_scaled_interp(pToTuTThreshold, Nbess);

    const string EoSTablePath = "EoS_table/EoSTable.dat";
    vector<chis> EoS = ReadTable(EoSTablePath);
    Interpolator2D tmpinterpchi1, tmpinterpchi2, tmpinterpchi3, tmpinterpchi4;

    int nx = 371; // To adapt if the table changes.
    //int nx = 771; // To adapt if the table changes.
    int ny = 701;
    vector<double> xs, ys; 
    extract_unique_xy_structured(EoS, nx,ny, xs, ys);

    tmpinterpchi1.nx = nx;tmpinterpchi2.nx = nx;tmpinterpchi3.nx = nx;tmpinterpchi4.nx = nx;
    tmpinterpchi1.ny = ny;tmpinterpchi2.ny = ny;tmpinterpchi3.ny = ny;tmpinterpchi4.ny = ny;

    tmpinterpchi1.xs = xs;tmpinterpchi2.xs = xs;tmpinterpchi3.xs = xs;tmpinterpchi4.xs = xs;
    tmpinterpchi1.ys = ys;tmpinterpchi2.ys = ys;tmpinterpchi3.ys = ys;tmpinterpchi4.ys = ys;

    tmpinterpchi1.F = ExtractChi1(EoS);
    tmpinterpchi2.F = ExtractChi2(EoS);
    tmpinterpchi3.F = ExtractChi3(EoS);
    tmpinterpchi4.F = ExtractChi4(EoS);


    static const Interpolator2D interpchi1 = tmpinterpchi1;
    static const Interpolator2D interpchi2 = tmpinterpchi2;
    static const Interpolator2D interpchi3 = tmpinterpchi3;
    static const Interpolator2D interpchi4 = tmpinterpchi4;


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
    size_t n_double_per_cell = 15;
    size_t n_store_double_per_cell = 7; 
    size_t cells_per_thread = n_cells / omp_get_max_threads() + 1;

    cout << "Total number of cells " << n_cells << endl;
    cout << "Start running on " <<  omp_get_max_threads() << " threads" << endl;
    cout << "\n" << endl;

    ////////////////////////////////////////////////////////////
    // Test chis interpolation

    //vector<double> u;
    //SurfaceElement cell;
    //vector<double> dsigma;
    //vector<double> uC;
    //vector<double> dsigmaC;
    //double T, muB, chi2, chi3 ,chi4, dV1, dV2;
    //double s1 = 0.0;
    //double s2 = 0.0;
    //double tau;
    //for(int i = 0; i<selectedCells.size(); i++){
    //    cell = selectedCells[i]; 
    //    T = cell.T_f;
    //    muB = cell.mu_B;
    //    chi2 = interpchi2.eval(T, muB);
    //    chi3 = interpchi3.eval(T, muB);
    //    chi4 = interpchi4.eval(T, muB);

    //    u = {cell.u[0], cell.u[1], cell.u[2], cell.u[3]};
    //    dsigma = {cell.s[0], cell.s[1], cell.s[2], cell.s[3]};
    //    uC = MilneToCartesian(u, cell);
    //    dsigmaC = MilneToCartesianSigmaCov(dsigma, cell);

    //    tau = cell.x[0];
    //    dV1 = uC[0] * dsigmaC[0] + uC[1] * dsigmaC[1] + uC[2] * dsigmaC[2] + uC[3] * dsigmaC[3]; 
    //    s1 += dV1 * cell.rho_B;
    //    dV2 = tau * (u[0]*dsigma[0] + u[1]*dsigma[1] + u[2]*dsigma[2] + u[3]*(dsigma[3]/tau)); 
    //    s2 += dV2 * cell.rho_B;

    ////    cout << T << " " << muB << " " << chi2 << " " << chi3 << " " << chi4 <<endl;
    //}
    //cout << s1 << " " << s2 << endl;


    ///// HERE MAIN CODE ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Prepare integration
    NumericalIntegration NumInt;
    vector<double> XLeg, WLeg, XLag, WLag;

    NumInt.GetGaussLegendreCT32(XLeg, WLeg);    
    NumInt.GetGaussLaguerreCT32(XLag, WLag);    

    int NLeg = XLeg.size();
    int NLag = XLag.size();


    vector<double> OUT, OUTSTORE;
    vector<bool> OUT_CHECK;
    OUT_CHECK.resize(n_cells);
    OUT.resize(n_cells * n_double_per_cell);
    OUTSTORE.resize(n_cells * n_store_double_per_cell);
    atomic<size_t> counter(0);

    //size_t DebugCell = 1411080;
    #pragma omp parallel
    {
        size_t local_count = 0;
        #pragma omp for schedule(static)
        for (size_t k = 0; k < selectedCells.size(); ++k) {
        //for (size_t k = DebugCell-100000; k < DebugCell+100000; ++k) {
            //bool OutGet = GetProbabilityProtons(Acc, FullSpace, selectedCells[k], &OUT[k * n_double_per_cell], XLeg, WLeg, XLag, WLag, NLeg, NLag, interpchi1, interpchi2, interpchi3, interpchi4, I1_scaled_interp);
            bool OutGet = GetSTOREDDATA(Acc, FullSpace, selectedCells[k], &OUTSTORE[k * n_store_double_per_cell], XLeg, WLeg, XLag, WLag, NLeg, NLag, I1_scaled_interp);
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
    //double pvec0, pvec1, pvec2;
    //double pvec3, pvec4, pvec5;
    //double pvec6, pvec7, pvec8;
    //double pvec9, pvec10, pvec11, pvec12, pvec13, pvec14;
    //double tau, x, y, eta;
    //ofstream outFile("p_values_"+energy+"_"+centrality+".txt");

    ////outFile << "# alpha netpAcc netpFull sumpAcc sumpFull protonAcc antiprotonAcc protonFull antiprotonFull chi2 dV chi3 dV chi4 dV tau x y eta dV check" << "\n"; 
    //for(int j = 0; j < OUT_CHECK.size(); j++){
    ////    pvec0 = OUT[j*n_double_per_cell]; // alpha net proton

    ////    OutGet = OUT_CHECK[j];
    //    outFile << pvec0 << "\n"; 
    ////}
    //outFile.close();

    //outFile << "# alpha netpAcc netpFull sumpAcc sumpFull protonAcc antiprotonAcc protonFull antiprotonFull chi2 dV chi3 dV chi4 dV tau x y eta dV check" << "\n"; 
    //for(int j = 0; j < OUT_CHECK.size(); j++){
    //    pvec0 = OUT[j*n_double_per_cell]; // alpha net proton
    //    pvec1 = OUT[j*n_double_per_cell+1]; // net proton Acc
    //    pvec2 = OUT[j*n_double_per_cell+2]; // net proton Full

    //    pvec3 = OUT[j*n_double_per_cell+3]; // sum proton Acc
    //    pvec4 = OUT[j*n_double_per_cell+4]; // sum proton Full
    //    pvec5 = OUT[j*n_double_per_cell+5]; // proton Acc

    //    pvec6 = OUT[j*n_double_per_cell+6]; // antiproton Acc
    //    pvec7 = OUT[j*n_double_per_cell+7]; // protons Full
    //    pvec8 = OUT[j*n_double_per_cell+8]; // antiprotons Full

    //    pvec9 = OUT[j*n_double_per_cell+9]; // chi2 dV
    //    pvec10 = OUT[j*n_double_per_cell+10]; // chi3 dV
    //    pvec11 = OUT[j*n_double_per_cell+11]; // chi4 dV
    //    pvec12 = OUT[j*n_double_per_cell+12]; // dV
    //    pvec13 = OUT[j*n_double_per_cell+13]; // rhoB * dV
    //    pvec14 = OUT[j*n_double_per_cell+14]; // chi1 * dV


    //    tau = selectedCells[j].x[0];
    //    x = selectedCells[j].x[1];
    //    y = selectedCells[j].x[2];
    //    eta = selectedCells[j].x[3];

    //    OutGet = OUT_CHECK[j];
    //    outFile << pvec0 << "  " // 0 alpha 
    //        << pvec1 << "  " // 1 net proton Acc
    //        << pvec2 << " " // 2  net proton Full
    //        << pvec3 << " " // 3 sum proton Acc
    //        << pvec4 << " " // 4 sum proton Full
    //        << pvec5 << " " // 5 proton Acc
    //        << pvec6 << " " // 6 antiproton Acc 
    //        << pvec7 << " " // 7 proton Full 
    //        << pvec8 << " " // 8 antiprotons full
    //        << pvec9 << " " // 9 chi2 dV
    //        << pvec10 << " " // 10 chi3 dV
    //        << pvec11 << " " // 11 chi4 dV
    //        << pvec13 << " " // 12 nB dV
    //        << pvec14 << " " // 13 chi1 dV
    //        << tau << " " // 14 tau 
    //        << x << " " // 15 x
    //        << y << " " // 16 y
    //        << eta << " " // 17 eta
    //        << pvec12 << " " // 18 dV
    //        << OutGet << "\n"; // 19 check
    //}
    //outFile.close();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Write in a file STORE.
    double Svec0, Svec1, Svec2, Svec3, Svec4, Svec5, Svec6;
    ofstream outFileS("STORED_values_"+energy+"_"+centrality+".txt");

    outFileS << "# T(GeV)    muB(GeV)    muQ(GeV)    muS(GeV)    alpha   nB(fm-3)    dV(fm3)" << "\n"; 
    for(int j = 0; j < OUT_CHECK.size(); j++){
        Svec0 = OUTSTORE[j*n_store_double_per_cell];   // T (GeV) 
        Svec1 = OUTSTORE[j*n_store_double_per_cell+1]; // muB (GeV)
        Svec2 = OUTSTORE[j*n_store_double_per_cell+2]; // muQ (GeV)
        Svec3 = OUTSTORE[j*n_store_double_per_cell+3]; // muS (GeV) 
        Svec4 = OUTSTORE[j*n_store_double_per_cell+4]; // alpha
        Svec5 = OUTSTORE[j*n_store_double_per_cell+5]; // nb hydro (fm-3)
        Svec6 = OUTSTORE[j*n_store_double_per_cell+6]; // dV (fm3)

        outFileS << Svec0 << "  " // 0 T  
            << Svec1 << "  "     // 1 muB
            << Svec2 << " "      // 2 muQ
            << Svec3 << " "      // 3 muS
            << Svec4 << " "      // 4 alppha
            << Svec5 << " "      // 5 nb
            << Svec6 << "\n";    // 6 dV
    }
    outFileS.close();

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

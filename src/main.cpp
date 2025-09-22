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



void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
    x.resize(n);
    w.resize(n);
    const double EPS = 1e-14;
    int m = (n + 1) / 2;

    double p1, p2;
    for (int i = 0; i < m; ++i) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double z1;
        do {
            p1 = 1.0;
            p2 = 0.0;
            for (int j = 0; j < n; ++j) {
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1);
            }
            double pp = n * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp;
        } while (std::fabs(z - z1) > EPS);

        x[i] = -z;
        x[n - 1 - i] = z;
        double wval = 2.0 / ((1.0 - z * z) *
                             std::pow(n * (z * p1 - p2) / (z * z - 1.0), 2));
        w[i] = wval;
        w[n - 1 - i] = wval;
    }
}


double integrand2piPhi(const std::vector<double>& u, const std::vector<double>& dsigma,
    double T, double m, double pT, double Y) {
    double mT = std::sqrt(pT * pT + m * m);
    double p0 = mT * std::cosh(Y);
    double pz = mT * std::sinh(Y);

    // exp[-(u0 p0 - u3 pz)/T]
    double expo1 = u[0] * p0 - u[3] * pz;
    double expfactor = std::exp(-expo1 / T);

    // transverse flow magnitude
    double a = u[1] * pT / T;
    double b = u[2] * pT / T;
    double r = std::sqrt(a * a + b * b);


    double I0 = boost::math::cyl_bessel_i(0, r);
    double I1 = boost::math::cyl_bessel_i(1, r);

    double term0 = dsigma[0] * p0 * 2. * M_PI * I0;
    double term1 = (r > 0.0 ? dsigma[1] * pT * 2. * M_PI * I1 * a / r : 0.0);
    double term2 = (r > 0.0 ? dsigma[2] * pT * 2. * M_PI * I1 * b / r : 0.0);
    double term3 = dsigma[3] * pz * 2. * M_PI * I0;

    return pT * expfactor * (term0 + term1 + term2 + term3);
}


double CalculateAcceptanceProbability2piPhi(const std::vector<double>& u,
    const std::vector<double>& dsigma,
    double T, double m,
    double Ymin, double Ymax,
    double pTmin, double pTmax,
    int nY = 32, int nPT = 32) {
    // Full denominator integral (all rapidities, all pT)
    double I_total = 0.0;
    // Restricted numerator integral (acceptance region)
    double I_accept = 0.0;

    // Gauss–Legendre nodes and weights
    std::vector<double> xY, wY, xPT, wPT;
    gauss_legendre(nY, xY, wY);
    gauss_legendre(nPT, xPT, wPT);

    // Loop over rapidity and pT
    for (int iY = 0; iY < nY; iY++) {
        double Y = 0.5 * (Ymax - Ymin) * xY[iY] + 0.5 * (Ymax + Ymin);
        double wYt = wY[iY] * 0.5 * (Ymax - Ymin);

        for (int ipT = 0; ipT < nPT; ipT++) {
            double pT = 0.5 * (pTmax - pTmin) * xPT[ipT] + 0.5 * (pTmax + pTmin);
            double wPTt = wPT[ipT] * 0.5 * (pTmax - pTmin);

            double val = integrand2piPhi(u, dsigma, T, m, pT, Y);
            double weight = wYt * wPTt;

            I_total += val * weight;

            // Example: restrict to a smaller acceptance window in Y and pT
            if (std::fabs(Y) < 1.0 && pT > 0.2 && pT < 2.0) {
                I_accept += val * weight;
            }
    }      
    }

    // Return ratio (probability)
    return (I_total > 0.0 ? I_accept / I_total : 0.0);
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
    cout << "reading freeze-out surface" << endl;

    vector<SurfaceElement> surface;

    bool boost_invariant = true;
    ostringstream surfdat_stream;
    surfdat_stream << Surfpath;

    // new counting, mac compatible ...
    int NCells = get_number_of_lines_of_binary_surface_file(surfdat_stream.str());

    cout << "NCells = " << NCells << endl;
    flush(cout);

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

double H(double pT, double y, double phi, double mu, double* u, double mp){
    return sqrt(mp*mp + pT*pT) * (u[0]*cosh(y) - u[3]*sinh(y)) - pT * (u[1] * cos(phi) + u[2] * sin(phi)) - mu;
}

double func(double H, double T, double y, double pT, double mp){
    return pT * exp(-H/T)/(cosh(y) * sqrt(mp * mp + pT * pT));
}

double commonIntegrandN0N3(double pT, double y, double* u, double T, double mu, double m){
    double mT = sqrt(pT * pT + m * m);
    //double I0 = jn(0, pT/T * sqrt(u[1] * u[1] + u[2] * u[2]));
    double I0 = boost::math::cyl_bessel_i(0, pT/T * sqrt(u[1] * u[1] + u[2] * u[2]));
    double EXP = exp(-mT / T * (cosh(y) * u[0] - sinh(y) * u[1]));
    return pT * mT * I0 * EXP;
} /// Perform integral over pT once for both N0 and N3: simply multiply integrals by cosh and sinh then perform integral on y

double commonIntegrandN1N2(double pT, double y, double* u, double T, double mu, double m){
    double mT = sqrt(pT * pT + m * m);
    //double I1 = jn(1, pT/T * sqrt(u[1] * u[1] + u[2] * u[2]));
    double I1 = boost::math::cyl_bessel_i(1, pT/T * sqrt(u[1] * u[1] + u[2] * u[2]));
    double EXP = exp(-mT / T * (cosh(y) * u[0] - sinh(y) * u[1]));
    return pT * pT * EXP * I1;
} // perform integral on both of pT and y and simplyx multiply by u[1]/aver2_u and u[2]/aver2_u; aver2_u = sqrt(u[1] * u[1] + u[2] * u[2])

double integrateCommonN0N3(double pTm, double pTM, int N, double y, double* u, double T, double mu, double m) {
    double dpT = (pTM - pTm) / N;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double pT = pTm + (i + 0.5) * dpT;
        sum += commonIntegrandN0N3(pT, y, u, T, mu, m);
    }
    return sum * dpT; 
}
vector<double> integrateN0N3(double ym, double yM, int Ny, double pTm, double pTM, int NpT, double* u, double T, double mu, double m){
    vector<double> out;
    double dy = (yM - ym) / Ny;
    double sum1 = 0.0;
    double sum2 = 0.0;

    double temp;
    for (int i = 0; i < Ny; ++i) {
        double y = ym + (i + 0.5) * dy;
        temp = integrateCommonN0N3(pTm, pTM, NpT, y, u, T, mu, m);
        sum1 += cosh(y) * temp;
        sum2 += sinh(y) * temp;
    }
    out.push_back(sum1 * 2 * M_PI * exp(mu/T));
    out.push_back(sum2 * 2 * M_PI * exp(mu/T));
    out.resize(out.size());
    return out; 
}


vector<double> integrateN1N2(double ym, double yM, int Ny, double pTm, double pTM, int NpT, double* u, double T, double mu, double m) {
    vector<double> out;
    double dy = (yM - ym) / Ny;
    double dpT = (pTM - pTm) / NpT;
    double sum = 0.0;

    for (int i = 0; i < Ny; ++i) {
        double y = ym + (i + 0.5) * dy;
        for (int j = 0; j < NpT; ++j) {
            double pT = pTm + (j + 0.5) * dpT;
            sum += commonIntegrandN1N2(pT, y, u, T, mu, m);
        }
    }
    double aver_u = sqrt(u[1]*u[1] + u[2]*u[2]);
    out.push_back(sum * 2 * M_PI * exp(mu/T) * u[1]/aver_u);
    out.push_back(sum * 2 * M_PI * exp(mu/T) * u[2]/aver_u);
    out.resize(out.size());
    return out;
}

double GetnB(Acceptance Acc, surfaceElement Surf, double m = 0.938, int Ny = 100, int NpT = 100){
    vector<double> N0_N3;vector<double> N1_N2;
    N0_N3 = integrateN0N3(Acc.ymin, Acc.ymax, Ny, Acc.pTmin, Acc.pTmax, NpT, Surf.u, Surf.T_f, Surf.mu_B, m);
    N1_N2 = integrateN1N2(Acc.ymin, Acc.ymax, Ny, Acc.pTmin, Acc.pTmax, NpT, Surf.u, Surf.T_f, Surf.mu_B, m);
    return 1/(2 * 2 * 2 * M_PI * M_PI * M_PI) * (Surf.s[0] * N0_N3[0] - Surf.s[1] * N1_N2[0] - Surf.s[2] * N1_N2[1] - Surf.s[3] * N0_N3[1]);
}


int main(int argc, char *argv[]) {
    const string SurfacePath = "surfacesFiles/"; 
    Acceptance Acc;
    Acc.pTmin = 0.4; Acc.pTmax = 2;
    Acc.ymin = -0.5; Acc.ymax = 0.5;

    Acceptance FullSpace;
    double num = 3.0;
    FullSpace.pTmin = 0.0; FullSpace.pTmax = num;
    FullSpace.ymin = -num; FullSpace.ymax = num;

    string energy = argv[1];
    string centrality = argv[2];

    const string fname = "AuAu"+energy+"/hydro_results_C"+centrality+"/surface_eps_0.26.dat";

    vector<SurfaceElement> surface;
    surface = ReadFreezeOutSurface(SurfacePath+fname); 





    // Example input

    // std::vector<double> u     = {1.0, 0.1, 0.0, 0.0};   // flow 4-velocity
    // std::vector<double> dsig  = {1.0, 0.0, 0.0, 0.0};   // surface normal
    // double T = 0.155;   // GeV
    // double m = 0.14;    // pion mass
    // double Ymin = -5.0, Ymax = 5.0;
    // double pTmin = 0.0, pTmax = 3.0;

    // double prob = CalculateAcceptanceProbability2piPhi(u, dsig, T, m, Ymin, Ymax, pTmin, pTmax);
    // std::cout << "Acceptance probability = " << prob << std::endl;

    double p;
    std::ofstream outFile("nb_values.txt");
    for (int i = 0; i < 1882601; ++i) {
        if(i%1000 ==0){
            cout << i/1882601.0 * 100 << endl;
        }

        //p = ProbaAcc(Acc, FullSpace, surface[i]);
        p = GetnB(FullSpace, surface[i]);
        outFile << p << " " << surface[i].rho_B << "\n";
    }

    outFile.close();
    return 0;
}
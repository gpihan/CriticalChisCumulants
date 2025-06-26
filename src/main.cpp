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
#include <random>

using namespace std;

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


typedef struct selectel {
    double p;
    double deltaVeff;
    double deltaVeffC;
    double u[4];
    double x[4];
    double sinheta;
    double cosheta;
} SelectEl;

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

//double fproduct(const double A[4], const double B[4], double tau) {
//    //return -A[0]*B[0] + A[1]*B[1] + A[2]*B[2] + tau * tau * A[3]*B[3];
//    return -A[0]*B[0] + A[1]*B[1] + A[2]*B[2] + tau * tau * A[3]*B[3];
//}
double getdVeff(const double u[4], const double da[4], double tau){
    return tau * (da[0] * u[0] + da[1] * u[1] + da[2] * u[2] + da[3] * u[3]/tau);
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




int main(int argc, char *argv[]) {
    const string SurfacePath = "surfacesFiles/"; 
    string energy = argv[1];
    string centrality = argv[2];

    bool CartesianCoord = true;

    const string fname = "AuAu"+energy+"/hydro_results_C"+centrality+"/surface_eps_0.26.dat";

    vector<SurfaceElement> surface;
    surface = ReadFreezeOutSurface(SurfacePath+fname); 

    double sumdVeff = 0.0;
    double sumdVeffC = 0.0;
    double sumNB = 0.0;
    double sumNBC = 0.0;
    double deltaVeff, deltaVeffC;
    long int Nd = 0;
    double uu0, uu1, uu2, uu3;
    double da0, da1, da2, da3;
    //double sh_eta_s, ch_eta_s;
    double sinheta, cosheta;

    double uc0, uc1, uc2, uc3;
    double dsigma0, dsigma1, dsigma2, dsigma3;
    double dsigmac0, dsigmac1, dsigmac2, dsigmac3;
    vector<double> surf_prob_vec;

    vector<SelectEl> SelSurf;

    for (auto& surface_element:surface){
        Nd+=1;
        //deltaVeff = getdVeff(surface_element.u, surface_element.s, surface_element.x[0]);

        // Get u mu in Milne coordinates
        uu0 = surface_element.u[0]; 
        uu1 = surface_element.u[1]; 
        uu2 = surface_element.u[2]; 
        uu3 = surface_element.u[3]; 

        // Get dsigma in Milne coordinates 
        da0 = surface_element.s[0];
        da1 = surface_element.s[1];
        da2 = surface_element.s[2];
        da3 = surface_element.s[3];

        deltaVeff = surface_element.x[0] * (da0 * uu0 + da1 * uu1 + da2 * uu2 + da3 * uu3 / surface_element.x[0]);

        // Get cosh and sinh eta
        sinheta = surface_element.sinh_eta_s;
        cosheta = surface_element.cosh_eta_s;

        // convert u and dsigma into cartesian coordinates
        uc0 = uu0 * cosheta + uu3 * sinheta;
        uc1 = uu1;
        uc2 = uu2;
        uc3 = uu0 * sinheta + uu3 * cosheta;

        dsigmac0 = surface_element.x[0] * cosheta * da0 - sinheta * da3;
        dsigmac1 = surface_element.x[0] * da1;
        dsigmac2 = surface_element.x[0] * da2;
        dsigmac3 = -(surface_element.x[0] * sinheta * da0 - cosheta * da3);

        // Get deltaVeff in cartesian
        deltaVeffC = uc0 * dsigmac0 + uc1 * dsigmac1 + uc2 * dsigmac2 + uc3 * dsigmac3;


        if(deltaVeffC > 0.0){
            sumdVeffC += deltaVeffC;
            sumNBC += surface_element.rho_B * deltaVeffC;

            SelectEl temp;
            temp.deltaVeffC = deltaVeffC; 

            temp.u[0] = surface_element.u[0];
            temp.u[1] = surface_element.u[1];
            temp.u[2] = surface_element.u[2];
            temp.u[3] = surface_element.u[3];

            temp.x[0] = surface_element.x[0];
            temp.x[1] = surface_element.x[1];
            temp.x[2] = surface_element.x[2];
            temp.x[3] = surface_element.x[3];

            temp.sinheta = sinheta;
            temp.cosheta = cosheta;

            SelSurf.push_back(temp);
        }
    }

    SelSurf.resize(SelSurf.size());

    int NB = sumNBC;

    cout << "V = " << sumdVeffC << " NB = " << (int) sumNBC << endl;


    // Calculate probabilities = dVeff/(sum dVeff)
    
    for(auto& el:SelSurf){
        el.p = el.deltaVeffC/sumdVeffC;
        surf_prob_vec.push_back(el.p);
    }

    surf_prob_vec.resize(surf_prob_vec.size());

    // Roll the dice NB time to get the NB surface cells. 
    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> dist(surf_prob_vec.begin(), surf_prob_vec.end());

    vector<int> results(NB);

    for (int i = 0; i < NB; ++i) {
        results[i] = dist(gen);
    }

    // writting the file with the selected cells
    std::ofstream outFile("cellValues.txt");
    outFile <<  "# t x y z u0 u1 u2 u3" << "\n";
    double t, z;
    SelectEl ele;
    for (int i = 0; i < NB; ++i) {
        ele = SelSurf[(int) results[i]];

        // Get u mu in Milne coordinates
        uu0 = ele.u[0]; 
        uu1 = ele.u[1]; 
        uu2 = ele.u[2]; 
        uu3 = ele.u[3]; 

        // Get cosh and sinh eta
        sinheta = ele.sinheta;
        cosheta = ele.cosheta;

        // convert u and dsigma into cartesian coordinates
        uc0 = uu0 * cosheta + uu3 * sinheta;
        uc1 = uu1;
        uc2 = uu2;
        uc3 = uu0 * sinheta + uu3 * cosheta;

        // Get surf cartesian x
        t = ele.x[0] * cosheta; 
        z = ele.x[0] * sinheta;

        outFile <<  t
            << " " << ele.x[1] 
            << " " << ele.x[2]
            << " " << z
            << " " << uc0
            << " " << uc1
            << " " << uc2
            << " " << uc3
            << "\n";
    }

    outFile.close();
    return 0;
}

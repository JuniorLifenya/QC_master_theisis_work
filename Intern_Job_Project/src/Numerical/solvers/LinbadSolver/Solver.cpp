#include <vector>
#include <complex>
#include <cmath>
#include <Eigen/Dense>

using ComplexMatrix = Eigen::MatrixXcd;
using complexvector = Eigen::VectorXcd;


class SimulationEngine: 
{

private:
    /* data Using test data for now*/
    double h_max = 1.0;
    double omega_gw = 3.14;

    ComplexMatrix H0; // Set value before use later as well
    ComplexMatrix H_int; // Set value before use later
    ComplexMatrix Sz,Sx,Sy; 
    ComplexMatrix psi0,psi_p1,psi_m1;
    

public: 
    // Time-dependent coefficient for Hamiltonia: h(t)
    double _strain_func(double t,double h){
        return h_max * std::sin(omega_gw * t);
    } // In python this was return args['h_max'] * np.sin(args['omega_gw'] * t)

    void run(double t){
        // 1. Evaluate the time-dependent Hamiltonian for this specific time 't'
        // In C++ (with Eigen), we can multiply a matrix by a double directly!
        ComplexMatrix H_current = H0 + (H_int*_strain_func(t));

        // Alternative We cannot simply have H = H0 + H_int(t), we must loop
        // for (double val: H_int_t){
        //     H.push_back(H0 + val);
        // } 
        // 'H' now contains {H0 + H_int, H0 + h(t)}

        std::vector<ComplexMatrix> e_ops = {
            psi0 * psi0.adjoint(),
            psi_p1 * psi_p1.adjoint(),
            psi_m1 * psi_m1.adjoint(),
            Sz,Sx,Sy
        };

        std::vector<ComplexMatrix> c_ops;
        if (/* condition for using mesolve */) {
            // Add your dissipation matrices here. For example:
            // c_ops.push_back(some_decay_matrix);
        }
        return 
    }

    
    
}


int main{






    return 0
};
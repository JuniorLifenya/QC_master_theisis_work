#include <vector>
#include <eigen-3.4.0>
#include <complex>
#include <Eigen/dense> 

typedef std::complex<double> = cd; 
using complexMatrix = Eigen::MatrixXcd;
using complexVector = Eigen::VectorxXcd; 


MatrixXd static_hamiltonian(double D, double gamma_e, double Bz)
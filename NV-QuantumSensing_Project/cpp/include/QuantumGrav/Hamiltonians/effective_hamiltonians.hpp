#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <complex>

using complexMatrix = Eigen::MatrixXcd;
using complexVector = Eigen::VectorXcd; 
typedef std::complex<double> = cd;


MatrixXd Build_TT_Hamiltonian(double h, int x , int p ,int N, double lambda, double xmin, double xmax)
{ 
    MatrixXd H = MatrixXd::Zero(N, N);       // Initialize the Hamiltonian matrix
    double dx = 0.1;                         // Spatial step size
    double hbar = 1.0, m = 1.0, omega = 1.0; // Constants for the harmonic oscillator

    vectorXcd p = -i* std::<vector> gradient = {dx,dy,dz};
    double H0 =  (hbar* hbar) / (2*m * dx*dx) + m; 
    double H_TT_int =(kappa/2*m)*(p.adjoint()*h*p + sigma/2*(B_eff))
    double H_TT_eff = H0 + H_TT_int
    
    for (int i = 0; i < N; i++)
    {
        H(i, i) = 2 * kin; // Diagonal elements
        if (i > 0)
            H(i, i - 1) = -kin; // Off-diagonal elements
        if (i < N - 1)
            H(i, i + 1) = -kin; // Off-diagonal elements
    }
    
    // Add anharmonic potential: V(x) = ½mω²x² + λx⁴
    for (int i = 0; i < N; i++)
    {
        double x = (i - N / 2.0) * dx;                                       // Position in the grid so center at x=0
        H(i, i) += 0.5 * m * omega * omega * x * x + lambda * x * x * x * x; // Harmonic potential term
    }
}


MatrixXd Build_FNC_Hamiltonian(double h, int x , int p ,int N, double lambda, double xmin, double xmax)
{

}
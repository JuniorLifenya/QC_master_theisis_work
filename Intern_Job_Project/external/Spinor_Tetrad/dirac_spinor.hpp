#pragma once
#include <eigen-3.4.0/Eigen/Dense>
#include <complex>
#include <iostream>
#include <vector>
#include <cmath>

//---- Utilities -----------------------------------

//--------------------------------------------------

// Dirac gamma matrices in Dirac Representation
// We can define a spinor as Eigen::Vector4cd
inline Matrix4cd gamma0()
{
    Matrix4cd g = Matrix4cd::Zero(); // Here we initialize
    g(0, 0) = 1;
    g(1, 1) = 1;
    g(2, 2) = -1;
    g(3, 3) = -1;
    return g;
}

inline Matrix4cd gammai(int i)
{
    // First define the gamma matrices
    Matrix2cd sigmaX, sigmaY, sigmaZ;
    sigmaX << 0, 1, 1, 0;
    sigmaY << 0, -complex<double>(0, 1), complex<double>(0, 1), 0;
    sigmaZ << 1, 0, 0, -1;

    // Store the sigma matrices in an array
    Matrix2cd sigma[3] = {sigmaX, sigmaY, sigmaZ};
    Matrix4cd g = Matrix4cd::Zero();
    g.block<2, 2>(0, 2) = sigma[i];
    g.block<2, 2>(2, 0) = -sigma[i];

    return g;
}

// The inline function provides a way to optimize the performance of the program by reducing
// the overhead related to a function call.When a function is specified like this the whole code
// of the inline function is inserted at each point the function is called,
// instead of performing a regular function call.
// So the inline keyword suggests the compiler that it should replace the functions call with the actual
// code of the function to avoid the overheads of the function call. its only a request to the compiler

// Another approach of just using the libraries
class DiracSpinor
{
public:
    DiracSpinor();                                // Default constructor, so by default the spinor is initialized to zero
    explicit DiracSpinor(const Vec4c &components); // Initializaed with components now

    // Lorentz transformation(boost + rotation )
    void applyLorentzTransform(const Mat4c &transform_matrix);

    // Time evolution under a Hamiltonian
    void evolve(const Mat4c &hamiltonian, double dt);

    // Get density matrix for decoherence calculations
    Mat4c densityMatrix() const;

    // Calculate expectation values
    cplx expectationValue(const Mat4c &operator_matrix) const;

    // Get current state
    const Vec4c &getState() const { return components_; }

private:
    Vec4c components_; // Spinor components
};

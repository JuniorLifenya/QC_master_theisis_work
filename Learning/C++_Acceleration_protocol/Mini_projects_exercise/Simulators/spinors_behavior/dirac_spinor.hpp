#pragma once
#include <eigen-3.4.0/Eigen/Dense> // Updated include path

// Dirac gamma matrices in Dirac Representation

using namespace Eigen;
using namespace std;

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
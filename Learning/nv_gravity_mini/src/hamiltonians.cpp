#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <filesystem> // Will be used for creating files and sending stuff to files

//------ Utilities ---------------------------------------------
using namespace Eigen;
using namespace std;
using Matrix4cd = Matrix<complex<double>, 4, 4>;
using Matrix2cd = Matrix<complex<double>, 2, 2>;
using Vector2cd = Vector<complex<double>, 2>;
using Vector4cd = Vector<complex<double>, 4>;
using Matrix4d = Matrix<double, 4, 4>;
using Matrix2d = Matrix<double, 2, 2>;
using Vector2d = Vector<double, 2>;
using Vector4d = Vector<double, 4>;

using Mat4c = Eigen::Matrix4cd;
using Mat2c = Eigen::Matrix2cd;
using Vec4c = Eigen::Vector4cd;
using Vec2c = Eigen::Vector2cd;

using Mat4 = Matrix4d;
using Mat2 = Matrix2d;
using Vec4 = Vector4d;
using Vec2 = Vector2d;

using cplx = complex<double>;

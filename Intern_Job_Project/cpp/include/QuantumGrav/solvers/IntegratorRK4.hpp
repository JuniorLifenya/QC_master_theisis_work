#include <iostream>
#include <cmath>
#include <vector>

using real = double;
using namespace std;

// Derivative for solving like dy/dx = (sin x - 5 y^2)/3
real dydx(real x, real y)
{
    return (std::sin(x) - 5.0 * y * y) / 3.0;
}

// Single‐step scalar RK4 from x0→xf
real rungeKuttaScalar(real x0, real y0, real xf, real h)
{
    int n = int((xf - x0) / h); // should be 2
    real y = y0;
    for (int i = 0; i < n; i++)
    {
        real k1 = h * dydx(x0, y);
        real k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1);
        real k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2);
        real k4 = h * dydx(x0 + h, y + k3);
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
        x0 += h;
    }
    return y;
}

// Vector‐based RK4 storing t[0…n], y[0…n]
void rungeKuttaVector(std::vector<real> &t,
                      std::vector<real> &y,
                      real x0, real y0, real h, int n)
{
    t[0] = x0;
    y[0] = y0;
    for (int i = 0; i < n; i++)
    {
        real ti = t[i], yi = y[i];
        real K1 = dydx(ti, yi);
        real K2 = dydx(ti + 0.5 * h, yi + 0.5 * h * K1);
        real K3 = dydx(ti + 0.5 * h, yi + 0.5 * h * K2);
        real K4 = dydx(ti + h, yi + h * K3);
        y[i + 1] = yi + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6.0;
        t[i + 1] = ti + h;
    }
}

int main()
{
    real x0 = 0.3, y0 = 5.0, xf = 0.9, h = 0.3;
    int n = int((xf - x0) / h); // 2 steps

    // — Scalar call —
    real y_scalar = rungeKuttaScalar(x0, y0, xf, h);
    std::cout << "Scalar RK4: y(0.9) = " << y_scalar << "\n";

    // — Vector call —
    std::vector<real> t(n + 1), y(n + 1);
    rungeKuttaVector(t, y, x0, y0, h, n);
    std::cout << "Vector RK4: y(0.9) = " << y[n] << "\n";

    return 0;
}
// Expected output:
// Scalar RK4: y(0.9) = -1261.5
// Code execution CT_exercising/MAIN_NumericalSolvers$ g++ -std=c++11 -I eigen runge_kutta_geeks4geeks_EXTENDED.cpp -o lol, then type : ./lol


// A sample differential equation "dy/dx = (x - y)/2"
double dydx(float x, float y)
{
    return (x - y) / (2);
}

// Finds value of y for a given x using step size h
// and initial value y0 at x0.
double rungeKutta(float x0, float y0, float xf, float h)
{
    // Count number of iterations using step size or
    // step height h
    double delta = (static_cast<double>((xf) - (x0)) / h);
    int n = static_cast<int>(round(delta));

    double k1,
        k2, k3, k4, k5;

    // Iterate for number of iterations
    double y = y0;
    for (int i = 1; i <= n; i++)
    {
        // Apply Runge Kutta Formulas to find
        // next value of y
        double k1 = h * dydx(x0, y);
        double k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1);
        double k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2);
        double k4 = h * dydx(x0 + h, y + k3);

        // Update next value of y
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        ;

        // Update next value of x
        x0 = x0 + h;
    }

    return y;
}

// Driver Code
int main()
{
    float x0 = 0, y0 = 1, xf = 2, h = 0.2;
    cout << "The value of y at x is : " << rungeKutta(x0, y0, xf, h);

    return 0;
}

// This code is contributed by code_hunt. Sources for information GeeksforGeeks.org
// Expected Output: The value of y at x is : 1.10364


// Declaration of functions
double f(const double &t, const double &x, const double &y);
double g(const double &t, const double &x, const double &y);
void Runge_Kutta(std::vector<double> &x, std::vector<double> &y, std::vector<double> &t, const double &h, const int &n);
void printVector(std::vector<double> &v);

int main()
{
    // This code solves a system of ordinary differential equations using the Runge-Kutta method.
    // For systems like this one here :
    // dx/dt = f(t,x,y,z) = x + 2 * y
    // dy/dt = g(t,x,y,z) = 3*x+2*y
    int n;
    std::cout << "Please enter the # of iterations:" << std::endl;
    std::cin >> n;

    std::vector<double> t(n + 1); // Can also be written more compactely as std:: vector<double> t(n),x(n),y(n);
    std::vector<double> x(n + 1);
    std::vector<double> y(n + 1);
    double h = 0.02;

    t[0] = 0;
    x[0] = 6;
    y[0] = 4;

    Runge_Kutta(x, y, t, h, n);

    std::cout << "t= ";
    printVector(t);
    std::cout << "x= ";
    printVector(x);
    std::cout << "y= ";
    printVector(y);

    return 0;
}

// Now lets try the problem with dydx=(sin(x)-5y^2)/3 and conditions: y(0.3) = 5 . Checking if y(0.9)= -1261.5
// Definition of functions
double f(const double &t, const double &x, const double &y)
{
    return x + 2 * y;
}

double g(const double &t, const double &x, const double &y)
{

    return 3 * x + 2 * y;
}

void Runge_Kutta(std::vector<double> &x, std::vector<double> &y, std::vector<double> &t, const double &h, const int &n)
{
    for (int i = 0; i < n - 1; i++)
    {
        double K1 = f(t[i], x[i], y[i]);
        double G1 = g(t[i], x[i], y[i]);

        double K2 = f(t[i] + h / 2., x[i] + h / 2. * K1, y[i] + h / 2. * G1);
        double G2 = g(t[i] + h / 2., x[i] + h / 2. * K1, y[i] + h / 2. * G1);

        double K3 = f(t[i] + h / 2., x[i] + h / 2. * K2, y[i] + h / 2. * G2);
        double G3 = g(t[i] + h / 2., x[i] + h / 2. * K2, y[i] + h / 2. * G2);

        double K4 = f(t[i] + h, x[i] + h * K3, y[i] + h * G3);
        double G4 = g(t[i] + h, x[i] + h * K3, y[i] + h * G3);

        x[i + 1] = x[i] + h / 6. * (K1 + 2 * K2 + 2 * K3 + K4);
        y[i + 1] = y[i] + h / 6. * (G1 + 2 * G2 + 2 * G3 + G4);
        t[i + 1] = t[i] + h;
    }
}

void printVector(std::vector<double> &v)
{
    std::cout << "[";
    for (double element : v)
    {
        std::cout << element << ",";
    }
    std::cout << "]" << std::endl;
}

// Source for this code is mainly a comprehensive youtube video, GeeksforGeeks and Stackoverflow.
// Code execution here : g++ -std=c++11 -I eigen Runge-Kutta_MatrixSys_for_CODE.cpp -o lol output:
//
#include <iostream>
#include <cmath>

class Complex
{
public:
    double real, imag;
    Complex(double r, double i) : real(r), imag(i) {}

    // Required operations for quantum:
    Complex operator*(const Complex &other) const
    {
        return Complex(real * other.real - imag * other.imag,
                       real * other.imag + imag * other.real);
    }

    double norm() const
    {
        return std::sqrt(real * real + imag * imag);
    }

    static Complex polar(double r, double theta)
    {
        return Complex(r * std::cos(theta), r * std::sin(theta));
    }
};

// Test with quantum state: e^{iπ}
int main()
{
    Complex i_pi = Complex::polar(1.0, M_PI);
    std::cout << "e^{iπ} = " << i_pi.real << " + " << i_pi.imag << "i\n";
}
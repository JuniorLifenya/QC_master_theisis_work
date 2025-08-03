#include <iostream>
#include <limits>
#include <cmath>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////

void celsius_farenheit_converter()
{
    cout << "Enter a temperature in Celsius: ";
    double celsius;
    cin >> celsius;
    double fahrenheit = (celsius * 9 / 5) + 32;
    cout << celsius << "°C is equal to " << fahrenheit << "°F " << endl;

    // Flush newline left in the buffer
    cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

void farenheit_celsius_converter()
{
    cout << "Enter a temperature in Farenheit: ";
    double farenheit;
    cin >> farenheit;
    double celsius = (farenheit - 32) / (1.8); // Cannot use 5/9 because it's integer division. Giving 0
    cout << farenheit << "°F is equal to " << celsius << "°C " << endl;
}

//////////////////////////////////////////////////////////////////////////////////////

void calc_sircle_area()
{
    cout << "Give me the radius of the circle in meters r = ";
    double radius;
    cin >> radius;
    double area = 3.14 * (pow(radius, radius));
    cout << "The area of the circle (\u03C0r\u00B2 ) is: " << area << " m\u00B2 " << endl;
}

//////////////////////////////////////////////////////////////////////////////////////

void square_function()
{
    cout << "Enter something you wanna square: ";
    double something_to_square;
    cin >> something_to_square;
    double square_function = pow(something_to_square, 2);
    cout << "The square of " << something_to_square << " is : " << square_function << endl;
}

//=======================================================================

void raiser_function()
{

    cout << "Enter something you wanna raise to some power: ";
    double something_power;
    cin >> something_power;
    cout << "Enter the power: ";
    double power;
    cin >> power;

    double power_function = pow(something_power, power);
    cout << something_power << " raised to the " << power << " is: " << power_function << endl;
}
/////////////////////////////////////////////////////////////////////////////

int main()
{

    // cout << "--- Celsius to Fahrenheit ---" << endl;
    // celsius_farenheit_converter();

    cout << " " << endl;

    // cout << "--- Fahrenheit to Celsius ---" << endl;
    // farenheit_celsius_converter();

    cout << " " << endl;

    cout << "---- Circle area ---" << endl;
    calc_sircle_area();

    cout << " " << endl;

    cout << "--- Square function ---" << endl;
    square_function();

    cout << " " << endl;

    cout << "--- Raiser function ---" << endl;
    raiser_function();

    cout << " " << endl;

    return 0;
}

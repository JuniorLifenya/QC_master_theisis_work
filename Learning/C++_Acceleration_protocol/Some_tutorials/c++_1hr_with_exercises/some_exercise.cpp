#include <iostream>
#include <limits>

using namespace std;

struct Person
{ // This is a struct, it's like a class but with default public access
    std::string name;
    int age;
};

//////////////////////////////////////////////////////////////////////////////////////

void swap_variables() // This is a function that swaps two variables
{
    int a = 1, b = 2;
    int temp = a;      // Saves a value into temp
    a = b;             // Now a COPIES the value of b now both a and b are 2, temp is 1
    b = temp;          // Now b takes temp's value which is a's previous value
    cout << a << endl; // prints 2
}

//////////////////////////////////////////////////////////////////////////////////////
void math_expression() // Calculate a simple expression
{
    double x = 10, y = 5;
    double z = (x + 10) / (3 * y);
    cout << z << endl;
}

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
    double celsius = (farenheit - 32) * (5 / 9);
    cout << farenheit << "°F is equal to " << celsius << "°C " << endl;
}

//////////////////////////////////////////////////////////////////////////////////////

int main()
{
    cout << "--- Swapping variables ---" << endl;
    swap_variables();

    cout << "--- Math expression ---" << endl;
    math_expression();

    cout << "--- Celsius to Fahrenheit ---" << endl;
    celsius_farenheit_converter();

    cout << "--- Fahrenheit to Celsius ---" << endl;
    farenheit_celsius_converter();

    return 0;
}

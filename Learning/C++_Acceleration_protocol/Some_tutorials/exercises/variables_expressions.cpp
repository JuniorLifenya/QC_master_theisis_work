#include <iostream>
#include <limits>
#include <cmath>

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
    cout << "Before swap: a = " << a << ", b = " << b << endl;
    int temp = a; // Saves a value into temp
    a = b;        // Now a COPIES the value of b now both a and b are 2, temp is 1
    b = temp;     // Now b takes temp's value which is a's previous value
    cout << "After swap: a = " << a << ", b = " << b << endl;
    // Outputs: a = 2, b = 1
}

//////////////////////////////////////////////////////////////////////////////////////
void math_expression() // Calculate a simple expression
{
    double x = 10, y = 5;
    double z = (x + 10) / (3 * y);
    cout << "The answer to the expression (x+1)/3y is: " << z << endl;
}

int main()
{

    cout << "--- Swapping variables ---" << endl;
    swap_variables();

    cout << " " << endl;

    cout << "--- Math expression ---" << endl;

    cout << " " << endl;

    math_expression();
}
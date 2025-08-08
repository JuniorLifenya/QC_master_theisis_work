#include <iostream> // for input/output, we can print and get input from the user
// iostream is one of the files from the standard library
// main is the entrypoint of our program.
#include <cmath>

namespace first
{
    int x = 0; // This is a variable in the first namespace

}

namespace second
{
    int x = 1; // This is a variable in the second namespace
}

// To use the variables from the namespaces, we can use the scope resolution operator ::
// Inside int main() we can access them like this:
// first::x and second::x

int main() // Int is the type of value returned by main.
{
    // ======================= Intro with printing ===========================

    std ::cout << "This is character out "; // No std::endl; makes no new line in terminal output

    //========================================================================
    //========================== Variables ===================================

    int some_size = 100;                            // ALWAYS Initialise variables, before using them
    double sales = 9.99;                            // double is a type of number with decimal places
    std ::cout << sales << some_size << std ::endl; // prints variables , std ::endl finishes the line with \n

    // Naming conventions (examples only — not used in code)
    int file_size; // Snake Case
    int FileSize;  // Pascal Case for naming classes
    int fileSize;  // Camel Case
    int iFileSize; // Hungarian Notation
    // Compiler will warm that these variables are declared but not used.

    //=======================================================================
    //=====================Constant Variables ===============================

    const double pi = 3.14;
    // pi = 0; <-- This will not compile, because pi is a constant and cannot be changed

    //=======================================================================
    //===================== Writing / Reading with console ==================

    int x = 10;
    std ::cout << "x = " << x << std ::endl; // Chaining output statements

    // Read input from user and store it in a specified/declared variable
    using namespace std;
    cout << " Enter a value: ";
    int value;    // Declare variable as integer before using it
    cin >> value; // cin is the input stream( reading ), >> is the extraction operator
    cout << "You entered: " << value << endl;
    // prints the value, if not integer , it will round to integer or print 0

    //=======================================================================
    //===================== Namespace =======================================
    // Namespace first appeared in C++ to avoid name conflicts.
    // std is the standard namespace, it contains all the standard library functions and classes.
    // To avoid having to redfine new variables like this

    int x = 0;
    int x = 1; // This will not compile, because x is already defined in the same scope
    // To avoid this, we can use different namespaces or use the std namespace.

    cout << first ::x << std ::endl;  // Accessing variable from first namespace
    cout << second ::x << std ::endl; // Accessing variable from second namespace

    //===================== Different namespace =============================

    using first::x;      // Can also declare namespace usage inside main
    using second::x;     // Can also declare namespace usage inside main
    using std::cin;      // Using cin from the std namespace
    using std::cout;     // Using cout from the std namespace
    using namespace std; // Using the entire std namespace ( all of the above )

    //=======================================================================
    //===================== Different namespace =============================

    return 0;
}
// std is short for standard,
// it's a namespace that contains many useful functions and classes.

//============================================================================

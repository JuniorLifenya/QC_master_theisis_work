#include <iostream> // for input/output, we can print and get input from the user
// iostream is one of the files from the standard library
// main is the entrypoint of our program.

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

    return 0;
}
// std is short for standard,
// it's a namespace that contains many useful functions and classes.

//============================================================================

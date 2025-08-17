#include <iostream> // for input/output, we can print and get input from the user
// iostream is one of the files from the standard library
// main is the entrypoint of our program.
#include <cmath>
#include <vector>
#include <cstdlib>
#include <random>
#include <ctime>
#include <string> // for string manipulation

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

    int x1 = 10;
    std ::cout << "x1 = " << x1 << std ::endl; // Chaining output statements

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

    // int x = 0; then right after we do
    // int x = 1; // This will not compile, because x is already defined in the same scope
    //  To avoid this, we can use different namespaces or use the std namespace.

    // cout << first ::x << std ::endl;  // Accessing variable from first namespace
    // cout << second ::x << std ::endl; // Accessing variable from second namespace

    //===================== Different namespace =============================

    // using first::x;      // Can also declare namespace usage inside main
    // using second::x;     // Can also declare namespace usage inside main
    using std::cin;      // Using cin from the std namespace
    using std::cout;     // Using cout from the std namespace
    using namespace std; // Using the entire std namespace ( all of the above )

    //=======================================================================
    //====================== Datatype =======================================

    // typedef std::vector<std::pair<std::string, int>> pairlist_t; // This could be a datatype for a pair-list
    //  But we can make it shorter and end it with identifier t
    // outsdide main we can declare typedef std::string text_t;
    // Then inside the main function we can use it like text_t firstName; instead of std::string firstName;
    // Other usage can be for defining stuff like integer or something more spefically like integer:
    // typedef int number_t;
    // Then we can use it like
    // Instead of using typedef , one should do use the (using keyboard instead)
    // using number_t = int; or something like
    // using text_t = std::string; this is more popular

    // So the typedef = reserved keyboard used to creat an additional name ( alias ) for another datatype
    // It's a new identifier for an existing type, helps readability and reduces typos

    //====================== Initializing variables =========================

    int a = 5;                       // Regular initialization
    double b = 3.14;                 // Floating-point initialization
    char c = 'A';                    // Character initialization
    string d = "Hello";              // String initialization
    bool e = true;                   // Boolean initialization (true or false)
    float f = 2.71f;                 // Float initialization
    long g = 1234567890L;            // Long initialization (so compiler dont treat it as a integer)
    short h = 12345;                 // Short initialization
    unsigned int i = 42;             // Unsigned integer initialization
    long long j = 123456789012345LL; // Long long initialization
    unsigned long k = 1234567890UL;  // Unsigned long initialization
    auto l = 1234567890;             // Auto initialization (deduces to int)
    decltype(l) m = 1234567890;      // Decltype initialization (same type as l)

    // SUPER USEFUL INITIALIZATION
    int number{4};              // Uniform initialization (C++11 and later)
    int zero_number{};          // Gives us out 0, and can be used to initialize other variables
    cout << number << "\n";     // Output: 4
    int binary_number = 0b1010; // Binary initialization (C++14 and later)
    int hex_number = 0xA;       // Hexadecimal initialization
    // cout << binary_number << "\n"; // Output: 10
    // cout << hex_number << "\n";    // Output: 10
    int octal_number = 012;       // Octal initialization (C++14 and later)
    cout << octal_number << "\n"; // Output: 10

    int large_number = 1'000'000; // Digit separators (C++14 and later)
    short another = large_number; // Implicit conversion to short
    cout << large_number << "\n"; // Output: 1000000
    cout << another << "\n";      // Output: 1000

    short num = 100;    // This one takes 2 bytes
    int _another = num; // This one takes 4 bytes

    //========================================================================
    //===================== Type Casting ====================================

    double x2 = 5.567;
    int y = static_cast<int>(x2);    // Explicit conversion from double to int
    cout << y << "\n";               // Output: 5
    float z = static_cast<float>(y); // Explicit conversion from int to float
    cout << z << "\n";               // Output: 5.0

    //========================================================================
    cout << "The tutorial is over bye bye!\n";

    //=========================================================================

    return 0;
}

// std is short for standard,
// it's a namespace that contains many useful functions and classes.

//============================================================================

#include <string>
#include <iostream>

using namespace std; // Using the standard namespace to avoid std:: prefix

//=======================================================================
//===================== Writing / Reading with console ==================

void writeToConsole(const std::string &message)
{
    std::cout << message << std::endl;
    std::cout << message << "\n"; // Alternative way to print a message, does not flush the output/buffer .
    std::cout.flush();            // Explicitly flush the output buffer makes message appear immediately
}

std::string readFromConsole()
{
    std::string input;
    std::getline(std::cin, input);
    return input;
}
//=========================================================================
int main()
{
    writeToConsole("Enter your name: ");
    std::string name = readFromConsole();
    writeToConsole("Hello, " + name + "!");

    // Example of using the readFromConsole function
    writeToConsole("Type something and press Enter: ");
    std::string userInput = readFromConsole();
    writeToConsole("You typed: " + userInput);

    return 0; // Return 0 to indicate successful execution
}
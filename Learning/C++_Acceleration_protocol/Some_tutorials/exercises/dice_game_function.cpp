#include <random>
#include <ctime>
#include <iostream>
#include <cstdlib>

int main()
{
    const short minval = 1;
    const short maxval = 6;

    srand(time(0));
    int dice_roll = rand() % (maxval - minval + 1) + minval;
    std::cout << "You rolled a " << dice_roll << std::endl;
}
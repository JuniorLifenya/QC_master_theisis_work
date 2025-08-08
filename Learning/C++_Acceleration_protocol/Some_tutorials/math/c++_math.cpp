#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

int main()
{
    //=================== Now we do math ====================================

    int x = 10;
    int y = 3;
    int addition = x + y; // Here x and y are called operands
    std ::cout << addition << std ::endl;

    //=======================================================================

    int division_incorect = x / y; // Division is integer division, so it will return 3. incorect
    double z = 10;
    double division_correct = z / y; // Casting x or y to double, so it will return 3.33333
    std ::cout << division_correct << std ::endl;

    //=======================================================================

    int modulus = x % y;         // modulus is the remainder of division 10 - (3*(fit into 10 = 3)) = 1
    int modulus_reverse = y % x; // modulus_reverse is the remainder of division 3 - (10*(fit into 3 = 0)) = 3
    std ::cout << modulus << std ::endl;

    //=======================================================================
    //=========Only Increment and decrement( arithmetic operators ) =========

    int a = 10;
    a = a + 1; // a = 11, but we can also write it like
    a += 1;    // a = 12
    a++;       // a = 12+1= 13
    int k = 10;
    int b = k++; // b = 10, y = 11, because k++ returns the value of a before increment! so increment accours "later"
    int c = ++k;
    // c = 12, k = 12, because ++a returns the value

    std ::cout << "k: " << k << " " << "b: " << b << std ::endl; // Here b = 10, k = 11
    std ::cout << "c: " << c << " " << "k: " << k << std ::endl; // c = 1+k but k = 11 + 1 = 12
    std ::cout << "A is equal to " << a << std ::endl;

    //=======================================================================
    //==================== Order of operations  =============================
    double tot = 1 + 2 * 3;           // 1 + 6 = 7
    double parantheses = (1 + 2) * 3; // 3 * 3 = 9

    //=======================================================================
    //===================== Using other part of standard library () =========
    using namespace std;

    double flor_usage = floor(3.982);
    cout << "floor_usage: " << flor_usage << endl;
    // floor is a function from standard library, that returns the largest integer less than or equal to a giv

    //=======================================================================
    //====================== Arithmetic operators ===========================

    return 0;
}
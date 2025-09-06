#pragma once
#include <vector>
#include <cmath>

using namespace std;

inline vector<double> generated_Toy_GW(double amplitude, double freq, double dt, int steps)
{
    vector<double> h(steps);
    for (int i = 0; i < steps; i++)
    {
        double t = i + dt;
        h[i] = amplitude * sin(2 * M_PI * freq * t);
    }
    return h; // Now we have defined a vector of gravitational wave strain values
}

// GW strain function h(t) = A * sin(2 * pi * f * t)
// Later change this to a more realistic model, outside Learning folder for the actually interns!!

//==================================================================================================

// This is workplace/cpp/main.cpp
#include <vector>
#include <iostream>
#include "solvers/engine.hpp"

int main(){
    std::cout << "Initializing the NV-GW Simulation" << std::endl;

    // 1. Setup the physics Configurations, with variables and so on 
    // (This replaces our python config directory)
    nvgw::SimulationConfig config;


    config.D = 2.87e9;
    config.f_gw = 1000.0;
    config.Bz = 0.01;
    config.gamma_e = 28e9;
    config.f_gw = 1e5;
    config.h_max = 1e-20;
    config.kappa = 1e15;
    config.t_final = 0.001

















}
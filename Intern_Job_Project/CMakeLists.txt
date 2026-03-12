cmake_minimum_required(VERSION 3.14)
project(QuantumEngine)

# Require C++17
set(CMAKE_CXX_STANDARD 17)

# Find PyBind11 and Eigen3
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Create the Python module
pybind11_add_module(gw_cpp_engine src/cpp/bindings.cpp src/cpp/engine.cpp)

# Link Eigen3
target_link_libraries(gw_cpp_engine PRIVATE Eigen3::Eigen)
# intern_project/python/bindings/nv_bindings.cpp
#include <pybind11/pybind11.h>
#include "nv_hamiltonian.hpp"

PYBIND11_MODULE(nv_physics, m) {
    py::class_<NVHamiltonian>(m, "NVHamiltonian")
        .def(py::init<double, double>())
        .def("build", &NVHamiltonian::build);
}
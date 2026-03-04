#include <../pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

// pybind11 module definition here and usage 

namespace py = pybind11;

PYBIND11_MODULE(nvgw_cpp,m){

    py::class_<NVSpin1>(m,"NVSpin1")
        .def(py::init<double, double , double>())
        .def("static_hamiltonian", &NVSpin1::static_hamiltonian)
        .def("strain_operator",&NVSpin1::strain_operator);

    m.def("sesolve", &sesolve, "Solve Schrødinger equation");

}
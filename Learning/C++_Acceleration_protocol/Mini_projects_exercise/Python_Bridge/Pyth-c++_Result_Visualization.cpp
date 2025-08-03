#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(quantum_sensors, m)
{
    py::class_<QuantumSensor>(m, "QuantumSensor")
        .def(py::init<>())
        .def("load_data", &QuantumSensor::loadData)
        .def("analyze", &QuantumSensor::applyKalmanFilter);

    m.def("simulate_decoherence", &run_simulation);
}
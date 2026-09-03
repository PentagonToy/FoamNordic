#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "foamnordic/runtime/longship.hpp"
#include "foamnordic/runtime/placement.hpp"
#include "artifact_bindings.hpp"
#include "resident_bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_native, module) {
    module.doc() = "Private bindings for stable FoamNordic native facades";

    using foamnordic::native::DataPath;
    using foamnordic::native::DataPathPreference;
    using foamnordic::native::HostPlacement;
    using foamnordic::native::InferenceDevice;
    using foamnordic::native::LongshipPlan;
    using foamnordic::native::LongshipRequest;
    using foamnordic::native::PlacementPlan;
    using foamnordic::native::PlacementRequest;

    nb::enum_<HostPlacement>(module, "HostPlacement")
        .value("AUTOMATIC", HostPlacement::automatic)
        .value("ATTACHED", HostPlacement::attached)
        .value("CENTRAL", HostPlacement::central);

    nb::enum_<InferenceDevice>(module, "InferenceDevice")
        .value("CPU", InferenceDevice::cpu)
        .value("GPU", InferenceDevice::gpu);

    nb::enum_<DataPathPreference>(module, "DataPathPreference")
        .value("AUTOMATIC", DataPathPreference::automatic)
        .value("SHARED_MEMORY", DataPathPreference::shared_memory)
        .value("UNIX_SOCKET", DataPathPreference::unix_socket)
        .value("UCX", DataPathPreference::ucx)
        .value("TCP", DataPathPreference::tcp);

    nb::enum_<DataPath>(module, "DataPath")
        .value("SHARED_MEMORY", DataPath::shared_memory)
        .value("UNIX_SOCKET", DataPath::unix_socket)
        .value("UCX", DataPath::ucx)
        .value("TCP", DataPath::tcp);

    nb::class_<PlacementRequest>(
        module,
        "PlacementRequest",
        "Mutable native input used to resolve ModelHost placement.")
        .def(nb::init<>())
        .def_rw("placement", &PlacementRequest::placement)
        .def_rw("device", &PlacementRequest::device)
        .def_rw("data_path", &PlacementRequest::data_path)
        .def_rw("solver_nodes", &PlacementRequest::solver_nodes)
        .def_rw("central_host_nodes", &PlacementRequest::central_host_nodes)
        .def_rw("solver_nodes_have_device", &PlacementRequest::solver_nodes_have_device)
        .def_rw("shared_memory_available", &PlacementRequest::shared_memory_available)
        .def_rw("unix_socket_available", &PlacementRequest::unix_socket_available)
        .def_rw("ucx_available", &PlacementRequest::ucx_available)
        .def("validate", &PlacementRequest::validate);

    nb::class_<PlacementPlan>(
        module,
        "PlacementPlan",
        "Read-only native placement decision.")
        .def_ro("placement", &PlacementPlan::placement)
        .def_ro("data_path", &PlacementPlan::data_path)
        .def_ro("host_instances", &PlacementPlan::host_instances)
        .def_ro("same_allocation", &PlacementPlan::same_allocation)
        .def_ro("same_node", &PlacementPlan::same_node)
        .def_ro("coupled_lifetime", &PlacementPlan::coupled_lifetime)
        .def_ro("reason", &PlacementPlan::reason);

    nb::class_<LongshipRequest>(
        module,
        "LongshipRequest",
        "Mutable native input for Longship resource planning.")
        .def(nb::init<>())
        .def_rw("name", &LongshipRequest::name)
        .def_rw("solver_nodes", &LongshipRequest::solver_nodes)
        .def_rw("solver_tasks", &LongshipRequest::solver_tasks)
        .def_rw("solver_cpus_per_task", &LongshipRequest::solver_cpus_per_task)
        .def_rw("host_cpus_per_node", &LongshipRequest::host_cpus_per_node)
        .def_rw("use_model_host", &LongshipRequest::use_model_host)
        .def_rw("placement", &LongshipRequest::placement)
        .def("validate", &LongshipRequest::validate);

    nb::class_<LongshipPlan>(
        module,
        "LongshipPlan",
        "Read-only native resource plan with lifecycle invariants.")
        .def_ro("name", &LongshipPlan::name)
        .def_ro("placement", &LongshipPlan::placement)
        .def_ro("allocation_nodes", &LongshipPlan::allocation_nodes)
        .def_ro("solver_tasks", &LongshipPlan::solver_tasks)
        .def_ro("solver_tasks_per_node", &LongshipPlan::solver_tasks_per_node)
        .def_ro("solver_cpus_per_task", &LongshipPlan::solver_cpus_per_task)
        .def_ro("host_tasks", &LongshipPlan::host_tasks)
        .def_ro("host_cpus_per_task", &LongshipPlan::host_cpus_per_task)
        .def_ro("allocation_cpus_per_node", &LongshipPlan::allocation_cpus_per_node)
        .def_ro("host_starts_first", &LongshipPlan::host_starts_first)
        .def_ro("fail_together", &LongshipPlan::fail_together);

    module.def(
        "resolve_placement",
        &foamnordic::native::resolve_placement,
        "request"_a,
        "Validate a placement request and return the resolved native plan.");
    module.def(
        "plan_longship",
        &foamnordic::native::plan_longship,
        "request"_a,
        "Validate resources and return the native Longship allocation plan.");
    bind_artifacts(module);
    bind_resident(module);
}

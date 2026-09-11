// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\ /  \|  V  | |  \| |/__\| _ \ _\| |/ _/
 * | _| \/ | /\ | \_/ | | | ' | \/ | v / v | | \__
 * |_| \__/|_||_|_| |_| |_|\__|\__/|_|_\__/|_|\__/
 *
 * FoamNordic
 */
// clang-format on
#include "resident_bindings.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

#include "foamnordic/backend/inference/exchange.hpp"
#include "foamnordic/backend/inference/manifest.hpp"
#include "foamnordic/backend/inference/model.hpp"
#include "foamnordic/backend/inference/worker.hpp"
#include "foamnordic/backend/inference/compiled.hpp"
#include "foamnordic/backend/inference/load.hpp"
#include "foamnordic/backend/inference/forest.hpp"
#include "foamnordic/backend/connectors/plugin.hpp"
#include "foamnordic/fjord/endpoint.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

std::uint64_t output_width(const foamnordic::inference::ModelArtifact& artifact) {
    std::uint64_t result = 0;
    for (const auto& field : artifact.contract.outputs) {
        if (result > std::numeric_limits<std::uint64_t>::max() - field.components) {
            throw std::overflow_error("Python model output width overflowed.");
        }
        result += field.components;
    }
    return result;
}

class PythonPackedKernel final : public foamnordic::inference::PackedModelKernel {
public:
    PythonPackedKernel(nb::object evaluator, std::uint64_t width)
        : evaluator_(std::move(evaluator)), output_width_(width) {}

    [[nodiscard]] bindnordic::BackendCapabilities capabilities()
        const noexcept override {
        return {
            .concurrent_evaluation = false,
            .row_independent = true,
            .owns_compute_threads = true,
        };
    }

    bindnordic::Tensor evaluate(
        bindnordic::TensorView features,
        std::uint64_t exchange_index,
        double physical_time,
        std::uint32_t rank) override {
        features.validate();
        if (features.shape.size() != 2) {
            throw std::invalid_argument("Python model features must be rank two.");
        }
        nb::gil_scoped_acquire acquire;
        try {
            auto* data = reinterpret_cast<char*>(
                const_cast<std::byte*>(features.bytes.data()));
            nb::object memory = nb::steal<nb::object>(PyMemoryView_FromMemory(
                data,
                static_cast<Py_ssize_t>(features.bytes.size()),
                PyBUF_READ));
            if (!memory.is_valid()) {
                throw nb::python_error();
            }
            const char* dtype =
                features.element == bindnordic::Element::float32
                    ? "float32"
                    : "float64";
            nb::object result = evaluator_(
                memory,
                features.shape[0],
                features.shape[1],
                dtype,
                exchange_index,
                physical_time,
                rank);
            Py_buffer buffer{};
            if (PyObject_GetBuffer(result.ptr(), &buffer, PyBUF_CONTIG_RO) < 0) {
                throw nb::python_error();
            }
            const auto bytes = features.shape[0] * output_width_
                               * bindnordic::element_size(features.element);
            if (buffer.len < 0 || static_cast<std::uint64_t>(buffer.len) != bytes) {
                PyBuffer_Release(&buffer);
                throw std::invalid_argument(
                    "Python model returned an incorrect packed byte count.");
            }
            std::vector<std::byte> copied(bytes);
            std::memcpy(copied.data(), buffer.buf, copied.size());
            PyBuffer_Release(&buffer);
            return {
                "bindnordic.predictions",
                features.element,
                {features.shape[0], output_width_},
                std::move(copied),
            };
        } catch (const nb::python_error& error) {
            const std::string message = error.what();
            PyErr_Clear();
            throw std::runtime_error(
                "Python model evaluation failed: " + message);
        }
    }

private:
    nb::object evaluator_;
    std::uint64_t output_width_;
};

class ReadyMarker {
public:
    explicit ReadyMarker(const std::filesystem::path& path) : path_(path) {
        if (path_.empty()) {
            return;
        }
        const auto descriptor = ::open(
            path_.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0660);
        if (descriptor < 0) {
            throw std::system_error(
                errno,
                std::generic_category(),
                "Cannot create ModelHost readiness marker");
        }
        const auto identity = std::to_string(static_cast<long long>(::getpid())) + "\n";
        const auto written = ::write(descriptor, identity.data(), identity.size());
        const auto saved_error = errno;
        ::close(descriptor);
        if (written != static_cast<ssize_t>(identity.size())) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
            throw std::system_error(
                saved_error == 0 ? EIO : saved_error,
                std::generic_category(),
                "Cannot write ModelHost readiness marker");
        }
    }

    ~ReadyMarker() {
        if (!path_.empty()) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }

private:
    std::filesystem::path path_;
};

class SharedExecutionCapacity {
public:
    explicit SharedExecutionCapacity(std::uint32_t units)
        : pool(std::make_shared<
               foamnordic::inference::ExecutionCapacityPool>(units)) {}

    std::shared_ptr<foamnordic::inference::ExecutionCapacityPool> pool;
};

}  // namespace

void bind_resident(nb::module_& module) {
    module.def("_load_connector_plugin", [](const std::string& path) {
        foamnordic::backend::load_connector_plugin(path);
    });
    nb::class_<SharedExecutionCapacity>(module, "_ExecutionCapacity")
        .def(nb::init<std::uint32_t>(), "units"_a);
    module.def("_evaluate_native_model", [](const std::string& manifest,
        const std::string& library, nb::bytes data, std::uint64_t rows,
        std::uint64_t columns, std::uint32_t threads) {
        using namespace foamnordic::inference;
        std::unique_ptr<PackedModelKernel> kernel;
        if (library.empty()) {
            kernel = std::make_unique<ForestKernel>(read_bundle_payload(manifest), threads);
        } else kernel = std::make_unique<CompiledKernel>(library, threads);
        bindnordic::Tensor input{"input", bindnordic::Element::float64,
            {rows, columns}, std::vector<std::byte>(data.size())};
        std::memcpy(input.bytes.data(), data.c_str(), data.size());
        bindnordic::Tensor result;
        { nb::gil_scoped_release release; result = kernel->evaluate(input.view(), 0, 0.0); }
        return nb::bytes(reinterpret_cast<const char*>(result.bytes.data()), result.bytes.size());
    });
    module.def("run_native_worker", [](const std::string& address,
        const std::string& manifest, std::uint32_t connections,
        const std::string& ready_file, bool shared_memory, std::uint32_t threads,
        std::uint64_t receive_bytes, std::uint64_t profile_every,
        const std::string& program, SharedExecutionCapacity* capacity) {
        using namespace foamnordic::inference;
        WorkerOptions options;
        options.connections = connections; options.model_threads = threads;
        options.shared_memory = shared_memory; options.maximum_receive_bytes = receive_bytes;
        options.profile_every = profile_every;
        options.program = program;
        EvaluateAllCells bypass;
        auto shared = capacity == nullptr
                          ? std::make_shared<ExecutionCapacityPool>(threads)
                          : capacity->pool;
        auto loaded = load_model(
            manifest,
            {
                threads,
                std::min(connections, threads),
                options.maximum_working_bytes,
                profile_every,
                std::move(shared),
                threads,
            });
        ModelWorker worker(
            foamnordic::fjord::FjordAddress::parse(address),
            std::move(loaded.artifact), bypass, *loaded.kernel, options);
        ReadyMarker ready(ready_file);
        nb::gil_scoped_release release;
        worker.run();
    });
    module.def("run_compiled_worker", [](const std::string& address,
        const std::string& manifest, const std::string& library,
        std::uint32_t connections, const std::string& ready_file,
        bool shared_memory, std::uint32_t threads, std::uint64_t receive_bytes,
        std::uint64_t profile_every, const std::string& program,
        SharedExecutionCapacity* capacity) {
        using namespace foamnordic::inference;
        auto artifact = read_manifest(manifest);
        if (artifact.format != ModelFormat::compiled)
            throw std::invalid_argument("Expected a compiled model manifest.");
        WorkerOptions options;
        options.connections = connections;
        options.model_threads = threads;
        options.shared_memory = shared_memory;
        options.maximum_receive_bytes = receive_bytes;
        options.profile_every = profile_every;
        options.program = program;
        CompiledKernel packed(library, threads);
        auto shared = capacity == nullptr
                          ? std::make_shared<ExecutionCapacityPool>(threads)
                          : capacity->pool;
        ArtifactModelKernel kernel(artifact, packed,
            {std::min(connections, threads), options.maximum_working_bytes,
             profile_every, std::move(shared), threads});
        EvaluateAllCells bypass;
        ModelWorker worker(foamnordic::fjord::FjordAddress::parse(address), artifact,
            bypass, kernel, options);
        ReadyMarker ready(ready_file);
        nb::gil_scoped_release release;
        worker.run();
    });
    module.def(
        "run_python_worker",
        [](const std::string& address,
           const std::string& manifest_path,
           nb::object evaluator,
           std::uint32_t connections,
           const std::string& ready_file,
           bool shared_memory,
           std::uint32_t model_threads,
           std::uint64_t maximum_receive_bytes,
           std::uint64_t profile_every,
           const std::string& program,
           SharedExecutionCapacity* capacity) {
            auto artifact = foamnordic::inference::read_manifest(manifest_path);
            if (artifact.format == foamnordic::inference::ModelFormat::onnx) {
                throw std::invalid_argument(
                    "ONNX artifacts must use the native ONNX connector.");
            }
            foamnordic::inference::WorkerOptions options;
            options.connections = connections;
            options.model_threads = model_threads;
            options.shared_memory = shared_memory;
            options.maximum_receive_bytes = maximum_receive_bytes;
            options.profile_every = profile_every;
            options.program = program;
            PythonPackedKernel packed(std::move(evaluator), output_width(artifact));
            auto shared = capacity == nullptr
                              ? std::make_shared<
                                    foamnordic::inference::ExecutionCapacityPool>(
                                    model_threads)
                              : capacity->pool;
            foamnordic::inference::ArtifactModelKernel kernel(
                artifact,
                packed,
                {
                    std::min(connections, model_threads),
                    options.maximum_working_bytes,
                    profile_every,
                    std::move(shared),
                    model_threads,
                });
            foamnordic::inference::EvaluateAllCells bypass;
            foamnordic::inference::ModelWorker worker(
                foamnordic::fjord::FjordAddress::parse(address),
                artifact,
                bypass,
                kernel,
                options);
            std::cout << "[FoamNordic] Model worker ready: "
                      << worker.address().text() << std::endl;
            ReadyMarker ready(ready_file);
            {
                nb::gil_scoped_release release;
                worker.run();
            }
        },
        "address"_a,
        "manifest_path"_a,
        "evaluator"_a,
        "connections"_a = 1,
        "ready_file"_a = "",
        "shared_memory"_a = true,
        "model_threads"_a = 1,
        "maximum_receive_bytes"_a = 16ULL * 1024ULL * 1024ULL * 1024ULL,
        "profile_every"_a = 0,
        "program"_a = "model",
        "capacity"_a = nb::none(),
        "Run a managed Python model behind the native Fjord worker.");
}

#include "resident_bindings.hpp"

#include <nanobind/stl/string.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

#include "foamnordic/backend/inference/closure.hpp"
#include "foamnordic/backend/inference/manifest.hpp"
#include "foamnordic/backend/inference/model.hpp"
#include "foamnordic/backend/inference/worker.hpp"
#include "foamnordic/fjord/endpoint.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

std::uint64_t output_width(const foamnordic::closure::ModelArtifact& artifact) {
    std::uint64_t result = 0;
    for (const auto& field : artifact.contract.outputs) {
        if (result > std::numeric_limits<std::uint64_t>::max() - field.components) {
            throw std::overflow_error("Python model output width overflowed.");
        }
        result += field.components;
    }
    return result;
}

class PythonPackedKernel final : public foamnordic::closure::PackedModelKernel {
public:
    PythonPackedKernel(nb::object evaluator, std::uint64_t width)
        : evaluator_(std::move(evaluator)), output_width_(width) {}

    foamnordic::fjord::Tensor evaluate(
        foamnordic::fjord::TensorView features,
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
                features.element == foamnordic::fjord::Element::float32
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
                               * foamnordic::fjord::element_size(features.element);
            if (buffer.len < 0 || static_cast<std::uint64_t>(buffer.len) != bytes) {
                PyBuffer_Release(&buffer);
                throw std::invalid_argument(
                    "Python model returned an incorrect packed byte count.");
            }
            std::vector<std::byte> copied(bytes);
            std::memcpy(copied.data(), buffer.buf, copied.size());
            PyBuffer_Release(&buffer);
            return {
                "foamnordic.predictions",
                features.element,
                {features.shape[0], output_width_},
                std::move(copied),
                exchange_index,
                physical_time,
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
                "Cannot create ClosureHost readiness marker");
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
                "Cannot write ClosureHost readiness marker");
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

}  // namespace

void bind_resident(nb::module_& module) {
    module.def(
        "run_python_worker",
        [](const std::string& address,
           const std::string& manifest_path,
           nb::object evaluator,
           std::uint32_t connections,
           const std::string& ready_file,
           bool shared_memory) {
            auto artifact = foamnordic::closure::read_manifest(manifest_path);
            if (artifact.format == foamnordic::closure::ModelFormat::onnx) {
                throw std::invalid_argument(
                    "ONNX artifacts must use the native ONNX connector.");
            }
            PythonPackedKernel packed(std::move(evaluator), output_width(artifact));
            foamnordic::closure::ArtifactModelKernel kernel(artifact, packed);
            foamnordic::closure::EvaluateEveryCell bypass;
            foamnordic::closure::WorkerOptions options;
            options.connections = connections;
            options.shared_memory = shared_memory;
            foamnordic::closure::NativeClosureWorker worker(
                foamnordic::fjord::FjordAddress::parse(address),
                artifact,
                bypass,
                kernel,
                options);
            std::cout << "[FoamNordic] Closure worker ready: "
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
        "Run a managed Python model behind the native Fjord worker.");
}

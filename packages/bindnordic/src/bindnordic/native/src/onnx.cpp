// clang-format off
/*
 *  ___ __   __  __ __   __  _  __  ___ __  _  ___
 * | __/__\ /  \|  V  | |  \| |/__\| _ \ _\| |/ _/
 * | _| \/ | /\ | \_/ | | | ' | \/ | v / v | | \__
 * |_| \__/|_||_|_| |_| |_|\__|\__/|_|_\__/|_|\__/
 *
 * BindNordic
 */
// clang-format on

#include "bindnordic/onnx.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace bindnordic {
namespace {

static_assert(ORT_API_VERSION == 28, "BindNordic requires ONNX Runtime API 28.");

ONNXTensorElementDataType onnx_element(Element element) {
    switch (element) {
        case Element::float32:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case Element::float64:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case Element::int32:
        case Element::int64:
            throw std::invalid_argument("BindNordic ONNX inference requires floating point.");
    }
    throw std::invalid_argument("BindNordic ONNX element type is unsupported.");
}

std::vector<std::int64_t> onnx_shape(const std::vector<std::uint64_t>& shape) {
    std::vector<std::int64_t> result;
    result.reserve(shape.size());
    for (const auto extent : shape) {
        if (extent > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("BindNordic tensor shape exceeds ONNX limits.");
        }
        result.push_back(static_cast<std::int64_t>(extent));
    }
    return result;
}

void validate_model_shape(
    const std::vector<std::int64_t>& model_shape,
    std::uint64_t feature_count,
    const char* direction) {
    if (model_shape.size() != 2
        || (model_shape[1] >= 0
            && static_cast<std::uint64_t>(model_shape[1]) != feature_count)) {
        throw std::invalid_argument(
            std::string("ONNX ") + direction
            + " must be a rank-two [cell, feature] tensor.");
    }
}

template <class Value>
Ort::Value input_tensor(
    const Ort::MemoryInfo& memory,
    TensorView features,
    const std::vector<std::int64_t>& shape,
    std::vector<Value>& fallback) {
    const auto address = reinterpret_cast<std::uintptr_t>(features.bytes.data());
    Value* data = nullptr;
    if (address % alignof(Value) == 0) {
        // ONNX Runtime borrows CPU input storage for the duration of Run().
        // Its C++ API predates std::span<const T> and requires a mutable pointer.
        data = const_cast<Value*>(
            reinterpret_cast<const Value*>(features.bytes.data()));
    } else {
        fallback.resize(features.bytes.size() / sizeof(Value));
        std::memcpy(fallback.data(), features.bytes.data(), features.bytes.size());
        data = fallback.data();
    }
    return Ort::Value::CreateTensor<Value>(
        memory,
        data,
        features.bytes.size() / sizeof(Value),
        shape.data(),
        shape.size());
}

template <class Value>
Ort::Value output_tensor(
    const Ort::MemoryInfo& memory,
    std::vector<std::byte>& bytes,
    const std::vector<std::int64_t>& shape) {
    return Ort::Value::CreateTensor<Value>(
        memory,
        reinterpret_cast<Value*>(bytes.data()),
        bytes.size() / sizeof(Value),
        shape.data(),
        shape.size());
}

}  // namespace

void OnnxOptions::validate() const {
    if (intra_op_threads < 1 || inter_op_threads < 1) {
        throw std::invalid_argument("ONNX Runtime thread counts must be positive.");
    }
}

class OnnxPackedKernel::Implementation {
public:
    Implementation(const std::filesystem::path& model_path, OnnxOptions options)
        : environment_(ORT_LOGGING_LEVEL_WARNING, "BindNordic"),
          options_(make_options(options)),
          session_(std::make_unique<Ort::Session>(
              environment_, model_path.c_str(), options_)),
          memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        initialize();
    }

    Implementation(std::vector<std::byte> model_bytes, OnnxOptions options)
        : model_bytes_(std::move(model_bytes)),
          environment_(ORT_LOGGING_LEVEL_WARNING, "BindNordic"),
          options_(make_options(options)),
          session_(std::make_unique<Ort::Session>(
              environment_, model_bytes_.data(), model_bytes_.size(), options_)),
          memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        if (model_bytes_.empty()) {
            throw std::invalid_argument("Embedded ONNX payload must not be empty.");
        }
        initialize();
    }

    Implementation(std::span<const std::byte> model_bytes, OnnxOptions options)
        : model_bytes_view_(model_bytes),
          environment_(ORT_LOGGING_LEVEL_WARNING, "BindNordic"),
          options_(make_options(options)),
          session_(std::make_unique<Ort::Session>(
              environment_, model_bytes_view_.data(), model_bytes_view_.size(), options_)),
          memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        if (model_bytes_view_.empty()) {
            throw std::invalid_argument("Embedded ONNX payload must not be empty.");
        }
        initialize();
    }

    void initialize() {
        const std::string runtime_version = OrtGetApiBase()->GetVersionString();
        if (runtime_version != "1.28.0") {
            throw std::runtime_error(
                "BindNordic requires ONNX Runtime 1.28.0; found "
                + runtime_version + '.');
        }
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
            throw std::invalid_argument(
                "BindNordic packed ONNX models require exactly one input and output.");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_->GetInputNameAllocated(0, allocator).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
        const auto input_type = session_->GetInputTypeInfo(0);
        const auto output_type = session_->GetOutputTypeInfo(0);
        const auto input_info = input_type.GetTensorTypeAndShapeInfo();
        const auto output_info = output_type.GetTensorTypeAndShapeInfo();
        input_element_ = input_info.GetElementType();
        output_element_ = output_info.GetElementType();
        input_shape_ = input_info.GetShape();
        output_shape_ = output_info.GetShape();
        if (output_shape_.size() != 2 || output_shape_[1] == 0) {
            throw std::invalid_argument(
                "BindNordic packed ONNX output must be rank-two [cell, feature].");
        }
    }

    Tensor evaluate(
        TensorView features,
        std::uint64_t exchange_index,
        double physical_time) {
        features.validate();
        if (features.shape.size() != 2) {
            throw std::invalid_argument("ONNX features do not match their contract.");
        }
        const auto element = onnx_element(features.element);
        if (input_element_ != element || output_element_ != element) {
            throw std::invalid_argument(
                "ONNX model dtype does not match BindNordic packed features.");
        }
        validate_model_shape(input_shape_, features.shape[1], "input");
        const auto shape = onnx_shape(features.shape);
        std::vector<float> float_input;
        std::vector<double> double_input;
        Ort::Value input{nullptr};
        if (features.element == Element::float32) {
            input = input_tensor(memory_, features, shape, float_input);
        } else {
            input = input_tensor(memory_, features, shape, double_input);
        }
        const char* input_names[]{input_name_.c_str()};
        const char* output_names[]{output_name_.c_str()};
        Ort::Value output{nullptr};
        std::vector<std::byte> bytes;
        if (output_shape_[1] > 0) {
            const std::vector<std::int64_t> bound_shape{
                shape[0], output_shape_[1]};
            const auto row_bytes = static_cast<std::uint64_t>(output_shape_[1])
                                   * element_size(features.element);
            if (shape[0] < 0
                || static_cast<std::uint64_t>(shape[0])
                       > std::numeric_limits<std::size_t>::max() / row_bytes) {
                throw std::overflow_error("ONNX output allocation exceeds host limits.");
            }
            bytes.resize(static_cast<std::size_t>(shape[0]) * row_bytes);
            if (features.element == Element::float32) {
                output = output_tensor<float>(memory_, bytes, bound_shape);
            } else {
                output = output_tensor<double>(memory_, bytes, bound_shape);
            }
            session_->Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input,
                1,
                output_names,
                &output,
                1);
        } else {
            auto outputs = session_->Run(
                Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
            if (outputs.size() != 1 || !outputs.front().IsTensor()) {
                throw std::runtime_error(
                    "ONNX Runtime did not return one tensor output.");
            }
            output = std::move(outputs.front());
        }
        const auto output_info = output.GetTensorTypeAndShapeInfo();
        const auto output_shape = output_info.GetShape();
        if (output_shape.size() != 2 || output_shape[0] != shape[0]
            || output_shape[1] <= 0 || output_info.GetElementType() != element) {
            throw std::runtime_error("ONNX Runtime returned invalid output metadata.");
        }
        if ((output_shape_[0] >= 0 && output_shape_[0] != output_shape[0])
            || (output_shape_[1] >= 0 && output_shape_[1] != output_shape[1])) {
            throw std::runtime_error(
                "ONNX Runtime output does not match the model declaration.");
        }
        if (bytes.empty()) {
            const auto byte_count = output_info.GetElementCount()
                                    * element_size(features.element);
            bytes.resize(byte_count);
            if (features.element == Element::float32) {
                std::memcpy(bytes.data(), output.GetTensorData<float>(), bytes.size());
            } else {
                std::memcpy(bytes.data(), output.GetTensorData<double>(), bytes.size());
            }
        }
        return {
            "bindnordic.predictions",
            features.element,
            {
                static_cast<std::uint64_t>(output_shape[0]),
                static_cast<std::uint64_t>(output_shape[1]),
            },
            std::move(bytes),
        };
    }

private:
    static Ort::SessionOptions make_options(OnnxOptions options) {
        options.validate();
        Ort::SessionOptions result;
        result.SetIntraOpNumThreads(options.intra_op_threads);
        result.SetInterOpNumThreads(options.inter_op_threads);
        result.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        result.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        return result;
    }

    std::vector<std::byte> model_bytes_;
    std::span<const std::byte> model_bytes_view_;
    Ort::Env environment_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_;
    std::string input_name_;
    std::string output_name_;
    ONNXTensorElementDataType input_element_{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    ONNXTensorElementDataType output_element_{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    std::vector<std::int64_t> input_shape_;
    std::vector<std::int64_t> output_shape_;
};

OnnxPackedKernel::OnnxPackedKernel(
    const std::filesystem::path& model_path,
    OnnxOptions options)
    : implementation_(std::make_unique<Implementation>(model_path, options)) {}

OnnxPackedKernel::OnnxPackedKernel(
    std::vector<std::byte> model_bytes,
    OnnxOptions options)
    : implementation_(
          std::make_unique<Implementation>(std::move(model_bytes), options)) {}

OnnxPackedKernel::OnnxPackedKernel(
    std::span<const std::byte> model_bytes,
    OnnxOptions options)
    : implementation_(std::make_unique<Implementation>(model_bytes, options)) {}

OnnxPackedKernel::~OnnxPackedKernel() = default;
OnnxPackedKernel::OnnxPackedKernel(OnnxPackedKernel&&) noexcept = default;
OnnxPackedKernel& OnnxPackedKernel::operator=(OnnxPackedKernel&&) noexcept = default;

BackendCapabilities onnx_backend_capabilities() noexcept {
    // One session owns its configured ORT thread pool. Keeping Run() serial
    // prevents request concurrency from multiplying that CPU budget.
    return {
        .concurrent_evaluation = false,
        .row_independent = false,
        .owns_compute_threads = true,
        .input_memory = MemorySpace::host,
        .output_memory = MemorySpace::host,
        .persistent_device_state = false,
        .completion = CompletionMode::synchronous,
        .workspace_recreated_on_shape_change = true,
        .preferred_batch_rows = 0,
        .alignment_bytes = alignof(float),
    };
}

BackendCapabilities OnnxPackedKernel::capabilities() const noexcept {
    return onnx_backend_capabilities();
}

Tensor OnnxPackedKernel::evaluate(
    TensorView features,
    std::uint64_t exchange_index,
    double physical_time,
    std::uint32_t /*rank*/) {
    return implementation_->evaluate(features, exchange_index, physical_time);
}

}  // namespace bindnordic

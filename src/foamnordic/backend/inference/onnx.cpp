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

#include "foamnordic/backend/inference/onnx.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace foamnordic::closure {
namespace {

static_assert(ORT_API_VERSION == 28, "FoamNordic requires ONNX Runtime API 28.");

ONNXTensorElementDataType onnx_element(fjord::Element element) {
    switch (element) {
        case fjord::Element::float32:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case fjord::Element::float64:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case fjord::Element::int32:
        case fjord::Element::int64:
            throw std::invalid_argument("FoamNordic ONNX inference requires floating point.");
    }
    throw std::invalid_argument("FoamNordic ONNX element type is unsupported.");
}

std::vector<std::int64_t> onnx_shape(const std::vector<std::uint64_t>& shape) {
    std::vector<std::int64_t> result;
    result.reserve(shape.size());
    for (const auto extent : shape) {
        if (extent > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("FoamNordic tensor shape exceeds ONNX limits.");
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

}  // namespace

void OnnxOptions::validate() const {
    if (intra_op_threads < 1 || inter_op_threads < 1) {
        throw std::invalid_argument("ONNX Runtime thread counts must be positive.");
    }
}

class OnnxPackedKernel::Implementation {
public:
    Implementation(const std::filesystem::path& model_path, OnnxOptions options)
        : environment_(ORT_LOGGING_LEVEL_WARNING, "FoamNordic"),
          options_(make_options(options)),
          session_(environment_, model_path.c_str(), options_),
          memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        const std::string runtime_version = OrtGetApiBase()->GetVersionString();
        if (runtime_version != "1.28.0") {
            throw std::runtime_error(
                "FoamNordic requires ONNX Runtime 1.28.0; found "
                + runtime_version + '.');
        }
        if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
            throw std::invalid_argument(
                "FoamNordic packed ONNX models require exactly one input and output.");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_.GetInputNameAllocated(0, allocator).get();
        output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
        const auto input_type = session_.GetInputTypeInfo(0);
        const auto output_type = session_.GetOutputTypeInfo(0);
        const auto input_info = input_type.GetTensorTypeAndShapeInfo();
        const auto output_info = output_type.GetTensorTypeAndShapeInfo();
        input_element_ = input_info.GetElementType();
        output_element_ = output_info.GetElementType();
        input_shape_ = input_info.GetShape();
        output_shape_ = output_info.GetShape();
        if (output_shape_.size() != 2 || output_shape_[1] == 0) {
            throw std::invalid_argument(
                "FoamNordic packed ONNX output must be rank-two [cell, feature].");
        }
    }

    fjord::Tensor evaluate(
        fjord::TensorView features,
        std::uint64_t exchange_index,
        double physical_time) {
        features.validate();
        if (features.shape.size() != 2 || features.time_index != exchange_index
            || std::abs(features.physical_time - physical_time) > 1.0e-12) {
            throw std::invalid_argument("ONNX features do not match their exchange.");
        }
        const auto element = onnx_element(features.element);
        if (input_element_ != element || output_element_ != element) {
            throw std::invalid_argument(
                "ONNX model dtype does not match FoamNordic packed features.");
        }
        validate_model_shape(input_shape_, features.shape[1], "input");
        const auto shape = onnx_shape(features.shape);
        std::vector<float> float_input;
        std::vector<double> double_input;
        Ort::Value input{nullptr};
        if (features.element == fjord::Element::float32) {
            float_input.resize(features.bytes.size() / sizeof(float));
            std::memcpy(float_input.data(), features.bytes.data(), features.bytes.size());
            input = Ort::Value::CreateTensor<float>(
                memory_,
                float_input.data(),
                float_input.size(),
                shape.data(),
                shape.size());
        } else {
            double_input.resize(features.bytes.size() / sizeof(double));
            std::memcpy(double_input.data(), features.bytes.data(), features.bytes.size());
            input = Ort::Value::CreateTensor<double>(
                memory_,
                double_input.data(),
                double_input.size(),
                shape.data(),
                shape.size());
        }
        const char* input_names[]{input_name_.c_str()};
        const char* output_names[]{output_name_.c_str()};
        auto outputs = session_.Run(
            Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
        if (outputs.size() != 1 || !outputs.front().IsTensor()) {
            throw std::runtime_error("ONNX Runtime did not return one tensor output.");
        }
        const auto output_info = outputs.front().GetTensorTypeAndShapeInfo();
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
        const auto byte_count = output_info.GetElementCount()
                                * fjord::element_size(features.element);
        std::vector<std::byte> bytes(byte_count);
        if (features.element == fjord::Element::float32) {
            std::memcpy(bytes.data(), outputs.front().GetTensorData<float>(), bytes.size());
        } else {
            std::memcpy(bytes.data(), outputs.front().GetTensorData<double>(), bytes.size());
        }
        return {
            "foamnordic.predictions",
            features.element,
            {
                static_cast<std::uint64_t>(output_shape[0]),
                static_cast<std::uint64_t>(output_shape[1]),
            },
            std::move(bytes),
            exchange_index,
            physical_time,
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

    Ort::Env environment_;
    Ort::SessionOptions options_;
    Ort::Session session_;
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

OnnxPackedKernel::~OnnxPackedKernel() = default;
OnnxPackedKernel::OnnxPackedKernel(OnnxPackedKernel&&) noexcept = default;
OnnxPackedKernel& OnnxPackedKernel::operator=(OnnxPackedKernel&&) noexcept = default;

fjord::Tensor OnnxPackedKernel::evaluate(
    fjord::TensorView features,
    std::uint64_t exchange_index,
    double physical_time,
    std::uint32_t /*rank*/) {
    return implementation_->evaluate(features, exchange_index, physical_time);
}

}  // namespace foamnordic::closure

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

#include "foamnordic/backend/inference/compiled.hpp"
#include <dlfcn.h>
#include <limits>
#include <stdexcept>

namespace foamnordic::inference {
CompiledKernel::CompiledKernel(const std::filesystem::path& path, std::uint32_t threads)
    : threads_(threads) {
    if (!threads) throw std::invalid_argument("Compiled thread budget must be positive.");
    library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error(dlerror());
    try {
        auto input = reinterpret_cast<std::size_t (*)()>(dlsym(library_, "foamnordic_input_width"));
        auto output = reinterpret_cast<std::size_t (*)()>(dlsym(library_, "foamnordic_output_width"));
        predict_ = reinterpret_cast<Predict>(dlsym(library_, "foamnordic_predict_parallel"));
        serial_ = reinterpret_cast<SerialPredict>(dlsym(library_, "foamnordic_predict"));
        if (!input || !output || (!predict_ && !serial_)) throw std::runtime_error("Incomplete compiled model ABI.");
        inputs_ = input(); outputs_ = output();
        if (!inputs_ || !outputs_) throw std::runtime_error("Empty compiled model contract.");
    } catch (...) { dlclose(library_); library_ = nullptr; throw; }
}
CompiledKernel::~CompiledKernel() { if (library_) dlclose(library_); }
fjord::Tensor CompiledKernel::evaluate(fjord::TensorView input, std::uint64_t index,
                                      double time, std::uint32_t) {
    input.validate();
    if (input.element != fjord::Element::float64 || input.shape.size() != 2
        || input.shape[1] != inputs_) throw std::invalid_argument("Compiled input contract mismatch.");
    const auto rows = input.shape[0];
    if (outputs_ > std::numeric_limits<std::size_t>::max() / sizeof(double)
        || rows > std::numeric_limits<std::size_t>::max() / sizeof(double) / outputs_)
        throw std::length_error("Compiled output size overflow.");
    fjord::Tensor output{"foamnordic.predictions", fjord::Element::float64,
        {rows, outputs_}, std::vector<std::byte>(rows * outputs_ * sizeof(double)), index, time};
    if (rows) {
        const auto* source = reinterpret_cast<const double*>(input.bytes.data());
        auto* destination = reinterpret_cast<double*>(output.bytes.data());
        const auto status = predict_ ? predict_(source, rows, inputs_, destination, outputs_, threads_)
                                     : serial_(source, rows, inputs_, destination, outputs_);
        if (status != 0) throw std::runtime_error("Compiled model evaluation failed.");
    }
    return output;
}
}

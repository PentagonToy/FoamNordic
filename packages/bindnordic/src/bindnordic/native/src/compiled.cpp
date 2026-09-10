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

#include "bindnordic/compiled.hpp"
#include <dlfcn.h>
#include <limits>
#include <stdexcept>

namespace bindnordic {
CompiledKernel::CompiledKernel(const std::filesystem::path& path, std::uint32_t threads)
    : threads_(threads) {
    if (!threads) throw std::invalid_argument("Compiled thread budget must be positive.");
    library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error(dlerror());
    try {
        const auto symbol = [this](const char* current, const char* legacy) {
            auto* value = dlsym(library_, current);
            return value ? value : dlsym(library_, legacy);
        };
        auto input = reinterpret_cast<std::size_t (*)()>(
            symbol("bindnordic_input_width", "foamnordic_input_width"));
        auto output = reinterpret_cast<std::size_t (*)()>(
            symbol("bindnordic_output_width", "foamnordic_output_width"));
        predict_ = reinterpret_cast<Predict>(
            symbol("bindnordic_predict_parallel", "foamnordic_predict_parallel"));
        serial_ = reinterpret_cast<SerialPredict>(
            symbol("bindnordic_predict", "foamnordic_predict"));
        if (!input || !output || (!predict_ && !serial_)) throw std::runtime_error("Incomplete compiled model ABI.");
        inputs_ = input(); outputs_ = output();
        if (!inputs_ || !outputs_) throw std::runtime_error("Empty compiled model contract.");
    } catch (...) { dlclose(library_); library_ = nullptr; throw; }
}
CompiledKernel::~CompiledKernel() { if (library_) dlclose(library_); }
Tensor CompiledKernel::evaluate(TensorView input, std::uint64_t index,
                                      double time, std::uint32_t) {
    static_cast<void>(index);
    static_cast<void>(time);
    input.validate();
    if (input.element != Element::float64 || input.shape.size() != 2
        || input.shape[1] != inputs_) throw std::invalid_argument("Compiled input contract mismatch.");
    const auto rows = input.shape[0];
    if (outputs_ > std::numeric_limits<std::size_t>::max() / sizeof(double)
        || rows > std::numeric_limits<std::size_t>::max() / sizeof(double) / outputs_)
        throw std::length_error("Compiled output size overflow.");
    Tensor output{"bindnordic.predictions", Element::float64,
        {rows, outputs_}, std::vector<std::byte>(rows * outputs_ * sizeof(double))};
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

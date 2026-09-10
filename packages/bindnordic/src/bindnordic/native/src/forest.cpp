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

#include "bindnordic/forest.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace bindnordic {
ForestKernel::ForestKernel(std::span<const std::byte> data, std::uint32_t threads)
    : threads_(threads) {
    if (!threads || data.size() < 32 || std::memcmp(data.data(), "FNOFOR1\0", 8))
        throw std::invalid_argument("Invalid forest header.");
    std::size_t position = 8;
    auto read = [&]() {
        if (data.size() - position < 8) throw std::invalid_argument("Truncated forest.");
        std::uint64_t result = 0;
        for (int i = 0; i < 8; ++i) result |= std::uint64_t(std::to_integer<unsigned char>(data[position++])) << (i * 8);
        return result;
    };
    width_ = read(); const auto trees = read(); const auto count = read();
    if (!width_ || !trees || trees > (data.size() - position) / 8)
        throw std::invalid_argument("Invalid forest dimensions.");
    for (std::uint64_t i = 0; i < trees; ++i) roots_.push_back(read());
    if (!count || count != (data.size() - position) / 48 || (data.size() - position) % 48)
        throw std::invalid_argument("Invalid forest node count.");
    for (std::uint64_t i = 0; i < count; ++i) {
        Node node{std::bit_cast<std::int64_t>(read()), std::bit_cast<std::int64_t>(read()),
                  std::bit_cast<std::int64_t>(read()), std::bit_cast<double>(read()),
                  std::bit_cast<double>(read()), false};
        const auto missing = read(); node.missing_left = missing;
        const bool leaf = node.left == -1 && node.right == -1;
        if (missing > 1 || !std::isfinite(node.threshold) || !std::isfinite(node.value)
            || (!leaf && (node.left <= static_cast<std::int64_t>(i)
                || node.right <= static_cast<std::int64_t>(i)
                || std::uint64_t(node.left) >= count || std::uint64_t(node.right) >= count
                || node.feature < 0 || std::uint64_t(node.feature) >= width_)))
            throw std::invalid_argument("Invalid forest node or cycle.");
        nodes_.push_back(node);
    }
    for (auto root : roots_) if (root >= count) throw std::invalid_argument("Invalid forest root.");
}
Tensor ForestKernel::evaluate(TensorView input, std::uint64_t index,
                                    double time, std::uint32_t) {
    static_cast<void>(index);
    static_cast<void>(time);
    input.validate();
    if (input.element != Element::float64 || input.shape.size() != 2 || input.shape[1] != width_)
        throw std::invalid_argument("Forest input contract mismatch.");
    const auto rows = input.shape[0];
    if (rows > std::numeric_limits<std::size_t>::max() / sizeof(double))
        throw std::length_error("Forest output size overflow.");
    Tensor result{"bindnordic.predictions", Element::float64,
        {rows, 1}, std::vector<std::byte>(rows * sizeof(double))};
    if (!rows) return result;
    const auto workers = std::min<std::uint64_t>(rows < 256 ? 1 : threads_, rows);
    executor_.run(workers, [&](std::size_t worker) {
        for (auto row = worker; row < rows; row += workers) {
            double sum = 0;
            for (auto root : roots_) {
                auto node = root;
                while (nodes_[node].left != -1) {
                    const auto& n = nodes_[node];
                    double value;
                    std::memcpy(&value, input.bytes.data() + (row * width_ + n.feature) * sizeof(double), sizeof(double));
                    const float narrowed = static_cast<float>(value);
                    node = (std::isnan(narrowed) ? n.missing_left : narrowed <= n.threshold) ? n.left : n.right;
                }
                sum += nodes_[node].value;
            }
            sum /= roots_.size();
            std::memcpy(result.bytes.data() + row * sizeof(double), &sum, sizeof(double));
        }
    });
    return result;
}
}

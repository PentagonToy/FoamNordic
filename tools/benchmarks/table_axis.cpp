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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

int main() {
    constexpr std::size_t points = 4097;
    constexpr std::size_t lookups = 10'000'000;
    constexpr double start = -2.0;
    constexpr double step = 0.001;
    std::vector<double> axis(points);
    for (std::size_t index = 0; index < points; ++index) {
        axis[index] = start + step * index;
    }
    std::mt19937_64 generator(42);
    std::uniform_real_distribution<double> distribution(axis.front(), axis.back());
    std::vector<double> coordinates(lookups);
    for (auto& coordinate : coordinates) coordinate = distribution(generator);

    volatile std::size_t checksum = 0;
    const auto measure = [&](auto locate) {
        const auto began = std::chrono::steady_clock::now();
        for (const auto coordinate : coordinates) checksum += locate(coordinate);
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - began).count();
    };
    const auto explicit_seconds = measure([&](double coordinate) {
        const auto upper = std::upper_bound(axis.begin(), axis.end(), coordinate);
        return upper == axis.end() ? points - 2
                                   : static_cast<std::size_t>(upper - axis.begin() - 1);
    });
    const auto uniform_seconds = measure([&](double coordinate) {
        return std::min(
            points - 2,
            static_cast<std::size_t>(
                std::max(0.0, std::floor((coordinate - start) / step))));
    });
    std::cout << "explicit_seconds=" << explicit_seconds
              << " uniform_seconds=" << uniform_seconds
              << " speedup=" << explicit_seconds / uniform_seconds
              << " checksum=" << checksum << '\n';
}

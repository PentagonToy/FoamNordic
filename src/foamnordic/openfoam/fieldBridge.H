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

#ifndef Foam_foamNordic_fieldBridge_H
#define Foam_foamNordic_fieldBridge_H

#include "fvMesh.H"
#include "word.H"

#include "foamnordic/fjord/tensor.hpp"

#include <cstdint>

namespace Foam::foamNordic {

[[nodiscard]] foamnordic::fjord::TensorView inputFieldView(
    const fvMesh& mesh,
    const word& name,
    std::uint64_t exchangeIndex,
    double physicalTime);

[[nodiscard]] foamnordic::fjord::MutableTensorView outputFieldView(
    const fvMesh& mesh,
    const word& name,
    std::uint64_t exchangeIndex,
    double physicalTime);

void correctFieldBoundary(const fvMesh& mesh, const word& name);

}  // namespace Foam::foamNordic

#endif

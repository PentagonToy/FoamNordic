/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Provider: ESI-OpenCFD (www.openfoam.com)        |
|   \\  /    A nd           | Extension: FoamNordic                           |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/

/*
 * This file is part of FoamNordic.
 * SPDX-License-Identifier: GPL-3.0-only
 */
#ifndef Foam_foamNordic_fieldBridge_H
#define Foam_foamNordic_fieldBridge_H

#include "fvMesh.H"
#include "scalarField.H"
#include "word.H"

#include "foamnordic/fjord/tensor.hpp"

#include <cstdint>

namespace Foam::foamNordic {

[[nodiscard]] foamnordic::fjord::TensorView inputFieldView(
    const fvMesh& mesh,
    const word& name,
    std::uint64_t exchangeIndex,
    double physicalTime,
    scalarField* scratch = nullptr);

[[nodiscard]] foamnordic::fjord::TensorView inputPatchView(
    const fvMesh& mesh,
    const word& name,
    const word& patch,
    std::uint64_t exchangeIndex,
    double physicalTime);

[[nodiscard]] foamnordic::fjord::MutableTensorView outputFieldView(
    const fvMesh& mesh,
    const word& name,
    std::uint64_t exchangeIndex,
    double physicalTime,
    scalarField* scratch = nullptr);

[[nodiscard]] foamnordic::fjord::MutableTensorView outputPatchView(
    const fvMesh& mesh,
    const word& name,
    const word& patch,
    std::uint64_t exchangeIndex,
    double physicalTime);

void commitOutputField(
    const fvMesh& mesh,
    const word& name,
    const scalarField* scratch = nullptr);

void correctFieldBoundary(const fvMesh& mesh, const word& name);

}  // namespace Foam::foamNordic

#endif

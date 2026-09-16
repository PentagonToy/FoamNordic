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
#ifndef Foam_foamNordic_operations_expression_H
#define Foam_foamNordic_operations_expression_H

#include <string>
#include <vector>

namespace Foam
{
namespace foamNordic
{
namespace operations
{

struct Expression
{
    std::string name;
    std::vector<Expression> arguments;

    [[nodiscard]] bool primitive() const noexcept;
    [[nodiscard]] std::string canonical() const;

    [[nodiscard]] static Expression parse(const std::string& source);
};

} // End namespace operations
} // End namespace foamNordic
} // End namespace Foam

#endif

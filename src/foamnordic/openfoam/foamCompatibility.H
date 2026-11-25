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
#ifndef Foam_foamNordic_foamCompatibility_H
#define Foam_foamNordic_foamCompatibility_H

#include "Time.H"
#include "dictionary.H"
#include "tmp.H"

#include <type_traits>
#include <utility>

#ifdef FOAMNORDIC_FOUNDATION
#include "surfaceFieldsFwd.H"
#endif

namespace Foam::foamNordic
{

template<class Value>
Value dictionaryValue(const dictionary& dict, const word& key)
{
#ifdef FOAMNORDIC_FOUNDATION
    return dict.lookup<Value>(key);
#else
    return dict.get<Value>(key);
#endif
}

template<class Value>
Value dictionaryValue
(
    const dictionary& dict,
    const word& key,
    const Value& fallback
)
{
#ifdef FOAMNORDIC_FOUNDATION
    return dict.lookupOrDefault<Value>(key, fallback);
#else
    return dict.getOrDefault<Value>(key, fallback);
#endif
}

template<class Value, class... Args>
tmp<Value> makeTemporary(Args&&... args)
{
#ifdef FOAMNORDIC_FOUNDATION
    return tmp<Value>(new Value(std::forward<Args>(args)...));
#else
    return tmp<Value>::New(std::forward<Args>(args)...);
#endif
}

template<class FieldType>
void correctBoundaryConditions(FieldType& field)
{
#ifdef FOAMNORDIC_FOUNDATION
    // Foundation surface patches do not have an evaluation step.
    if constexpr (!std::is_same_v<typename FieldType::GeoMesh_, surfaceMesh>)
#endif
    {
        field.correctBoundaryConditions();
    }
}

inline word currentTimeName(const Time& runTime)
{
#ifdef FOAMNORDIC_FOUNDATION
    return runTime.name();
#else
    return runTime.timeName();
#endif
}

} // End namespace Foam::foamNordic

#endif

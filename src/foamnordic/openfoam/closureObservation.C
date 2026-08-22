/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Provider:  ESI-OpenCFD (www.openfoam.com)        |
|   \\  /    A nd           | Extension: FoamNordic                           |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/

/*
 * This file is part of FoamNordic.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "closureObservation.H"

#include "closureSession.H"
#include "fieldBridge.H"

#include "IOstreams.H"

#include "foamnordic/fjord/endpoint.hpp"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace Foam::foamNordic {
namespace {

foamnordic::adapter::ObservationRetention retention(const dictionary& dict) {
    const auto maxRecords = dict.getOrDefault<label>("maxRecords", 64);
    const auto maxBytes = dict.getOrDefault<label>("maxBytes", 256 * 1024);
    if (maxRecords <= 0 || maxBytes <= 0) {
        throw std::invalid_argument(
            "FoamNordic observation maxRecords and maxBytes must be positive.");
    }
    const auto overflow = dict.getOrDefault<word>("overflow", "dropOldest");
    foamnordic::adapter::ObservationOverflow policy;
    if (overflow == "dropOldest") {
        policy = foamnordic::adapter::ObservationOverflow::drop_oldest;
    } else if (overflow == "dropNewest") {
        policy = foamnordic::adapter::ObservationOverflow::drop_newest;
    } else {
        throw std::invalid_argument(
            "FoamNordic observation overflow must be dropOldest or dropNewest.");
    }
    return {
        static_cast<std::size_t>(maxRecords),
        static_cast<std::size_t>(maxBytes),
        policy,
    };
}

foamnordic::adapter::CompiledObservationPlan compilePlan(
    const wordList& fields,
    const dictionary& dict) {
    const auto every = dict.getOrDefault<label>("every", 1);
    const auto offset = dict.getOrDefault<label>("offset", 0);
    if (every <= 0 || offset < 0) {
        throw std::invalid_argument(
            "FoamNordic observation every must be positive and offset non-negative.");
    }
    foamnordic::adapter::ObservationPlan declaration;
    std::unordered_set<std::string> unique;
    for (const auto& field : fields) {
        const std::string name(field.c_str());
        if (name.empty() || !unique.insert(name).second) {
            throw std::invalid_argument(
                "FoamNordic observation fields must be non-empty and unique.");
        }
        declaration.observe(
            name,
            {
                static_cast<std::uint64_t>(every),
                static_cast<std::uint64_t>(offset),
            });
    }
    return declaration.compile();
}

}  // namespace

ClosureObservation::ClosureObservation(const dictionary& dict)
    : fields_(dict.get<wordList>("fields")),
      plan_(compilePlan(fields_, dict)),
      publisher_(std::make_unique<foamnordic::adapter::ObservationPublisher>(
          foamnordic::fjord::connect(
              foamnordic::fjord::FjordAddress::parse(
                  resolveRankAddress(dict.get<string>("address")).c_str())),
          retention(dict))) {}

std::unique_ptr<ClosureObservation> ClosureObservation::create(
    const dictionary& closureDict) {
    if (!closureDict.found("observation")) {
        return nullptr;
    }
    return std::unique_ptr<ClosureObservation>(
        new ClosureObservation(closureDict.subDict("observation")));
}

void ClosureObservation::publish(
    const fvMesh& mesh,
    const Time& time,
    std::uint64_t exchangeIndex) noexcept {
    if (!enabled_) {
        return;
    }
    try {
        foamnordic::adapter::ReadOnlyFieldMap views;
        views.reserve(fields_.size());
        for (const auto& field : fields_) {
            views.emplace(
                field.c_str(),
                inputFieldView(
                    mesh,
                    field,
                    exchangeIndex,
                    static_cast<double>(time.value())));
        }
        auto record = plan_.execute(
            exchangeIndex,
            static_cast<double>(time.value()),
            views);
        if (record) {
            static_cast<void>(publisher_->try_publish(std::move(*record)));
        }
        if (!publisher_->healthy()) {
            throw std::runtime_error(publisher_->failure());
        }
    } catch (const std::exception& error) {
        enabled_ = false;
        Info<< "[FoamNord] Observation disabled: " << error.what() << nl;
        publisher_->stop();
    }
}

void ClosureObservation::shutdown() noexcept {
    enabled_ = false;
    if (publisher_) {
        publisher_->stop();
    }
}

}  // namespace Foam::foamNordic

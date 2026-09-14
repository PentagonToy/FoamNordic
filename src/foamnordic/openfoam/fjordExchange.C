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
#include "fjordExchange.H"
#include "fieldProgramObservation.H"
#include "fieldProgramSession.H"
#include "fieldBridge.H"
#include "operations/frame.H"

#include "addToRunTimeSelectionTable.H"

#include "foamnordic/backend/adapter/exchange.hpp"
#include "foamnordic/backend/adapter/sequence.hpp"
#include <chrono>
#include <utility>
#include <vector>

namespace Foam::functionObjects {

defineTypeNameAndDebug(fjordExchange, 0);
addToRunTimeSelectionTable(functionObject, fjordExchange, dictionary);

fjordExchange::fjordExchange(
    const word& name,
    const Time& runTime,
    const dictionary& dict)
:
    fvMeshFunctionObject(name, runTime, dict) {
    sequence_ = std::make_unique<foamnordic::adapter::ExchangeSequence>();
    read(dict);
    if (nativeAlgebra_) {
        nativeMonitor_ =
            std::make_unique<foamNordic::NativeAlgebraMonitor>(dict);
    } else {
        observation_ = foamNordic::FieldProgramObservation::create(dict);
        connectPeer();
    }
}

fjordExchange::~fjordExchange() {
    if (observation_) {
        observation_->shutdown();
        observation_.reset();
    }
    if (harbor_) {
        foamNordic::writeSolverTelemetry(name(), exchange_->profile_snapshot());
        try {
            harbor_->shutdown();
        } catch (...) {
        }
    }
    if (nativeMonitor_) {
        nativeMonitor_->shutdown();
    }
}

bool fjordExchange::read(const dictionary& dict) {
    fvMeshFunctionObject::read(dict);
    dict.readEntry("outputs", outputs_);
    nativeAlgebra_ = dict.findEntry("outputExpressions") != nullptr;
    if (nativeAlgebra_) {
        const List<string> expressions =
            dict.get<List<string>>("outputExpressions");
        outputExpressions_.clear();
        outputExpressions_.reserve(expressions.size());
        for (const auto& expression : expressions) {
            outputExpressions_.push_back(
                foamNordic::operations::Expression::parse(expression));
        }
    } else {
        dict.readEntry("inputs", inputs_);
    }
    inputPatches_ = dict.getOrDefault<wordList>(
        "inputPatches", wordList(inputs_.size(), "none"));
    outputPatches_ = dict.getOrDefault<wordList>(
        "outputPatches", wordList(outputs_.size(), "none"));
    inputKeys_ = dict.getOrDefault<wordList>("inputKeys", inputs_);
    outputKeys_ = dict.getOrDefault<wordList>("outputKeys", outputs_);
    if (!nativeAlgebra_) {
        dict.readEntry("address", address_);
        sharedMemory_ = dict.getOrDefault<bool>("sharedMemory", true);
        ucx_ = dict.getOrDefault<bool>("ucx", false);
        sessionId_ = static_cast<std::uint64_t>(
            dict.getOrDefault<label>("sessionId", 1));
    }
    const auto profileEvery = dict.getOrDefault<label>("profileEvery", 0);
    if (profileEvery < 0) {
        FatalIOErrorInFunction(dict)
            << "profileEvery must not be negative" << exit(FatalIOError);
    }
    profileEvery_ = static_cast<std::uint64_t>(profileEvery);
    const auto stage = dict.getOrDefault<word>(
        "exchangeStage", "timeStepStart");
    if (stage == "timeStepStart") {
        stage_ = Stage::timeStepStart;
    } else if (stage == "outerCorrector") {
        stage_ = Stage::outerCorrector;
    } else if (stage == "pressureCorrected") {
        stage_ = Stage::pressureCorrected;
    } else if (stage == "timeStepEnd") {
        stage_ = Stage::timeStepEnd;
    } else {
        FatalIOErrorInFunction(dict)
            << "exchangeStage must be timeStepStart, outerCorrector, "
               "pressureCorrected, or timeStepEnd"
            << exit(FatalIOError);
    }
    const auto cadence = dict.getOrDefault<word>("exchangeControl", "timeStep");
    if (cadence == "timeStep") {
        sequence_->configure(foamnordic::adapter::ExchangeCadence::time_step);
    } else if (cadence == "everyCall") {
        sequence_->configure(
            foamnordic::adapter::ExchangeCadence::every_call);
    } else {
        FatalIOErrorInFunction(dict)
            << "exchangeControl must be timeStep or everyCall"
            << exit(FatalIOError);
    }
    const bool invalidNative = nativeAlgebra_
        && (outputs_.empty() || outputExpressions_.size() != outputs_.size());
    const bool invalidResident = !nativeAlgebra_
        && (inputs_.empty() || outputs_.empty() || sessionId_ == 0
            || inputKeys_.size() != inputs_.size()
            || outputKeys_.size() != outputs_.size()
            || inputPatches_.size() != inputs_.size()
            || outputPatches_.size() != outputs_.size());
    if (invalidNative || invalidResident) {
        FatalIOErrorInFunction(dict)
            << "matching non-empty native expressions or resident keys/fields "
               "and a positive sessionId are required"
            << exit(FatalIOError);
    }
    return true;
}

void fjordExchange::connectPeer() {
    harbor_ = foamNordic::connectSession(
        address_, sharedMemory_, ucx_, sessionId_);
    foamnordic::adapter::ExchangeContract contract;
    for (const auto& key : inputKeys_) {
        contract.inputs.emplace_back(key.c_str());
    }
    for (const auto& key : outputKeys_) {
        contract.outputs.emplace_back(key.c_str());
    }
    exchange_ = std::make_unique<foamnordic::adapter::AtomicFieldExchange>(
        *harbor_, std::move(contract), profileEvery_);
}

bool fjordExchange::execute() {
    if (stage_ != Stage::timeStepStart) {
        return true;
    }
    return exchange();
}

bool fjordExchange::exchange() {
    try {
        const auto selected = sequence_->next(
            static_cast<std::uint64_t>(mesh_.time().timeIndex()));
        if (!selected) {
            return true;
        }
        if (nativeAlgebra_) {
            const auto started = std::chrono::steady_clock::now();
            const auto exchangeIndex = *selected;
            foamNordic::operations::OperationFrame frame(mesh_);
            std::vector<std::string> snapshots(outputs_.size());
            forAll(outputs_, index) {
                snapshots[index] = frame.snapshot(
                    outputExpressions_[index],
                    static_cast<std::size_t>(index));
            }
            forAll(outputs_, index) {
                frame.validateAssignment(
                    outputs_[index],
                    Foam::foamNordic::operations::Expression::parse(
                        snapshots[index]));
            }
            forAll(outputs_, index) {
                frame.assign(
                    outputs_[index],
                    Foam::foamNordic::operations::Expression::parse(
                        snapshots[index]));
            }
            forAll(outputs_, index) {
                frame.correctBoundary(
                    outputs_[index],
                    Foam::foamNordic::operations::Expression::parse(
                        snapshots[index]));
            }
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started);
            nativeMonitor_->record(
                mesh_, mesh_.time(), exchangeIndex,
                static_cast<std::uint64_t>(elapsed.count()));
            return true;
        }
        const auto exchangeIndex = *selected;
        const auto physicalTime = static_cast<double>(mesh_.time().value());
        const auto topologyGeneration = topology_.observe(mesh_);
        const auto exchangeStarted = std::chrono::steady_clock::now();
        foamnordic::adapter::InputFieldMap inputs;
        foamnordic::adapter::OutputFieldMap outputs;
        inputScratch_.clear();
        inputScratch_.resize(inputs_.size());
        outputScratch_.clear();
        outputScratch_.resize(outputs_.size());
        forAll(inputs_, index) {
            const auto& name = inputs_[index];
            const auto& key = inputKeys_[index];
            auto view = inputPatches_[index] == "none"
                ? foamNordic::inputFieldView(
                    mesh_,
                    name,
                    exchangeIndex,
                    physicalTime,
                    &inputScratch_[index])
                : foamNordic::inputPatchView(
                    mesh_,
                    name,
                    inputPatches_[index],
                    exchangeIndex,
                    physicalTime);
            view.name = key.c_str();
            view.topology_generation = topologyGeneration;
            inputs.emplace(key.c_str(), std::move(view));
        }
        forAll(outputs_, index) {
            const auto& name = outputs_[index];
            const auto& key = outputKeys_[index];
            auto view = outputPatches_[index] == "none"
                ? foamNordic::outputFieldView(
                    mesh_, name, exchangeIndex, physicalTime, &outputScratch_[index])
                : foamNordic::outputPatchView(
                    mesh_,
                    name,
                    outputPatches_[index],
                    exchangeIndex,
                    physicalTime);
            view.name = key.c_str();
            view.topology_generation = topologyGeneration;
            outputs.emplace(key.c_str(), std::move(view));
        }
        exchange_->execute(exchangeIndex, physicalTime, inputs, outputs);
        forAll(outputs_, index) {
            if (outputPatches_[index] == "none") {
                foamNordic::commitOutputField(
                    mesh_, outputs_[index], &outputScratch_[index]);
                foamNordic::correctFieldBoundary(mesh_, outputs_[index]);
            }
        }
        if (observation_) {
            const auto exchangeWait = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - exchangeStarted).count();
            observation_->publish(
                mesh_, mesh_.time(), exchangeIndex, topologyGeneration,
                exchangeWait);
        }
        return true;
    } catch (const std::exception& error) {
        FatalErrorInFunction << error.what() << exit(FatalError);
        return false;
    }
}

bool fjordExchange::write() {
    if (stage_ != Stage::timeStepEnd) {
        return true;
    }
    return exchange();
}

void fjordExchange::updateMesh(const mapPolyMesh& map) {
    fvMeshFunctionObject::updateMesh(map);
    topology_.markChanged();
}

}  // namespace Foam::functionObjects

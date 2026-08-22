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

#include "fjordExchange.H"
#include "closureSession.H"
#include "fieldBridge.H"

#include "addToRunTimeSelectionTable.H"

#include "foamnordic/backend/adapter/exchange.hpp"
#include "foamnordic/backend/adapter/sequence.hpp"
#include <utility>

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
    connectPeer();
}

fjordExchange::~fjordExchange() {
    if (harbor_) {
        try {
            harbor_->shutdown();
        } catch (...) {
        }
    }
}

bool fjordExchange::read(const dictionary& dict) {
    fvMeshFunctionObject::read(dict);
    dict.readEntry("inputs", inputs_);
    dict.readEntry("outputs", outputs_);
    inputKeys_ = dict.getOrDefault<wordList>("inputKeys", inputs_);
    outputKeys_ = dict.getOrDefault<wordList>("outputKeys", outputs_);
    dict.readEntry("address", address_);
    sharedMemory_ = dict.getOrDefault<bool>("sharedMemory", true);
    ucx_ = dict.getOrDefault<bool>("ucx", false);
    sessionId_ = static_cast<std::uint64_t>(
        dict.getOrDefault<label>("sessionId", 1));
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
    if (inputs_.empty() || outputs_.empty() || sessionId_ == 0
        || inputKeys_.size() != inputs_.size()
        || outputKeys_.size() != outputs_.size()) {
        FatalIOErrorInFunction(dict)
            << "matching non-empty keys/fields and a positive sessionId are required"
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
        *harbor_, std::move(contract));
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
        const auto exchangeIndex = *selected;
        const auto physicalTime = static_cast<double>(mesh_.time().value());
        foamnordic::adapter::InputFieldMap inputs;
        foamnordic::adapter::OutputFieldMap outputs;
        inputScratch_.clear();
        inputScratch_.resize(inputs_.size());
        outputScratch_.clear();
        outputScratch_.resize(outputs_.size());
        forAll(inputs_, index) {
            const auto& name = inputs_[index];
            const auto& key = inputKeys_[index];
            auto view = foamNordic::inputFieldView(
                mesh_,
                name,
                exchangeIndex,
                physicalTime,
                &inputScratch_[index]);
            view.name = key.c_str();
            inputs.emplace(key.c_str(), std::move(view));
        }
        forAll(outputs_, index) {
            const auto& name = outputs_[index];
            const auto& key = outputKeys_[index];
            auto view = foamNordic::outputFieldView(
                mesh_, name, exchangeIndex, physicalTime, &outputScratch_[index]);
            view.name = key.c_str();
            outputs.emplace(key.c_str(), std::move(view));
        }
        exchange_->execute(exchangeIndex, physicalTime, inputs, outputs);
        forAll(outputs_, index) {
            foamNordic::commitOutputField(
                mesh_, outputs_[index], &outputScratch_[index]);
            foamNordic::correctFieldBoundary(mesh_, outputs_[index]);
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

}  // namespace Foam::functionObjects

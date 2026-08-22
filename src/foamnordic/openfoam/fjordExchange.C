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
    dict.readEntry("address", address_);
    sharedMemory_ = dict.getOrDefault<bool>("sharedMemory", true);
    sessionId_ = static_cast<std::uint64_t>(
        dict.getOrDefault<label>("sessionId", 1));
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
    if (inputs_.empty() || outputs_.empty() || sessionId_ == 0) {
        FatalIOErrorInFunction(dict)
            << "inputs, outputs, and a positive sessionId are required"
            << exit(FatalIOError);
    }
    return true;
}

void fjordExchange::connectPeer() {
    harbor_ = foamNordic::connectSession(
        address_, sharedMemory_, sessionId_);
    foamnordic::adapter::ExchangeContract contract;
    for (const auto& name : inputs_) {
        contract.inputs.emplace_back(name.c_str());
    }
    for (const auto& name : outputs_) {
        contract.outputs.emplace_back(name.c_str());
    }
    exchange_ = std::make_unique<foamnordic::adapter::AtomicFieldExchange>(
        *harbor_, std::move(contract));
}

bool fjordExchange::execute() {
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
        for (const auto& name : inputs_) {
            inputs.emplace(
                name.c_str(),
                foamNordic::inputFieldView(
                    mesh_, name, exchangeIndex, physicalTime));
        }
        for (const auto& name : outputs_) {
            outputs.emplace(
                name.c_str(),
                foamNordic::outputFieldView(
                    mesh_, name, exchangeIndex, physicalTime));
        }
        exchange_->execute(exchangeIndex, physicalTime, inputs, outputs);
        for (const auto& name : outputs_) {
            foamNordic::correctFieldBoundary(mesh_, name);
        }
        return true;
    } catch (const std::exception& error) {
        FatalErrorInFunction << error.what() << exit(FatalError);
        return false;
    }
}

bool fjordExchange::write() {
    return true;
}

}  // namespace Foam::functionObjects

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

#ifndef Foam_functionObjects_fjordExchange_H
#define Foam_functionObjects_fjordExchange_H

#include "fvMeshFunctionObject.H"
#include "scalarField.H"
#include "wordList.H"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace foamnordic::adapter {
class AtomicFieldExchange;
class ExchangeSequence;
}

namespace foamnordic::fjord {
class Harbor;
}

namespace Foam::foamNordic {
class ClosureObservation;
}

namespace Foam::functionObjects {

class fjordExchange final : public fvMeshFunctionObject {
    wordList inputKeys_;
    wordList inputs_;
    wordList outputKeys_;
    wordList outputs_;
    string address_;
    bool sharedMemory_{true};
    bool ucx_{false};
    std::uint64_t sessionId_{1};
    std::unique_ptr<foamnordic::fjord::Harbor> harbor_;
    std::unique_ptr<foamnordic::adapter::AtomicFieldExchange> exchange_;
    std::unique_ptr<foamnordic::adapter::ExchangeSequence> sequence_;
    std::unique_ptr<foamNordic::ClosureObservation> observation_;
    std::vector<scalarField> inputScratch_;
    std::vector<scalarField> outputScratch_;

    enum class Stage { timeStepStart, outerCorrector, pressureCorrected, timeStepEnd };
    Stage stage_{Stage::timeStepStart};

    void connectPeer();
    bool exchange();

public:
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-missing-override"
#endif
    TypeName("foamNordicExchange");
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    fjordExchange(const word& name, const Time& runTime, const dictionary& dict);
    ~fjordExchange() override;

    fjordExchange(const fjordExchange&) = delete;
    void operator=(const fjordExchange&) = delete;

    bool read(const dictionary& dict) override;
    bool execute() override;
    bool write() override;
};

}  // namespace Foam::functionObjects

#endif

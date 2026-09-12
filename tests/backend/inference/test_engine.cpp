#include <cassert>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "foamnordic/backend/inference/engine.hpp"

namespace {

using foamnordic::inference::EngineAdvance;
using foamnordic::inference::EngineCostObservation;
using foamnordic::inference::EngineCostObserver;
using foamnordic::inference::EngineOutcome;
using foamnordic::inference::EngineOperation;
using foamnordic::inference::EngineStep;
using foamnordic::inference::ExecutionResources;
using foamnordic::inference::ProgramContract;
using foamnordic::inference::StatefulEngineBackend;
using foamnordic::inference::StatefulEngineCapabilities;
using foamnordic::inference::TensorMap;
using foamnordic::inference::TransactionalEngine;

class RecordingObserver final : public EngineCostObserver {
public:
    void observe(const EngineCostObservation& observation) noexcept override {
        observations.push_back(observation);
    }

    std::vector<EngineCostObservation> observations;
};

foamnordic::fjord::Tensor tensor(std::string name, double value) {
    std::vector<std::byte> bytes(sizeof(double));
    std::memcpy(bytes.data(), &value, sizeof(double));
    return {
        std::move(name),
        foamnordic::fjord::Element::float64,
        {1, 1},
        std::move(bytes),
    };
}

double value(const TensorMap& values, const char* name) {
    double result{};
    const auto& bytes = values.at(name).bytes;
    std::memcpy(&result, bytes.data(), sizeof(double));
    return result;
}

ProgramContract contract() {
    return {
        "stateful",
        {{"x", bindnordic::Element::float64, 1}},
        {{"y", bindnordic::Element::float64, 1}},
    };
}

class ProbeEngine final : public StatefulEngineBackend {
public:
    StatefulEngineCapabilities engine_capabilities() const noexcept override {
        return {true, true, survives_topology_change};
    }

    bindnordic::BackendCapabilities backend_capabilities()
        const noexcept override {
        return memory;
    }

    void initialise(const ProgramContract&, const ExecutionResources&) override {
        initialised = true;
    }

    EngineAdvance advance(
        const TensorMap& state,
        const TensorMap& inputs,
        const EngineStep&) override {
        if (fail) {
            throw std::runtime_error("probe failure");
        }
        const auto next = value(state, "state") + value(inputs, "x");
        TensorMap candidate;
        candidate.emplace("state", tensor("state", next));
        TensorMap outputs;
        outputs.emplace("y", tensor("y", next));
        return {std::move(candidate), std::move(outputs)};
    }

    std::vector<std::byte> checkpoint(const TensorMap& state) const override {
        return state.at("state").bytes;
    }

    TensorMap restore(std::span<const std::byte> payload) override {
        double restored{};
        std::memcpy(&restored, payload.data(), sizeof(double));
        TensorMap state;
        state.emplace("state", tensor("state", restored));
        return state;
    }

    void close() noexcept override { closed = true; }

    bindnordic::BackendCapabilities memory{};
    bool survives_topology_change{false};
    bool fail{false};
    bool initialised{false};
    bool closed{false};
};

TensorMap initial_state(double initial) {
    TensorMap state;
    state.emplace("state", tensor("state", initial));
    return state;
}

TensorMap inputs(double input) {
    TensorMap result;
    result.emplace("x", tensor("x", input));
    return result;
}

EngineStep step(std::uint64_t topology = 1) {
    return {1, 1, topology, 0.1, 0.01};
}

void test_commit_rollback_and_restore() {
    ProbeEngine backend;
    RecordingObserver observer;
    TransactionalEngine engine(contract(), {}, backend, &observer);
    engine.initialise(initial_state(2.0), 1);
    assert(backend.initialised);
    auto outputs = engine.advance(inputs(3.0), step());
    assert(value(outputs, "y") == 5.0);
    assert(value(engine.state(), "state") == 5.0);
    assert(engine.state_generation() == 1);
    assert(observer.observations.size() == 1);
    assert(observer.observations.back().outcome == EngineOutcome::completed);
    assert(observer.observations.back().input_bytes == sizeof(double));
    assert(observer.observations.back().state_bytes == sizeof(double));
    assert(observer.observations.back().output_bytes == sizeof(double));

    const auto saved = engine.checkpoint();
    assert(observer.observations.size() == 2);
    assert(observer.observations.back().operation == EngineOperation::checkpoint);
    assert(observer.observations.back().outcome == EngineOutcome::completed);
    backend.fail = true;
    try {
        static_cast<void>(engine.advance(inputs(4.0), step()));
        assert(false);
    } catch (const std::runtime_error&) {
    }
    assert(value(engine.state(), "state") == 5.0);
    assert(engine.state_generation() == 1);
    assert(observer.observations.size() == 3);
    assert(observer.observations.back().outcome == EngineOutcome::failed);

    backend.fail = false;
    static_cast<void>(engine.advance(inputs(4.0), step()));
    assert(value(engine.state(), "state") == 9.0);
    engine.restore(saved);
    assert(value(engine.state(), "state") == 5.0);
    assert(engine.state_generation() == 1);
    assert(observer.observations.back().operation == EngineOperation::restore);
    assert(observer.observations.back().outcome == EngineOutcome::completed);
}

void test_topology_change_is_explicit() {
    ProbeEngine backend;
    TransactionalEngine engine(contract(), {}, backend);
    engine.initialise(initial_state(1.0), 1);
    try {
        static_cast<void>(engine.advance(inputs(1.0), step(2)));
        assert(false);
    } catch (const std::runtime_error&) {
    }
    assert(value(engine.state(), "state") == 1.0);

    ProbeEngine persistent_backend;
    persistent_backend.survives_topology_change = true;
    TransactionalEngine persistent(contract(), {}, persistent_backend);
    persistent.initialise(initial_state(1.0), 1);
    static_cast<void>(persistent.advance(inputs(1.0), step(2)));
    assert(persistent.topology_generation() == 2);
}

void test_unsupported_memory_is_rejected() {
    ProbeEngine backend;
    backend.memory.input_memory = bindnordic::MemorySpace::device;
    try {
        TransactionalEngine engine(contract(), {}, backend);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
}

}  // namespace

int main() {
    test_commit_rollback_and_restore();
    test_topology_change_is_explicit();
    test_unsupported_memory_is_rejected();
}

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "foamnordic/backend/adapter/exchange.hpp"
#include "foamnordic/backend/adapter/field.hpp"
#include "foamnordic/backend/adapter/observation.hpp"
#include "foamnordic/backend/adapter/observation_stream.hpp"
#include "foamnordic/backend/adapter/plan.hpp"
#include "foamnordic/backend/adapter/port.hpp"
#include "foamnordic/backend/adapter/sequence.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

foamnordic::fjord::MutableTensorView velocity_view(
    std::array<double, 6>& values,
    std::uint64_t exchange_index = 1,
    double physical_time = 0.1) {
    return {
        "U",
        foamnordic::fjord::Element::float64,
        {2, 3},
        foamnordic::fjord::as_writable_bytes(std::span(values)),
        exchange_index,
        physical_time,
    };
}

void test_identity_is_an_in_place_no_op() {
    std::array<double, 6> velocity{1.0, -2.0, 3.0, 4.0, 5.0, 6.0};
    const auto original = velocity;
    foamnordic::adapter::FieldTransform identity;
    require(identity.is_identity(3), "Default field transform is not identity.");
    identity.apply_in_place(velocity_view(velocity));
    require(velocity == original, "Identity transform modified the LDC velocity field.");
}

void test_fused_component_transform_and_statistics() {
    std::array<double, 6> velocity{1.0, -2.0, 3.0, 4.0, 5.0, 6.0};
    foamnordic::adapter::FieldTransform transform{
        {2.0, 1.0, -1.0},
        {0.0, 1.0, 0.0},
        -5.0,
        5.0,
    };
    transform.apply_in_place(velocity_view(velocity));
    require(
        velocity == std::array<double, 6>{2.0, -1.0, -3.0, 5.0, 5.0, -5.0},
        "Fused component transform produced incorrect values.");

    const auto summary = foamnordic::adapter::statistics(velocity_view(velocity).read_only());
    require(summary.minimum == -5.0, "Native minimum is incorrect.");
    require(summary.maximum == 5.0, "Native maximum is incorrect.");
    require(std::abs(summary.mean - 0.5) < 1.0e-15, "Native mean is incorrect.");
    require(summary.count == 6, "Native statistic count is incorrect.");
}

void test_scalar_standardization() {
    std::array<float, 3> values{2.0F, 4.0F, 6.0F};
    foamnordic::fjord::MutableTensorView field{
        "delta",
        foamnordic::fjord::Element::float32,
        {3},
        foamnordic::fjord::as_writable_bytes(std::span(values)),
        2,
        0.2,
    };
    foamnordic::adapter::FieldTransform standardize{{0.5}, {-2.0}};
    standardize.apply_in_place(field);
    require(
        values == std::array<float, 3>{-1.0F, 0.0F, 1.0F},
        "Native scalar standardization is incorrect.");
}

void test_compiled_plan_executes_outside_python_loop() {
    foamnordic::adapter::ExecutionPlan declaration;
    declaration.modify("U", foamnordic::adapter::FieldTransform{{1.005}, {0.0}});
    const auto plan = declaration.compile();
    require(plan.size() == 1, "Compiled execution plan has an incorrect size.");

    foamnordic::adapter::ObservationPlan observation_declaration;
    observation_declaration.observe("U", {2, 1}).observe("p", {2, 1});
    const auto observations = observation_declaration.compile();

    std::array<double, 6> velocity{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::array<double, 2> pressure{-1.0, 2.0};
    std::size_t observation_count = 0;
    foamnordic::adapter::ObservationBuffer observation_buffer({1, 1'024});
    for (std::uint64_t exchange = 1; exchange <= 3; ++exchange) {
        const auto physical_time = static_cast<double>(exchange) * 0.1;
        foamnordic::fjord::MutableTensorView pressure_field{
            "p",
            foamnordic::fjord::Element::float64,
            {2},
            foamnordic::fjord::as_writable_bytes(std::span(pressure)),
            exchange,
            physical_time,
        };
        const foamnordic::adapter::MutableFieldMap fields{
            {"U", velocity_view(velocity, exchange, physical_time)},
            {"p", pressure_field},
        };
        plan.execute(exchange, physical_time, fields);
        const foamnordic::adapter::ReadOnlyFieldMap read_only_fields{
            {"U", fields.at("U").read_only()},
            {"p", fields.at("p").read_only()},
        };
        auto record = observations.execute(exchange, physical_time, read_only_fields);
        if (record) {
            observation_count += record->fields.size();
            require(
                observation_buffer.try_publish(std::move(*record)),
                "A sparse observation could not be published.");
        }
    }

    const auto expected = 1.005 * 1.005 * 1.005;
    require(
        std::abs(velocity.front() - expected) < 1.0e-15,
        "Compiled native plan did not modify U on every exchange.");
    require(observation_count == 4, "Observation cadence did not remain native and sparse.");
    require(
        observation_buffer.buffered_records() == 1
            && observation_buffer.dropped_records() == 1,
        "Drop-oldest observation retention is not bounded.");
    const auto latest = observation_buffer.try_pop_oldest();
    require(
        latest.has_value() && latest->exchange_index == 3,
        "Observation retention did not preserve the newest record.");
}

void test_execution_and_observation_ownership_are_separate() {
    foamnordic::adapter::ExecutionPlan execution;
    execution.modify("U", foamnordic::adapter::FieldTransform{{1.0}, {0.0}});
    bool duplicate_writer_rejected = false;
    try {
        execution.modify("U", foamnordic::adapter::FieldTransform{{2.0}, {0.0}});
    } catch (const std::invalid_argument&) {
        duplicate_writer_rejected = true;
    }
    require(duplicate_writer_rejected, "Execution plan accepted two writers for U.");

    foamnordic::adapter::ObservationBuffer buffer({1, 128});
    foamnordic::adapter::ObservationRecord oversized{
        1,
        0.1,
        {{std::string(256, 'U'), {0.0, 1.0, 0.5, 2}}},
    };
    require(!buffer.try_publish(std::move(oversized)), "Oversized observation was retained.");
    require(
        buffer.buffered_records() == 0 && buffer.dropped_records() == 1,
        "Oversized observation violated the memory boundary.");
}

void test_observation_stream_is_separate_from_closure_transport() {
    auto channels = foamnordic::fjord::local_channel_pair();
    foamnordic::adapter::ObservationPublisher publisher(
        std::move(channels.first), {4, 4'096});
    foamnordic::adapter::ObservationReceiver receiver(std::move(channels.second));

    require(
        publisher.try_publish({
            17,
            0.25,
            {
                {"U", {-2.0, 5.0, 1.5, 18}},
                {"p", {-1.0, 2.0, 0.25, 6}},
            },
        }),
        "Observation publisher rejected a valid compact record.");
    const auto received = receiver.receive(std::chrono::seconds(1));
    require(received.has_value(), "Longship observation receiver timed out.");
    require(
        received->exchange_index == 17
            && received->physical_time == 0.25
            && received->fields.size() == 2
            && received->fields[0].field == "U"
            && received->fields[0].values.minimum == -2.0
            && received->fields[0].values.maximum == 5.0
            && received->fields[0].values.mean == 1.5
            && received->fields[0].values.count == 18,
        "Observation stream changed native summary metadata.");
    publisher.stop();
    receiver.close();
    require(publisher.healthy(), "Healthy observation stream reported failure.");
}

void test_atomic_field_exchange_applies_only_committed_output() {
    auto channels = foamnordic::fjord::local_channel_pair();
    const foamnordic::fjord::HarborOptions options{
        foamnordic::fjord::HandshakeMode::disabled,
        std::chrono::milliseconds(1'000),
    };
    foamnordic::fjord::Harbor solver(std::move(channels.first), options);
    foamnordic::fjord::Harbor closure(std::move(channels.second), options);
    foamnordic::adapter::AtomicFieldExchange exchange(
        solver, {{"c_tilde", "c_var", "T_tilde"}, {"omega_c"}});

    std::thread worker([&closure] {
        std::vector<foamnordic::fjord::Tensor> inputs;
        while (true) {
            auto message = closure.receive_message();
            if (message.kind == foamnordic::fjord::RuneKind::tensor) {
                inputs.push_back(std::move(*message.tensor));
                continue;
            }
            require(
                message.kind == foamnordic::fjord::RuneKind::complete
                    && message.tensor_count == inputs.size(),
                "Atomic input commit did not contain the prepared fields.");
            break;
        }
        const std::array<double, 2> prediction{4.0, 8.0};
        const foamnordic::fjord::TensorView output{
            "omega_c",
            foamnordic::fjord::Element::float64,
            {2},
            foamnordic::fjord::as_bytes(std::span(prediction)),
            7,
            0.25,
        };
        closure.publish(7, std::span(&output, 1));
    });

    std::array<double, 2> progress{0.2, 0.8};
    std::array<double, 2> variance{0.01, 0.02};
    std::array<double, 2> temperature{800.0, 1400.0};
    std::array<double, 2> reaction_rate{-1.0, -1.0};
    const auto view = [](const char* name, std::array<double, 2>& values) {
        return foamnordic::fjord::MutableTensorView{
            name,
            foamnordic::fjord::Element::float64,
            {2},
            foamnordic::fjord::as_writable_bytes(std::span(values)),
            7,
            0.25,
        };
    };
    exchange.execute(
        7,
        0.25,
        {
            {"c_tilde", view("c_tilde", progress)},
            {"c_var", view("c_var", variance)},
            {"T_tilde", view("T_tilde", temperature)},
            {"omega_c", view("omega_c", reaction_rate)},
        });
    worker.join();
    require(
        reaction_rate == std::array<double, 2>{4.0, 8.0},
        "Committed closure output was not applied atomically.");
}

void test_incomplete_output_never_modifies_field() {
    auto channels = foamnordic::fjord::local_channel_pair();
    const foamnordic::fjord::HarborOptions options{
        foamnordic::fjord::HandshakeMode::disabled,
        std::chrono::milliseconds(1'000),
    };
    foamnordic::fjord::Harbor solver(std::move(channels.first), options);
    foamnordic::fjord::Harbor closure(std::move(channels.second), options);
    foamnordic::adapter::AtomicFieldExchange exchange(solver, {{"U"}, {"nut"}});

    std::thread worker([&closure] {
        static_cast<void>(closure.receive_message());
        static_cast<void>(closure.receive_message());
        const std::array<double, 2> prediction{9.0, 9.0};
        closure.send({
            "nut",
            foamnordic::fjord::Element::float64,
            {2},
            foamnordic::fjord::as_bytes(std::span(prediction)),
            3,
            0.1,
        });
        closure.complete(3, 2);
    });

    std::array<double, 2> velocity{1.0, 2.0};
    std::array<double, 2> viscosity{0.0, 0.0};
    const auto field = [](const char* name, std::array<double, 2>& values) {
        return foamnordic::fjord::MutableTensorView{
            name,
            foamnordic::fjord::Element::float64,
            {2},
            foamnordic::fjord::as_writable_bytes(std::span(values)),
            3,
            0.1,
        };
    };
    bool rejected = false;
    try {
        exchange.execute(
            3,
            0.1,
            {{"U", field("U", velocity)}, {"nut", field("nut", viscosity)}});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    worker.join();
    require(rejected, "An incomplete output commit was accepted.");
    require(
        viscosity == std::array<double, 2>{0.0, 0.0},
        "An incomplete output batch partially modified the solver field.");
}

void test_stale_output_never_modifies_field() {
    auto channels = foamnordic::fjord::local_channel_pair();
    const foamnordic::fjord::HarborOptions options{
        foamnordic::fjord::HandshakeMode::disabled,
        std::chrono::milliseconds(1'000),
    };
    foamnordic::fjord::Harbor solver(std::move(channels.first), options);
    foamnordic::fjord::Harbor closure(std::move(channels.second), options);
    foamnordic::adapter::AtomicFieldExchange exchange(
        solver, {{"U"}, {"nut"}});

    std::thread worker([&closure] {
        static_cast<void>(closure.receive_message());
        static_cast<void>(closure.receive_message());
        const std::array<double, 2> stalePrediction{9.0, 9.0};
        const foamnordic::fjord::TensorView staleOutput{
            "nut",
            foamnordic::fjord::Element::float64,
            {2},
            foamnordic::fjord::as_bytes(std::span(stalePrediction)),
            3,
            0.05,
        };
        closure.publish(3, std::span(&staleOutput, 1));
    });

    std::array<double, 2> velocity{1.0, 2.0};
    std::array<double, 2> viscosity{-1.0, -1.0};
    const auto field = [](const char* name, std::array<double, 2>& values) {
        return foamnordic::fjord::MutableTensorView{
            name,
            foamnordic::fjord::Element::float64,
            {2},
            foamnordic::fjord::as_writable_bytes(std::span(values)),
            3,
            0.1,
        };
    };
    bool rejected = false;
    try {
        exchange.execute(
            3,
            0.1,
            {{"U", field("U", velocity)}, {"nut", field("nut", viscosity)}});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    worker.join();
    require(rejected, "A stale closure output was accepted.");
    require(
        viscosity == std::array<double, 2>{-1.0, -1.0},
        "A stale closure output modified the solver field.");
}

void test_worker_error_never_modifies_field(bool sharedMemory) {
    auto channels = sharedMemory
                        ? foamnordic::fjord::shared_memory_channel_pair(8, 256)
                        : foamnordic::fjord::local_channel_pair();
    const foamnordic::fjord::HarborOptions options{
        foamnordic::fjord::HandshakeMode::disabled,
        std::chrono::milliseconds(1'000),
    };
    foamnordic::fjord::Harbor solver(std::move(channels.first), options);
    foamnordic::fjord::Harbor closure(std::move(channels.second), options);
    foamnordic::adapter::AtomicFieldExchange exchange(
        solver, {{"U"}, {"nut"}});

    std::thread worker([&closure] {
        static_cast<void>(closure.receive_message());
        static_cast<void>(closure.receive_message());
        closure.fail_exchange(3);
    });

    std::array<double, 2> velocity{1.0, 2.0};
    std::array<double, 2> viscosity{-1.0, -1.0};
    const auto field = [](const char* name, std::array<double, 2>& values) {
        return foamnordic::fjord::MutableTensorView{
            name,
            foamnordic::fjord::Element::float64,
            {2},
            foamnordic::fjord::as_writable_bytes(std::span(values)),
            3,
            0.1,
        };
    };
    bool rejected = false;
    try {
        exchange.execute(
            3,
            0.1,
            {{"U", field("U", velocity)}, {"nut", field("nut", viscosity)}});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    worker.join();
    require(rejected, "A native worker error was accepted as output.");
    require(
        viscosity == std::array<double, 2>{-1.0, -1.0},
        "A native worker error modified the solver field.");
}

void test_pimple_exchange_sequence() {
    foamnordic::adapter::ExchangeSequence time_steps;
    require(time_steps.next(1) == 1, "First timestep index is incorrect.");
    require(
        !time_steps.next(1).has_value(),
        "Repeated PIMPLE outer corrector created a duplicate timestep exchange.");
    require(time_steps.next(2) == 2, "Next timestep index is incorrect.");

    foamnordic::adapter::ExchangeSequence closure_calls(
        foamnordic::adapter::ExchangeCadence::every_call);
    require(closure_calls.next(1) == 0, "First closure call is incorrect.");
    require(closure_calls.next(1) == 1, "Second closure call is incorrect.");
    require(closure_calls.next(2) == 2, "Closure call index is not monotonic.");
}

void test_closure_exchange_runs_on_every_call() {
    auto channels = foamnordic::fjord::local_channel_pair();
    const foamnordic::fjord::HarborOptions options{
        foamnordic::fjord::HandshakeMode::disabled,
        std::chrono::milliseconds(1'000),
    };
    foamnordic::fjord::Harbor solver(std::move(channels.first), options);
    foamnordic::fjord::Harbor closure(std::move(channels.second), options);
    foamnordic::adapter::BlockingClosureExchange exchange(
        solver, {{"U"}, {"U"}});

    std::thread worker([&closure] {
        for (std::uint64_t expected = 0; expected < 2; ++expected) {
            const auto input = closure.receive_message();
            const auto complete = closure.receive_message();
            require(
                input.tensor.has_value()
                    && input.exchange_index == expected
                    && input.tensor->solver_time_index == 7
                    && complete.kind == foamnordic::fjord::RuneKind::complete
                    && complete.exchange_index == expected,
                "Per-call worker received an incorrect transaction.");
            const auto view = input.tensor->view();
            closure.publish(expected, std::span(&view, 1));
        }
    });

    std::array<double, 6> velocity{1.0, -2.0, 3.0, 4.0, -5.0, 6.0};
    const auto first = exchange.execute(
        7, 0.25, {{"U", velocity_view(velocity, 7, 0.25)}});
    const auto second = exchange.execute(
        7, 0.25, {{"U", velocity_view(velocity, 7, 0.25)}});
    worker.join();
    require(
        first == 0 && second == 1,
        "Repeated closure calls did not receive distinct exchange indices.");
}

void test_generic_solver_closure_port() {
    auto channels = foamnordic::fjord::local_channel_pair();
    const foamnordic::fjord::HarborOptions options{
        foamnordic::fjord::HandshakeMode::disabled,
        std::chrono::milliseconds(1'000),
    };
    foamnordic::fjord::Harbor solver(std::move(channels.first), options);
    foamnordic::fjord::Harbor closure(std::move(channels.second), options);
    foamnordic::adapter::ClosurePort port(
        solver, {{"grad(U)", "delta"}, {"nut"}});

    std::thread worker([&closure] {
        for (std::uint64_t exchange = 0; exchange < 2; ++exchange) {
            const auto gradient = closure.receive_message();
            const auto delta = closure.receive_message();
            const auto complete = closure.receive_message();
            require(
                gradient.tensor.has_value() && delta.tensor.has_value()
                    && gradient.tensor->solver_time_index == 9
                    && delta.tensor->solver_time_index == 9
                    && complete.exchange_index == exchange,
                "Generic closure port did not publish one complete call.");
            const std::array<double, 2> prediction{
                0.01 + static_cast<double>(exchange),
                0.02 + static_cast<double>(exchange),
            };
            const foamnordic::fjord::TensorView output{
                "nut",
                foamnordic::fjord::Element::float64,
                {2},
                foamnordic::fjord::as_bytes(std::span(prediction)),
                exchange,
                0.4,
                9,
            };
            closure.publish(exchange, std::span(&output, 1));
        }
    });

    const std::array<double, 18> gradient{};
    const std::array<double, 2> delta{0.1, 0.2};
    std::array<double, 2> nut{};
    const auto invoke = [&] {
        auto call = port.begin(9, 0.4);
        call.provide({
                "grad(U)",
                foamnordic::fjord::Element::float64,
                {2, 9},
                foamnordic::fjord::as_bytes(std::span(gradient)),
                9,
                0.4,
            })
            .provide({
                "delta",
                foamnordic::fjord::Element::float64,
                {2},
                foamnordic::fjord::as_bytes(std::span(delta)),
                9,
                0.4,
            })
            .receive({
                "nut",
                foamnordic::fjord::Element::float64,
                {2},
                foamnordic::fjord::as_writable_bytes(std::span(nut)),
                9,
                0.4,
            });
        return call.commit();
    };
    require(invoke() == 0, "First generic closure call is incorrect.");
    require(invoke() == 1, "Second generic closure call is incorrect.");
    worker.join();
    require(
        nut == std::array<double, 2>{1.01, 1.02},
        "Generic closure output was not committed to its solver field.");
}

}  // namespace

int main() {
    test_identity_is_an_in_place_no_op();
    test_fused_component_transform_and_statistics();
    test_scalar_standardization();
    test_compiled_plan_executes_outside_python_loop();
    test_execution_and_observation_ownership_are_separate();
    test_observation_stream_is_separate_from_closure_transport();
    test_atomic_field_exchange_applies_only_committed_output();
    test_incomplete_output_never_modifies_field();
    test_stale_output_never_modifies_field();
    test_worker_error_never_modifies_field(false);
    test_worker_error_never_modifies_field(true);
    test_pimple_exchange_sequence();
    test_closure_exchange_runs_on_every_call();
    test_generic_solver_closure_port();
}

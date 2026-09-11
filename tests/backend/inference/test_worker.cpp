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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "foamnordic/backend/inference/affine.hpp"
#include "foamnordic/backend/inference/manifest.hpp"
#include "foamnordic/backend/inference/model.hpp"
#include "foamnordic/backend/inference/worker.hpp"
#include "foamnordic/fjord/endpoint.hpp"
#include "foamnordic/fjord/harbor.hpp"
#ifdef FOAMNORDIC_HAVE_UCX
#include "foamnordic/fjord/ucx_channel.hpp"
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

foamnordic::fjord::Tensor scalar_field(
    std::string name,
    const std::vector<double>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(double));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return {
        std::move(name),
        foamnordic::fjord::Element::float64,
        {static_cast<std::uint64_t>(values.size())},
        std::move(bytes),
        4,
        0.2,
    };
}

std::vector<double> values_of(const foamnordic::fjord::Tensor& tensor) {
    std::vector<double> values(tensor.bytes.size() / sizeof(double));
    std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
    return values;
}

template<class Value>
void append_value(std::vector<std::byte>& bytes, Value value) {
    const auto offset = bytes.size();
    bytes.resize(offset + sizeof(Value));
    std::memcpy(bytes.data() + offset, &value, sizeof(Value));
}

void write_bytes(
    const std::filesystem::path& path,
    const std::vector<std::byte>& bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(stream.good(), "Could not create a temporary table payload.");
    stream.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    require(stream.good(), "Could not write a temporary table payload.");
}

foamnordic::fjord::Harbor connect_rank(
    const foamnordic::fjord::FjordAddress& address,
    std::uint32_t rank,
    std::uint32_t peers) {
    foamnordic::fjord::Harbor client(foamnordic::fjord::connect(address));
    const auto session = client.connect_session({
        4096,
        foamnordic::fjord::Capability::uds
            | foamnordic::fjord::Capability::shm,
        rank,
        peers,
        4096,
    });
    if (foamnordic::fjord::any(
            session.capabilities & foamnordic::fjord::Capability::shm)) {
        client.accept_shared_memory();
    }
    return client;
}

void verify_affine_exchange(foamnordic::fjord::Harbor& client) {
    client.send(scalar_field("a", {1.0, 3.0}).view());
    client.send(scalar_field("b", {4.0, 5.0}).view());
    client.complete(4, 2);

    auto message = client.receive_message();
    require(
        message.kind == foamnordic::fjord::RuneKind::tensor
            && message.tensor.has_value(),
        "Resident worker did not return a prediction tensor.");
    const auto values = values_of(*message.tensor);
    require(
        values.size() == 2
            && std::abs(values[0] + 1.5) < 1.0e-12
            && std::abs(values[1] - 1.5) < 1.0e-12,
        "Resident worker returned incorrect affine predictions.");
    std::uint64_t completed_exchange = 0;
    require(
        client.receive_control(&completed_exchange)
                == foamnordic::fjord::RuneKind::complete
            && completed_exchange == 4,
        "Resident worker returned an incorrect atomic completion.");
}

void test_resident_worker_lifecycle() {
    const auto socket_path = std::filesystem::temp_directory_path()
                             / ("foamnordic-worker-"
                                + std::to_string(std::random_device{}()) + ".sock");
    const foamnordic::inference::ProgramContract contract{
        "resident-affine",
        {
            {"a", foamnordic::fjord::Element::float64, 1},
            {"b", foamnordic::fjord::Element::float64, 1},
        },
        {{"result", foamnordic::fjord::Element::float64, 1}},
    };
    foamnordic::inference::DenseAffineKernel affine(2, 1, {2.0, -1.0}, {0.5});
    foamnordic::inference::ArtifactModelKernel kernel(
        {
            foamnordic::inference::ModelFormat::onnx,
            "reference.onnx",
            contract,
            {},
            std::nullopt,
            std::nullopt,
        },
        affine);
    foamnordic::inference::EvaluateAllCells bypass;
    std::exception_ptr worker_failure;

    {
        foamnordic::inference::ModelWorker worker(
            foamnordic::fjord::FjordAddress::local(socket_path.string()),
            {
                foamnordic::inference::ModelFormat::onnx,
                "reference.onnx",
                contract,
                {},
                std::nullopt,
                std::nullopt,
            },
            bypass,
            kernel);
        std::thread worker_thread([&] {
            try {
                worker.run();
            } catch (...) {
                worker_failure = std::current_exception();
            }
        });

        foamnordic::fjord::Harbor client(
            foamnordic::fjord::connect(worker.address()));
        const auto session = client.connect_session({
            2048,
            foamnordic::fjord::Capability::uds
                | foamnordic::fjord::Capability::shm,
            0,
            1,
            4096,
        });
        if (foamnordic::fjord::any(
                session.capabilities & foamnordic::fjord::Capability::shm)) {
            client.accept_shared_memory();
        }
        client.send(scalar_field("a", {1.0, 3.0}).view());
        client.send(scalar_field("b", {4.0, 5.0}).view());
        client.complete(4, 2);

        auto message = client.receive_message();
        require(
            message.kind == foamnordic::fjord::RuneKind::tensor
                && message.tensor.has_value(),
            "Resident worker did not return a prediction tensor.");
        const auto values = values_of(*message.tensor);
        require(
            values.size() == 2
                && std::abs(values[0] + 1.5) < 1.0e-12
                && std::abs(values[1] - 1.5) < 1.0e-12,
            "Resident worker returned incorrect affine predictions.");
        std::uint64_t completed_exchange = 0;
        require(
            client.receive_control(&completed_exchange)
                    == foamnordic::fjord::RuneKind::complete
                && completed_exchange == 4,
            "Resident worker returned an incorrect atomic completion.");
        client.shutdown();
        worker_thread.join();
        if (worker_failure) {
            std::rethrow_exception(worker_failure);
        }
    }
    require(
        !std::filesystem::exists(socket_path),
        "Resident worker left its Unix socket behind.");
}

void test_table_bundle_runs_through_resident_worker() {
    const auto root = std::filesystem::temp_directory_path();
    const auto nonce = std::to_string(std::random_device{}());
    const auto socket_path = root / ("foamnordic-table-worker-" + nonce + ".sock");
    const auto payload_path = root / ("foamnordic-table-worker-" + nonce + ".table");
    const auto bundle_path = root / ("foamnordic-table-worker-" + nonce + ".fnom");

    std::vector<std::byte> payload;
    for (const char character : std::string("FNOTAB1\0", 8)) {
        append_value(payload, character);
    }
    append_value(payload, std::uint8_t{2});
    append_value(payload, std::uint8_t{1});
    append_value(payload, std::uint16_t{0});
    append_value(payload, std::uint64_t{1});
    append_value(payload, std::uint8_t{0});
    append_value(payload, std::uint64_t{2});
    append_value(payload, 0.0);
    append_value(payload, 1.0);
    append_value(payload, 0.0);
    append_value(payload, 2.0);
    write_bytes(payload_path, payload);

    foamnordic::inference::write_bundle(
        bundle_path,
        {
            foamnordic::inference::ModelFormat::table,
            payload_path.filename().string(),
            {
                "table-worker",
                {{"x", foamnordic::fjord::Element::float64, 1}},
                {{"result", foamnordic::fjord::Element::float64, 1}},
            },
            {},
            std::nullopt,
            std::nullopt,
            "table",
        },
        payload_path);

    foamnordic::inference::EvaluateAllCells bypass;
    std::exception_ptr worker_failure;
    {
        foamnordic::inference::ModelWorker worker(
            foamnordic::fjord::FjordAddress::local(socket_path.string()),
            bundle_path,
            bypass);
        std::thread worker_thread([&] {
            try {
                worker.run();
            } catch (...) {
                worker_failure = std::current_exception();
            }
        });

        foamnordic::fjord::Harbor client(
            foamnordic::fjord::connect(worker.address()));
        const auto session = client.connect_session({
            4096,
            foamnordic::fjord::Capability::uds
                | foamnordic::fjord::Capability::shm,
            0,
            1,
            4096,
        });
        if (foamnordic::fjord::any(
                session.capabilities & foamnordic::fjord::Capability::shm)) {
            client.accept_shared_memory();
        }
        client.send(scalar_field("x", {0.25, 0.75}).view());
        client.complete(4, 1);

        const auto message = client.receive_message();
        require(
            message.kind == foamnordic::fjord::RuneKind::tensor
                && message.tensor.has_value(),
            "Table worker did not return a prediction tensor.");
        const auto actual = values_of(*message.tensor);
        require(
            actual.size() == 2
                && std::abs(actual[0] - 0.5) < 1.0e-12
                && std::abs(actual[1] - 1.5) < 1.0e-12,
            "Table worker returned incorrect interpolated values.");
        std::uint64_t completed_exchange = 0;
        require(
            client.receive_control(&completed_exchange)
                    == foamnordic::fjord::RuneKind::complete
                && completed_exchange == 4,
            "Table worker returned an incorrect atomic completion.");
        client.shutdown();
        worker_thread.join();
        if (worker_failure) {
            std::rethrow_exception(worker_failure);
        }
    }

    std::error_code ignored;
    std::filesystem::remove(payload_path, ignored);
    std::filesystem::remove(bundle_path, ignored);
    require(
        !std::filesystem::exists(socket_path),
        "Table worker left its Unix socket behind.");
}

void test_one_resident_host_serves_multiple_solver_ranks() {
    const auto socket_path = std::filesystem::temp_directory_path()
                             / ("foamnordic-worker-multi-"
                                + std::to_string(std::random_device{}()) + ".sock");
    const foamnordic::inference::ProgramContract contract{
        "resident-multi-rank",
        {
            {"a", foamnordic::fjord::Element::float64, 1},
            {"b", foamnordic::fjord::Element::float64, 1},
        },
        {{"result", foamnordic::fjord::Element::float64, 1}},
    };
    foamnordic::inference::DenseAffineKernel affine(2, 1, {2.0, -1.0}, {0.5});
    foamnordic::inference::ArtifactModelKernel kernel(
        {
            foamnordic::inference::ModelFormat::onnx,
            "reference.onnx",
            contract,
            {},
            std::nullopt,
            std::nullopt,
        },
        affine);
    foamnordic::inference::EvaluateAllCells bypass;
    std::exception_ptr worker_failure;

    {
        foamnordic::inference::WorkerOptions options;
        options.connections = 2;
        options.maximum_receive_bytes = 32;
        foamnordic::inference::ModelWorker worker(
            foamnordic::fjord::FjordAddress::local(socket_path.string()),
            {
                foamnordic::inference::ModelFormat::onnx,
                "reference.onnx",
                contract,
                {},
                std::nullopt,
                std::nullopt,
            },
            bypass,
            kernel,
            options);
        std::thread worker_thread([&] {
            try {
                worker.run();
            } catch (...) {
                worker_failure = std::current_exception();
            }
        });

        auto first = connect_rank(worker.address(), 0, 2);
        auto second = connect_rank(worker.address(), 1, 2);
        std::exception_ptr first_failure, second_failure;
        std::thread first_request([&] {
            try { verify_affine_exchange(first); }
            catch (...) { first_failure = std::current_exception(); }
        });
        std::thread second_request([&] {
            try { verify_affine_exchange(second); }
            catch (...) { second_failure = std::current_exception(); }
        });
        first_request.join();
        second_request.join();
        first.shutdown();
        second.shutdown();
        worker_thread.join();
        if (first_failure) { std::rethrow_exception(first_failure); }
        if (second_failure) { std::rethrow_exception(second_failure); }
        if (worker_failure) {
            std::rethrow_exception(worker_failure);
        }
    }
    require(
        !std::filesystem::exists(socket_path),
        "Multi-rank resident host left its Unix socket behind.");
}

void test_worker_rejects_an_empty_model_cpu_budget() {
    foamnordic::inference::WorkerOptions options;
    options.model_threads = 0;
    bool rejected = false;
    try {
        options.validate();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Resident worker accepted zero model threads.");

    options.model_threads = 1;
    options.maximum_working_bytes = 0;
    rejected = false;
    try {
        options.validate();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Resident worker accepted a zero working-memory budget.");

    options.maximum_working_bytes = 1;
    options.maximum_receive_bytes = 0;
    rejected = false;
    try {
        options.validate();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Resident worker accepted a zero receive-memory budget.");
}

#ifdef FOAMNORDIC_HAVE_UCX
void test_resident_worker_upgrades_tcp_control_to_ucx() {
    const foamnordic::inference::ProgramContract contract{
        "resident-ucx",
        {
            {"a", foamnordic::fjord::Element::float64, 1},
            {"b", foamnordic::fjord::Element::float64, 1},
        },
        {{"result", foamnordic::fjord::Element::float64, 1}},
    };
    foamnordic::inference::DenseAffineKernel affine(2, 1, {2.0, -1.0}, {0.5});
    foamnordic::inference::ArtifactModelKernel kernel(
        {
            foamnordic::inference::ModelFormat::onnx,
            "reference.onnx",
            contract,
            {},
            std::nullopt,
            std::nullopt,
        },
        affine);
    foamnordic::inference::EvaluateAllCells bypass;
    foamnordic::inference::WorkerOptions options;
    options.shared_memory = false;
    options.ucx = true;
    options.ucx_host = "127.0.0.1";
    std::exception_ptr worker_failure;

    auto port_reservation =
        foamnordic::fjord::FjordListener::network("127.0.0.1", 0);
    const auto worker_address = port_reservation.address();
    port_reservation.close();

    foamnordic::inference::ModelWorker worker(
        worker_address,
        {
            foamnordic::inference::ModelFormat::onnx,
            "reference.onnx",
            contract,
            {},
            std::nullopt,
            std::nullopt,
        },
        bypass,
        kernel,
        options);
    std::thread worker_thread([&] {
        try {
            worker.run();
        } catch (...) {
            worker_failure = std::current_exception();
        }
    });

    foamnordic::fjord::Harbor client(
        foamnordic::fjord::connect(worker.address()));
    const auto selected = client.connect_session({
        4096,
        foamnordic::fjord::Capability::tcp
            | foamnordic::fjord::Capability::ucx,
        0,
        1,
        4096,
    });
    require(
        foamnordic::fjord::any(
            selected.capabilities & foamnordic::fjord::Capability::ucx),
        "Resident worker did not select UCX.");
    client.accept_ucx();
    verify_affine_exchange(client);
    client.shutdown();
    worker_thread.join();
    if (worker_failure) {
        std::rethrow_exception(worker_failure);
    }
}
#endif

}  // namespace

int main() {
    test_resident_worker_lifecycle();
    test_table_bundle_runs_through_resident_worker();
    test_one_resident_host_serves_multiple_solver_ranks();
    test_worker_rejects_an_empty_model_cpu_budget();
#ifdef FOAMNORDIC_HAVE_UCX
    test_resident_worker_upgrades_tcp_control_to_ucx();
#endif
}

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

#include "foamnordic/fjord/ucx_channel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>

#include <ucp/api/ucp.h>

namespace foamnordic::fjord {
namespace {

[[noreturn]] void ucx_failure(const char* operation, ucs_status_t status) {
    throw std::runtime_error(
        std::string(operation) + ": " + ucs_status_string(status));
}

void require_ucx(const char* operation, ucs_status_t status) {
    if (status != UCS_OK) {
        ucx_failure(operation, status);
    }
}

struct ResolvedAddress {
    sockaddr_storage storage{};
    socklen_t length{0};
};

ResolvedAddress resolve_address(
    const std::string& host,
    std::uint16_t port,
    bool passive) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = passive ? AI_PASSIVE : 0;
    addrinfo* addresses = nullptr;
    const auto service = std::to_string(port);
    const auto status = ::getaddrinfo(
        host.empty() ? nullptr : host.c_str(),
        service.c_str(),
        &hints,
        &addresses);
    if (status != 0) {
        throw std::runtime_error(
            std::string("Cannot resolve Fjord UCX address: ")
            + ::gai_strerror(status));
    }

    ResolvedAddress result;
    for (auto* candidate = addresses; candidate != nullptr;
         candidate = candidate->ai_next) {
        if (candidate->ai_addrlen <= sizeof(result.storage)) {
            std::memcpy(
                &result.storage, candidate->ai_addr, candidate->ai_addrlen);
            result.length = static_cast<socklen_t>(candidate->ai_addrlen);
            break;
        }
    }
    ::freeaddrinfo(addresses);
    if (result.length == 0) {
        throw std::runtime_error("Fjord UCX address has no usable socket form.");
    }
    return result;
}

std::uint16_t address_port(const sockaddr_storage& address) {
    if (address.ss_family == AF_INET) {
        return ntohs(reinterpret_cast<const sockaddr_in*>(&address)->sin_port);
    }
    if (address.ss_family == AF_INET6) {
        return ntohs(reinterpret_cast<const sockaddr_in6*>(&address)->sin6_port);
    }
    throw std::runtime_error("UCX listener returned an unsupported address family.");
}

std::string numeric_host(const sockaddr_storage& address) {
    char host[NI_MAXHOST]{};
    const auto length = address.ss_family == AF_INET
                            ? sizeof(sockaddr_in)
                            : sizeof(sockaddr_in6);
    const auto status = ::getnameinfo(
        reinterpret_cast<const sockaddr*>(&address),
        static_cast<socklen_t>(length),
        host,
        sizeof(host),
        nullptr,
        0,
        NI_NUMERICHOST);
    if (status != 0) {
        throw std::runtime_error(
            std::string("Cannot inspect Fjord UCX listener: ")
            + ::gai_strerror(status));
    }
    return host;
}

struct UcxRuntime {
    ucp_context_h context{nullptr};
    ucp_worker_h worker{nullptr};

    UcxRuntime() {
        ucp_config_t* config = nullptr;
        require_ucx(
            "Cannot read UCX configuration",
            ucp_config_read(nullptr, nullptr, &config));

        ucp_params_t context_parameters{};
        context_parameters.field_mask =
            UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_NAME;
        context_parameters.features = UCP_FEATURE_STREAM;
        context_parameters.name = "FoamNordic Fjord";
        const auto context_status =
            ucp_init(&context_parameters, config, &context);
        ucp_config_release(config);
        require_ucx("Cannot initialize FoamNordic UCX context", context_status);

        ucp_worker_params_t worker_parameters{};
        worker_parameters.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        worker_parameters.thread_mode = UCS_THREAD_MODE_SINGLE;
        const auto worker_status =
            ucp_worker_create(context, &worker_parameters, &worker);
        if (worker_status != UCS_OK) {
            ucp_cleanup(context);
            context = nullptr;
            ucx_failure("Cannot create FoamNordic UCX worker", worker_status);
        }
    }

    ~UcxRuntime() {
        if (worker != nullptr) {
            ucp_worker_destroy(worker);
        }
        if (context != nullptr) {
            ucp_cleanup(context);
        }
    }
};

struct RequestState {
    bool complete{false};
    ucs_status_t status{UCS_INPROGRESS};
    std::size_t length{0};
};

void send_complete(
    void*,
    ucs_status_t status,
    void* user_data) {
    auto& state = *static_cast<RequestState*>(user_data);
    state.status = status;
    state.complete = true;
}

void receive_complete(
    void*,
    ucs_status_t status,
    std::size_t length,
    void* user_data) {
    auto& state = *static_cast<RequestState*>(user_data);
    state.status = status;
    state.length = length;
    state.complete = true;
}

void progress_briefly(ucp_worker_h worker) {
    if (ucp_worker_progress(worker) == 0) {
        std::this_thread::yield();
    }
}

}  // namespace

struct UcxChannel::Impl {
    std::shared_ptr<UcxRuntime> runtime;
    ucp_ep_h endpoint{nullptr};
    std::atomic<bool> interrupted{false};
    std::atomic<ucs_status_t> peer_status{UCS_OK};

    explicit Impl(std::shared_ptr<UcxRuntime> selected_runtime)
        : runtime(std::move(selected_runtime)) {}

    static void peer_error(void* argument, ucp_ep_h, ucs_status_t status) {
        auto& implementation = *static_cast<Impl*>(argument);
        implementation.peer_status.store(status, std::memory_order_release);
    }

    void check_active() const {
        if (endpoint == nullptr
            || interrupted.load(std::memory_order_acquire)) {
            throw std::runtime_error("Fjord UCX channel is closed.");
        }
        const auto status = peer_status.load(std::memory_order_acquire);
        if (status != UCS_OK) {
            ucx_failure("Fjord UCX peer failed", status);
        }
    }

    void wait_request(void* request, RequestState& state, const char* operation) {
        if (request == nullptr) {
            return;
        }
        if (UCS_PTR_IS_ERR(request)) {
            ucx_failure(operation, UCS_PTR_STATUS(request));
        }

        bool canceled = false;
        while (!state.complete) {
            if (interrupted.load(std::memory_order_acquire) && !canceled) {
                ucp_request_cancel(runtime->worker, request);
                canceled = true;
            }
            progress_briefly(runtime->worker);
        }
        ucp_request_free(request);
        if (state.status != UCS_OK) {
            ucx_failure(operation, state.status);
        }
        check_active();
    }

    void close() noexcept {
        interrupted.store(true, std::memory_order_release);
        if (endpoint == nullptr) {
            return;
        }

        ucp_request_param_t parameters{};
        // Fjord operations complete synchronously before close. Force release
        // avoids making destruction depend on the peer continuing to progress;
        // both endpoints use UCP_ERR_HANDLING_MODE_PEER as UCX requires.
        parameters.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        parameters.flags = UCP_EP_CLOSE_FLAG_FORCE;
        auto* request = ucp_ep_close_nbx(endpoint, &parameters);
        endpoint = nullptr;
        if (request == nullptr || UCS_PTR_IS_ERR(request)) {
            return;
        }
        while (ucp_request_check_status(request) == UCS_INPROGRESS) {
            static_cast<void>(ucp_worker_progress(runtime->worker));
        }
        ucp_request_free(request);
    }
};

struct UcxListener::Impl {
    std::shared_ptr<UcxRuntime> runtime{std::make_shared<UcxRuntime>()};
    ucp_listener_h listener{nullptr};
    ucp_conn_request_h connection{nullptr};
    FjordAddress public_address;

    explicit Impl(FjordAddress address) : public_address(std::move(address)) {}

    ~Impl() { close(); }

    static void connection_request(
        ucp_conn_request_h request,
        void* argument) {
        auto& implementation = *static_cast<Impl*>(argument);
        if (implementation.connection == nullptr) {
            implementation.connection = request;
        } else {
            static_cast<void>(ucp_listener_reject(
                implementation.listener, request));
        }
    }

    void close() noexcept {
        if (listener != nullptr) {
            if (connection != nullptr) {
                static_cast<void>(ucp_listener_reject(listener, connection));
                connection = nullptr;
            }
            ucp_listener_destroy(listener);
            listener = nullptr;
        }
    }
};

UcxChannel::UcxChannel(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

UcxChannel::~UcxChannel() { close(); }

void UcxChannel::write_all(std::span<const std::byte> bytes) {
    implementation_->check_active();
    if (bytes.empty()) {
        return;
    }

    RequestState state;
    ucp_request_param_t parameters{};
    parameters.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
                              | UCP_OP_ATTR_FIELD_DATATYPE
                              | UCP_OP_ATTR_FIELD_USER_DATA;
    parameters.datatype = ucp_dt_make_contig(1);
    parameters.cb.send = send_complete;
    parameters.user_data = &state;
    auto* request = ucp_stream_send_nbx(
        implementation_->endpoint,
        bytes.data(),
        bytes.size(),
        &parameters);
    implementation_->wait_request(request, state, "Fjord UCX write failed");
}

void UcxChannel::read_all(std::span<std::byte> bytes) {
    implementation_->check_active();
    if (bytes.empty()) {
        return;
    }

    RequestState state;
    std::size_t immediate_length = 0;
    ucp_request_param_t parameters{};
    parameters.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
                              | UCP_OP_ATTR_FIELD_DATATYPE
                              | UCP_OP_ATTR_FIELD_FLAGS
                              | UCP_OP_ATTR_FIELD_USER_DATA;
    parameters.datatype = ucp_dt_make_contig(1);
    parameters.flags = UCP_STREAM_RECV_FLAG_WAITALL;
    parameters.cb.recv_stream = receive_complete;
    parameters.user_data = &state;
    auto* request = ucp_stream_recv_nbx(
        implementation_->endpoint,
        bytes.data(),
        bytes.size(),
        &immediate_length,
        &parameters);
    if (request == nullptr) {
        state.length = immediate_length;
    } else {
        implementation_->wait_request(
            request, state, "Fjord UCX read failed");
    }
    if (state.length != bytes.size()) {
        throw std::runtime_error("Fjord UCX read returned a partial stream.");
    }
}

bool UcxChannel::wait_readable(std::chrono::milliseconds timeout) {
    if (timeout.count() < 0) {
        throw std::invalid_argument("Fjord readiness timeout must not be negative.");
    }
    implementation_->check_active();
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        ucp_stream_poll_ep_t ready{};
        const auto count = ucp_stream_worker_poll(
            implementation_->runtime->worker, &ready, 1, 0);
        if (count < 0) {
            ucx_failure(
                "Fjord UCX readiness wait failed",
                static_cast<ucs_status_t>(count));
        }
        if (count > 0 && ready.ep == implementation_->endpoint) {
            return true;
        }
        implementation_->check_active();
        progress_briefly(implementation_->runtime->worker);
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

void UcxChannel::interrupt() noexcept {
    implementation_->interrupted.store(true, std::memory_order_release);
}

void UcxChannel::close() noexcept {
    if (implementation_) {
        implementation_->close();
    }
}

UcxListener::UcxListener(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

UcxListener UcxListener::network(
    const std::string& host,
    std::uint16_t port) {
    const auto native = resolve_address(host, port, true);
    auto implementation = std::make_unique<Impl>(FjordAddress{
        FjordKind::ucx, host, port});

    ucp_listener_params_t parameters{};
    parameters.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR
                            | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    parameters.sockaddr.addr =
        reinterpret_cast<const sockaddr*>(&native.storage);
    parameters.sockaddr.addrlen = native.length;
    parameters.conn_handler.cb = Impl::connection_request;
    parameters.conn_handler.arg = implementation.get();
    require_ucx(
        "Cannot create Fjord UCX listener",
        ucp_listener_create(
            implementation->runtime->worker,
            &parameters,
            &implementation->listener));

    ucp_listener_attr_t attributes{};
    attributes.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
    require_ucx(
        "Cannot inspect Fjord UCX listener",
        ucp_listener_query(implementation->listener, &attributes));
    const auto public_host = host.empty()
                                 ? numeric_host(attributes.sockaddr)
                                 : host;
    implementation->public_address = FjordAddress::ucx(
        public_host, address_port(attributes.sockaddr));
    return UcxListener(std::move(implementation));
}

UcxListener::~UcxListener() { close(); }

UcxListener::UcxListener(UcxListener&&) noexcept = default;

UcxListener& UcxListener::operator=(UcxListener&&) noexcept = default;

std::unique_ptr<FjordChannel> UcxListener::accept(
    std::chrono::milliseconds timeout) {
    if (timeout.count() <= 0) {
        throw std::invalid_argument("Fjord UCX accept timeout must be positive.");
    }
    if (!implementation_ || implementation_->listener == nullptr) {
        throw std::runtime_error("Fjord UCX listener is closed.");
    }
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (implementation_->connection == nullptr) {
        progress_briefly(implementation_->runtime->worker);
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("Fjord UCX connection timed out.");
        }
    }

    auto channel = std::make_unique<UcxChannel::Impl>(
        implementation_->runtime);
    ucp_ep_params_t parameters{};
    parameters.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST
                            | UCP_EP_PARAM_FIELD_ERR_HANDLER
                            | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    parameters.conn_request =
        std::exchange(implementation_->connection, nullptr);
    parameters.err_handler.cb = UcxChannel::Impl::peer_error;
    parameters.err_handler.arg = channel.get();
    parameters.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    require_ucx(
        "Cannot accept Fjord UCX endpoint",
        ucp_ep_create(
            implementation_->runtime->worker,
            &parameters,
            &channel->endpoint));
    return std::unique_ptr<FjordChannel>(
        new UcxChannel(std::move(channel)));
}

FjordAddress UcxListener::address() const {
    if (!implementation_ || implementation_->listener == nullptr) {
        throw std::runtime_error("Fjord UCX listener is closed.");
    }
    return implementation_->public_address;
}

void UcxListener::close() noexcept {
    if (implementation_) {
        implementation_->close();
    }
}

std::unique_ptr<FjordChannel> connect_ucx(const FjordAddress& address) {
    address.validate();
    if (address.kind != FjordKind::ucx) {
        throw std::invalid_argument("Fjord UCX connect requires an ucx:// address.");
    }
    const auto native = resolve_address(address.location, address.port, false);
    auto channel = std::make_unique<UcxChannel::Impl>(
        std::make_shared<UcxRuntime>());

    ucp_ep_params_t parameters{};
    parameters.field_mask = UCP_EP_PARAM_FIELD_FLAGS
                            | UCP_EP_PARAM_FIELD_SOCK_ADDR
                            | UCP_EP_PARAM_FIELD_ERR_HANDLER
                            | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    parameters.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    parameters.sockaddr.addr =
        reinterpret_cast<const sockaddr*>(&native.storage);
    parameters.sockaddr.addrlen = native.length;
    parameters.err_handler.cb = UcxChannel::Impl::peer_error;
    parameters.err_handler.arg = channel.get();
    parameters.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    require_ucx(
        "Cannot connect Fjord UCX endpoint",
        ucp_ep_create(
            channel->runtime->worker,
            &parameters,
            &channel->endpoint));
    return std::unique_ptr<FjordChannel>(
        new UcxChannel(std::move(channel)));
}

}  // namespace foamnordic::fjord

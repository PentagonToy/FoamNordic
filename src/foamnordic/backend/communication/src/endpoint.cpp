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

#include "foamnordic/fjord/endpoint.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace foamnordic::fjord {
namespace {

[[noreturn]] void system_failure(const std::string& operation) {
    throw std::runtime_error(operation + ": " + std::strerror(errno));
}

void close_descriptor(int descriptor) noexcept {
    if (descriptor >= 0) {
        ::close(descriptor);
    }
}

void configure_tcp(int descriptor) {
    const int enabled = 1;
    if (::setsockopt(descriptor, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) != 0) {
        system_failure("Cannot enable TCP_NODELAY for Fjord");
    }
}

sockaddr_un unix_address(const std::string& path) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (path.empty() || path.size() >= sizeof(address.sun_path)) {
        throw std::invalid_argument("Fjord Unix socket path is empty or too long.");
    }
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    return address;
}

void remove_stale_socket(const std::string& path) {
    struct stat status {};
    if (::lstat(path.c_str(), &status) != 0) {
        if (errno == ENOENT) {
            return;
        }
        system_failure("Cannot inspect Fjord Unix socket path");
    }
    if (!S_ISSOCK(status.st_mode)) {
        throw std::runtime_error(
            "Fjord refuses to replace a non-socket path: " + path);
    }
    if (::unlink(path.c_str()) != 0) {
        system_failure("Cannot remove stale Fjord Unix socket");
    }
}

std::uint16_t socket_port(int descriptor) {
    sockaddr_storage storage{};
    socklen_t length = sizeof(storage);
    if (::getsockname(descriptor, reinterpret_cast<sockaddr*>(&storage), &length) != 0) {
        system_failure("Cannot inspect Fjord listener address");
    }
    if (storage.ss_family == AF_INET) {
        return ntohs(reinterpret_cast<const sockaddr_in*>(&storage)->sin_port);
    }
    if (storage.ss_family == AF_INET6) {
        return ntohs(reinterpret_cast<const sockaddr_in6*>(&storage)->sin6_port);
    }
    return 0;
}

}  // namespace

FjordAddress FjordAddress::local(std::string path) {
    FjordAddress address{FjordKind::unix_socket, std::move(path), 0};
    address.validate();
    return address;
}

FjordAddress FjordAddress::network(std::string host, std::uint16_t port) {
    FjordAddress address{FjordKind::tcp, std::move(host), port};
    address.validate();
    return address;
}

FjordAddress FjordAddress::ucx(std::string host, std::uint16_t port) {
    FjordAddress address{FjordKind::ucx, std::move(host), port};
    address.validate();
    return address;
}

FjordAddress FjordAddress::parse(const std::string& text) {
    constexpr std::string_view unix_prefix = "unix://";
    constexpr std::string_view tcp_prefix = "tcp://";
    constexpr std::string_view ucx_prefix = "ucx://";
    if (text.starts_with(unix_prefix)) {
        return local(text.substr(unix_prefix.size()));
    }
    const auto is_tcp = text.starts_with(tcp_prefix);
    const auto is_ucx = text.starts_with(ucx_prefix);
    if (!is_tcp && !is_ucx) {
        throw std::invalid_argument(
            "Fjord address must begin with unix://, tcp://, or ucx://.");
    }
    const auto prefix_size = is_tcp ? tcp_prefix.size() : ucx_prefix.size();
    const auto endpoint = text.substr(prefix_size);
    const auto separator = endpoint.rfind(':');
    if (separator == std::string::npos || separator == 0
        || separator + 1 == endpoint.size()) {
        throw std::invalid_argument("Fjord TCP address must contain host and port.");
    }
    const auto port_text = endpoint.substr(separator + 1);
    std::size_t consumed = 0;
    const auto parsed = std::stoul(port_text, &consumed);
    if (consumed != port_text.size() || parsed > 65535) {
        throw std::invalid_argument("Fjord network port is invalid.");
    }
    const auto host = endpoint.substr(0, separator);
    return is_tcp ? network(host, static_cast<std::uint16_t>(parsed))
                  : ucx(host, static_cast<std::uint16_t>(parsed));
}

std::string FjordAddress::text() const {
    validate();
    if (kind == FjordKind::unix_socket) {
        return "unix://" + location;
    }
    const auto scheme = kind == FjordKind::tcp ? "tcp://" : "ucx://";
    return scheme + location + ":" + std::to_string(port);
}

void FjordAddress::validate() const {
    if (location.empty()) {
        throw std::invalid_argument("Fjord address location must not be empty.");
    }
    if (kind == FjordKind::unix_socket && port != 0) {
        throw std::invalid_argument("A Fjord Unix address must not specify a port.");
    }
    if (kind != FjordKind::unix_socket && port == 0) {
        throw std::invalid_argument("A Fjord network address requires a non-zero port.");
    }
}

FjordListener::FjordListener(int descriptor, FjordAddress address, bool owns_path)
    : descriptor_(descriptor), address_(std::move(address)), owns_path_(owns_path) {}

FjordListener FjordListener::local(const std::string& path, int backlog) {
    if (backlog < 1) {
        throw std::invalid_argument("Fjord listener backlog must be positive.");
    }
    const auto native = unix_address(path);
    const int descriptor = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (descriptor < 0) {
        system_failure("Cannot create Fjord Unix listener");
    }
    remove_stale_socket(path);
    if (::bind(descriptor, reinterpret_cast<const sockaddr*>(&native), sizeof(native)) != 0) {
        close_descriptor(descriptor);
        system_failure("Cannot bind Fjord Unix listener");
    }
    if (::listen(descriptor, backlog) != 0) {
        close_descriptor(descriptor);
        ::unlink(path.c_str());
        system_failure("Cannot listen on Fjord Unix socket");
    }
    return FjordListener(descriptor, FjordAddress::local(path), true);
}

FjordListener FjordListener::network(
    const std::string& host,
    std::uint16_t port,
    int backlog) {
    if (backlog < 1) {
        throw std::invalid_argument("Fjord listener backlog must be positive.");
    }
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* addresses = nullptr;
    const auto service = std::to_string(port);
    const int result = ::getaddrinfo(host.empty() ? nullptr : host.c_str(), service.c_str(), &hints, &addresses);
    if (result != 0) {
        throw std::runtime_error(std::string("Cannot resolve Fjord listener: ") + gai_strerror(result));
    }

    int descriptor = -1;
    for (auto* candidate = addresses; candidate != nullptr; candidate = candidate->ai_next) {
        descriptor = ::socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
        if (descriptor < 0) {
            continue;
        }
        const int enabled = 1;
        ::setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
        if (::bind(descriptor, candidate->ai_addr, candidate->ai_addrlen) == 0
            && ::listen(descriptor, backlog) == 0) {
            break;
        }
        close_descriptor(descriptor);
        descriptor = -1;
    }
    ::freeaddrinfo(addresses);
    if (descriptor < 0) {
        system_failure("Cannot bind Fjord TCP listener");
    }
    const auto actual_port = socket_port(descriptor);
    const auto public_host = host.empty() ? std::string("0.0.0.0") : host;
    return FjordListener(
        descriptor, FjordAddress::network(public_host, actual_port), false);
}

FjordListener::~FjordListener() { close(); }

FjordListener::FjordListener(FjordListener&& other) noexcept
    : descriptor_(std::exchange(other.descriptor_, -1)),
      address_(std::move(other.address_)),
      owns_path_(std::exchange(other.owns_path_, false)) {}

FjordListener& FjordListener::operator=(FjordListener&& other) noexcept {
    if (this != &other) {
        close();
        descriptor_ = std::exchange(other.descriptor_, -1);
        address_ = std::move(other.address_);
        owns_path_ = std::exchange(other.owns_path_, false);
    }
    return *this;
}

std::unique_ptr<FjordChannel> FjordListener::accept() {
    int descriptor = -1;
    do {
        descriptor = ::accept(descriptor_, nullptr, nullptr);
    } while (descriptor < 0 && errno == EINTR);
    if (descriptor < 0) {
        system_failure("Cannot accept Fjord connection");
    }
    if (address_.kind == FjordKind::tcp) {
        configure_tcp(descriptor);
    }
    return std::make_unique<SocketChannel>(descriptor);
}

FjordAddress FjordListener::address() const { return address_; }

void FjordListener::close() noexcept {
    close_descriptor(descriptor_);
    descriptor_ = -1;
    if (owns_path_ && !address_.location.empty()) {
        ::unlink(address_.location.c_str());
        owns_path_ = false;
    }
}

std::unique_ptr<FjordChannel> connect(const FjordAddress& address) {
    address.validate();
    if (address.kind == FjordKind::unix_socket) {
        const auto native = unix_address(address.location);
        const int descriptor = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (descriptor < 0) {
            system_failure("Cannot create Fjord Unix client");
        }
        if (::connect(descriptor, reinterpret_cast<const sockaddr*>(&native), sizeof(native)) != 0) {
            close_descriptor(descriptor);
            system_failure("Cannot connect to Fjord Unix listener");
        }
        return std::make_unique<SocketChannel>(descriptor);
    }

    if (address.kind == FjordKind::ucx) {
        throw std::invalid_argument(
            "Use connect_ucx for a Fjord UCX address.");
    }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    const auto service = std::to_string(address.port);
    const int result = ::getaddrinfo(address.location.c_str(), service.c_str(), &hints, &addresses);
    if (result != 0) {
        throw std::runtime_error(std::string("Cannot resolve Fjord endpoint: ") + gai_strerror(result));
    }
    int descriptor = -1;
    for (auto* candidate = addresses; candidate != nullptr; candidate = candidate->ai_next) {
        descriptor = ::socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
        if (descriptor < 0) {
            continue;
        }
        if (::connect(descriptor, candidate->ai_addr, candidate->ai_addrlen) == 0) {
            break;
        }
        close_descriptor(descriptor);
        descriptor = -1;
    }
    ::freeaddrinfo(addresses);
    if (descriptor < 0) {
        system_failure("Cannot connect to Fjord TCP listener");
    }
    configure_tcp(descriptor);
    return std::make_unique<SocketChannel>(descriptor);
}

}  // namespace foamnordic::fjord

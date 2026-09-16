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
#include "fieldProgramSession.H"

#include "Pstream.H"

#include "foamnordic/fjord/endpoint.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace Foam
{
namespace foamNordic
{

std::uint64_t
MeshTopologySequence::observe(
    const fvMesh& mesh
)
{
    const string facesInstance(mesh.facesInstance());
    const bool changing = mesh.topoChanging();
    const bool changed = initialised_
        && (explicitlyChanged_ || facesInstance != facesInstance_
            || mesh.nCells() != cells_ || mesh.nFaces() != faces_
            || mesh.nPoints() != points_ || (changing && !wasChanging_));
    if (changed)
    {
        ++generation_;
    }
    facesInstance_ = facesInstance;
    cells_ = mesh.nCells();
    faces_ = mesh.nFaces();
    points_ = mesh.nPoints();
    wasChanging_ = changing;
    explicitlyChanged_ = false;
    initialised_ = true;
    return generation_;
}

void
MeshTopologySequence::markChanged() noexcept
{
    explicitlyChanged_ = true;
}

std::uint64_t
MeshTopologySequence::generation() const noexcept
{
    return generation_;
}

void
writeSolverTelemetry(
    const word& program,
    const foamnordic::adapter::SolverExchangeSnapshot& snapshot
)
{
    if (snapshot.samples == 0)
    {
        return;
    }
    const auto mean = [&snapshot](std::uint64_t total)
    {
        return total / snapshot.samples;
    };
    Info << "[FoamNordic] Solver telemetry: {\"program\":\"" << program
         << "\",\"rank\":" << Pstream::myProcNo()
         << ",\"samples\":" << snapshot.samples
         << ",\"mean_round_trip_nanoseconds\":"
         << mean(snapshot.round_trip_nanoseconds)
         << ",\"mean_request_publish_nanoseconds\":"
         << mean(snapshot.request_publish_nanoseconds)
         << ",\"mean_response_wait_nanoseconds\":"
         << mean(snapshot.response_wait_nanoseconds)
         << ",\"mean_field_commit_nanoseconds\":"
         << mean(snapshot.field_commit_nanoseconds) << '}' << nl;
}

void
writeNativeAlgebraTelemetry(
    const word& program,
    const std::uint64_t samples,
    const std::uint64_t evaluateAndCommitNanoseconds
)
{
    if (samples == 0)
    {
        return;
    }
    Info << "[FoamNordic] Native algebra telemetry: {\"program\":\"" << program
         << "\",\"rank\":" << Pstream::myProcNo() << ",\"samples\":" << samples
         << ",\"mean_evaluate_and_commit_nanoseconds\":"
         << evaluateAndCommitNanoseconds / samples << '}' << nl;
}

string
resolveRankAddress(
    const string& address
)
{
    std::string resolved(address.c_str());
    const std::string marker("{rank}");
    const auto position = resolved.find(marker);
    if (position != std::string::npos)
    {
        resolved.replace(
            position,
            marker.size(),
            std::to_string(Pstream::myProcNo())
        );
    }
    return resolved;
}

std::unique_ptr<foamnordic::fjord::Harbor>
connectSession(
    const string& address,
    bool sharedMemory,
    bool ucx,
    std::uint64_t sessionId
)
{
    if (address.empty() || sessionId == 0)
    {
        throw std::invalid_argument(
            "FoamNordic closure address and positive sessionId are required."
        );
    }
    const auto endpoint = foamnordic::fjord::FjordAddress::parse(
        resolveRankAddress(address).c_str()
    );
    auto harbor = std::make_unique<foamnordic::fjord::Harbor>(
        foamnordic::fjord::connect(endpoint)
    );
    auto capabilities
        = endpoint.kind == foamnordic::fjord::FjordKind::unix_socket
        ? foamnordic::fjord::Capability::uds
        : foamnordic::fjord::Capability::tcp;
    if (sharedMemory
        && endpoint.kind == foamnordic::fjord::FjordKind::unix_socket)
    {
        capabilities = capabilities | foamnordic::fjord::Capability::shm;
    }
    if (ucx)
    {
        if (endpoint.kind != foamnordic::fjord::FjordKind::tcp)
        {
            throw std::invalid_argument(
                "FoamNordic UCX mode requires a TCP control address."
            );
        }
#ifdef FOAMNORDIC_HAVE_UCX
        capabilities = capabilities | foamnordic::fjord::Capability::ucx;
#else
        throw std::runtime_error(
            "FoamNordic OpenFOAM adapter was built without UCX support."
        );
#endif
    }
    const auto selected = harbor->connect_session(
        {
            sessionId,
            capabilities,
            static_cast<std::uint32_t>(Pstream::myProcNo()),
            static_cast<std::uint32_t>(Pstream::nProcs()),
            16ULL * 1024ULL * 1024ULL * 1024ULL,
        }
    );
    if (ucx)
    {
        if (!foamnordic::fjord::any(
                selected.capabilities & foamnordic::fjord::Capability::ucx
            ))
        {
            throw std::runtime_error(
                "FoamNordic required UCX but ModelHost did not negotiate it."
            );
        }
#ifdef FOAMNORDIC_HAVE_UCX
        harbor->accept_ucx();
#endif
    }
    else if (sharedMemory
             && foamnordic::fjord::any(
                 selected.capabilities & foamnordic::fjord::Capability::shm
             ))
    {
        harbor->accept_shared_memory();
    }
    return harbor;
}

FieldProgramSession::FieldProgramSession(
    const dictionary& dict,
    foamnordic::adapter::ExchangeContract contract
)
{
    const auto address = dict.get<string>("address");
    const auto sharedMemory = dict.getOrDefault<bool>("sharedMemory", true);
    const auto ucx = dict.getOrDefault<bool>("ucx", false);
    const auto sessionId
        = static_cast<std::uint64_t>(dict.getOrDefault<label>("sessionId", 1));
    const auto configuredProfileEvery
        = dict.getOrDefault<label>("profileEvery", 0);
    if (configuredProfileEvery < 0)
    {
        throw std::invalid_argument(
            "FoamNordic profileEvery must not be negative."
        );
    }
    const auto profileEvery
        = static_cast<std::uint64_t>(configuredProfileEvery);
    harbor_ = connectSession(address, sharedMemory, ucx, sessionId);
    port_ = std::make_unique<foamnordic::adapter::FieldProgramPort>(
        *harbor_,
        std::move(contract),
        profileEvery
    );
}

FieldProgramSession::~FieldProgramSession()
{
    try
    {
        shutdown();
    }
    catch (...)
    {
    }
}

foamnordic::adapter::FieldInvocation
FieldProgramSession::begin(
    const fvMesh& mesh,
    const Time& time
)
{
    if (!port_)
    {
        throw std::logic_error("FoamNordic closure session is shut down.");
    }
    return port_->begin(
        static_cast<std::uint64_t>(time.timeIndex()),
        static_cast<double>(time.value()),
        topology_.observe(mesh)
    );
}

std::uint64_t
FieldProgramSession::topologyGeneration() const noexcept
{
    return topology_.generation();
}

foamnordic::adapter::SolverExchangeSnapshot
FieldProgramSession::profileSnapshot() const noexcept
{
    return port_ ? port_->profile_snapshot()
                 : foamnordic::adapter::SolverExchangeSnapshot {};
}

void
FieldProgramSession::shutdown()
{
    port_.reset();
    if (harbor_)
    {
        harbor_->shutdown();
        harbor_.reset();
    }
}

} // End namespace foamNordic
} // End namespace Foam

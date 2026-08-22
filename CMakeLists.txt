cmake_minimum_required(VERSION 3.20)

project(FoamNordic LANGUAGES CXX)

option(FOAMNORDIC_TESTS "Build FoamNordic native tests" ON)
option(FOAMNORDIC_BENCHMARKS "Build FoamNordic native benchmarks" OFF)
option(FOAMNORDIC_OPENFOAM_TOOLS "Build FoamNordic OpenFOAM test workers" OFF)
option(FOAMNORDIC_NETWORK_TOOLS "Build FoamNordic inter-node probes" OFF)
option(FOAMNORDIC_ONNX_RUNTIME "Build the native ONNX Runtime kernel" OFF)
option(FOAMNORDIC_RESIDENT_TOOLS "Build native resident worker executables" OFF)
set(FOAMNORDIC_ONNX_RUNTIME_ROOT "" CACHE PATH "ONNX Runtime 1.28.0 installation root")

add_subdirectory(src/foamnordic/runtime)
add_subdirectory(src/foamnordic/backend/communication)
add_subdirectory(src/foamnordic/backend/adapter)
add_subdirectory(src/foamnordic/backend/inference)

if(FOAMNORDIC_TESTS)
    enable_testing()
    add_subdirectory(tests/runtime)
    add_subdirectory(tests/backend/communication)
    add_subdirectory(tests/backend/adapter)
    add_subdirectory(tests/backend/inference)
    add_subdirectory(tests/openfoam)
endif()

if(FOAMNORDIC_BENCHMARKS)
    add_subdirectory(tools/benchmarks)
endif()

if(FOAMNORDIC_OPENFOAM_TOOLS)
    add_subdirectory(tools/openfoam)
endif()

if(FOAMNORDIC_NETWORK_TOOLS)
    add_subdirectory(tools/network)
endif()

if(FOAMNORDIC_RESIDENT_TOOLS)
    if(NOT FOAMNORDIC_ONNX_RUNTIME)
        message(FATAL_ERROR
            "FOAMNORDIC_RESIDENT_TOOLS currently requires FOAMNORDIC_ONNX_RUNTIME=ON")
    endif()
    add_subdirectory(tools/resident)
endif()

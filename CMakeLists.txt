cmake_minimum_required(VERSION 3.20)

project(FoamNordic VERSION 1.0.4 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(FOAMNORDIC_TESTS "Build FoamNordic native tests" ON)
option(FOAMNORDIC_BENCHMARKS "Build FoamNordic native benchmarks" OFF)
option(FOAMNORDIC_OPENFOAM_TOOLS "Build FoamNordic OpenFOAM test workers" OFF)
option(FOAMNORDIC_NETWORK_TOOLS "Build FoamNordic inter-node probes" OFF)
option(FOAMNORDIC_ONNX_RUNTIME "Build the native ONNX Runtime kernel" OFF)
option(FOAMNORDIC_RESIDENT_TOOLS "Build native resident worker executables" OFF)
option(FOAMNORDIC_UCX "Build the optional native UCX data plane" OFF)
option(FOAMNORDIC_PYTHON "Build the FoamNordic Python extension" OFF)
set(FOAMNORDIC_ONNX_RUNTIME_ROOT "" CACHE PATH "ONNX Runtime 1.28.0 installation root")
set(FOAMNORDIC_UCX_ROOT "" CACHE PATH "UCX installation root")

add_subdirectory(src/foamnordic/runtime)
add_subdirectory(src/foamnordic/backend/communication)
add_subdirectory(src/foamnordic/backend/adapter)
add_subdirectory(src/foamnordic/backend/inference)
add_subdirectory(tools/longship)

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

if(FOAMNORDIC_PYTHON)
    add_subdirectory(python)
endif()

# The runtime value types are the first stable C++ facade used by bindings and
# external launchers. Additional native libraries join this export only after
# their public ABI has been separated from transport and solver internals.
install(
    TARGETS foamnordic_runtime
    EXPORT FoamNordicTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Development
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    DIRECTORY src/foamnordic/runtime/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/foamnordic/runtime
    COMPONENT Development
    FILES_MATCHING PATTERN "*.hpp")
install(
    EXPORT FoamNordicTargets
    FILE FoamNordicTargets.cmake
    NAMESPACE FoamNordic::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FoamNordic
    COMPONENT Development)

configure_package_config_file(
    cmake/FoamNordicConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/FoamNordicConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FoamNordic)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/FoamNordicConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/FoamNordicConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/FoamNordicConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FoamNordic
    COMPONENT Development)

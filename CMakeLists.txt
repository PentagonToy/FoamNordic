cmake_minimum_required(VERSION 3.20)

project(FoamNordic VERSION 1.0.7 LANGUAGES CXX)

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

include(cmake/BindNordic.cmake)
foamnordic_add_bindnordic()

add_subdirectory(src/foamnordic/runtime)
add_subdirectory(src/foamnordic/backend/communication)
add_subdirectory(src/foamnordic/backend/adapter)
add_subdirectory(src/foamnordic/backend/inference)
add_subdirectory(tools/longship)

if(FOAMNORDIC_TESTS)
    enable_testing()
    add_subdirectory(packages/bindnordic/tests/native)
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

# Keep the definition of a complete runtime build inside CMake.  Callers build
# this aggregate target instead of reproducing an option-dependent target list.
add_custom_target(foamnordic_runtime_bundle
    DEPENDS
        foamnordic_adapter
        foamnordic_inference
        foamnordic-longship)
if(FOAMNORDIC_RESIDENT_TOOLS)
    add_dependencies(foamnordic_runtime_bundle foamnordic_model_worker)
endif()
if(FOAMNORDIC_ONNX_RUNTIME)
    add_dependencies(foamnordic_runtime_bundle foamnordic_onnx_connector)
endif()

if(FOAMNORDIC_PYTHON)
    add_subdirectory(packages/foamnordic)
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
    DIRECTORY src/foamnordic/backend/communication/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT Development
    FILES_MATCHING PATTERN "*.hpp")
install(
    DIRECTORY packages/bindnordic/src/bindnordic/native/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT Development
    FILES_MATCHING PATTERN "*.hpp")
install(
    DIRECTORY src/foamnordic/backend/adapter/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/foamnordic/backend/adapter
    COMPONENT Development
    FILES_MATCHING PATTERN "*.hpp")
install(
    FILES src/foamnordic/backend/connectors/plugin.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/foamnordic/backend/connectors
    COMPONENT Development)
install(
    FILES
        src/foamnordic/openfoam/closureHook.H
        src/foamnordic/openfoam/fieldBridge.H
        src/foamnordic/openfoam/fieldProgramObservation.H
        src/foamnordic/openfoam/fieldProgramSession.H
        src/foamnordic/openfoam/fjordExchange.H
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/foamnordic/openfoam
    COMPONENT Development)
install(
    DIRECTORY src/foamnordic/openfoam/operations/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/foamnordic/openfoam/operations
    COMPONENT Development
    FILES_MATCHING PATTERN "*.H")
install(
    DIRECTORY src/foamnordic/openfoam/models/progressRateFjord/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/foamnordic/openfoam/models/progressRateFjord
    COMPONENT Development
    FILES_MATCHING PATTERN "*.H")
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

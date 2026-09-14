cmake_minimum_required(VERSION 3.20)

project(FoamNordicBuildKit VERSION 1.0.8 LANGUAGES CXX)

include(GNUInstallDirs)

option(FOAMNORDIC_TESTS "Build FoamNordic native tests" OFF)
option(FOAMNORDIC_ONNX_RUNTIME "Build the native ONNX Runtime kernel" OFF)
option(FOAMNORDIC_RESIDENT_TOOLS "Build native resident worker executables" OFF)
option(FOAMNORDIC_UCX "Build the optional native UCX data plane" OFF)
set(FOAMNORDIC_ONNX_RUNTIME_ROOT "" CACHE PATH "ONNX Runtime 1.28.0 installation root")
set(FOAMNORDIC_UCX_ROOT "" CACHE PATH "UCX installation root")

include(cmake/BindNordic.cmake)
foamnordic_add_bindnordic()

add_subdirectory(src/foamnordic/runtime)
add_subdirectory(src/foamnordic/backend/communication)
add_subdirectory(src/foamnordic/backend/adapter)
add_subdirectory(src/foamnordic/backend/inference)
add_subdirectory(tools/longship)

if(FOAMNORDIC_RESIDENT_TOOLS)
    if(NOT FOAMNORDIC_ONNX_RUNTIME)
        message(FATAL_ERROR
            "FOAMNORDIC_RESIDENT_TOOLS currently requires FOAMNORDIC_ONNX_RUNTIME=ON")
    endif()
    add_subdirectory(tools/resident)
endif()

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

install(
    TARGETS
        foamnordic_runtime
        foamnordic_fjord
        foamnordic_adapter
        foamnordic_inference
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Development)
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

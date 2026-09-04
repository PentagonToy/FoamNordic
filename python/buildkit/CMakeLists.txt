cmake_minimum_required(VERSION 3.20)

project(FoamNordicBuildKit VERSION 1.0.5 LANGUAGES CXX)

include(GNUInstallDirs)

option(FOAMNORDIC_TESTS "Build FoamNordic native tests" OFF)
option(FOAMNORDIC_ONNX_RUNTIME "Build the native ONNX Runtime kernel" OFF)
option(FOAMNORDIC_RESIDENT_TOOLS "Build native resident worker executables" OFF)
option(FOAMNORDIC_UCX "Build the optional native UCX data plane" OFF)
set(FOAMNORDIC_ONNX_RUNTIME_ROOT "" CACHE PATH "ONNX Runtime 1.28.0 installation root")
set(FOAMNORDIC_UCX_ROOT "" CACHE PATH "UCX installation root")

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

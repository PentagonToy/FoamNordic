cmake_minimum_required(VERSION 3.20)

project(FoamNordicBuildKit VERSION 1.0.4 LANGUAGES CXX)

include(GNUInstallDirs)

option(FOAMNORDIC_UCX "Build the optional native UCX data plane" OFF)
set(FOAMNORDIC_UCX_ROOT "" CACHE PATH "UCX installation root")

add_subdirectory(src/foamnordic/runtime)
add_subdirectory(src/foamnordic/backend/communication)
add_subdirectory(src/foamnordic/backend/adapter)

install(
    TARGETS
        foamnordic_runtime
        foamnordic_fjord
        foamnordic_adapter
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

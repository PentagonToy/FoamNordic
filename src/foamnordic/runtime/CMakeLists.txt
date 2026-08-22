find_package(Threads REQUIRED)

add_library(
    foamnordic_runtime STATIC
    log.cpp
    longship.cpp
    longship_cli.cpp
    longship_supervisor.cpp
    placement.cpp)
add_library(FoamNordic::runtime ALIAS foamnordic_runtime)

target_compile_features(foamnordic_runtime PUBLIC cxx_std_20)
target_link_libraries(foamnordic_runtime PUBLIC Threads::Threads)
target_include_directories(foamnordic_runtime PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>)
set_target_properties(foamnordic_runtime PROPERTIES
    CXX_EXTENSIONS OFF
    EXPORT_NAME runtime
    POSITION_INDEPENDENT_CODE ON
)

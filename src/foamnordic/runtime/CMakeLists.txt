add_library(
    foamnordic_runtime STATIC
    log.cpp
    longship.cpp
    longship_supervisor.cpp
    placement.cpp)
add_library(FoamNordic::runtime ALIAS foamnordic_runtime)

target_compile_features(foamnordic_runtime PUBLIC cxx_std_20)
target_include_directories(foamnordic_runtime PUBLIC ${PROJECT_SOURCE_DIR}/src)
set_target_properties(foamnordic_runtime PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

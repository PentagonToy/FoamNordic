add_library(foamnordic_adapter STATIC
    exchange.cpp
    field.cpp
    observation.cpp
    observation_stream.cpp
    plan.cpp
    port.cpp
    sequence.cpp
)
add_library(FoamNordic::adapter ALIAS foamnordic_adapter)

target_compile_features(foamnordic_adapter PUBLIC cxx_std_20)
target_include_directories(foamnordic_adapter PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(foamnordic_adapter PUBLIC FoamNordic::fjord FoamNordic::runtime)
set_target_properties(foamnordic_adapter PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

find_package(Threads REQUIRED)

add_library(foamnordic_fjord STATIC
    src/endpoint.cpp
    src/channel.cpp
    src/harbor.cpp
    src/rune.cpp
    src/shm_ring.cpp
    src/shm_channel.cpp
)

add_library(FoamNordic::fjord ALIAS foamnordic_fjord)

target_compile_features(foamnordic_fjord PUBLIC cxx_std_20)
target_link_libraries(foamnordic_fjord PUBLIC Threads::Threads FoamNordic::runtime)
target_include_directories(foamnordic_fjord PUBLIC include)
set_target_properties(foamnordic_fjord PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

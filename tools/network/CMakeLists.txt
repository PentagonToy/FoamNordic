add_executable(foamnordic_fjord_network_probe fjord_probe.cpp)
target_link_libraries(
    foamnordic_fjord_network_probe PRIVATE FoamNordic::fjord)

add_executable(foamnordic_log_test test_log.cpp)
target_link_libraries(foamnordic_log_test PRIVATE FoamNordic::runtime)

add_test(NAME runtime.log COMMAND foamnordic_log_test)

add_executable(foamnordic_placement_test test_placement.cpp)
target_link_libraries(foamnordic_placement_test PRIVATE FoamNordic::runtime)

add_test(NAME runtime.placement COMMAND foamnordic_placement_test)

add_executable(foamnordic_longship_test test_longship.cpp)
target_link_libraries(foamnordic_longship_test PRIVATE FoamNordic::runtime)

add_test(NAME runtime.longship COMMAND foamnordic_longship_test)

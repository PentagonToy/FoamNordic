add_executable(foamnordic_log_test test_log.cpp)
target_link_libraries(foamnordic_log_test PRIVATE FoamNordic::runtime)

add_test(NAME runtime.log COMMAND foamnordic_log_test)

add_executable(foamnordic_placement_test test_placement.cpp)
target_link_libraries(foamnordic_placement_test PRIVATE FoamNordic::runtime)

add_test(NAME runtime.placement COMMAND foamnordic_placement_test)

add_executable(foamnordic_longship_test test_longship.cpp)
target_link_libraries(foamnordic_longship_test PRIVATE FoamNordic::runtime)

add_test(NAME runtime.longship COMMAND foamnordic_longship_test)
add_test(NAME runtime.longship_help COMMAND foamnordic-longship --help)

add_test(
    NAME runtime.longship_template
    COMMAND
        ${CMAKE_COMMAND}
        -DTEMPLATE=${PROJECT_SOURCE_DIR}/tools/template/slurm/longship.sbatch.in
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test_longship_template.cmake)

add_executable(
    foamnordic_openfoam_expression_test
    test_expression.cpp
    ${PROJECT_SOURCE_DIR}/src/foamnordic/openfoam/operations/expression.C)

target_include_directories(
    foamnordic_openfoam_expression_test
    PRIVATE ${PROJECT_SOURCE_DIR}/src/foamnordic/openfoam/operations)
target_compile_features(
    foamnordic_openfoam_expression_test
    PRIVATE cxx_std_20)

add_test(
    NAME openfoam.operations.expression
    COMMAND foamnordic_openfoam_expression_test)

add_test(
    NAME openfoam.closure_dictionary
    COMMAND
        ${CMAKE_COMMAND}
        -DINPUT=${PROJECT_SOURCE_DIR}/src/foamnordic/template/openfoam/closureDict.in
        -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/foamnordicClosureDict
        -DCONFIGURE_SCRIPT=${PROJECT_SOURCE_DIR}/tools/openfoam/configureClosureDict.cmake
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test_closure_dictionary.cmake)

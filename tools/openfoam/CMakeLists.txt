add_executable(foamnordic_openfoam_echo echo_worker.cpp)
target_link_libraries(foamnordic_openfoam_echo PRIVATE FoamNordic::fjord)

if(FOAMNORDIC_ONNX_RUNTIME)
    add_executable(foamnordic_openfoam_onnx_fixture onnx_fixture.cpp)
    target_link_libraries(
        foamnordic_openfoam_onnx_fixture PRIVATE FoamNordic::closure)
endif()

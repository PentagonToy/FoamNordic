add_executable(foamnordic_closure_test test_closure.cpp)
target_link_libraries(foamnordic_closure_test PRIVATE FoamNordic::closure)

add_test(NAME backend.inference.closure COMMAND foamnordic_closure_test)

add_executable(foamnordic_closure_runner_test test_runner.cpp)
target_link_libraries(foamnordic_closure_runner_test PRIVATE FoamNordic::closure)

add_test(NAME backend.inference.runner COMMAND foamnordic_closure_runner_test)

add_executable(foamnordic_artifact_test test_artifact.cpp)
target_link_libraries(foamnordic_artifact_test PRIVATE FoamNordic::closure)

add_test(NAME backend.inference.artifact COMMAND foamnordic_artifact_test)

add_executable(foamnordic_closure_worker_test test_worker.cpp)
target_link_libraries(foamnordic_closure_worker_test PRIVATE FoamNordic::closure)

add_test(NAME backend.inference.worker COMMAND foamnordic_closure_worker_test)

if(FOAMNORDIC_ONNX_RUNTIME)
    add_executable(foamnordic_onnx_test test_onnx.cpp)
    target_link_libraries(foamnordic_onnx_test PRIVATE FoamNordic::closure)

    add_test(NAME backend.inference.onnx COMMAND foamnordic_onnx_test)
endif()

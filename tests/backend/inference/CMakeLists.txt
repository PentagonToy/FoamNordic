add_executable(foamnordic_inference_test test_exchange.cpp)
target_link_libraries(foamnordic_inference_test PRIVATE FoamNordic::inference)

add_test(NAME backend.inference.exchange COMMAND foamnordic_inference_test)

add_executable(foamnordic_inference_runner_test test_runner.cpp)
target_link_libraries(foamnordic_inference_runner_test PRIVATE FoamNordic::inference)

add_test(NAME backend.inference.runner COMMAND foamnordic_inference_runner_test)

add_executable(foamnordic_artifact_test test_artifact.cpp)
target_link_libraries(foamnordic_artifact_test PRIVATE FoamNordic::inference)

add_test(NAME backend.inference.artifact COMMAND foamnordic_artifact_test)

add_executable(foamnordic_inference_worker_test test_worker.cpp)
target_link_libraries(foamnordic_inference_worker_test PRIVATE FoamNordic::inference)

add_test(NAME backend.inference.worker COMMAND foamnordic_inference_worker_test)

if(FOAMNORDIC_ONNX_RUNTIME)
    add_executable(foamnordic_onnx_test test_onnx.cpp)
    target_link_libraries(foamnordic_onnx_test PRIVATE FoamNordic::inference)

    add_test(NAME backend.inference.onnx COMMAND foamnordic_onnx_test)
endif()

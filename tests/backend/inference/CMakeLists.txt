add_executable(foamnordic_inference_test test_exchange.cpp)
target_link_libraries(foamnordic_inference_test PRIVATE FoamNordic::inference)

add_test(NAME backend.inference.exchange COMMAND foamnordic_inference_test)

add_executable(foamnordic_inference_runner_test test_runner.cpp)
target_link_libraries(foamnordic_inference_runner_test PRIVATE FoamNordic::inference)

add_test(NAME backend.inference.runner COMMAND foamnordic_inference_runner_test)

add_executable(foamnordic_artifact_test test_artifact.cpp)
target_link_libraries(foamnordic_artifact_test PRIVATE FoamNordic::inference)

add_test(NAME backend.inference.artifact COMMAND foamnordic_artifact_test)

add_executable(foamnordic_forest_test test_forest.cpp)
target_link_libraries(foamnordic_forest_test PRIVATE FoamNordic::inference)
add_test(NAME backend.inference.forest COMMAND foamnordic_forest_test)
set_tests_properties(backend.inference.forest PROPERTIES TIMEOUT 20)

add_executable(foamnordic_admission_test test_admission.cpp)
target_link_libraries(foamnordic_admission_test PRIVATE FoamNordic::inference)
add_test(NAME backend.inference.admission COMMAND foamnordic_admission_test)
set_tests_properties(backend.inference.admission PROPERTIES TIMEOUT 20)

add_executable(foamnordic_inference_worker_test test_worker.cpp)
target_link_libraries(foamnordic_inference_worker_test PRIVATE FoamNordic::inference)

add_test(NAME backend.inference.worker COMMAND foamnordic_inference_worker_test)

if(FOAMNORDIC_ONNX_RUNTIME)
    add_executable(foamnordic_onnx_test test_onnx.cpp)
    target_link_libraries(foamnordic_onnx_test PRIVATE FoamNordic::inference)

    add_test(NAME backend.inference.onnx COMMAND foamnordic_onnx_test)
endif()

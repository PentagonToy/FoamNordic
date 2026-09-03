add_library(foamnordic_inference STATIC
    ../connectors/registry.cpp
    affine.cpp
    artifact.cpp
    exchange.cpp
    load.cpp
    manifest.cpp
    model.cpp
    runner.cpp
    scaler.cpp
    worker.cpp
)
add_library(FoamNordic::inference ALIAS foamnordic_inference)

target_compile_features(foamnordic_inference PUBLIC cxx_std_20)
target_include_directories(foamnordic_inference PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(foamnordic_inference PUBLIC FoamNordic::fjord FoamNordic::runtime)
set_target_properties(foamnordic_inference PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if(FOAMNORDIC_ONNX_RUNTIME)
    find_path(FOAMNORDIC_ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS "${FOAMNORDIC_ONNX_RUNTIME_ROOT}/include"
        REQUIRED
        NO_DEFAULT_PATH
    )
    find_library(FOAMNORDIC_ONNXRUNTIME_LIBRARY
        NAMES onnxruntime libonnxruntime
        HINTS "${FOAMNORDIC_ONNX_RUNTIME_ROOT}/lib"
        REQUIRED
        NO_DEFAULT_PATH
    )
    target_sources(foamnordic_inference PRIVATE onnx.cpp)
    target_sources(foamnordic_inference PRIVATE
        ../connectors/onnx/connector.cpp)
    target_include_directories(foamnordic_inference PRIVATE
        "${FOAMNORDIC_ONNXRUNTIME_INCLUDE_DIR}"
    )
    target_link_libraries(foamnordic_inference PRIVATE
        "${FOAMNORDIC_ONNXRUNTIME_LIBRARY}"
    )
    target_compile_definitions(foamnordic_inference PUBLIC
        FOAMNORDIC_HAS_ONNXRUNTIME=1
    )
endif()

add_library(foamnordic_closure STATIC
    ../connectors/registry.cpp
    affine.cpp
    artifact.cpp
    closure.cpp
    load.cpp
    manifest.cpp
    model.cpp
    runner.cpp
    scaler.cpp
    worker.cpp
)
add_library(FoamNordic::closure ALIAS foamnordic_closure)

target_compile_features(foamnordic_closure PUBLIC cxx_std_20)
target_include_directories(foamnordic_closure PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(foamnordic_closure PUBLIC FoamNordic::fjord FoamNordic::runtime)
set_target_properties(foamnordic_closure PROPERTIES
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
    target_sources(foamnordic_closure PRIVATE onnx.cpp)
    target_sources(foamnordic_closure PRIVATE
        ../connectors/onnx/connector.cpp)
    target_include_directories(foamnordic_closure PRIVATE
        "${FOAMNORDIC_ONNXRUNTIME_INCLUDE_DIR}"
    )
    target_link_libraries(foamnordic_closure PRIVATE
        "${FOAMNORDIC_ONNXRUNTIME_LIBRARY}"
    )
    target_compile_definitions(foamnordic_closure PUBLIC
        FOAMNORDIC_HAS_ONNXRUNTIME=1
    )
endif()

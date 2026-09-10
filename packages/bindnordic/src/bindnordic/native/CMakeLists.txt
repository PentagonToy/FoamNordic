add_library(bindnordic_models STATIC
    src/affine.cpp
    src/artifact.cpp
    src/compiled.cpp
    src/contract.cpp
    src/forest.cpp
    src/manifest.cpp
    src/scaler.cpp
    src/table.cpp
)
add_library(BindNordic::models ALIAS bindnordic_models)

target_compile_features(bindnordic_models PUBLIC cxx_std_20)
target_include_directories(bindnordic_models PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(bindnordic_models PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(bindnordic_models PROPERTIES
    CXX_EXTENSIONS OFF
    EXPORT_NAME models
    POSITION_INDEPENDENT_CODE ON
)

if(BINDNORDIC_ONNX_RUNTIME)
    find_path(BINDNORDIC_ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS "${BINDNORDIC_ONNX_RUNTIME_ROOT}/include"
        REQUIRED
        NO_DEFAULT_PATH
    )
    find_library(BINDNORDIC_ONNXRUNTIME_LIBRARY
        NAMES onnxruntime libonnxruntime
        HINTS "${BINDNORDIC_ONNX_RUNTIME_ROOT}/lib"
        REQUIRED
        NO_DEFAULT_PATH
    )
    target_sources(bindnordic_models PRIVATE src/onnx.cpp)
    target_include_directories(bindnordic_models PRIVATE
        "${BINDNORDIC_ONNXRUNTIME_INCLUDE_DIR}"
    )
    target_link_libraries(bindnordic_models PRIVATE
        "${BINDNORDIC_ONNXRUNTIME_LIBRARY}"
    )
    target_compile_definitions(bindnordic_models PUBLIC BINDNORDIC_HAS_ONNXRUNTIME=1)
endif()

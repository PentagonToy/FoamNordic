add_library(foamnordic_inference STATIC
    ../connectors/registry.cpp
    exchange.cpp
    execution.cpp
    smedja.cpp
    load.cpp
    model.cpp
    runner.cpp
    worker.cpp
)
add_library(FoamNordic::inference ALIAS foamnordic_inference)

target_compile_features(foamnordic_inference PUBLIC cxx_std_20)
target_include_directories(foamnordic_inference PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(foamnordic_inference PUBLIC FoamNordic::fjord FoamNordic::runtime BindNordic::models)
target_link_libraries(foamnordic_inference PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(foamnordic_inference PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if(FOAMNORDIC_ONNX_RUNTIME)
    target_sources(foamnordic_inference PRIVATE
        ../connectors/onnx/connector.cpp)
    target_compile_definitions(foamnordic_inference PUBLIC
        FOAMNORDIC_HAS_ONNXRUNTIME=1
    )
endif()

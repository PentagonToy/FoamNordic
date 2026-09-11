add_library(foamnordic_inference STATIC
    ../connectors/registry.cpp
    ../connectors/plugin.cpp
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

    add_library(foamnordic_onnx_connector SHARED
        ../connectors/onnx/plugin.cpp)
    target_compile_features(foamnordic_onnx_connector PRIVATE cxx_std_20)
    target_include_directories(foamnordic_onnx_connector PRIVATE
        ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(foamnordic_onnx_connector PRIVATE BindNordic::models)
    set_target_properties(foamnordic_onnx_connector PROPERTIES
        CXX_EXTENSIONS OFF
        OUTPUT_NAME foamnordic_onnx_connector)
    if(APPLE)
        set_target_properties(foamnordic_onnx_connector PROPERTIES
            INSTALL_RPATH "@loader_path")
    elseif(UNIX)
        set_target_properties(foamnordic_onnx_connector PROPERTIES
            INSTALL_RPATH "$ORIGIN")
    endif()
    install(
        TARGETS foamnordic_onnx_connector
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Runtime
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime)
endif()

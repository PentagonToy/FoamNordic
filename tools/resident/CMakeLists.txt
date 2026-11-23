add_executable(foamnordic_model_worker main.cpp)
target_link_libraries(foamnordic_model_worker PRIVATE FoamNordic::inference)
target_compile_features(foamnordic_model_worker PRIVATE cxx_std_20)
set_target_properties(foamnordic_model_worker PROPERTIES CXX_EXTENSIONS OFF)
file(RELATIVE_PATH FOAMNORDIC_MODEL_WORKER_LIBRARY_PATH
    "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
if(APPLE)
    set_target_properties(foamnordic_model_worker PROPERTIES
        INSTALL_RPATH "@loader_path/${FOAMNORDIC_MODEL_WORKER_LIBRARY_PATH}")
elseif(UNIX)
    set_target_properties(foamnordic_model_worker PROPERTIES
        INSTALL_RPATH "$ORIGIN/${FOAMNORDIC_MODEL_WORKER_LIBRARY_PATH}")
endif()
install(
    TARGETS foamnordic_model_worker
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Runtime)
get_filename_component(
    FOAMNORDIC_ONNXRUNTIME_LIBRARY_DIR
    "${FOAMNORDIC_ONNXRUNTIME_LIBRARY}"
    DIRECTORY)
install(
    DIRECTORY "${FOAMNORDIC_ONNXRUNTIME_LIBRARY_DIR}/"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT Runtime
    FILES_MATCHING
        PATTERN "libonnxruntime.so*"
        PATTERN "libonnxruntime*.dylib")

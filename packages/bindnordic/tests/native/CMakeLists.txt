add_executable(bindnordic_native_models_test test_models.cpp)
target_link_libraries(bindnordic_native_models_test PRIVATE BindNordic::models)
target_compile_options(bindnordic_native_models_test PRIVATE $<$<CONFIG:Release>:-UNDEBUG>)
add_test(NAME native.models COMMAND bindnordic_native_models_test)

add_executable(foamnordic-longship main.cpp)
target_link_libraries(foamnordic-longship PRIVATE FoamNordic::runtime)
target_compile_features(foamnordic-longship PRIVATE cxx_std_20)
set_target_properties(foamnordic-longship PROPERTIES CXX_EXTENSIONS OFF)

if(FOAMNORDIC_PYTHON)
    install(
        TARGETS foamnordic-longship
        RUNTIME DESTINATION foamnordic/bin
        COMPONENT Python)
else()
    install(
        TARGETS foamnordic-longship
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Runtime)
endif()

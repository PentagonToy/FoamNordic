find_package(Python 3.11...<3.13 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_native
    src/artifact_bindings.cpp
    src/main.cpp
    src/resident_bindings.cpp
)
target_link_libraries(_native PRIVATE FoamNordic::runtime FoamNordic::closure)
target_compile_features(_native PRIVATE cxx_std_20)
set_target_properties(_native PROPERTIES CXX_EXTENSIONS OFF)

install(TARGETS _native LIBRARY DESTINATION foamnordic COMPONENT Python)
install(
    FILES
        ${PROJECT_SOURCE_DIR}/src/foamnordic/template/openfoam/observation.in
        ${PROJECT_SOURCE_DIR}/src/foamnordic/template/openfoam/derivedSchemes.json
        ${PROJECT_SOURCE_DIR}/src/foamnordic/template/openfoam/turbulenceProperties.fjord.in
    DESTINATION foamnordic/templates/openfoam
    COMPONENT Python)
install(
    FILES ${PROJECT_SOURCE_DIR}/src/foamnordic/template/large_banner.txt
    DESTINATION foamnordic/templates
    COMPONENT Python)
install(
    FILES
        ${PROJECT_SOURCE_DIR}/src/foamnordic/template/slurm/sailing.sbatch.in
        ${PROJECT_SOURCE_DIR}/src/foamnordic/template/slurm/submit.sh.in
    DESTINATION foamnordic/templates/slurm
    COMPONENT Python)

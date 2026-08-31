find_package(Threads REQUIRED)

set(foamnordic_fjord_sources
    src/endpoint.cpp
    src/channel.cpp
    src/harbor.cpp
    src/rune.cpp
    src/shm_ring.cpp
    src/shm_channel.cpp
)

if(FOAMNORDIC_UCX)
    find_path(FOAMNORDIC_UCX_INCLUDE_DIR
        NAMES ucp/api/ucp.h
        HINTS "${FOAMNORDIC_UCX_ROOT}/include"
        REQUIRED)
    find_library(FOAMNORDIC_UCP_LIBRARY
        NAMES ucp
        HINTS "${FOAMNORDIC_UCX_ROOT}/lib" "${FOAMNORDIC_UCX_ROOT}/lib64"
        REQUIRED)
    find_library(FOAMNORDIC_UCS_LIBRARY
        NAMES ucs
        HINTS "${FOAMNORDIC_UCX_ROOT}/lib" "${FOAMNORDIC_UCX_ROOT}/lib64"
        REQUIRED)

    add_library(foamnordic_ucx INTERFACE)
    add_library(FoamNordic::ucx ALIAS foamnordic_ucx)
    target_include_directories(
        foamnordic_ucx INTERFACE "${FOAMNORDIC_UCX_INCLUDE_DIR}")
    target_link_libraries(
        foamnordic_ucx INTERFACE
        "${FOAMNORDIC_UCP_LIBRARY}"
        "${FOAMNORDIC_UCS_LIBRARY}")
    list(APPEND foamnordic_fjord_sources src/ucx_channel.cpp)
    message(STATUS
        "FoamNordic UCX data plane: enabled (${FOAMNORDIC_UCP_LIBRARY})")
else()
    message(STATUS "FoamNordic UCX data plane: disabled")
endif()

add_library(foamnordic_fjord STATIC ${foamnordic_fjord_sources})

add_library(FoamNordic::fjord ALIAS foamnordic_fjord)

target_compile_features(foamnordic_fjord PUBLIC cxx_std_20)
target_link_libraries(foamnordic_fjord PUBLIC Threads::Threads FoamNordic::runtime)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # shm_open/shm_unlink live in librt on the oldest glibc supported by
    # manylinux wheels.  Newer glibc folds them into libc, but retaining the
    # explicit dependency keeps the binary portable across both layouts.
    target_link_libraries(foamnordic_fjord PUBLIC rt)
endif()
target_include_directories(foamnordic_fjord PUBLIC include)
if(FOAMNORDIC_UCX)
    target_compile_definitions(foamnordic_fjord PUBLIC FOAMNORDIC_HAVE_UCX=1)
    target_link_libraries(foamnordic_fjord PUBLIC FoamNordic::ucx)
endif()
set_target_properties(foamnordic_fjord PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

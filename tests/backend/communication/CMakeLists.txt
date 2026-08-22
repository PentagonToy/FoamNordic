add_executable(foamnordic_fjord_test test_fjord.cpp)
target_link_libraries(foamnordic_fjord_test PRIVATE FoamNordic::fjord)

add_test(NAME backend.communication.fjord COMMAND foamnordic_fjord_test)

add_executable(foamnordic_shm_ring_test test_shm_ring.cpp)
target_link_libraries(foamnordic_shm_ring_test PRIVATE FoamNordic::fjord)

add_test(NAME backend.communication.shm_ring COMMAND foamnordic_shm_ring_test)

set(FOAMNORDIC_ADDRESS "unix:///tmp/foamnordic-closure.sock")
set(FOAMNORDIC_SESSION_ID 2026)
set(FOAMNORDIC_SHARED_MEMORY true)
set(FOAMNORDIC_UCX false)
set(FOAMNORDIC_INPUT_KEYS pressure_laplacian)
set(FOAMNORDIC_INPUT_EXPRESSIONS "\"laplacian(p)\"")
set(FOAMNORDIC_OUTPUT_FIELDS nut)
set(FOAMNORDIC_PROBE_EXPRESSION "laplacian(p)")
set(FOAMNORDIC_PROBE_OUTPUT nut)
set(FOAMNORDIC_PROBE_SCALE 1.005)
set(FOAMNORDIC_PROBE_SEED 0.25)
set(FOAMNORDIC_PROBE_EXPECT_FAILURE true)

include("${CONFIGURE_SCRIPT}")
file(READ "${OUTPUT}" document)

foreach(expected IN ITEMS
        "address          \"unix:///tmp/foamnordic-closure.sock\";"
        "sessionId        2026;"
        "sharedMemory     true;"
        "ucx              false;"
        "    pressure_laplacian"
        "    \"laplacian(p)\""
        "probeExpression  \"laplacian(p)\";"
        "probeOutput      nut;"
        "probeScale       1.005;"
        "probeSeed        0.25;"
        "probeExpectFailure true;")
    string(FIND "${document}" "${expected}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR
            "Generated closure dictionary is missing: ${expected}")
    endif()
endforeach()

string(FIND "${document}" "@FOAMNORDIC_" placeholder)
if(NOT placeholder EQUAL -1)
    message(FATAL_ERROR
        "Generated closure dictionary contains an unresolved placeholder")
endif()

if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
    message(FATAL_ERROR "INPUT and OUTPUT are required")
endif()

foreach(variable IN ITEMS
        FOAMNORDIC_ADDRESS
        FOAMNORDIC_SESSION_ID)
    if(NOT DEFINED ${variable})
        message(FATAL_ERROR "${variable} is required")
    endif()
endforeach()

configure_file("${INPUT}" "${OUTPUT}" @ONLY)

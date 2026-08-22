file(READ "${TEMPLATE}" document)

foreach(expected IN ITEMS
        "exec @LONGSHIP_EXECUTABLE@"
        "@HOST_READY_ARGUMENTS@"
        "--readiness-timeout-ms @READINESS_TIMEOUT_MS@"
        "--termination-grace-ms @TERMINATION_GRACE_MS@"
        "--ntasks=@HOST_TASKS@"
        "--ntasks-per-node=@SOLVER_TASKS_PER_NODE@"
        "@HOST_COMMAND@"
        "@SOLVER_COMMAND@")
    string(FIND "${document}" "${expected}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Longship template is missing: ${expected}")
    endif()
endforeach()

foreach(obsolete IN ITEMS "wait -n" "trap cleanup" "@HOST_READY_COMMAND@")
    string(FIND "${document}" "${obsolete}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR
            "Longship template retains duplicate lifecycle logic: ${obsolete}")
    endif()
endforeach()

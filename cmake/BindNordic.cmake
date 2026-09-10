function(foamnordic_add_bindnordic)
    if(TARGET BindNordic::models)
        return()
    endif()

    set(BINDNORDIC_SOURCE_DIR "" CACHE PATH "BindNordic native source directory")
    if(NOT BINDNORDIC_SOURCE_DIR)
        set(_bindnordic_monorepo
            "${CMAKE_CURRENT_SOURCE_DIR}/packages/bindnordic/src/bindnordic/native")
        if(EXISTS "${_bindnordic_monorepo}/CMakeLists.txt")
            set(BINDNORDIC_SOURCE_DIR "${_bindnordic_monorepo}")
        else()
            find_package(Python 3.11...<3.13 REQUIRED COMPONENTS Interpreter)
            execute_process(
                COMMAND "${Python_EXECUTABLE}" -c
                    "from pathlib import Path; import bindnordic; print(Path(bindnordic.__file__).resolve().parent / 'native')"
                RESULT_VARIABLE _bindnordic_status
                OUTPUT_VARIABLE _bindnordic_package
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(_bindnordic_status EQUAL 0
                AND EXISTS "${_bindnordic_package}/CMakeLists.txt")
                set(BINDNORDIC_SOURCE_DIR "${_bindnordic_package}")
            endif()
        endif()
    endif()

    if(NOT EXISTS "${BINDNORDIC_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "BindNordic native sources were not found. Install BindNordic or set BINDNORDIC_SOURCE_DIR.")
    endif()

    set(BINDNORDIC_ONNX_RUNTIME "${FOAMNORDIC_ONNX_RUNTIME}" CACHE BOOL "" FORCE)
    set(BINDNORDIC_ONNX_RUNTIME_ROOT "${FOAMNORDIC_ONNX_RUNTIME_ROOT}" CACHE PATH "" FORCE)
    add_subdirectory("${BINDNORDIC_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/bindnordic-native")
endfunction()

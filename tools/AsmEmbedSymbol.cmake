function(usage)
    message("Write an assembly file to <output> that embeds <symbol> as data with the"
                   " contents from <input>.")
    message("Usage: cmake -DSYMBOL=<symbol> -DINPUT_BIN=<input> -DOUTPUT=<output>")
endfunction()

foreach(PARAM_NAME IN ITEMS "SYMBOL" "INPUT_BIN" "OUTPUT")
    if(NOT DEFINED ${PARAM_NAME})
        usage()
        message(FATAL_ERROR "${PARAM_NAME} is required.\n"
                            "Please call the script as 'cmake -D${PARAM_NAME}=<value> ... -P <script>'")
    endif()
endforeach()

# The name of this file is written into the output
get_filename_component(GENERATOR_NAME ${CMAKE_CURRENT_LIST_FILE} NAME)

configure_file(${CMAKE_CURRENT_LIST_DIR}/asm_embed_symbol.S.in ${OUTPUT} @ONLY)

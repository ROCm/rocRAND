# Creates PREFIX_VERSION[_MAJOR|_MINOR|_PATCH] variables
function(create_version_vars PREFIX VERSION_MAJOR VERSION_MINOR VERSION_PATCH)
    set(${PREFIX}_VERSION_MAJOR ${VERSION_MAJOR} PARENT_SCOPE)
    set(${PREFIX}_VERSION_MINOR ${VERSION_MINOR} PARENT_SCOPE)
    set(${PREFIX}_VERSION_PATCH ${VERSION_PATCH} PARENT_SCOPE)
    set(${PREFIX}_VERSION
        "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
        PARENT_SCOPE
    )
    math(EXPR temp "${VERSION_MAJOR} * 100000 + ${VERSION_MINOR} * 100 + ${VERSION_PATCH}")
    set(${PREFIX}_VERSION_NUMBER ${temp} PARENT_SCOPE)
endfunction()

#
function(package_set_postinst_prerm LIB_NAMES LIB_DIRS INCLUDE_DIRS)
    list(LENGTH LIB_NAMES len1)
    list(LENGTH LIB_DIRS  len2)
    if(NOT (len1 EQUAL len2))
        message(FATAL_ERROR "LIB_NAMES and LIB_DIRS must have same size")
    endif()
    math(EXPR len3 "${len1} - 1")

    set(POSTINST_SOURCE "")
    set(PRERM_SOURCE "")
    foreach(val RANGE ${len3})
        list(GET LIB_NAMES ${val} lib_name)
        list(GET LIB_DIRS  ${val} lib_dir)
        list(GET INCLUDE_DIRS ${val} inc_dir)

        set(POSTINST_SOURCE "${POSTINST_SOURCE}\nmkdir -p ${inc_dir}/../../include/")
        set(POSTINST_SOURCE "${POSTINST_SOURCE}\nmkdir -p ${lib_dir}/../../lib/cmake/${lib_name}")
        set(POSTINST_SOURCE "${POSTINST_SOURCE}\necho \"${lib_dir}\" > /etc/ld.so.conf.d/${lib_name}.conf")
        set(POSTINST_SOURCE "${POSTINST_SOURCE}\nln -sr ${inc_dir} ${inc_dir}/../../include/${lib_name}")
        set(POSTINST_SOURCE "${POSTINST_SOURCE}\nln -sr ${lib_dir}/lib${lib_name}.so ${lib_dir}/../../lib/lib${lib_name}.so")
        set(POSTINST_SOURCE "${POSTINST_SOURCE}\nln -sr ${lib_dir}/cmake/${lib_name} ${lib_dir}/../../lib/cmake/${lib_name}\n")

        set(PRERM_SOURCE "${PRERM_SOURCE}\nrm /etc/ld.so.conf.d/${lib_name}.conf")
        set(PRERM_SOURCE "${PRERM_SOURCE}\nunlink ${inc_dir}/../../include/${lib_name}")
        set(PRERM_SOURCE "${PRERM_SOURCE}\nunlink ${lib_dir}/../../lib/lib${lib_name}.so")
        set(PRERM_SOURCE "${PRERM_SOURCE}\nunlink ${lib_dir}/../../lib/cmake/${lib_name}/${lib_name}")
        set(PRERM_SOURCE "${PRERM_SOURCE}\nrm -d ${lib_dir}/../../lib/cmake/${lib_name}\n")
    endforeach()
    file(WRITE ${PROJECT_BINARY_DIR}/deb/postinst ""
        "#!/bin/bash\n"
        "${POSTINST_SOURCE}\n"
        "ldconfig\n"
    )
    file(WRITE ${PROJECT_BINARY_DIR}/deb/prerm ""
        "#!/bin/bash\n"
        "${PRERM_SOURCE}\n"
        "ldconfig\n"
    )

    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${PROJECT_BINARY_DIR}/deb/postinst;${PROJECT_BINARY_DIR}/deb/prerm" PARENT_SCOPE)
    set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/deb/postinst" PARENT_SCOPE)
    set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/deb/prerm" PARENT_SCOPE)
endfunction()
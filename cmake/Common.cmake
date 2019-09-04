#
function(package_set_postinst_prerm LIB_NAMES LIB_DIRS INCLUDE_DIRS)
    list(LENGTH LIB_NAMES len1)
    list(LENGTH LIB_DIRS  len2)
    if(NOT (len1 EQUAL len2))
        message(FATAL_ERROR "LIB_NAMES and LIB_DIRS must have same size")
    endif()
    math(EXPR len3 "${len1} - 1")

    set(POSTINST_SOURCE "")
    set(PREINST_SOURCE "")
    set(PRERM_SOURCE "")
    set(PRERM_RPM_SOURCE "")
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
	#For preinstall script, first argument is 1 for install and 2 for upgrade
	#Skip removal of symlinks if install command is called
        set(PREINST_SOURCE "${PREINST_SOURCE}\nif [ $1 == 2 ]; then")
        set(PREINST_SOURCE "${PREINST_SOURCE}\n\trm -f /etc/ld.so.conf.d/${lib_name}.conf")
        set(PREINST_SOURCE "${PREINST_SOURCE}\n\tunlink ${inc_dir}/../../include/${lib_name}")
        set(PREINST_SOURCE "${PREINST_SOURCE}\n\tunlink ${lib_dir}/../../lib/lib${lib_name}.so")
        set(PREINST_SOURCE "${PREINST_SOURCE}\n\tunlink ${lib_dir}/../../lib/cmake/${lib_name}/${lib_name}")
        set(PREINST_SOURCE "${PREINST_SOURCE}\n\trm -f ${lib_dir}/lib${lib_name}.so*")
        set(PREINST_SOURCE "${PREINST_SOURCE}\n\trm -d ${lib_dir}/../../lib/cmake/${lib_name}\n")
        set(PREINST_SOURCE "${PREINST_SOURCE}\nfi")

        set(PRERM_SOURCE "${PRERM_SOURCE}\nrm -f /etc/ld.so.conf.d/${lib_name}.conf")
        set(PRERM_SOURCE "${PRERM_SOURCE}\nunlink ${inc_dir}/../../include/${lib_name}")
        set(PRERM_SOURCE "${PRERM_SOURCE}\nunlink ${lib_dir}/../../lib/lib${lib_name}.so")
        set(PRERM_SOURCE "${PRERM_SOURCE}\nunlink ${lib_dir}/../../lib/cmake/${lib_name}/${lib_name}")
        set(PRERM_SOURCE "${PRERM_SOURCE}\nrm -d ${lib_dir}/../../lib/cmake/${lib_name}\n")

	#For pre uninstall script, first argument is 0 for uninstall and 1 for upgrade
	#Skip removal of symlinks if upgrade command is called (only for RPM)
        set(PRERM_RPM_SOURCE "${PRERM_RPM_SOURCE}\nif [ $1 == 0 ]; then")
        set(PRERM_RPM_SOURCE "${PRERM_RPM_SOURCE}\n\trm -f /etc/ld.so.conf.d/${lib_name}.conf")
        set(PRERM_RPM_SOURCE "${PRERM_RPM_SOURCE}\n\tunlink ${inc_dir}/../../include/${lib_name}")
        set(PRERM_RPM_SOURCE "${PRERM_RPM_SOURCE}\n\tunlink ${lib_dir}/../../lib/lib${lib_name}.so")
        set(PRERM_RPM_SOURCE "${PRERM_RPM_SOURCE}\n\tunlink ${lib_dir}/../../lib/cmake/${lib_name}/${lib_name}")
        set(PRERM_RPM_SOURCE "${PRERM_RPM_SOURCE}\n\trm -d ${lib_dir}/../../lib/cmake/${lib_name}\n")
        set(PRERM_RPM_SOURCE "${PRERM_RPM_SOURCE}\nfi")
    endforeach()
    #For Deb package, POSTINST_SOURCE and PRERM_SOURCE are used
    #For RPM package, POSTINST_SOURCE, PREINST_SOURCE, PRERM_RPM_SOURCE are used
    file(WRITE ${PROJECT_BINARY_DIR}/deb/postinst ""
        "#!/bin/bash\n"
        "${POSTINST_SOURCE}\n"
        "ldconfig\n"
    )
    file(WRITE ${PROJECT_BINARY_DIR}/deb/preinst ""
        "#!/bin/bash\n"
        "${PREINST_SOURCE}\n"
        "ldconfig\n"
    )
    file(WRITE ${PROJECT_BINARY_DIR}/deb/prerm ""
        "#!/bin/bash\n"
        "${PRERM_SOURCE}\n"
        "ldconfig\n"
    )
    file(WRITE ${PROJECT_BINARY_DIR}/deb/prermrpm ""
        "#!/bin/bash\n"
        "${PRERM_RPM_SOURCE}\n"
        "ldconfig\n"
    )

    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${PROJECT_BINARY_DIR}/deb/postinst;${PROJECT_BINARY_DIR}/deb/prerm" PARENT_SCOPE)
    set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/deb/postinst" PARENT_SCOPE)
    set(CPACK_RPM_PRE_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/deb/preinst" PARENT_SCOPE)
    set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/deb/prermrpm" PARENT_SCOPE)
endfunction()

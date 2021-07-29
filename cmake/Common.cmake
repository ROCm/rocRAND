#
set(ROCM_DISABLE_LDCONFIG OFF CACHE BOOL "")

function(package_set_postinst_prerm LIB_NAMES LIB_DIRS INCLUDE_DIRS SOVERSIONS)
    list(LENGTH LIB_NAMES len1)
    list(LENGTH LIB_DIRS  len2)
    if(NOT (len1 EQUAL len2))
        message(FATAL_ERROR "LIB_NAMES and LIB_DIRS must have same size")
    endif()
    math(EXPR len3 "${len1} - 1")

    set(POSTINST_DEVEL_SOURCE "")
    set(PREINST_DEVEL_SOURCE "")
    set(PRERM_DEVEL_SOURCE "")
    set(PRERM_DEVEL_RPM_SOURCE "")

    set(POSTINST_RUNTIME_SOURCE "")
    set(PREINST_RUNTIME_SOURCE "")
    set(PRERM_RUNTIME_SOURCE "")
    set(PRERM_RUNTIME_RPM_SOURCE "")

    set(POSTINST_SOURCE "")
    set(PREINST_SOURCE "")
    set(PRERM_SOURCE "")
    set(PRERM_RPM_SOURCE "")

    if(ROCM_USE_DEV_COMPONENT)
        set(PI_D_S POSTINST_DEVEL_SOURCE)
        set(RI_D_S PREINST_DEVEL_SOURCE)
        set(PR_D_S PRERM_DEVEL_SOURCE)
        set(PR_D_SR PRERM_DEVEL_RPM_SOURCE)

        set(PI_R_S POSTINST_RUNTIME_SOURCE)
        set(RI_R_S PREINST_RUNTIME_SOURCE)
        set(PR_R_S PRERM_RUNTIME_SOURCE)
        set(PR_R_SR PRERM_RUNTIME_RPM_SOURCE)
    else()
        set(PI_D_S POSTINST_SOURCE)
        set(RI_D_S PREINST_SOURCE)
        set(PR_D_S PRERM_SOURCE)
        set(PR_D_SR PRERM_RPM_SOURCE)

        set(PI_R_S POSTINST_SOURCE)
        set(RI_R_S PREINST_SOURCE)
        set(PR_R_S PRERM_SOURCE)
        set(PR_R_SR PRERM_RPM_SOURCE)
    endif()

    foreach(val RANGE ${len3})
        list(GET LIB_NAMES ${val} lib_name)
        list(GET LIB_DIRS  ${val} lib_dir)
        list(GET INCLUDE_DIRS ${val} inc_dir)
        list(GET SOVERSIONS ${val} so_ver)

        rocm_version_regex_parse("^([0-9]+).*" LIB_VERSION_MAJOR "${so_ver}")
        set (LIB_VERSION_STRING "${so_ver}.0")
        if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
            set (LIB_VERSION_STRING "${so_ver}.$ENV{ROCM_LIBPATCH_VERSION}")
        endif()

        set(${PI_D_S} "${${PI_D_S}}\nmkdir -p ${inc_dir}/../../include/")
        set(${PI_R_S} "${${PI_R_S}}\nmkdir -p ${inc_dir}/../../include/")
        set(${PI_D_S} "${${PI_D_S}}\nmkdir -p ${lib_dir}/../../lib/cmake/${lib_name}")
        set(${PI_R_S} "${${PI_R_S}}\nmkdir -p ${lib_dir}/../../lib/cmake/${lib_name}")
        if(NOT ${ROCM_DISABLE_LDCONFIG})
            set(${PI_R_S} "${${PI_R_S}}\necho \"${lib_dir}\" > /etc/ld.so.conf.d/${lib_name}.conf")
        endif()
        set(${PI_D_S} "${${PI_D_S}}\nln -sr ${inc_dir} ${inc_dir}/../../include/${lib_name}")
        set(${PI_D_S} "${${PI_D_S}}\nln -sr ${lib_dir}/lib${lib_name}.so ${lib_dir}/../../lib/lib${lib_name}.so")
        set(${PI_R_S} "${${PI_R_S}}\nln -sr ${lib_dir}/lib${lib_name}.so.${LIB_VERSION_MAJOR} ${lib_dir}/../../lib/lib${lib_name}.so.${LIB_VERSION_MAJOR}")
        set(${PI_R_S} "${${PI_R_S}}\nln -sr ${lib_dir}/lib${lib_name}.so.${LIB_VERSION_STRING} ${lib_dir}/../../lib/lib${lib_name}.so.${LIB_VERSION_STRING}")
        set(${PI_D_S} "${${PI_D_S}}\nln -sr ${lib_dir}/cmake/${lib_name} ${lib_dir}/../../lib/cmake/${lib_name}\n")
	#For preinstall script, first argument is 1 for install and 2 for upgrade
	#Skip removal of symlinks if install command is called
        set(${RI_D_S} "${${RI_D_S}}\nif [ $1 == 2 ]; then")
        set(${RI_R_S} "${${RI_R_S}}\nif [ $1 == 2 ]; then")
        set(${RI_R_S} "${${RI_R_S}}\n\trm -f /etc/ld.so.conf.d/${lib_name}.conf")
        set(${RI_D_S} "${${RI_D_S}}\n\tunlink ${inc_dir}/../../include/${lib_name}")
        set(${RI_D_S} "${${RI_D_S}}\n\tunlink ${lib_dir}/../../lib/lib${lib_name}.so")
        set(${RI_R_S} "${${RI_R_S}}\n\tunlink ${lib_dir}/../../lib/lib${lib_name}.so.${LIB_VERSION_MAJOR}")
        set(${RI_R_S} "${${RI_R_S}}\n\tunlink ${lib_dir}/../../lib/lib${lib_name}.so.${LIB_VERSION_STRING}")
        set(${RI_D_S} "${${RI_D_S}}\n\tunlink ${lib_dir}/../../lib/cmake/${lib_name}/${lib_name}")
        set(${RI_D_S} "${${RI_D_S}}\n\trm -f ${lib_dir}/lib${lib_name}.so")
        set(${RI_R_S} "${${RI_R_S}}\n\trm -f ${lib_dir}/lib${lib_name}.so.*")
        set(${RI_D_S} "${${RI_D_S}}\n\trm -d ${lib_dir}/../../lib/cmake/${lib_name}\n")
        set(${RI_D_S} "${${RI_D_S}}\nfi")
        set(${RI_R_S} "${${RI_R_S}}\nfi")

        set(${PR_R_S} "${${PR_R_S}}\nrm -f /etc/ld.so.conf.d/${lib_name}.conf")
        set(${PR_D_S} "${${PR_D_S}}\nunlink ${inc_dir}/../../include/${lib_name}")
        set(${PR_D_S} "${${PR_D_S}}\nunlink ${lib_dir}/../../lib/lib${lib_name}.so")
        set(${PR_R_S} "${${PR_R_S}}\nunlink ${lib_dir}/../../lib/lib${lib_name}.so.${LIB_VERSION_MAJOR}")
        set(${PR_R_S} "${${PR_R_S}}\nunlink ${lib_dir}/../../lib/lib${lib_name}.so.${LIB_VERSION_STRING}")
        set(${PR_D_S} "${${PR_D_S}}\nunlink ${lib_dir}/../../lib/cmake/${lib_name}/${lib_name}")
        set(${PR_D_S} "${${PR_D_S}}\nrm -d ${lib_dir}/../../lib/cmake/${lib_name}\n")

	#For pre uninstall script, first argument is 0 for uninstall and 1 for upgrade
	#Skip removal of symlinks if upgrade command is called (only for RPM)
        set(${PR_D_SR} "${${PR_D_SR}}\nif [ $1 == 0 ]; then")
        set(${PR_R_SR} "${${PR_R_SR}}\nif [ $1 == 0 ]; then")
        set(${PR_R_SR} "${${PR_R_SR}}\n\trm -f /etc/ld.so.conf.d/${lib_name}.conf")
        set(${PR_D_SR} "${${PR_D_SR}}\n\tunlink ${inc_dir}/../../include/${lib_name}")
        set(${PR_D_SR} "${${PR_D_SR}}\n\tunlink ${lib_dir}/../../lib/lib${lib_name}.so")
        set(${PR_R_SR} "${${PR_R_SR}}\n\tunlink ${lib_dir}/../../lib/lib${lib_name}.so.${LIB_VERSION_MAJOR}")
        set(${PR_R_SR} "${${PR_R_SR}}\n\tunlink ${lib_dir}/../../lib/lib${lib_name}.so.${LIB_VERSION_STRING}")
        set(${PR_D_SR} "${${PR_D_SR}}\n\tunlink ${lib_dir}/../../lib/cmake/${lib_name}/${lib_name}")
        set(${PR_D_SR} "${${PR_D_SR}}\n\trm -d ${lib_dir}/../../lib/cmake/${lib_name}\n")
        set(${PR_D_SR} "${${PR_D_SR}}\nfi")
    endforeach()
    #For Deb package, POSTINST_SOURCE and PRERM_SOURCE are used
    #For RPM package, POSTINST_SOURCE, PREINST_SOURCE, PRERM_RPM_SOURCE are used
    if(ROCM_USE_DEV_COMPONENT)
        file(WRITE ${PROJECT_BINARY_DIR}/deb/runtime/postinst ""
            "#!/bin/bash\n"
            "${POSTINST_RUNTIME_SOURCE}\n"
            "ldconfig\n"
        )
        file(WRITE ${PROJECT_BINARY_DIR}/deb/devel/postinst ""
            "#!/bin/bash\n"
            "${POSTINST_DEVEL_SOURCE}\n"
            "ldconfig\n"
        )

        file(WRITE ${PROJECT_BINARY_DIR}/deb/runtime/preinst ""
            "#!/bin/bash\n"
            "${PREINST_RUNTIME_SOURCE}\n"
            "ldconfig\n"
        )
        file(WRITE ${PROJECT_BINARY_DIR}/deb/devel/preinst ""
            "#!/bin/bash\n"
            "${PREINST_DEVEL_SOURCE}\n"
            "ldconfig\n"
        )

        file(WRITE ${PROJECT_BINARY_DIR}/deb/runtime/prerm ""
            "#!/bin/bash\n"
            "${PRERM_RUNTIME_SOURCE}\n"
            "ldconfig\n"
        )
        file(WRITE ${PROJECT_BINARY_DIR}/deb/devel/prerm ""
            "#!/bin/bash\n"
            "${PRERM_DEVEL_SOURCE}\n"
            "ldconfig\n"
        )

        file(WRITE ${PROJECT_BINARY_DIR}/deb/runtime/prermrpm ""
            "#!/bin/bash\n"
            "${PRERM_RUNTIME_RPM_SOURCE}\n"
            "ldconfig\n"
        )
        file(WRITE ${PROJECT_BINARY_DIR}/deb/devel/prermrpm ""
            "#!/bin/bash\n"
            "${PRERM_DEVEL_RPM_SOURCE}\n"
            "ldconfig\n"
        )
        set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${PROJECT_BINARY_DIR}/deb/runtime/postinst;${PROJECT_BINARY_DIR}/deb/runtime/prerm" PARENT_SCOPE)
        set(CPACK_DEBIAN_DEVEL_PACKAGE_CONTROL_EXTRA "${PROJECT_BINARY_DIR}/deb/devel/postinst;${PROJECT_BINARY_DIR}/deb/devel/prerm" PARENT_SCOPE)
        set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/deb/runtime/postinst" PARENT_SCOPE)
        set(CPACK_RPM_DEVEL_POST_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/deb/devel/postinst" PARENT_SCOPE)
        set(CPACK_RPM_PRE_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/deb/runtime/preinst" PARENT_SCOPE)
        set(CPACK_RPM_DEVEL_PRE_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/deb/devel/preinst" PARENT_SCOPE)
        set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/deb/runtime/prermrpm" PARENT_SCOPE)
        set(CPACK_RPM_DEVEL_PRE_UNINSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/deb/devel/prermrpm" PARENT_SCOPE)
    else()
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
    endif()
endfunction()

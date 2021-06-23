# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CPackComponent)
unset(IE_CPACK_COMPONENTS_ALL CACHE)

set(IE_CPACK_IE_DIR       deployment_tools/inference_engine)

#
# ie_cpack_set_library_dir()
#
# Set library directory for cpack
#
function(ie_cpack_set_library_dir)
    if(WIN32)
        set(IE_CPACK_LIBRARY_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH_FOLDER}/${CMAKE_BUILD_TYPE} PARENT_SCOPE)
        set(IE_CPACK_RUNTIME_PATH ${IE_CPACK_IE_DIR}/bin/${ARCH_FOLDER}/${CMAKE_BUILD_TYPE} PARENT_SCOPE)
        set(IE_CPACK_ARCHIVE_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH_FOLDER}/${CMAKE_BUILD_TYPE} PARENT_SCOPE)
    else()
        set(IE_CPACK_LIBRARY_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH_FOLDER} PARENT_SCOPE)
        set(IE_CPACK_RUNTIME_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH_FOLDER} PARENT_SCOPE)
        set(IE_CPACK_ARCHIVE_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH_FOLDER} PARENT_SCOPE)
    endif()
endfunction()

ie_cpack_set_library_dir()

#
# ie_cpack_add_component(NAME ...)
#
# Wraps original `cpack_add_component` and adds component to internal IE list
#
macro(ie_cpack_add_component NAME)
    list(APPEND IE_CPACK_COMPONENTS_ALL ${NAME})
    set(IE_CPACK_COMPONENTS_ALL "${IE_CPACK_COMPONENTS_ALL}" CACHE STRING "" FORCE)
    cpack_add_component(${NAME} ${ARGN})
endmacro()

macro(ie_cpack)
    set(CPACK_GENERATOR "TGZ")
    string(REPLACE "/" "_" CPACK_PACKAGE_VERSION "${CI_BUILD_NUMBER}")
    if(WIN32)
        set(CPACK_PACKAGE_NAME inference-engine_${CMAKE_BUILD_TYPE})
    else()
        set(CPACK_PACKAGE_NAME inference-engine)
    endif()

    foreach(ver IN LISTS MAJOR MINOR PATCH)
        if(DEFINED IE_VERSION_${ver})
            set(CPACK_PACKAGE_VERSION_${ver} ${IE_VERSION_${ver}})
        endif()
    endforeach()

    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenVINO toolkit")
    set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED OFF)
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
    # OFF - single archive, OFF - multiple packages per component
    # set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    set(CPACK_PACKAGE_VENDOR "Intel Corporation")
    set(CPACK_VERBATIM_VARIABLES ON)
    set(CPACK_COMPONENTS_ALL ${ARGN})
    set(CPACK_STRIP_FILES ON)
    set(CPACK_THREADS 8)

    if(OS_FOLDER)
        set(CPACK_SYSTEM_NAME "${OS_FOLDER}")
    endif()

    include(CPack)
endmacro()

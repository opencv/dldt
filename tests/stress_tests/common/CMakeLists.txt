# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME StressTestsCommon)

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

add_library(${TARGET_NAME} STATIC ${SRC} ${HDR})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#
# Adds compiler flags to C / C++ sources
#
macro(add_compiler_flags)
    foreach(flag ${ARGN})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}")
    endforeach()
endmacro()

if(EXISTS "${OpenVINO_SOURCE_DIR}/thirdparty/gflags")
    function(add_gflags)
        set(GFLAGS_IS_SUBPROJECT TRUE)
        set(HAVE_SYS_STAT_H 1)
        set(HAVE_INTTYPES_H 1)
        set(INTTYPES_FORMAT C99)
        set(BUILD_TESTING OFF)
        set(BUILD_SHARED_LIBS OFF)
        add_subdirectory(${OpenVINO_SOURCE_DIR}/thirdparty/gflags
                         ${CMAKE_CURRENT_BINARY_DIR}/gflags_build
                         EXCLUDE_FROM_ALL)
        set_target_properties(gflags_nothreads_static PROPERTIES FOLDER thirdparty)
    endfunction()
    add_gflags()
endif()

#
# Google Tests framework
#
if(EXISTS "${OpenVINO_SOURCE_DIR}/thirdparty/gtest")
    function(add_gtest_libraries)
        set(gtest_force_shared_crt ON CACHE BOOL "disable static CRT for google test")
        if (UNIX)
            add_compiler_flags(-Wno-undef)
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                add_compiler_flags(-Wno-deprecated-copy)
            endif()
        endif()

        set(BUILD_SHARED_LIBS OFF)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        add_subdirectory(${OpenVINO_SOURCE_DIR}/thirdparty/gtest
                         ${CMAKE_CURRENT_BINARY_DIR}/gtest_build
                         EXCLUDE_FROM_ALL)

        get_target_property(gtest_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
        set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gtest_include_dirs}")

        get_target_property(gmock_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
        set_target_properties(gmock PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gmock_include_dirs};${gmock_SOURCE_DIR}/include")

        # If we have specified /Z7 option, remove -Zi option which comes from gtest
        if (WIN32)
            set(gtest_targets "gtest;gtest_main;gmock;gmock_main")
            foreach(target_name ${gtest_targets})
                if(TARGET "${target_name}")
                    get_target_property(_target_cxx_flags ${target_name} COMPILE_OPTIONS)
                    if(_target_cxx_flags)
                        if(CMAKE_CXX_FLAGS_DEBUG MATCHES ".+/Z7.+" OR CMAKE_CXX_FLAGS_RELWITHDEBINFO MATCHES ".+/Z7.+")
                            string(REPLACE "-Zi" " " _target_cxx_flags ${_target_cxx_flags})
                            message(STATUS "Removing -Zi flag from target " ${target_name})
                            set_target_properties(${target_name} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
                        endif()
                    endif()
                endif()
            endforeach()
        endif()
        set_target_properties(gtest gtest_main gmock gmock_main
                              PROPERTIES FOLDER thirdparty)
    endfunction()
    add_gtest_libraries()
endif()

if(EXISTS "${OpenVINO_SOURCE_DIR}/thirdparty/pugixml")
    function(ie_build_pugixml)
        set(BUILD_SHARED_LIBS OFF)
        add_subdirectory(${OpenVINO_SOURCE_DIR}/thirdparty/pugixml
                         ${CMAKE_CURRENT_BINARY_DIR}/pugixml_build
                         EXCLUDE_FROM_ALL)
        set_property(TARGET pugixml-static PROPERTY EXPORT_NAME pugixml)
    endfunction()
    ie_build_pugixml()
endif()

target_link_libraries(${TARGET_NAME}
    PUBLIC
        gtest
        pugixml
        ${InferenceEngine_LIBRARIES}
        gflags
    PRIVATE
        gtest_main)

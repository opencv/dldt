﻿// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <fstream>
#include <string>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include <file_utils.h>
#include <details/ie_exception.hpp>

#ifndef _WIN32
# include <limits.h>
# include <unistd.h>
# include <dlfcn.h>
#else
# if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#  error "Only WINAPI_PARTITION_DESKTOP is supported, because of GetModuleHandleEx[A|W]"
# endif
# include <Windows.h>
#endif

long long FileUtils::fileSize(const char* charfilepath) {
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widefilename = FileUtils::multiByteCharToWString(charfilepath);
    const wchar_t* fileName = widefilename.c_str();
#else
    const char* fileName = charfilepath;
#endif
    std::ifstream in(fileName, std::ios_base::binary | std::ios_base::ate);
    return in.tellg();
}

namespace InferenceEngine {

namespace {

template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C> >
std::basic_string<C> getPathName(const std::basic_string<C>& s) {
    size_t i = s.rfind(FileUtils::FileTraits<C>::FileSeparator, s.length());
    if (i != std::string::npos) {
        return (s.substr(0, i));
    }

    return {};
}

}  // namespace

static std::string getIELibraryPathA() {
#if defined(_WIN32) && (!defined(WINAPI_FAMILY) || WINAPI_PARTITION_DESKTOP)
    CHAR ie_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPSTR>(getIELibraryPath), &hm)) {
        THROW_IE_EXCEPTION << "GetModuleHandle returned " << GetLastError();
    }
    GetModuleFileNameA(hm, (LPSTR)ie_library_path, sizeof(ie_library_path));
    return getPathName(std::string(ie_library_path));
#elif defined(__APPLE__) || defined(__linux__)
# ifdef USE_STATIC_IE
#  ifdef __APPLE__
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getIELibraryPath), &info);
    std::string path = getPathName(std::string(info.dli_fname)).c_str();
#  else
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    std::string path = getPathName(std::string(result, (count > 0) ? count : 0));
#  endif  // __APPLE__
    return FileUtils::makePath(path, std::string( "lib"));
# else
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getIELibraryPath), &info);
    return getPathName(std::string(info.dli_fname)).c_str();
# endif  // USE_STATIC_IE
#endif  // _WIN32
    return {};
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT

std::wstring getIELibraryPathW() {
#if defined(_WIN32) && (!defined(WINAPI_FAMILY) || WINAPI_PARTITION_DESKTOP)
    WCHAR ie_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(getIELibraryPath), &hm)) {
        THROW_IE_EXCEPTION << "GetModuleHandle returned " << GetLastError();
    }
    GetModuleFileNameW(hm, (LPWSTR)ie_library_path, sizeof(ie_library_path) / sizeof(ie_library_path[0]));
    return getPathName(std::wstring(ie_library_path));
#elif defined(__linux__) || defined(__APPLE__)
    return ::FileUtils::multiByteCharToWString(getIELibraryPathA().c_str());
#endif
    return {};
}

#endif  // ENABLE_UNICODE_PATH_SUPPORT

std::string getIELibraryPath() {
#ifdef ENABLE_UNICODE_PATH_SUPPORT
    return FileUtils::wStringtoMBCSstringChar(getIELibraryPathW());
#else
    return getIELibraryPathA();
#endif
}

}  // namespace InferenceEngine

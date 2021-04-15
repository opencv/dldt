// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include <windows.h>
#include <memory>
#include <vector>
#include "ie_system_conf.h"
#include "ie_parallel.hpp"
#include "ie_parallel_custom_arena.hpp"

namespace InferenceEngine {
int getNumberOfCPUCores(bool bigCoresOnly) {
    const int fallback_val = parallel_get_max_threads();
    DWORD sz = 0;
    // querying the size of the resulting structure, passing the nullptr for the buffer
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &sz) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return fallback_val;

    std::unique_ptr<uint8_t[]> ptr(new uint8_t[sz]);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr.get()), &sz))
        return fallback_val;

    int phys_cores = 0;
    size_t offset = 0;
    do {
        offset += reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr.get() + offset)->Size;
        phys_cores++;
    } while (offset < sz);

    #if TBB_HYBRID_CPUS_SUPPORT_PRESENT // TBB has hybrid CPU aware task_arena api
    auto core_types = custom::info::core_types();
    for (auto tp : core_types) {
        printf("type: %d %d concurency\n", tp, custom::info::default_concurrency(custom::task_arena::constraints{}.set_core_type(tp)));
    }
    if (bigCoresOnly && core_types.size() > 1) /*Hybrid CPU*/ {
        const auto little_cores = core_types.front();
        // assuming the Little cores feature no hyper-threading
        printf("original getNumberOfCPUCores: %d \n", phys_cores);
        phys_cores -= custom::info::default_concurrency(custom::task_arena::constraints{}.set_core_type(little_cores));
        // TODO: REMOVE THE DEBUG PRINTF
        printf("patched getNumberOfCPUCores: %d \n", phys_cores);
    }
    #endif
    return phys_cores;
}

#if !(IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
// OMP/SEQ threading on the Windows doesn't support NUMA
std::vector<int> getAvailableNUMANodes() { return std::vector<int>(1, 0); }
#endif

}  // namespace InferenceEngine

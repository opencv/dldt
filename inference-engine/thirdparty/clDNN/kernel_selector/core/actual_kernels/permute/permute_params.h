// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// permute_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct permute_params : public base_params {
    permute_params() : base_params(KernelType::PERMUTE) {}

    std::vector<uint16_t> order;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// permute_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct permute_optional_params : optional_params {
    permute_optional_params() : optional_params(KernelType::PERMUTE) {}
};
}  // namespace kernel_selector

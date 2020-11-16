﻿// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once

#include "activation_kernel_base.h"

#include <vector>

namespace kernel_selector {
class ActivationKernelRef : public ActivationKernelBase {
public:
    ActivationKernelRef() : ActivationKernelBase("activation_ref") {}
    virtual ~ActivationKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    JitConstants GetJitConstants(const activation_params& params, DispatchData dispatchData) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {FusedOpType::QUANTIZE,
                FusedOpType::SCALE,
                FusedOpType::ACTIVATION};
    }
};
}  // namespace kernel_selector

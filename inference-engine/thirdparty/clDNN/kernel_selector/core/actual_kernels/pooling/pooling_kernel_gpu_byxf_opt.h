// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pooling_kernel_base.h"

namespace kernel_selector {
class PoolingKernelGPUByxfOpt : public PoolingKernelBase {
public:
    PoolingKernelGPUByxfOpt() : PoolingKernelBase("pooling_gpu_byxf_opt") {}
    virtual ~PoolingKernelGPUByxfOpt() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params&, const optional_params&) const override;
    JitConstants GetJitConstants(const pooling_params& params, DispatchData dispatchData) const override;
    DispatchData SetDefault(const pooling_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }
};
}  // namespace kernel_selector

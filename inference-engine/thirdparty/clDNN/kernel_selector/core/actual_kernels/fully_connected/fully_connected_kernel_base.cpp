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


#include "fully_connected_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <vector>
#include <algorithm>

namespace kernel_selector {
JitConstants FullyConnectedKernelBase::GetJitConstants(const fully_connected_params& params,
                                                       const FullyConnectedKernelBase::DispatchData&) const {
    JitConstants jit = WeightBiasKernelBase::GetJitConstants(params);
    const auto& input = params.inputs[0];
    const auto x_size = input.LogicalSize() / input.Batch().v;

    jit.AddConstant(MakeJitConstant("INPUT0_ELEMENTS_COUNT", x_size));

    return jit;
}

FullyConnectedKernelBase::DispatchData FullyConnectedKernelBase::SetDefault(const fully_connected_params& params,
                                                                            int) const {
    DispatchData dispatchData;
    dispatchData.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    // Determine global work sizes.
    dispatchData.gws0 = params.output.LogicalSize();
    dispatchData.gws1 = dispatchData.gws2 = 1;

    // Find largest positive local work size that is divider for global work size.
    dispatchData.lws0 = std::min(std::max(dispatchData.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
    while (dispatchData.gws0 % dispatchData.lws0 != 0) {
        --dispatchData.lws0;
    }
    dispatchData.lws1 = dispatchData.lws2 = 1;

    return dispatchData;
}

KernelsData FullyConnectedKernelBase::GetCommonKernelsData(const Params &params,
                                                           const optional_params &options,
                                                           DataLayout dl,
                                                           WeightsLayout wl,
                                                           float estimated_time,
                                                           const std::string exeMode,
                                                           int autoTuneIndex) const {
    if (!Validate(params, options)) {
        return KernelsData();
    }

    const auto& orgParams = static_cast<const fully_connected_params&>(params);
    const auto& orgOptParams = static_cast<const fully_connected_optional_params&>(options);

    if (orgParams.inputs[0].GetLayout() == DataLayout::bfzyx && orgParams.inputs[0].GetLayout() != dl)
        return KernelsData();

    bool bProperInput = orgParams.inputs[0].GetLayout() == dl;
    if (!bProperInput && !orgParams.inputs[0].PitchesDifferFromLogicalDims()) {
        bProperInput = (dl == DataLayout::fb && orgParams.inputs[0].GetLayout() == DataLayout::fyxb) ||
                       (dl == DataLayout::bf && orgParams.inputs[0].GetLayout() == DataLayout::bfyx);
    }

    const bool bSupportedInput = orgOptParams.allowInputReordering || bProperInput;

    if (!bSupportedInput) {
        return KernelsData();
    }

    KernelData kd = KernelData::Default<fully_connected_params>(params);
    fully_connected_params& newParams = *static_cast<fully_connected_params*>(kd.params.get());

    if (!bProperInput) {
        newParams.inputs[0] = newParams.inputs[0].TransformIgnorePadding(dl);
        kd.reorderInput = true;
    }

    bool succeed = UpdateWeightsParams(newParams,
                                       options,
                                       wl,
                                       kd.weightsReorderParams,
                                       GetSupportedKey());

    if (!succeed) {
        return {};
    }

    kd.kernels.resize(1);

    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);

    const DispatchData runInfo = SetDefault(newParams, autoTuneIndex);
    auto cldnn_jit = GetJitConstants(newParams, runInfo);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    uint32_t fused_deps_total = 0;
    for (auto& fused_dep : newParams.fused_ops) {
        for (int i = 0; i < static_cast<int>(fused_dep.dep_size); i++) {
            fused_deps_total++;
        }
    }

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     runInfo,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     exeMode,
                     true,
                     !orgParams.bias.empty(),
                     1,
                     fused_deps_total);

    // TODO Pass estimated time only through DispatchData
    kd.estimatedTime = estimated_time;
    kd.autoTuneIndex = autoTuneIndex;
    return {kd};
}

std::string FullyConnectedKernelBase::GetAutoTuneOptions(int autoTuneIndex) const {
    if ((autoTuneIndex >= 0) && (autoTuneIndex < static_cast<int>(autoTuneOptions.size()))) {
        return autoTuneOptions[autoTuneIndex];
    }

    return DEFAULT;
}

KernelsData FullyConnectedKernelBase::GetTunedKernelsDataByIndex(const Params &params,
                                                                 const optional_params &options,
                                                                 DataLayout dl,
                                                                 WeightsLayout wl,
                                                                 float estimated_time,
                                                                 const int autoTuneIndex) const {
    return GetCommonKernelsData(params,
                                options,
                                dl,
                                wl,
                                estimated_time,
                                GetAutoTuneOptions(autoTuneIndex),
                                autoTuneIndex);
}


JitConstants FullyConnectedKernelBase::GetFusedPrimitivesJitConstants(const fully_connected_params&, const DispatchData&) const {
    return {};
}

bool FullyConnectedKernelBase::Validate(const Params& p, const optional_params&) const {
    const fully_connected_params& params = static_cast<const fully_connected_params&>(p);

    if (params.GetType() != KernelType::FULLY_CONNECTED) {
        return false;
    }

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

Datatype FullyConnectedKernelBase::GetActivationType(const fully_connected_params& params) const {
    if (params.quantization != QuantizationType::NONE)
        return Datatype::F32;

    return GetUnitType(params);
}

}  // namespace kernel_selector

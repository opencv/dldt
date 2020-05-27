/*
// Copyright (c) 2020 Intel Corporation
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
*/

#include "gemm_kernel_mmad_int8.h"

namespace kernel_selector {
ParamsKey GemmKernelMMADint8::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableTensorPitches();
    k.EnableQuantization(QuantizationType::SYMMETRIC);

    return k;
}

JitConstants GemmKernelMMADint8::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);
    GemmTuningData td = SetTuningParams(params);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", td.simd_size));
    jit.Merge(MakeTypeJitConstants(Datatype::INT32, "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(Datatype::F32, "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(params.inputs[0].GetDType() == Datatype::INT8 ? Datatype::INT32 : Datatype::UINT32, "PACKED_INPUT0"));
    jit.Merge(MakeTypeJitConstants(params.inputs[1].GetDType() == Datatype::INT8 ? Datatype::INT32 : Datatype::UINT32, "PACKED_INPUT1"));
    jit.AddConstant(MakeJitConstant("TILE_NUM", td.tile_num));
    jit.AddConstant(MakeJitConstant("TILE_SIZE_M", td.simd_size * td.tile_num));
    jit.AddConstant(MakeJitConstant("TILE_SIZE_N", td.simd_size));
    jit.AddConstant(MakeJitConstant("TILE_SIZE_K", td.simd_size * td.pack_size));
    jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS_M", td.size_m % (td.simd_size * td.tile_num)));
    jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS_N", td.size_n % td.simd_size));
    jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS_K", td.size_k % (td.simd_size * td.pack_size)));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf = { "", {"b", "f", "y", "x"}, "dequantized", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

GemmKernelBase::DispatchData GemmKernelMMADint8::SetDefault(const gemm_params& params) const {
    const auto& output = params.output;
    auto total_batches = output.LogicalSize() / (output.X().v * output.Y().v);

    DispatchData kd;
    GemmTuningData td = SetTuningParams(params);

    std::vector<size_t> global = { Align(output.X().v, td.simd_size),
                                   Align(output.Y().v, td.simd_size * td.tile_num) / (td.simd_size * td.tile_num),
                                   total_batches };

    std::vector<size_t> local = { td.simd_size, 1, 1 };

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

GemmKernelMMADint8::GemmTuningData GemmKernelMMADint8::InitGemmTuningData(const gemm_params& params) const {
    GemmTuningData tuning_data;

    tuning_data.size_m = params.output.Y().v;
    tuning_data.size_n = params.output.X().v;
    tuning_data.size_k = params.transpose_input0 ? params.inputs[0].Y().v : params.inputs[0].X().v;

    return tuning_data;
}

inline size_t GemmKernelMMADint8::GetMmadOperationsNumber(const GemmTuningData& tuning_data) const {
    return tuning_data.size_m * tuning_data.size_n * tuning_data.size_k;
}

bool GemmKernelMMADint8::HasLeftovers(const GemmTuningData& tuning_data, int tile_size) const {
    if (tile_size == 32) {
        return tuning_data.size_m % 32 || tuning_data.size_n % 16 || tuning_data.size_k % 64;
    } else if (tile_size == 16) {
        return tuning_data.size_m % 16 || tuning_data.size_n % 16 || tuning_data.size_k % 64;
    } else if (tile_size == 8) {
        return tuning_data.size_m % 8 || tuning_data.size_n % 8 || tuning_data.size_k % 32;
    } else {
        return true;
    }
}

GemmKernelMMADint8::GemmTuningData GemmKernelMMADint8::SetTuningParams(const gemm_params& params) const {
    GemmTuningData tuning_data = InitGemmTuningData(params);
    auto mmad_operations_number = GetMmadOperationsNumber(tuning_data);

    bool leftovers_simd16x2 = HasLeftovers(tuning_data, 16*2);
    bool leftovers_simd16 = HasLeftovers(tuning_data, 16);
    bool leftovers_simd8 = HasLeftovers(tuning_data, 8);

    bool small_matrices = mmad_operations_number <= 128 * 128 * 128;
    bool average_matrices = mmad_operations_number <= 448 * 448 * 448;
    bool very_big_matrices = mmad_operations_number >= 1024 * 1024 * 1024;
    bool no_input2 = params.inputs.size() == 3 ? false : true;

    size_t simd_size = 16;
    size_t tile_num = 1;

    if (!leftovers_simd16x2 && very_big_matrices && no_input2)
        { simd_size = 16; tile_num = 2; }
    else if (leftovers_simd16 && !leftovers_simd8)
        { simd_size = 8; }
    else if ((!params.transpose_input0 && !params.transpose_input1) && average_matrices)
        { simd_size = 8; }
    else if (small_matrices)
        { simd_size = 8; }
    else
        { simd_size = 16; }

    tuning_data.simd_size = simd_size;
    tuning_data.tile_num = tile_num;

    return tuning_data;
}

KernelsData GemmKernelMMADint8::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return KernelsData();
    }

    const auto& prim_params = static_cast<const gemm_params&>(params);

    auto run_info = GemmKernelMMADint8::SetDefault(prim_params);
    KernelData k_data = KernelData::Default<gemm_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel,
                     run_info,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     DEFAULT,
                     false,
                     false,
                     (uint32_t)prim_params.inputs.size(),
                     GetFusedPrimitiveInputsCount(params));

    GemmTuningData tuning_data = InitGemmTuningData(prim_params);
    auto mmad_operations_number = GetMmadOperationsNumber(tuning_data);

    k_data.estimatedTime = mmad_operations_number < 4096 ? DONT_USE_IF_HAVE_SOMETHING_ELSE : FORCE_PRIORITY_3;

    return {k_data};
}

bool GemmKernelMMADint8::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    const auto& gmm_params = static_cast<const gemm_params&>(params);
    auto input0_type = gmm_params.inputs[0].GetDType();
    auto input1_type = gmm_params.inputs[1].GetDType();

    if ((input0_type != Datatype::UINT8 && input0_type != Datatype::INT8) ||
        (input1_type != Datatype::UINT8 && input1_type != Datatype::INT8))
        return false;

    return true;
}
}  // namespace kernel_selector

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scale_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"
#include "cldnn/runtime/error_handler.hpp"

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct scale_gpu : typed_primitive_gpu_impl<scale> {
    using parent = typed_primitive_gpu_impl<scale>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<scale_gpu>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<scale>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);
        args.inputs = {instance.input_memory_ptr(), instance.scale_memory()};
        args.output = instance.output_memory_ptr();

        if (_outer.bias_term()) {
            args.inputs.push_back(instance.bias_memory());
        }
        return args;
    }

public:
    static primitive_impl* create(const scale_node& arg) {
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg);
        auto ew_optional_params =
            get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        ew_params.inputs.push_back(convert_data_tensor(arg.scale_in().get_output_layout()));

        ew_params.operations.push_back({{kernel_selector::eltwise_params::InputType::Buffer(0),
                                         kernel_selector::eltwise_params::InputType::Buffer(1)},
                                        kernel_selector::eltwise_mode::MUL});

        if (arg.bias_term()) {
            ew_params.inputs.push_back(convert_data_tensor(arg.bias().get_output_layout()));
            ew_params.operations.push_back({{kernel_selector::eltwise_params::InputType::Intermediate(0),
                                             kernel_selector::eltwise_params::InputType::Buffer(2)},
                                            kernel_selector::eltwise_mode::ADD});
        }

        ew_params.layoutBased = true;

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto scale = new scale_gpu(arg, best_kernels[0]);

        return scale;
    }
};

namespace detail {

attach_scale_gpu::attach_scale_gpu() {
    auto val_fw = scale_gpu::create;

    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::yxfb), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::byxf), val_fw);

    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);

    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), val_fw);

    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfwzyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfwzyx), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfwzyx), val_fw);

    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::b_fs_yx_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv16), val_fw);

    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::fs_b_yx_fsv32), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::fs_b_yx_fsv32), val_fw);

    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_yx_bsv16_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bs_fs_yx_bsv16_fsv16), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bs_fs_yx_bsv16_fsv16), val_fw);

    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv4), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv4), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv4), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv4), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::b_fs_yx_fsv4), val_fw);

    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv32), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv32), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv32), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv32), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::b_fs_yx_fsv32), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_zyx_fsv32), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_zyx_fsv32), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_zyx_fsv32), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_zyx_fsv32), val_fw);
    implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::b_fs_zyx_fsv32), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn

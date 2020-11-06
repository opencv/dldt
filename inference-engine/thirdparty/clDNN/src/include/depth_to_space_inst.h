/*
// Copyright (c) 2019-2020 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "cldnn/primitives/depth_to_space.hpp"
#include "primitive_inst.h"
#include "kernel_selector/core/actual_kernels/depth_to_space/depth_to_space_kernel_base.h"

#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<depth_to_space> : public typed_program_node_base<depth_to_space> {
    using parent = typed_program_node_base<depth_to_space>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::shared_ptr<kernel_selector::fuse_params> get_fuse_params() const override {
        return std::make_shared<kernel_selector::depth_to_space_fuse_params>();
    }
};

using depth_to_space_node = typed_program_node<depth_to_space>;

template <>
class typed_primitive_inst<depth_to_space> : public typed_primitive_inst_base<depth_to_space> {
    using parent = typed_primitive_inst_base<depth_to_space>;

public:
    static layout calc_output_layout(depth_to_space_node const& node);
    static std::string to_string(depth_to_space_node const& node);

public:
    typed_primitive_inst(network_impl& network, depth_to_space_node const& desc);
};

using depth_to_space_inst = typed_primitive_inst<depth_to_space>;
}  // namespace cldnn

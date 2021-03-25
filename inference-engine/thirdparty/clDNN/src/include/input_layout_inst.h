// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/input_layout.hpp"
#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {
struct memory_impl;

template <>
struct typed_program_node<input_layout> : public typed_program_node_base<input_layout> {
    using parent = typed_program_node_base<input_layout>;
    using parent::parent;

    typed_program_node(const std::shared_ptr<input_layout> prim, program_impl& prog);
};

using input_layout_node = typed_program_node<input_layout>;

template <>
class typed_primitive_inst<input_layout> : public typed_primitive_inst_base<input_layout> {
    using parent = typed_primitive_inst_base<input_layout>;

public:
    static layout calc_output_layout(input_layout_node const& node) { return node.get_primitive()->layout; }
    static std::string to_string(input_layout_node const& node);

public:
    typed_primitive_inst(network_impl& network, input_layout_node const& node);

    void set_data(memory_impl& mem);
};

using input_layout_inst = typed_primitive_inst<input_layout>;

}  // namespace cldnn

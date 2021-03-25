// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "data_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"

#include "json_object.h"
#include <string>
#include <memory>
#include <algorithm>

namespace cldnn {
primitive_type_id data::type_id() {
    static primitive_type_base<data> instance;
    return &instance;
}

namespace {
memory_impl::ptr attach_or_copy_data(network_impl& network, memory_impl& mem) {
    auto& engine = network.get_engine();
    if (mem.is_allocated_by(engine))
        return (memory_impl::ptr) &mem;

    memory_impl::ptr result = engine.allocate_memory(mem.get_layout(), network.get_id(), false);
    mem_lock<char> src(mem);
    mem_lock<char> dst(result);
    std::copy(src.begin(), src.end(), dst.begin());
    return result;
}
}  // namespace

data_node::typed_program_node(const std::shared_ptr<data> dprim, program_impl& prog)
    : parent(dprim, prog), mem(dprim->mem.get()) {
    constant = true;
    can_share_buffer(false);
    recalc_output_layout(false);
}

void data_node::attach_memory(memory_impl& new_mem, bool invalidate_users_if_changed) {
    mem = (memory_impl::ptr) &new_mem;
    recalc_output_layout(invalidate_users_if_changed);
}

std::string data_inst::to_string(data_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);
    return primitive_description.str();
}

data_inst::typed_primitive_inst(network_impl& network, data_node const& node)
    : parent(network, node, *attach_or_copy_data(network, node.get_attached_memory())) {}

}  // namespace cldnn

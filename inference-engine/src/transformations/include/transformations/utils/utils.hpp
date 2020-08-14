// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <assert.h>
#include <vector>
#include <limits>

#include <transformations_visibility.hpp>
#include <ngraph/op/util/op_annotations.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/opsets/opset3.hpp>

namespace ngraph {
namespace op {
namespace util {

template <class T>
bool normalize_single_value(std::vector<T> vec, float & value) {
    for (const auto & val : vec) {
        if (val != *vec.begin()) return false;
    }

    float ref_val = *vec.begin();

    if (ref_val < std::numeric_limits<float>::lowest() || ref_val > std::numeric_limits<float>::max()) {
        return false;
    }

    value = ref_val;
    return true;
}

template <class T>
bool has_op_with_type(const std::shared_ptr<const ngraph::Function> &function) {
    for (const auto & op : function->get_ops()) {
        if (std::dynamic_pointer_cast<T>(op)) {
            return true;
        }
    }
    return false;
}

inline std::string create_ie_output_name(const ngraph::Output<ngraph::Node>& output) {
    const auto& prev_layer = output.get_node_shared_ptr();
    std::string out_name = prev_layer->get_friendly_name();
    if (prev_layer->get_output_size() != 1)
        out_name += "." + std::to_string(output.get_index());
    return out_name;
}

TRANSFORMATIONS_API bool get_single_value(const std::shared_ptr<op::v0::Constant> & const_node, float & value);

TRANSFORMATIONS_API std::shared_ptr<ngraph::Node> normalize_constant(const std::shared_ptr<op::v0::Constant> & constant,
                                                                           const PartialShape & shape);

TRANSFORMATIONS_API std::shared_ptr<ngraph::Node> broadcastTo(const Output<Node>& input, const Shape& shape);

TRANSFORMATIONS_API std::shared_ptr<ngraph::Node> reshapeTo(const Output<Node> & input, const Shape& shape);

TRANSFORMATIONS_API bool constantIsEqualTo(const std::shared_ptr<ngraph::op::v0::Constant>& const_node, float value, float eps = 1e-5);

TRANSFORMATIONS_API bool has_f16_constants(const std::shared_ptr<const ngraph::Function> &function);

}  // namespace util
}  // namespace op
}  // namespace ngraph

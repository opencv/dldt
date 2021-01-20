// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/selu_ie.hpp"

#include <algorithm>
#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "../itt.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::SeluIE::type_info;

op::SeluIE::SeluIE(const Output<Node> & input,
                   const float alpha,
                   const float gamma)
        : Op({input}), gamma(gamma), alpha(alpha) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::SeluIE::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(SeluIE_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<SeluIE>(new_args.at(0), alpha, gamma);
}

void op::SeluIE::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(SeluIE_validate_and_infer_types);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool op::SeluIE::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(SeluIE_visit_attributes);
    visitor.on_attribute("alpha", alpha);
    visitor.on_attribute("gamma", gamma);
    return true;
}

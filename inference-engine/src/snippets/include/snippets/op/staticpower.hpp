// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/op/op.hpp>
#include <ngraph/op/power.hpp>
#include <snippets/snippets_isa.hpp>

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface PowerStatic
 * @brief Generated by Canonicalization for a spasical case of power innstruction which has constant power value
 * @ingroup snippets
 */
class TRANSFORMATIONS_API PowerStatic : public ngraph::op::v1::Power {
public:
    NGRAPH_RTTI_DECLARATION;

    PowerStatic() : Power() {
    }

    PowerStatic(const Output<Node>& arg0,
            const Output<Node>& arg1,
            const ngraph::op::AutoBroadcastSpec& auto_broadcast =
                ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY)) : Power(arg0, arg1, auto_broadcast) {
        NGRAPH_CHECK(!!std::dynamic_pointer_cast<ngraph::snippets::op::Scalar>(arg1.get_node_shared_ptr()), "second argument must be scalar constant.");
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<PowerStatic>(new_args.at(0), new_args.at(1), this->get_autob());
    }
};

} // namespace op
} // namespace snippets
} // namespace ngraph
// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/leaky_relu_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::LeakyReluFusion, "LeakyReluFusion", 0);

ngraph::pass::LeakyReluFusion::LeakyReluFusion() {
    MATCHER_SCOPE(LeakyReluFusion);
    auto data_pattern = ngraph::pattern::any_input();
    auto alpha_patter = ngraph::pattern::any_input(pattern::has_static_shape());
    auto multiply_pattern = ngraph::pattern::wrap_type<opset8::Multiply>({data_pattern, alpha_patter}, pattern::consumers_count(1));
    auto max_pattern = ngraph::pattern::wrap_type<opset8::Maximum>({data_pattern, multiply_pattern}, pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map.at(data_pattern);
        const auto & original_alpha_patter = pattern_map.at(alpha_patter);
        
        if (shape_size(original_alpha_patter.get_shape()) != 1)
            return false;

        auto leaky_relu = register_new_node<ngraph::opset8::PRelu>(data, original_alpha_patter);
        auto maximum = pattern_map.at(max_pattern);
        leaky_relu->set_friendly_name(maximum.get_node()->get_friendly_name());

        copy_runtime_info({
                            pattern_map.at(multiply_pattern).get_node_shared_ptr(),
                            maximum.get_node_shared_ptr()
                          },
                          leaky_relu);
        replace_node(maximum.get_node_shared_ptr(), leaky_relu);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(max_pattern, matcher_name);
    this->register_matcher(m, callback);
}

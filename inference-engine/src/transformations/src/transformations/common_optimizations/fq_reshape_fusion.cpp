// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_reshape_fusion.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>

ngraph::pass::FakeQuantizeReshapeFusion::FakeQuantizeReshapeFusion() {
    const auto fq_node_p = ngraph::pattern::wrap_type<opset4::FakeQuantize>(
            {ngraph::pattern::wrap_type<opset4::Constant>(), // for weights only
             ngraph::pattern::any_input(),
             ngraph::pattern::any_input(),
             ngraph::pattern::any_input(),
             ngraph::pattern::any_input()},
            pattern::consumers_count(1));
    const auto reshape_node_p = ngraph::pattern::wrap_type<opset4::Reshape>(
            {fq_node_p, ngraph::pattern::any_input()}, pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
        const auto &pattern_map = m.get_pattern_value_map();
        const auto fq_node = pattern_map.at(fq_node_p).get_node_shared_ptr();
        if (fq_node->is_dynamic())
            return false;
        const auto &reshape_node = pattern_map.at(reshape_node_p).get_node_shared_ptr();
        const auto &reshape_pattern_node = reshape_node->get_input_node_shared_ptr(1);
        const auto &original_data_rank = fq_node->get_input_shape(0).size();
        OutputVector renewed_inputs = {reshape_node->clone_with_new_inputs({fq_node->input_value(0), reshape_pattern_node})};
        for (auto i = 1; i < 5; ++i) {
            Output<Node> limit_input = fq_node->input_value(i);
            auto limit_shape = limit_input.get_shape();
            NGRAPH_CHECK(limit_shape.size() <= original_data_rank, "FakeQuantize limit input has unexpected rank");
            if (limit_shape.size() < original_data_rank) // aligning limit rank with data rank
                limit_shape.insert(limit_shape.begin(), original_data_rank - limit_shape.size(), uint64_t(1));
            NGRAPH_CHECK(limit_shape.size() == original_data_rank, "FakeQuantize limit input has unexpected rank");
            const auto &limit_size = std::accumulate(limit_shape.begin(), limit_shape.end(), 1, std::multiplies<int64_t>());
            const auto &max_element_idx = std::distance(limit_shape.begin(), std::max_element(limit_shape.begin(), limit_shape.end()));
            const auto &max_element = limit_shape[max_element_idx];
            if (max_element == limit_size) { // per-tensor / per-channel limit
                auto new_limit_shape = reshape_node->get_output_shape(0);
                std::transform(new_limit_shape.begin(), new_limit_shape.end(), new_limit_shape.begin(),
                               [max_element](uint64_t &dim) { return dim == max_element ? max_element : 1; });
                const auto &new_limit_size = std::accumulate(new_limit_shape.begin(), new_limit_shape.end(), 1, std::multiplies<int64_t>());
                if (new_limit_size == limit_size) { // we tracked future channel placement
                    if (new_limit_shape == limit_input.get_shape())
                        renewed_inputs.push_back(limit_input);
                    else
                        renewed_inputs.push_back(reshape_node->copy_with_new_inputs(
                                {limit_input, opset4::Constant::create(element::i64, {new_limit_shape.size()}, new_limit_shape)}));
                    continue;
                }
            }
            // resulting FQ will become or already is more than per-tensor / per-channel
            return false;
        }
        for (auto &new_input : renewed_inputs)
            copy_runtime_info({reshape_node, fq_node}, new_input.get_node_shared_ptr());
        const auto new_fq_node = fq_node->clone_with_new_inputs(renewed_inputs);
        replace_node(reshape_node, new_fq_node);
        new_fq_node->set_friendly_name(fq_node->get_friendly_name());
        copy_runtime_info({fq_node, reshape_node}, new_fq_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_node_p, "FakeQuantizeReshapeFusion");
    this->register_matcher(m, callback);
}

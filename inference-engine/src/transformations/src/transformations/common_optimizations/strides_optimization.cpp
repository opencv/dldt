// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include <transformations/common_optimizations/strides_optimization.hpp>
#include <transformations/rt_info/strides_property.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/variant.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::StridesOptimization, "StridesOptimization", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvStridesPropagation, "ConvStridesPropagation", 0);

static bool can_propagate_conv_stride(const std::shared_ptr<ngraph::Node>& conv) {
    const auto& kernel_shape = conv->input_value(1).get_shape();
    return std::all_of(kernel_shape.begin() + 2, kernel_shape.end(), [] (size_t s) -> bool { return s == 1; });
}

static std::tuple<ngraph::Strides, bool> check_next_ops(const std::vector<ngraph::Input<ngraph::Node>>& next_ops) {
    std::vector<ngraph::Strides> strides;
    for (const auto& op : next_ops) {
        if (!has_strides_prop(op)) {
            return std::make_tuple(ngraph::Strides{}, false);
        }
        strides.push_back(get_strides_prop(op));
    }
    bool all_ops_are_valid = std::all_of(strides.begin(), strides.end(),
                                         [&strides] (const ngraph::Strides& s) -> bool {
                                             bool all_ones = std::all_of(s.begin(), s.end(), [] (size_t i) -> bool { return i == 1; });
                                             return s == strides[0] && !all_ones;
                                        });
    return std::make_tuple(strides[0], all_ops_are_valid);
}

static void insert_pooling(const ngraph::Output<ngraph::Node>& first, ngraph::Input<ngraph::Node>& second, const ngraph::Strides& strides) {
    auto first_node = first.get_node_shared_ptr();
    auto rank = first.get_partial_shape().rank();
    if (rank.is_static() && static_cast<size_t>(rank.get_length()) < strides.size() + 2) {
        size_t diff = strides.size() + 2 - static_cast<size_t>(rank.get_length());
        auto ones = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{diff}, std::vector<int64_t>(diff, 1));
        auto current_shape = std::make_shared<ngraph::opset7::ShapeOf>(first);
        auto new_shape = std::make_shared<ngraph::opset7::Concat>(ngraph::OutputVector{ones, current_shape}, 0);
        first_node = std::make_shared<ngraph::opset7::Reshape>(first_node, new_shape, false);
    }
    auto pool = std::make_shared<ngraph::opset7::MaxPool>(first_node, strides, ngraph::Shape{}, ngraph::Shape{}, ngraph::Shape(strides.size(), 1));
    second.replace_source_output(pool);
}

static void handle_not_equal_stride_props(std::vector<ngraph::Input<ngraph::Node>>&& next_ops) {
    for (auto& op : next_ops) {
        if (!has_strides_prop(op))
            continue;
        auto strides = get_strides_prop(op);
        bool are_strides_ones = std::all_of(strides.begin(), strides.end(),
                                            [] (size_t s) -> bool { return s == 1; });
        if (!are_strides_ones) {
            auto conv = dynamic_cast<ngraph::opset7::Convolution*>(op.get_node());
            if (conv) {
                conv->set_strides(strides);
            } else {
                insert_pooling(op.get_source_output(), op, strides);
            }
        }
    }
}

ngraph::pass::ConvStridesPropagation::ConvStridesPropagation() {
    MATCHER_SCOPE(ConvStridesPropagation);
    auto data = pattern::any_input();
    auto weights = pattern::any_input(pattern::has_static_shape());
    auto conv_pattern = pattern::wrap_type<opset7::Convolution>({data, weights});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto conv = std::dynamic_pointer_cast<opset7::Convolution>(m.get_match_root());
        if (!conv)
            return false;

        auto conv_strides = conv->get_strides();
        Strides strides_ones(conv_strides.size(), 1);
        auto next_ops = op::util::get_node_target_inputs(conv);
        bool all_ops_are_valid;
        Strides strides;
        std::tie(strides, all_ops_are_valid) = check_next_ops(next_ops);

        if (!all_ops_are_valid) {
            handle_not_equal_stride_props(std::move(next_ops));
        } else {
            std::transform(conv_strides.begin(), conv_strides.end(), strides.begin(), conv_strides.begin(),
                    [] (size_t s1, size_t s2) -> size_t { return s1 * s2; });
        }

        if (can_propagate_conv_stride(conv)) {
            conv->set_strides(strides_ones);
            insert_strides_prop(conv->input(0), conv_strides);
        } else {
            conv->set_strides(conv_strides);
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::SupportedNodesStridesPropagation, "SupportedNodesStridesPropagation", 0);

ngraph::pass::SupportedNodesStridesPropagation::SupportedNodesStridesPropagation() {
    MATCHER_SCOPE(SupportedNodesStridesPropagation);
    auto root = std::make_shared<pattern::op::Or>(
            OutputVector{pattern::wrap_type<op::util::UnaryElementwiseArithmetic>(),
                         pattern::wrap_type<op::util::BinaryElementwiseArithmetic>()});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        auto next_ops = op::util::get_node_target_inputs(node);
        bool all_ops_are_valid;
        Strides strides;
        std::tie(strides, all_ops_are_valid) = check_next_ops(next_ops);

        if (!all_ops_are_valid) {
            return false;
        }

        for (auto& input : node->inputs()) {
            insert_strides_prop(input, strides);
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(root, matcher_name);
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::UnsupportedNodesStridesPropagation, "UnsupportedNodesStridesPropagation", 0);

ngraph::pass::UnsupportedNodesStridesPropagation::UnsupportedNodesStridesPropagation() {
    MATCHER_SCOPE(UnsupportedNodesStridesPropagation);
    auto root = pattern::any_input();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        auto next_ops = op::util::get_node_target_inputs(node);
        handle_not_equal_stride_props(std::move(next_ops));

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(root, matcher_name);
    this->register_matcher(m, callback);
}

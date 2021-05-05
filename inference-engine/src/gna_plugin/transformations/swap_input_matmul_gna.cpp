// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <numeric>
#include <transformations/swap_input_matmul_gna.hpp>


NGRAPH_RTTI_DEFINITION(SwapInputMatMul, "SwapInputMatMul", 0);

SwapInputMatMul::SwapInputMatMul() {
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset1::MatMul>({ngraph::pattern::any_input(
            ngraph::pattern::has_static_shape()), ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                                     ngraph::pattern::has_static_shape());
    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<ngraph::opset1::MatMul>(m.get_match_root());
        if (!matmul) {
            return false;
        }

        auto input_a = matmul->input(0).get_source_output();
        auto input_b = matmul->input(1).get_source_output();

        ngraph::Shape shape_input_a = input_a.get_shape();

        auto create_transpose = [this](ngraph::Output<ngraph::Node> node, const std::string& transpose_name) -> std::shared_ptr<ngraph::Node> {
            ngraph::Shape output_shape = node.get_node_shared_ptr()->get_shape();

            std::vector<size_t> transpose_order(output_shape.size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto transpose = register_new_node<ngraph::opset1::Transpose>(
                    node, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape {transpose_order.size()}, transpose_order));
            transpose->set_friendly_name(transpose_name);
            return transpose;
        };

        ngraph::NodeVector new_ops;

        // Skip FakeQuantize layers
        std::shared_ptr<ngraph::Node> input_a_skip_fq = input_a.get_node_shared_ptr();
        if (std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(input_a_skip_fq)) {
            input_a_skip_fq = input_a_skip_fq->input_value(0).get_node_shared_ptr();
        }

        std::shared_ptr<ngraph::Node> input_b_skip_fq = input_b.get_node_shared_ptr();
        if (std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(input_b_skip_fq)) {
            input_b_skip_fq = input_b_skip_fq->input_value(0).get_node_shared_ptr();
        }

        if (std::dynamic_pointer_cast<ngraph::opset1::Constant>(input_a_skip_fq) &&
            !std::dynamic_pointer_cast<ngraph::opset1::Constant>(input_b_skip_fq)) {
            if (shape_input_a[0] < 8 || ((shape_input_a[0] % 8 != 0 || shape_input_a[1] % 8 != 0))) {
                return false;
            }
            auto new_matmul = std::make_shared<ngraph::opset1::MatMul>(input_b, input_a, !matmul->get_transpose_b(), !matmul->get_transpose_a());
            new_matmul->set_friendly_name(matmul->get_friendly_name() + "/swap_inputs");
            new_ops.push_back(new_matmul);
            auto traspose_output = create_transpose(new_matmul,  matmul->get_friendly_name());
            new_ops.push_back(traspose_output);

            ngraph::copy_runtime_info(matmul, new_ops);
            ngraph::replace_node(matmul, traspose_output);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "SwapInputMatMul");
    this->register_matcher(m, callback);
}

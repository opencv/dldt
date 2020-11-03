// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_depth_to_space.hpp"
#include "transformations/itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertDepthToSpace, "ConvertDepthToSpace", 0);

ngraph::pass::ConvertDepthToSpace::ConvertDepthToSpace() {
    IETRANSFORM_SCOPE(ConvertDepthToSpace,
        auto dts_node = ngraph::pattern::wrap_type<ngraph::opset1::DepthToSpace>();
        ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
            auto dts_node = std::dynamic_pointer_cast<ngraph::opset1::DepthToSpace> (m.get_match_root());
            if (!dts_node || m_transformation_callback(dts_node)) {
                return false;
            }

            auto input = dts_node->input(0).get_source_output().get_node_shared_ptr();

            /*
            * In this transformation we decompose DepthToSpace operation to the next sequence of ops:
            * Reshape(shape_begin)->Transpose(order)->Reshape(shape_end)
            *
            * if mode equal to blocks_first
            * shape_begin = [N, block_size, block_size, ..., block_size, C / (block_size ^ K), D1, D2, ..., DK]
            *
            * if mode equal to depth_first
            * shape_begin = [N, C / (block_size ^ K), block_size, block_size, ..., block_size, D1, D2, ..., DK]
            *
            */

            auto input_shape = dts_node->input(0).get_shape();
            auto spatial_dims = input_shape.size() - 2;
            auto block_size = dts_node->get_block_size();
            auto mode = dts_node->get_mode();

            // Calculate Reshape shape_begin
            std::vector<int64_t> shape_begin{static_cast<int64_t>(input_shape[0])};
            auto C = input_shape[1];
            for (size_t i = 0; i < spatial_dims; ++i) {
                shape_begin.push_back(block_size);
                C /= block_size;
            }

            switch (mode) {
                case ngraph::op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
                    shape_begin.push_back(C);
                    break;
                case ngraph::op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
                    shape_begin.insert(shape_begin.begin() + 1, C);
                    break;
            }

            for (size_t i = 0; i < spatial_dims; ++i) {
                shape_begin.push_back(input_shape[2 + i]);
            }

            // Calculate Transpose order
            std::vector<int64_t> order{0};
            switch (mode) {
                case ngraph::op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
                    order.push_back(spatial_dims + 1);
                    for (size_t i = 1; i <= spatial_dims; ++i) {
                        order.push_back(spatial_dims + 1 + i);
                        order.push_back(i);
                    }
                    break;
                case ngraph::op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
                    order.push_back(1);
                    for (size_t i = 1; i <= spatial_dims; ++i) {
                        order.push_back(spatial_dims + 1 + i);
                        order.push_back(i + 1);
                    }
                    break;
            }

            // Calculate Reshape shape_end
            std::vector<int64_t> shape_end{static_cast<int64_t>(input_shape[0]), static_cast<int64_t>(C)};
            for (size_t i = 0; i < spatial_dims; ++i) {
                shape_end.push_back(block_size * input_shape[2 + i]);
            }

            auto create_constant = [](std::vector<int64_t > & v) -> std::shared_ptr<op::Constant> {
                return op::Constant::create(element::i64, Shape{v.size()}, v);
            };

            auto reshape_begin = std::make_shared<ngraph::opset1::Reshape>(input, create_constant(shape_begin), true);
            auto transpose = std::make_shared<ngraph::opset1::Transpose>(reshape_begin, create_constant(order));
            auto reshape_end = std::make_shared<ngraph::opset1::Reshape>(transpose, create_constant(shape_end), true);
            reshape_end->set_friendly_name(dts_node->get_friendly_name());
            ngraph::copy_runtime_info(dts_node, {reshape_begin, transpose, reshape_end});
            ngraph::replace_node(dts_node, reshape_end);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(dts_node, matcher_name);
        this->register_matcher(m, callback);
        return;
    )
}

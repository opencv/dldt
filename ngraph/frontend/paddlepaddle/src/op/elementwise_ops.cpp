//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <map>

#include <ngraph/opsets/ngraph::opset6.hpp>
#include "elementwise_ops.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                template <typename T>
                NamedOutputs elementwise_ops(const NodeContext& node)
                {
                    auto x = node.get_ng_input("X");
                    auto y = node.get_ng_input("Y");

                    auto axis = node.get_attribute<int>("axis");

                    PDPD_NODE_VALIDATION_CHECK(ngraph::frontend::ErrorCode::OP_VALIDATION_FAILED,
                                               node,
                                               x.get_partial_shape().rank().is_static(),
                                               "elementwise_ops: X rank must be static!");
                    PDPD_NODE_VALIDATION_CHECK(ngraph::frontend::ErrorCode::OP_VALIDATION_FAILED,
                                               node,
                                               y.get_partial_shape().rank().is_static(),
                                               "elementwise_ops: Y rank must be static!");
                    int64_t x_rank = x.get_partial_shape().rank().get_length();
                    int64_t y_rank = y.get_partial_shape().rank().get_length();

                    if ((axis == -1) || (axis == x_rank - 1) || (x_rank == y_rank))
                    {
                        return node.default_single_output_mapping({std::make_shared<T>(x, y)},
                                                                  {"Out"});
                    }
                    else
                    {
                        // This broadcast can be implemented by either ngraph::Reshape or
                        // ngraph::Broadcast. Since PDPD implicates y_shape is a subsequence of
                        // x_shape starting from axis, to use ngraph::Reshape like Paddle2ONNX,
                        // which is more friendly to PnP.
                        auto broadcast_shape = std::vector<int64_t>(x_rank, 1);
                        PartialShape y_shape = y.get_partial_shape();
                        int32_t i = 0;
                        for (auto it = y_shape.begin(); it != y_shape.end(); ++i, ++it)
                            broadcast_shape[axis + i] = (*it).get_length();

                        auto reshape_node = ngraph::opset6::Constant::create(
                            element::i64, Shape{broadcast_shape.size()}, broadcast_shape);
                        auto y_node = std::make_shared<ngraph::opset6::Reshape>(y, reshape_node, false);
                        return node.default_single_output_mapping({std::make_shared<T>(x, y_node)},
                                                                  {"Out"});
                    }
                }

                //
                NamedOutputs elementwise_add(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Add>(node_context);
                }

                NamedOutputs elementwise_sub(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Subtract>(node_context);
                }

                NamedOutputs elementwise_mul(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Multiply>(node_context);
                }

                NamedOutputs elementwise_div(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Divide>(node_context);
                }

                NamedOutputs elementwise_min(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Minimum>(node_context);
                }

                NamedOutputs elementwise_max(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Maximum>(node_context);
                }

                NamedOutputs elementwise_pow(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Power>(node_context);
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
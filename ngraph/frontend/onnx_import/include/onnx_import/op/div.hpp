//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#pragma once

#include <memory>

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"
#include "onnx_import/core/node.hpp"
#include "onnx_import/default_opset.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline OutputVector div(const Node& node)
                {
                    const Output<ngraph::Node> lhs_node = node.get_ng_inputs().at(0);
                    Output<ngraph::Node> rhs_node = node.get_ng_inputs().at(1);
                    auto lhs_rank = lhs_node.get_shape().size();
                    auto rhs_rank = rhs_node.get_shape().size();
                    auto axis = node.get_attribute_value<std::int64_t>("axis", lhs_rank - rhs_rank);
                    bool broadcast = node.get_attribute_value<std::int64_t>("broadcast", 0);
                    if (broadcast)
                    {
                        // Unidirectional broadcast right node to left shape.
                        rhs_node = std::make_shared<default_opset::Broadcast>(
                            rhs_node,
                            std::make_shared<default_opset::ShapeOf>(lhs_node),
                            default_opset::Constant::create(element::i64, Shape{1}, {axis}));
                    }
                    return {std::make_shared<default_opset::Divide>(
                        lhs_node, rhs_node, ngraph::op::AutoBroadcastSpec::NONE)};
                }

            } // namespace set_1

            namespace set_7
            {
                inline OutputVector div(const Node& node)
                {
                    return {std::make_shared<default_opset::Divide>(node.get_ng_inputs().at(0),
                                                                    node.get_ng_inputs().at(1))};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph

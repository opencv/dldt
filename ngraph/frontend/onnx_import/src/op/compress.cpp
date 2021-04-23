// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/validation_util.hpp"
#include "op/compress.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector compress(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    auto condition = std::make_shared<default_opset::Convert>(node.get_ng_inputs().at(1), element::u8);

                    int64_t axis = 0;
                    if(node.has_attribute("axis"))
                    {
                        axis = node.get_attribute_value<int64_t>("axis");
                        axis = ngraph::normalize_axis(node.get_description(),
                               axis,
                               data.get_partial_shape().rank());
                    }
                    else
                    {
                        data = std::make_shared<default_opset::Squeeze>(ngraph::builder::opset1::flatten(data, axis));
                    }
                    auto axis_node = default_opset::Constant::create(element::i64, Shape{}, {axis});
                    auto zero_node = default_opset::Constant::create(element::i64, Shape{}, {0});
                    auto result = std::make_shared<default_opset::Gather>(
                        data, 
                        std::make_shared<default_opset::Squeeze>(
                            std::make_shared<default_opset::NonZero>(
                                condition
                            ),
                            zero_node
                        ),
                        axis_node
                    );

                    return {result};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph

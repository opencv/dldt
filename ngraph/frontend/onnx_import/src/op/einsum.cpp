// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "op/einsum.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_12
            {
                OutputVector einsum(const Node& node)
                {
                    const std::string& equation{
                        node.get_attribute_value<std::string>("equation")};

                    return OutputVector{std::make_shared<default_opset::Einsum>(node.get_ng_inputs(), equation)};
                }

            } // namespace set_12

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph

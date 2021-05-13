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

#include "transpose2.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs transpose2(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    auto perm = node.get_attribute<std::vector<int>>("axis");

                    auto rank =
                        static_cast<unsigned long>(data.get_partial_shape().rank().get_length());

                    std::cout << perm.size() << std::endl;
                    std::cout << data.get_partial_shape().rank() << ":" << rank << std::endl;

                    PDPD_NODE_VALIDATION_CHECK(ngraph::frontend::ErrorCode::OP_VALIDATION_FAILED,
                                               node,
                                               perm.size() == rank,
                                               "transpose2: axis size must equal to data rank!");

                    auto input_order = opset6::Constant::create(element::i64, {rank}, perm);
                    return node.default_single_output_mapping(
                        {std::make_shared<opset6::Transpose>(data, input_order)}, {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph

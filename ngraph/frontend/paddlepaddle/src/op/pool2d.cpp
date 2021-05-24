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

#include "pool2d.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs pool2d(const NodeContext& node)
                {
                    // TODO : resolve padding according to spec
                    auto data = node.get_ng_input("X");
                    auto pooling_type = node.get_attribute<std::string>("pooling_type");
                    auto global_pooling = node.get_attribute<bool>("global_pooling");
                    auto adaptive = node.get_attribute<bool>("adaptive");
                    auto kernel_shape = node.get_attribute<std::vector<int32_t>>("ksize");
                    if (pooling_type == "max" && !global_pooling)
                    {
                        auto strides = node.get_attribute<std::vector<int32_t>>("strides");
                        auto paddings = node.get_attribute<std::vector<int32_t>>("paddings");
                        auto rounding_type = node.get_attribute<bool>("ceil_mode")
                                                 ? ngraph::op::RoundingType::CEIL
                                                 : ngraph::op::RoundingType::FLOOR;
                        return node.default_single_output_mapping(
                            {std::make_shared<ngraph::opset6::MaxPool>(
                                data,
                                ngraph::Strides(strides.begin(), strides.end()),
                                ngraph::Shape(paddings.begin(), paddings.end()),
                                ngraph::Shape(paddings.begin(), paddings.end()),
                                ngraph::Shape(kernel_shape.begin(), kernel_shape.end()),
                                rounding_type)},
                            {"Out"});
                    }
                    else if (pooling_type == "avg" &&
                             (global_pooling ||
                              (adaptive && all_of(kernel_shape.begin(),
                                                  kernel_shape.end(),
                                                  [](int32_t s) { return s == 1; }))))
                    {
                        // TODO : resolve axes according to rank
                        auto axes =
                            ngraph::opset6::Constant::create(ngraph::element::i64, {2}, {2, 3});
                        return node.default_single_output_mapping(
                            {std::make_shared<ngraph::opset6::ReduceMean>(data, axes, true)},
                            {"Out"});
                    }
                    else
                    {
                        throw std::runtime_error("Unsupported pooling type");
                    }
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
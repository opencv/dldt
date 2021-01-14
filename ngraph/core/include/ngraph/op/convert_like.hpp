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

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise type conversion operation.
            class NGRAPH_API ConvertLike : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ConvertLike", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a conversion operation.
                ConvertLike() = default;
                /// \brief Constructs a conversion operation.
                /// \param data  Node that produces the input tensor.
                /// \param like  Node which provides the target type information for the conversion.
                ConvertLike(const Output<Node>& data, const Output<Node>& like);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
    }
}

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

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Batched average pooling operation.
            ///
            class NGRAPH_API AvgPool : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs a batched average pooling operation.
                AvgPool() = default;

                ///
                /// \brief      Constructs a batched average pooling operation.
                ///
                /// \param      arg            The output producing the input data batch tensor.<br>
                ///                            `[d1, dn]`
                /// \param      strides        The strides.<br> `[n]`
                /// \param      pads_begin     The beginning of padding shape.<br> `[n]`
                /// \param      pads_end       The end of padding shape.<br> `[n]`
                /// \param      kernel         The kernel shape.<br> `[n]`
                /// \param      exclude_pad    If false then averages include padding elements, each
                ///                            treated as the number zero.  If true, padding
                ///                            elements
                ///                            are entirely ignored when computing averages.
                /// \param      rounding_type  Whether to use ceiling or floor rounding type while
                ///                            computing output shape.
                /// \param      auto_pad       Padding type to use for additional padded dimensions
                ///
                AvgPool(const Output<Node>& arg,
                        const Strides& strides,
                        const Shape& pads_begin,
                        const Shape& pads_end,
                        const Shape& kernel,
                        bool exclude_pad,
                        op::RoundingType rounding_type,
                        const PadType& auto_pad);

                ///
                /// \brief      Constructs a batched average pooling operation.
                ///
                /// \param      arg            The output producing the input data batch tensor.<br>
                ///                            `[d1, dn]`
                /// \param      strides        The strides.<br> `[n]`
                /// \param      pads_begin     The beginning of padding shape.<br> `[n]`
                /// \param      pads_end       The end of padding shape.<br> `[n]`
                /// \param      kernel         The kernel shape.<br> `[n]`
                /// \param      exclude_pad    If false then averages include padding elements, each
                ///                            treated as the number zero.  If true, padding
                ///                            elements
                ///                            are entirely ignored when computing averages.
                /// \param      rounding_type  Whether to use ceiling or floor rounding type while
                ///                            computing output shape.
                ///
                AvgPool(const Output<Node>& arg,
                        const Strides& strides,
                        const Shape& pads_begin,
                        const Shape& pads_end,
                        const Shape& kernel,
                        bool exclude_pad,
                        op::RoundingType rounding_type);

                size_t get_version() const override { return 1; }
                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The kernel shape.
                const Shape& get_kernel() const;
                void set_kernel(const Shape& kernel);
                /// \return The strides.
                const Strides& get_strides() const;
                void set_strides(const Strides& strides);
                /// \return The beginning of padding shape.
                const Shape& get_pads_begin() const;
                void set_pads_begin(const Shape& pads_begin);
                /// \return The end of padding shape.
                const Shape& get_pads_end() const;
                void set_pads_end(const Shape& pads_end);
                bool get_exclude_pad() const;
                void set_exclude_pad(bool exclude_pad);
                /// \return The pad type for pooling.
                const PadType& get_auto_pad() const;
                void set_auto_pad(const PadType& auto_pad);
                op::RoundingType get_rounding_type() const;
                void set_rounding_type(op::RoundingType rounding_type);
                /// \return The default value for AvgPool.
                virtual std::shared_ptr<Node> get_default_value() const override;

            protected:
                Shape m_kernel;
                Strides m_strides;
                Shape m_pads_begin;
                Shape m_pads_end;
                bool m_exclude_pad{true};
                PadType m_auto_pad{PadType::EXPLICIT};
                op::RoundingType m_rounding_type{op::RoundingType::FLOOR};
            };
        } // namespace v1

        using v1::AvgPool;
    } // namespace op
} // namespace ngraph

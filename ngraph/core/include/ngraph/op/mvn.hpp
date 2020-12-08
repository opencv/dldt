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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        NGRAPH_SUPPRESS_DEPRECATED_START

        namespace v0
        {
            /// \brief Operator performing Mean Variance Normalization
            ///
            class NGRAPH_API MVN : public ngraph::op::util::FusedOp
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                MVN() = default;
                /// \brief Constructs an MVN operation.
                ///
                /// \param data Input tensor with data
                /// \param normalize_variance flag that denotes whether to perform variance
                ///                           normalization.
                /// \param across_channels flag that denotes if mean values are shared across
                /// channels.
                /// \param eps the number to be added to the variance to avoid division by zero when
                ///            normalizing the value
                ///
                MVN(const Output<Node>& data,
                    bool across_channels = true,
                    bool normalize_variance = true,
                    double eps = 1e-9);

                /// \brief Constructs an MVN operation.
                ///
                /// \param data Input tensor with data
                /// \param reduction_axes A list of axes, along which to reduce.
                /// \param normalize_variance flag that denotes whether to perform variance
                ///                           normalization.
                /// \param eps the number to be added to the variance to avoid division by zero when
                ///            normalizing the value
                ///
                MVN(const Output<Node>& data,
                    AxisSet reduction_axes,
                    bool normalize_variance = true,
                    double eps = 1e-9);

                virtual OutputVector decompose_op() const override;

                virtual void validate_and_infer_types() override;

                virtual bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                double get_eps() const { return m_eps; }
                bool get_across_channels() const { return m_across_channels; }
                bool get_normalize_variance() const { return m_normalize_variance; }
                AxisSet get_reduction_axes() const { return m_reduction_axes; }
                void set_reduction_axes(AxisSet axes) { m_reduction_axes = axes; }
            private:
                double m_eps = 1e-9;
                bool m_across_channels;
                bool m_normalize_variance;
                AxisSet m_reduction_axes;
            };
        }
        using v0::MVN;

        NGRAPH_SUPPRESS_DEPRECATED_END

        /// \brief Specifies how eps is applied in MVN
        enum class MVNEpsMode
        {
            // Apply eps inside sqrt
            INSIDE_SQRT,
            // Apply eps outside sqrt
            OUTSIDE_SQRT
        };

        NGRAPH_API
        std::ostream& operator<<(std::ostream& s, const MVNEpsMode& type);

        namespace v6
        {
            /// \brief Operator performing Mean Variance Normalization
            ///
            class NGRAPH_API MVN : public ngraph::op::Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                MVN() = default;
                /// \brief Constructs an MVN operation.
                ///
                /// \param data Input tensor with data
                /// \param reduction_axes A list of axes, along which to reduce.
                /// \param normalize_variance flag that denotes whether to perform variance
                ///                           normalization.
                /// \param eps the number to be added to the variance to avoid division by zero when
                ///            normalizing the value
                /// \param eps_mode the mode of applying epsilon
                ///
                MVN(const Output<Node>& data,
                    const Output<Node>& reduction_axes,
                    bool normalize_variance = true,
                    double eps = 1e-9,
                    MVNEpsMode eps_mode = MVNEpsMode::INSIDE_SQRT);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                double get_eps() const { return m_eps; }
                bool get_normalize_variance() const { return m_normalize_variance; }
                MVNEpsMode get_eps_mode() const { return m_eps_mode; }
            private:
                bool m_normalize_variance = true;
                double m_eps = 1e-9;
                MVNEpsMode m_eps_mode = MVNEpsMode::INSIDE_SQRT;
            };
        } // namespace v6
    }     // namespace op

    template <>
    class NGRAPH_API AttributeAdapter<op::MVNEpsMode>
        : public EnumAttributeAdapterBase<op::MVNEpsMode>
    {
    public:
        AttributeAdapter(op::MVNEpsMode& value)
            : EnumAttributeAdapterBase<op::MVNEpsMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::MVNEpsMode>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph

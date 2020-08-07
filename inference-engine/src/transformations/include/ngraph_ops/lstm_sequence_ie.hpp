// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <transformations_visibility.hpp>

#include "ngraph/opsets/opset4.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
class TRANSFORMATIONS_API LSTMSequenceIE : public Op, public ngraph::op::util::RNNCellBase {
public:
    static constexpr NodeTypeInfo type_info{"LSTMSequenceIE", 1};

    const NodeTypeInfo &get_type_info() const override { return type_info; }

    LSTMSequenceIE() = delete;

    LSTMSequenceIE(const Output <Node> &X,
                   const Output <Node> &H_t,
                   const Output <Node> &C_t,
                   const Output <Node> &sequence_lengths,
                   const Output <Node> &B,
                   const Output <Node> &WR,
                   size_t hidden_size,
                   ngraph::opset4::LSTMSequence::direction lstm_direction,
                   const std::vector<std::string> &activations,
                   const std::vector<float> &activations_alpha,
                   const std::vector<float> &activations_beta,
                   float clip);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector &new_args) const override;

    void validate_and_infer_types() override;

    ngraph::opset4::LSTMSequence::direction get_direction() { return m_direction; }

    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    ngraph::opset4::LSTMSequence::direction m_direction;
};
}  // namespace op
}  // namespace ngraph

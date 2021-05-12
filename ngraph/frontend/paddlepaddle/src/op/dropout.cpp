// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "dropout.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs dropout (const NodeContext& node) {
    auto data = node.get_ng_input("X");
    auto dropout_prob = ngraph::opset6::Constant::create(ngraph::element::f32, {1}, {1 - node.get_attribute<float>("dropout_prob")});
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Multiply>(data, dropout_prob)}, {"Out"});
}

}}}}
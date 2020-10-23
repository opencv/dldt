// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fold_fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FoldFakeQuantizeFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& constShape,
    const std::vector<float>& constValues,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto constant = std::make_shared<ngraph::opset1::Constant>(precision, constShape, constValues);

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        constant, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);
    fakeQuantize->set_friendly_name("fakeQuantize");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{}, "FoldFakeQuantizeFunction");
}

std::shared_ptr<ngraph::Function> FoldFakeQuantizeFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& constShape,
    const std::vector<float>& constValues) {
    const std::shared_ptr<Node> constant = std::make_shared<ngraph::opset1::Constant>(precision, constShape, constValues);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(constant) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{}, "FoldFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph

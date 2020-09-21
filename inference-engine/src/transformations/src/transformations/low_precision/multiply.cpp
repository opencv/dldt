﻿// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/multiply.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/common/dequantization_op.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void MultiplyTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Multiply>(pass, context);
}

bool MultiplyTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto multiply = m.get_match_root();
    if (!LayerTransformation::canBeTransformed(context, multiply)) {
        return false;
    }

    multiply = separateInStandaloneBranch(multiply);
    auto newMultiply = multiply;

    auto fold_fake_quantizes = [](std::shared_ptr<Node>& multiply, const size_t index) {
        auto fakeQuantizeOnWeights = as_type_ptr<opset1::FakeQuantize>(multiply->get_input_node_shared_ptr(index));
        if (fakeQuantizeOnWeights != nullptr) {
            auto result = NetworkHelper::fold_fake_quantize(fakeQuantizeOnWeights);
            if (is_type<opset1::Constant>(result)) {
                replace_node(fakeQuantizeOnWeights, result);
            }
        }
    };

    fold_fake_quantizes(multiply, 0ul);
    fold_fake_quantizes(multiply, 1ul);

    const int fullPathIndex = getNotEmpty(multiply);

    if (fullPathIndex == -1) {
        const auto multiplyBranch = getMultiplyConstBranch(multiply);

        if (multiplyBranch.first == -1 || multiplyBranch.second == -1)
            return false;

        auto multiplyParent = multiply->get_input_node_shared_ptr(multiplyBranch.first);
        auto constParent = multiply->get_input_node_shared_ptr(multiplyBranch.first == 0 ? 1 : 0);
        auto multiplyParentParent = multiplyParent->get_input_node_shared_ptr(multiplyBranch.second);
        auto multiplyParentConst = multiplyParent->get_input_node_shared_ptr(multiplyBranch.second == 0 ? 1 : 0);

        newMultiply = std::make_shared<op::TypeRelaxed<opset1::Multiply>>(
            std::vector<ngraph::element::Type>{ element::f32, element::f32 },
            std::vector<ngraph::element::Type>{element::f32},
            ngraph::op::TemporaryReplaceOutputType(multiplyParentParent, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(fold<opset1::Multiply>(multiplyParentConst, constParent), element::f32).get());

        NetworkHelper::copyInfo(multiplyParent, newMultiply);
        NetworkHelper::copyInfo(multiply, newMultiply);
    } else {
        const int emptyPathIndex = fullPathIndex == 0 ? 1 : 0;

        FakeQuantizeDequantization dequantizationEmptyPath = NetworkHelper::getDequantization(multiply, emptyPathIndex);
        if ((updatePrecisions && !dequantizationEmptyPath.empty() && !dequantizationEmptyPath.isLowPrecision()) ||
            (dequantizationEmptyPath.multiply == nullptr && dequantizationEmptyPath.subtract == nullptr)) {
            return false;
        }

        FakeQuantizeDequantization dequantizationFullPath = NetworkHelper::getDequantization(multiply, fullPathIndex);
        if (updatePrecisions && !dequantizationFullPath.empty() && !dequantizationFullPath.isLowPrecision()) {
            return false;
        }

        std::shared_ptr<Node> subtractValuesEmptyPath;
        std::shared_ptr<Node> multiplyValuesEmptyPath;
        std::tie(subtractValuesEmptyPath, multiplyValuesEmptyPath) = NetworkHelper::createEmptyValues(dequantizationEmptyPath);

        // check if empty path shifts are not zero
        if (!NetworkHelper::isZeroConst(subtractValuesEmptyPath)) {
            return false;
        }

        std::shared_ptr<Node> subtractValuesFullPath;
        std::shared_ptr<Node> multiplyValuesFullPath;
        std::tie(subtractValuesFullPath, multiplyValuesFullPath) = NetworkHelper::createEmptyValues(dequantizationFullPath);


        // before: Y = (SC1 * (X1 - SH1)) * (SC2 * X2)
        // after : Y = (SC1' * (X1 - SH1)) * (X2) , where :
        //         SC1' = SC1 * SC2
        std::shared_ptr<Node> newMultiplyValuesFullPath = fold<opset1::Multiply>(multiplyValuesEmptyPath, multiplyValuesFullPath);
        std::vector<Output<Node>> inputs{ {}, {} };
        inputs[emptyPathIndex] = dequantizationEmptyPath.data;
        inputs[fullPathIndex] = std::make_shared<DequantizationMultiply>(
            dequantizationFullPath.subtract == nullptr ?
                (dequantizationFullPath.convert == nullptr ?
                    dequantizationFullPath.data : dequantizationFullPath.convert) :
                dequantizationFullPath.subtract,
            newMultiplyValuesFullPath);

        newMultiply = multiply->clone_with_new_inputs(inputs);
    }

    replace_node(multiply, newMultiply);
    updateOutput(context, newMultiply, multiply);

    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph

﻿// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/depth_to_space.hpp"

#include <memory>
#include "low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

DepthToSpaceTransformation::DepthToSpaceTransformation(const Params& params) : TransparentBaseTransformation(params) {
    auto matcher = make_op_pattern<opset1::DepthToSpace>({ make_op_label<ngraph::opset1::Multiply>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (!op || m_transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "DepthToSpaceTransformation");
    this->register_matcher(m, callback);
}

void DepthToSpaceTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::DepthToSpace>({ make_op_label<ngraph::opset1::Multiply>() }));
}

bool DepthToSpaceTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> depthToSpace = m.get_match_root();
    if (!canBeTransformed(context, depthToSpace)) {
        return false;
    }

    depthToSpace = NetworkHelper::separateInStandaloneBranch(depthToSpace);
    moveDequantizationAfter(context, depthToSpace, NetworkHelper::getDequantization(depthToSpace), true);
    return true;
}

bool DepthToSpaceTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

bool DepthToSpaceTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(layer);
    if (dequantization.multiply != nullptr) {
        auto multiplyConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1));
        if (!NetworkHelper::isScalarLike(multiplyConst)) {
            return false;
        }
    }

    if (dequantization.subtract != nullptr) {
        if (!NetworkHelper::isScalarLike(dequantization.subtractConstant)) {
            return false;
        }
    }

    return true;
}

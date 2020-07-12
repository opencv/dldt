﻿// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/convert.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void ConvertTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Convert>(pass, context);
}

void ConvertTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(m.get_match_root());

    const ngraph::element::Type precisionBefore = convert->get_input_element_type(0);
    const ngraph::element::Type precisionAfter = convert->get_output_element_type(0);

    std::shared_ptr<opset1::Subtract> subtract = std::make_shared<op::TypeRelaxed<opset1::Subtract>>(
        convert->get_input_node_shared_ptr(0),
        std::make_shared<opset1::Constant>(precisionBefore, Shape{}, std::vector<size_t>({ 0 })));
    subtract->set_output_type(0, precisionAfter, convert->get_output_partial_shape(0));

    replace_node(convert, subtract);

    subtract->set_friendly_name(convert->get_friendly_name());
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph

// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/constant.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MultiplyBranch {
public:
    Shape inputShape;
    ngraph::builder::subgraph::Constant constant;
    ngraph::element::Type precisionBeforeDequantization;
    ngraph::builder::subgraph::DequantizationOperations dequantization;
};

inline std::ostream& operator<<(std::ostream& out, const MultiplyBranch& branch) {
    return out << "_" << branch.constant << "_" << branch.precisionBeforeDequantization << "_" << branch.dequantization;
}

class MultiplyValues {
public:
    MultiplyBranch branch1;
    MultiplyBranch branch2;
    bool isDequantization;
};

inline std::ostream& operator<<(std::ostream& out, const MultiplyValues& values) {
    return out << "_" << values.branch1 << "_" << values.branch2 << (values.isDequantization ? "_isDequantization" : "");
}

class MultiplyFunction {
public:
    static std::shared_ptr<ngraph::Function> get(const ngraph::Shape& inputShape, const MultiplyValues& actualValues);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph

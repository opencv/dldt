// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/constant.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"

namespace LayerTestsDefinitions {

class ConvolutionQDqTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
    ngraph::builder::subgraph::DequantizationOperations::Convert convertOnData;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOnData;

    ngraph::builder::subgraph::Constant constantOnWeights;
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    ngraph::builder::subgraph::DequantizationOperations::Convert convertOnWeights;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOnWeights;

    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    ConvolutionQDqTransformationParam
> ConvolutionQDqTransformationParams;

class ConvolutionQDqTransformation :
    public testing::WithParamInterface<ConvolutionQDqTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionQDqTransformationParams> obj);

protected:
    void SetUp() override;

    void Run() override;
};

}  // namespace LayerTestsDefinitions

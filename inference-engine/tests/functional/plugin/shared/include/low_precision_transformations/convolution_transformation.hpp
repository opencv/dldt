// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    InferenceEngine::details::LayerTransformation::Params,
    bool, // fqOnActivations
    bool  // fqOnWeights
> ConvolutionTransformationParams;

class ConvolutionTransformation :
    public testing::WithParamInterface<ConvolutionTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionTransformationParams> obj);

protected:
    void SetUp() override;

private:
    void validate() override;
};

}  // namespace LayerTestsDefinitions

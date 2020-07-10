// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "ngraph_functions/low_precision_transformations/fuse_fake_quantize_and_scale_shift_function.hpp"
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

// ngraph::builder::subgraph::FakeQuantizeOnData
typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    InferenceEngine::details::LayerTransformation::Params,
    ngraph::builder::subgraph::FakeQuantizeOnData> FuseFakeQuantizeAndScaleShiftTransformationParams;

class FuseFakeQuantizeAndScaleShiftTransformation :
    public testing::WithParamInterface<FuseFakeQuantizeAndScaleShiftTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseFakeQuantizeAndScaleShiftTransformationParams> obj);

protected:
    void SetUp() override;

private:
    void validate();
};

}  // namespace LayerTestsDefinitions

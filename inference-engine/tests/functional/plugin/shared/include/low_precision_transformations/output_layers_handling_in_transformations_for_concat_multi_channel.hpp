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
    bool> OutputLayersHandlingInTransformationsParams;

class OutputLayersHandlingInTransformationsForConcatMultiChannel :
    public testing::WithParamInterface<LayerTestsUtils::LayerTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj);
    InferenceEngine::Blob::Ptr generateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;

private:
    void validate() override;
};

}  // namespace LayerTestsDefinitions

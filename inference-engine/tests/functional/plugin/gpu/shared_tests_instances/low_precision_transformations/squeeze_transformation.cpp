// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/squeeze_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::pass::low_precision;

namespace {
    const std::vector<ngraph::element::Type> netPrecisions = {
        ngraph::element::f32,
        ngraph::element::f16
    };


    const std::vector<LayerTransformation::Params> trasformationParamValues = {
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8().setUpdatePrecisions(false),
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8().setUpdatePrecisions(true),
    };

    const std::vector<LayerTestsDefinitions::SqueezeTransformationParam> params = {
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 3 },
            { 1, 3, 5, 1}
        },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 2, 3 },
            { 1, 1, 1, 1 }
        },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 3 },
            { 1, 64, 32, 1 }
        },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 2.0, 3.0 },
            { 1, 32, 1, 1 }
        }
    };

    INSTANTIATE_TEST_CASE_P(smoke_LPT, SqueezeTransformation,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GPU),
            ::testing::ValuesIn(trasformationParamValues),
            ::testing::ValuesIn(params)),
        SqueezeTransformation::getTestCaseName);
}  // namespace

// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/convolution_qdq_transformation.hpp"
#include "low_precision_transformations/convolution_with_incorrect_weights.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(true),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
};

const std::vector<LayerTestsDefinitions::ConvolutionQDqTransformationParam> params = {
    // Actual:
    //
    // FQ
    //  |FP32
    //  |
    // Convert    Convert   Constant  Constant
    //  |U8        |U8       |U8       |U8
    //  |          |         |         |
    // Convert    Convert   Convert   Convert
    //   \FP32    /FP32      \FP32    /FP32
    //    \      /            \      /
    //    Subtract  Constant  Subtract  Constant
    //      \FP32   /FP32       \FP32   /FP32
    //       \     /             \     /
    //       Multiply           Multiply
    //         \FP32           /FP32
    //          \             /
    //            Convolution
    //
    // Transformed:
    //
    //  FQ        Constant
    //   \U8      /U8
    //    \      /
    //    Subtract   Constant
    //      \FP32    /I8
    //       \      /
    //       Convolution  Constant
    //         \FP32      /FP32
    //          \        /
    //           Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { 0.f }, { 255.f }, { 0.f }, { 255.f }, ngraph::element::u8 },
        { ngraph::element::u8, false },
        {
            { ngraph::element::f32, false },
            { {128.f}, ngraph::element::f32, {}, false, 1ul, ngraph::element::u8, true },
            { {0.02f}, ngraph::element::f32, {}, false }
        },
        {{0.5f}, ngraph::element::i8},
        {},
        {},
        {
            { ngraph::element::f32, false },
            { {128.f}, ngraph::element::f32, {}, false, 1ul, ngraph::element::i8, true },
            { {0.03f}, ngraph::element::f32, {}, false }
        },
        "output_original",
        "U8"
    },
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 16, 16 },
    { 4, 3, 16, 16 }
};

INSTANTIATE_TEST_CASE_P(DISABLED_smoke_LPT, ConvolutionQDqTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ConvolutionQDqTransformation::getTestCaseName);
}  // namespace

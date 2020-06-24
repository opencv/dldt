// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/range.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<float> start = { 1.0f, 1.2f };
const std::vector<float> stop = { 5.0f, 5.2f };
const std::vector<float> step = { 1.0f, 0.1f };

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(Basic, RangeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(start),
                                ::testing::ValuesIn(stop),
                                ::testing::ValuesIn(step),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        RangeLayerTest::getTestCaseName);
}  // namespace

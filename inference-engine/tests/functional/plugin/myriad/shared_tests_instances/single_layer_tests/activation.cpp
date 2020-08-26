// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;
namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<ActivationTypes> activationTypes = {
        Sigmoid,
        Tanh,
        Relu,
        Exp,
        Log,
        Gelu,
        Mish,
        SoftPlus
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 50}, {{}}},
        {{1, 128}, {{}}},
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(activationTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(CommonTestUtils::combineShapes<size_t>(basic)),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)
);


INSTANTIATE_TEST_CASE_P(Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

}  // namespace

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;
namespace {
// Common params
const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U8
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sigmoid,               {}},
        {Tanh,                  {}},
        {Relu,                  {}},
        {Exp,                   {}},
        {Log,                   {}},
        {Sign,                  {}},
        {Abs,                   {}},
        {Gelu,                  {}},
        {Clamp,                 {{-2.0f, 2.0f}}},
        {Negative,              {}},
        {Acos,                  {}},
        {Asin,                  {}},
        {Atan,                  {}},
        {Cos,                   {}},
        {Cosh,                  {}},
        {Floor,                 {}},
        {Sin,                   {}},
        {Sinh,                  {}},
        {Sqrt,                  {}},
        {Tan,                   {}},
        {Elu,                   {{0.1f}}},
        {Erf,                   {}},
        {HardSigmoid,           {{0.2f, 0.5f}}},
        {Selu,                  {{1.6732f, 1.0507f}}},
        {Ceiling,               {}},
        {Mish,                  {}},
        {HSwish,                {}},
        {SoftPlus,              {}},
        {HSigmoid,              {}},
        {Swish,                 {{0.5f}}},
        {RoundHalfToEven,       {}},
        {RoundHalfAwayFromZero, {}}
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
    {PReLu, {{}}}
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 50}, {{}}},
        {{1, 128}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
        {{1, 50}, {{1}, {50}}},
        {{1, 128}, {{1}, {128}}},

        // Broadcast check
        {{3, 2}, {{1}, {2}, {3, 2}}},
        {{3, 2, 5}, {{1}, {2}, {5}, {2, 5}, {3, 1, 5}, {1, 2, 1}, {1, 1, 5}, {3, 1, 1}, {3, 2, 5}}},
        {{2, 1, 2}, {{2}, {2, 1, 1}}},
        {{3, 2, 5, 7}, {{1}, {7}, {2}, {5, 7}, {2, 5, 7}, {2, 1, 1}, {1, 2, 1, 1}, {3, 2, 1, 1}, {3, 2, 5, 7}}},
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

const auto basicPreluCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationParamTypes)),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(preluBasic)),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(smoke_Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Activation_Basic_Prelu_Const, ActivationLayerTest, basicPreluCases, ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_Activation_Basic_Prelu_Param, ActivationParamLayerTest, basicPreluCases, ActivationLayerTest::getTestCaseName);
}  // namespace

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph_functions/builders.hpp>
#include <transformations/common_optimizations/tiling.hpp>
#include <transformations/serialize.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {
struct SoftMaxConfig {
    InferenceEngine::SizeVector  inputShape;
    size_t axis;
};

typedef std::tuple<
    InferenceEngine::Precision,         // netPrecision
    SoftMaxConfig,                      // softmaxTestConfig
    std::string,                        // targetDevice
    CPUSpecificParams
> softmaxCPUTestParams;

class SoftMaxLayerCPUTest : public testing::WithParamInterface<softmaxCPUTestParams>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<softmaxCPUTestParams>& obj) {
        CPUSpecificParams cpuParams;
        InferenceEngine::Precision netPrecision;
        SoftMaxConfig config;
        std::string targetDevice;
        std::tie(netPrecision, config, targetDevice, cpuParams) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "IS=" << CommonTestUtils::vec2str(config.inputShape) << "_";
        result << "axis=" << config.axis << "_";
        result << "trgDev=" << targetDevice;
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        SoftMaxConfig config;
        CPUSpecificParams cpuParams;
        std::tie(netPrecision, config, targetDevice, cpuParams) = this->GetParam();

        inPrc = outPrc = netPrecision;

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }
        selectedType.push_back('_');
        selectedType += inPrc.name();

        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, {config.inputShape});

        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        // const auto softMax = std::make_shared<ngraph::opset1::Softmax>(paramOuts.at(0), config.axis);
        auto convolutionNode = ngraph::builder::makeConvolution(paramOuts.front(), ngPrc, {2, 2}, {config.axis, config.axis}, {0, 0},
                                                                {0, 0}, {1, 1}, ngraph::op::PadType::VALID, 6);

        function = makeNgraphFunction(ngPrc, params, convolutionNode, "Convolution");
    }
};

TEST_P(SoftMaxLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Softmax");
}

namespace {
//not optimized cpu spec
const auto notOptimizedCPUSpec = CPUSpecificParams{{}, {}, {}, "ref_any"};

const std::vector<SoftMaxConfig> optimizedConfigsFP32 {
        {InferenceEngine::SizeVector{1, 3, 64, 64}, 1},
        {InferenceEngine::SizeVector{1, 3, 64, 64}, 2},
        {InferenceEngine::SizeVector{1, 3, 64, 64}, 3},
        {InferenceEngine::SizeVector{1, 3, 64, 64}, 4},
};

const std::vector<SoftMaxConfig> notOptimizedConfigsFP32 {
        {InferenceEngine::SizeVector{1, 100}, 0},
        {InferenceEngine::SizeVector{10, 10}, 0},
        {InferenceEngine::SizeVector{10, 10, 10}, 0},
        {InferenceEngine::SizeVector{10, 10, 10}, 1},
};

const auto OptimizedParams = testing::Combine(
        testing::Values(Precision::FP32),
        testing::ValuesIn(optimizedConfigsFP32),
        testing::Values(CommonTestUtils::DEVICE_CPU),
        testing::Values(emptyCPUSpec));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax_Optimized_CPU, SoftMaxLayerCPUTest, OptimizedParams, SoftMaxLayerCPUTest::getTestCaseName);

const auto NotOptimizedParams = testing::Combine(
        testing::Values(Precision::FP32, Precision::BF16),
        testing::ValuesIn(notOptimizedConfigsFP32),
        testing::Values(CommonTestUtils::DEVICE_CPU),
        testing::Values(notOptimizedCPUSpec));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax_CPU, SoftMaxLayerCPUTest, NotOptimizedParams, SoftMaxLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
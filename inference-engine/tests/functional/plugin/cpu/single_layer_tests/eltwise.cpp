// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/eltwise.hpp>
#include <ngraph_functions/builders.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::EltwiseTestParams,
        CPUSpecificParams,
        Precision, // CNN input precision
        Precision> // CNN output precision
        EltwiseLayerCPUTestParamsSet;

class EltwiseLayerCPUTest : public testing::WithParamInterface<EltwiseLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::EltwiseTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        Precision inputPrecision, outputPrecision;
        std::tie(basicParamsSet, cpuParams, inputPrecision, outputPrecision) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::EltwiseLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::EltwiseTestParams>(
                basicParamsSet, 0));

        result << "_" << "CNNinpPrc=" << inputPrecision.name();
        result << "_" << "CNNoutPrc=" << outputPrecision.name();

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() {
        LayerTestsDefinitions::EltwiseTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams, inPrc, outPrc) = this->GetParam();

        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        ngraph::helpers::InputLayerType secondaryInputType;
        CommonTestUtils::OpType opType;
        ngraph::helpers::EltwiseTypes eltwiseType;
        std::map<std::string, std::string> additional_config;
        std::tie(inputShapes, eltwiseType, secondaryInputType, opType, netPrecision, targetDevice, additional_config) = basicParamsSet;
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        std::string strExpectedPrc;
        if (Precision::BF16 == netPrecision) {
            strExpectedPrc = "BF16";
        } else if (Precision::FP32 == netPrecision) {
            strExpectedPrc = "FP32";
        }

        std::string isaType;
        if (with_cpu_x86_avx512f()) {
            isaType = "jit_avx512";
        } else if (with_cpu_x86_avx2()) {
            isaType = "jit_avx2";
        } else if (with_cpu_x86_sse42()) {
            isaType = "jit_sse42";
        } else {
            isaType = "ref";
        }
        selectedType = isaType + "_" + strExpectedPrc;

        std::vector<size_t> inputShape1, inputShape2;
        if (inputShapes.size() == 1) {
            inputShape1 = inputShape2 = inputShapes.front();
        } else if (inputShapes.size() == 2) {
            inputShape1 = inputShapes.front();
            inputShape2 = inputShapes.back();
        } else {
            THROW_IE_EXCEPTION << "Incorrect number of input shapes";
        }

        configuration.insert(additional_config.begin(), additional_config.end());
        auto input = ngraph::builder::makeParams(ngPrc, {inputShape1});

        std::vector<size_t> shape_input_secondary;
        switch (opType) {
            case CommonTestUtils::OpType::SCALAR: {
                shape_input_secondary = std::vector<size_t>({1});
                break;
            }
            case CommonTestUtils::OpType::VECTOR:
                shape_input_secondary = inputShape2;
                break;
            default:
                FAIL() << "Unsupported Secondary operation type";
        }

        std::shared_ptr<ngraph::Node> secondaryInput;
        if (eltwiseType == ngraph::helpers::EltwiseTypes::DIVIDE ||
            eltwiseType == ngraph::helpers::EltwiseTypes::FLOOR_MOD ||
            eltwiseType == ngraph::helpers::EltwiseTypes::MOD) {
            std::vector<float> data(ngraph::shape_size(shape_input_secondary));
            data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(shape_input_secondary), 10, 2);
            secondaryInput = ngraph::builder::makeConstant(ngPrc, shape_input_secondary, data);
        } else {
            secondaryInput = ngraph::builder::makeInputLayer(ngPrc, secondaryInputType, shape_input_secondary);
            if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
                input.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
            }
        }

        auto eltwise = ngraph::builder::makeEltwise(input[0], secondaryInput, eltwiseType);
        eltwise->get_rt_info() = getCPUInfo();
        function = std::make_shared<ngraph::Function>(eltwise, input, "Eltwise");
    }
};

TEST_P(EltwiseLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "Eltwise");
}

namespace {

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesBinInp = { // Always two input nodes
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::DIVIDE,
        ngraph::helpers::EltwiseTypes::FLOOR_MOD,
        ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesDiffInp = { // Different number of input nodes depending on optimizations
        ngraph::helpers::EltwiseTypes::POWER,
        ngraph::helpers::EltwiseTypes::MOD
};
// Withing the test scope we don't need any implicit bf16 optimisations, so let's run the network as is.
std::map<std::string, std::string> additional_config = {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO}};

std::vector<Precision> bf16InpOutPrc = {Precision::BF16, Precision::FP32};

std::vector<CPUSpecificParams> filterCPUSpecificParams(std::vector<CPUSpecificParams>& paramsVector) {
    auto adjustBlockedFormatByIsa = [](std::vector<cpu_memory_format_t>& formats) {
        for (int i = 0; i < formats.size(); i++) {
            if (formats[i] == nChw16c)
                formats[i] = nChw8c;
            if (formats[i] == nCdhw16c)
                formats[i] = nCdhw8c;
        }
    };

    if (!with_cpu_x86_avx512f()) {
        for (auto& param : paramsVector) {
            adjustBlockedFormatByIsa(std::get<0>(param));
            adjustBlockedFormatByIsa(std::get<1>(param));
        }
    }

    return paramsVector;
}

std::vector<std::vector<std::vector<size_t>>> inShapes_4D = {
        {{2, 4, 4, 1}},
        {{2, 17, 5, 4}},
        {{2, 17, 5, 4}, {1, 17, 1, 1}},
        {{2, 17, 5, 1}, {1, 17, 1, 4}},
};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c, nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw, nchw}, {nchw}, {}, {})
};

const auto params_4D_FP32 = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inShapes_4D),
            ::testing::ValuesIn(eltwiseOpTypesBinInp),
            ::testing::ValuesIn(secondaryInputTypes),
            ::testing::ValuesIn(opTypes),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_4D_FP32_MemOrder, EltwiseLayerCPUTest, params_4D_FP32, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_FP32_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_4D),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::Values(emptyCPUSpec),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_4D_FP32, EltwiseLayerCPUTest, params_4D_FP32_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_BF16 = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_4D),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::Values(InferenceEngine::Precision::BF16),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::Values(emptyCPUSpec),
        ::testing::ValuesIn(bf16InpOutPrc),
        ::testing::ValuesIn(bf16InpOutPrc));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_4D_BF16_MemOrder, EltwiseLayerCPUTest, params_4D_BF16, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_BF16_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_4D),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::Values(InferenceEngine::Precision::BF16),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::Values(emptyCPUSpec),
        ::testing::ValuesIn(bf16InpOutPrc),
        ::testing::ValuesIn(bf16InpOutPrc));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_4D_BF16, EltwiseLayerCPUTest, params_4D_BF16_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

std::vector<std::vector<std::vector<size_t>>> inShapes_5D = {
        {{2, 4, 3, 4, 1}},
        {{2, 17, 7, 5, 4}},
        {{2, 17, 6, 5, 4}, {1, 17, 6, 1, 1}},
        {{2, 17, 6, 5, 1}, {1, 17, 1, 1, 4}},
};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c, nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ndhwc, ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw, ncdhw}, {ncdhw}, {}, {})
};

const auto params_5D_FP32 = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inShapes_5D),
            ::testing::ValuesIn(eltwiseOpTypesBinInp),
            ::testing::ValuesIn(secondaryInputTypes),
            ::testing::ValuesIn(opTypes),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_5D_FP32_MemOrder, EltwiseLayerCPUTest, params_5D_FP32, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_BF16 = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_5D),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::Values(InferenceEngine::Precision::BF16),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        ::testing::ValuesIn(bf16InpOutPrc),
        ::testing::ValuesIn(bf16InpOutPrc));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_5D_BF16_MemOrder, EltwiseLayerCPUTest, params_5D_BF16, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_FP32_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_5D),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::Values(emptyCPUSpec),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_5D_FP32, EltwiseLayerCPUTest, params_5D_FP32_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_BF16_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_5D),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::Values(InferenceEngine::Precision::BF16),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::Values(emptyCPUSpec),
        ::testing::ValuesIn(bf16InpOutPrc),
        ::testing::ValuesIn(bf16InpOutPrc));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_5D_BF16, EltwiseLayerCPUTest, params_5D_BF16_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

std::vector<std::vector<std::vector<size_t>>> inShapes_4D_Blocked_Planar = {
        {{2, 17, 31, 3}, {2, 1, 31, 3}},
        {{2, 17, 5, 1}, {2, 1, 1, 4}},
};

std::vector<CPUSpecificParams> cpuParams_4D_Blocked_Planar = {
        CPUSpecificParams({nChw16c, nchw}, {nChw16c}, {}, {}),
};

const auto params_4D_FP32_Blocked_Planar = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inShapes_4D_Blocked_Planar),
            ::testing::ValuesIn(eltwiseOpTypesBinInp),
            ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
            ::testing::ValuesIn(opTypes),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Planar)),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_4D_FP32_Blocked_Planar, EltwiseLayerCPUTest, params_4D_FP32_Blocked_Planar, EltwiseLayerCPUTest::getTestCaseName);


std::vector<std::vector<std::vector<size_t>>> inShapes_4D_Planar_Blocked = {
        {{2, 1, 31, 3}, {2, 17, 31, 3}},
        {{2, 1, 1, 4}, {2, 17, 5, 1}},
};

std::vector<CPUSpecificParams> cpuParams_4D_Planar_Blocked = {
        CPUSpecificParams({nchw, nChw16c}, {nChw16c}, {}, {}),
};

const auto params_4D_FP32_Planar_Blocked = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inShapes_4D_Planar_Blocked),
            ::testing::ValuesIn(eltwiseOpTypesBinInp),
            ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
            ::testing::ValuesIn(opTypes),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Planar_Blocked)),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_4D_FP32_Planar_Blocked, EltwiseLayerCPUTest, params_4D_FP32_Planar_Blocked, EltwiseLayerCPUTest::getTestCaseName);


std::vector<std::vector<std::vector<size_t>>> inShapes_5D_Blocked_Planar = {
        {{2, 17, 31, 4, 3}, {2, 1, 31, 1, 3}},
        {{2, 17, 5, 3, 1}, {2, 1, 1, 3, 4}},
};

std::vector<CPUSpecificParams> cpuParams_5D_Blocked_Planar = {
        CPUSpecificParams({nCdhw16c, ncdhw}, {nCdhw16c}, {}, {}),
};

const auto params_5D_FP32_Blocked_Planar = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inShapes_5D_Blocked_Planar),
            ::testing::ValuesIn(eltwiseOpTypesBinInp),
            ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
            ::testing::ValuesIn(opTypes),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Blocked_Planar)),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_5D_FP32_Blocked_Planar, EltwiseLayerCPUTest, params_5D_FP32_Blocked_Planar, EltwiseLayerCPUTest::getTestCaseName);


std::vector<std::vector<std::vector<size_t>>> inShapes_5D_Planar_Blocked = {
        {{2, 1, 31, 1, 3}, {2, 17, 31, 4, 3}},
        {{2, 1, 1, 3, 4}, {2, 17, 5, 3, 1}},
};

std::vector<CPUSpecificParams> cpuParams_5D_Planar_Blocked = {
        CPUSpecificParams({ncdhw, nCdhw16c}, {nCdhw16c}, {}, {}),
};

const auto params_5D_FP32_Planar_Blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_5D_Planar_Blocked),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Planar_Blocked)),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_5D_FP32_Planar_Blocked, EltwiseLayerCPUTest, params_5D_FP32_Planar_Blocked, EltwiseLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions

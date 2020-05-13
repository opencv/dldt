// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bfloat16_helpers.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <functional>
#include <map>
#include <utility>

#include <ie_core.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

namespace LayerTestsDefinitions {

class Scaleshift_x3_ConvEltwiseRelu : public BasicBF16Test {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
        //
        //   scaleshift (FP32)
        //        |
        //       Conv (BF16)   scaleshift (FP32
        //
        //         \         /
        //
        //        Eltwise (Fused to Conv)
        //          |
        //         ReLU   (Fused to Conv)
        //           |
        //       scaleshift  (FP32)

        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        auto channelsCount = inputShapes[1];

        // multiply
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{inputShapes});
        std::shared_ptr<ngraph::opset1::Constant> const1 = nullptr;
        if (netPrecision == Precision::FP32) {
            const1 = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            const1 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(BFloat16Helpers::reducePrecisionBitwiseS(2.0f)) });
        }
        auto mulNode = std::make_shared<opset1::Multiply>(input1, const1);

        // add
        std::shared_ptr<ngraph::opset1::Constant> const2 = nullptr;
        if (netPrecision == Precision::FP32) {
            const2 = opset1::Constant::create(ntype, Shape{1}, { 1.0f });
        } else {
            const2 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(BFloat16Helpers::reducePrecisionBitwiseS(1.0f)) });
        }
        auto addNode = std::make_shared<opset1::Add>(mulNode, const2);
        addNode->set_friendly_name("Add_1");

        // convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode = nullptr;
        ngraph::Shape convFilterShape = { channelsCount, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(channelsCount * channelsCount * 3 * 3);
            BFloat16Helpers::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(channelsCount * channelsCount * 3 * 3);
            BFloat16Helpers::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode1 = std::make_shared<ngraph::opset1::Convolution>(
            addNode, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
            ngraph::CoordinateDiff({ 1, 1 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode1->set_friendly_name("Convolution_1");

        // multiply
        std::shared_ptr<ngraph::opset1::Constant> const3 = nullptr;
        if (netPrecision == Precision::FP32) {
            const3 = opset1::Constant::create(ntype, Shape{1}, { 3.0f });
        } else {
            const3 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(BFloat16Helpers::reducePrecisionBitwiseS(3.0f)) });
        }
        auto mulNode2 = std::make_shared<opset1::Multiply>(input1, const3);

        // add
        std::shared_ptr<ngraph::opset1::Constant> const4 = nullptr;
        if (netPrecision == Precision::FP32) {
            const4 = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            const4 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(BFloat16Helpers::reducePrecisionBitwiseS(2.0f)) });
        }
        auto addNode2 = std::make_shared<opset1::Add>(mulNode2, const4);
        addNode2->set_friendly_name("Add_2");

        // Eltwise, i.e. Add
        auto eltNode = std::make_shared<opset1::Add>(convNode1, addNode2);
        eltNode->set_friendly_name("ELT_1");

        // ReLU
        auto reluNode =  std::make_shared<opset1::Relu>(eltNode);
        reluNode->set_friendly_name("RELU_1");

        // multiply
        std::shared_ptr<ngraph::opset1::Constant> const5 = nullptr;
        if (netPrecision == Precision::FP32) {
            const5 = opset1::Constant::create(ntype, Shape{1}, { 4.0f });
        } else {
            const5 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(BFloat16Helpers::reducePrecisionBitwiseS(4.0f)) });
        }
        auto mulNode3 = std::make_shared<opset1::Multiply>(reluNode, const5);

        // add
        std::shared_ptr<ngraph::opset1::Constant> const6 = nullptr;
        if (netPrecision == Precision::FP32) {
            const6 = opset1::Constant::create(ntype, Shape{1}, { 3.0f });
        } else {
            const6 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(BFloat16Helpers::reducePrecisionBitwiseS(3.0f)) });
        }
        auto addNode3 = std::make_shared<opset1::Add>(mulNode3, const6);
        addNode3->set_friendly_name("Add_3");

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{addNode3}, ngraph::ParameterVector{input1});
    }

    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE1:
        threshold = 2e-1;

        // STAGE2:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Add_1"] = "FP32";
        expectedPrecisions["Convolution_1"] = "BF16";
        expectedPrecisions["Add_2"] = "FP32";
        expectedPrecisions["ELT_1"] = "ndef";
        expectedPrecisions["RELU_1"] = "ndef";
        expectedPrecisions["Add_3"] = "FP32";
    }
};

TEST_P(Scaleshift_x3_ConvEltwiseRelu, CompareWithRefImpl) {
    test();
};

INSTANTIATE_TEST_CASE_P(FP32_bfloat16_NoReshape, Scaleshift_x3_ConvEltwiseRelu,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Scaleshift_x3_ConvEltwiseRelu::getTestCaseName);

INSTANTIATE_TEST_CASE_P(BF16_bfloat16_NoReshape, Scaleshift_x3_ConvEltwiseRelu,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::BF16),
                            ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Scaleshift_x3_ConvEltwiseRelu::getTestCaseName);



}  // namespace LayerTestsDefinitions

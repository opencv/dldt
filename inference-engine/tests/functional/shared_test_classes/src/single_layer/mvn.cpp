// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/mvn.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string MvnLayerTest::getTestCaseName(const testing::TestParamInfo<mvnParams>& obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    bool acrossChannels, normalizeVariance;
    double eps;
    std::string targetDevice;
    std::tie(inputShapes, inputPrecision, acrossChannels, normalizeVariance, eps, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Precision=" << inputPrecision.name() << "_";
    result << "AcrossChannels=" << (acrossChannels ? "TRUE" : "FALSE") << "_";
    result << "NormalizeVariance=" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
    result << "Epsilon=" << eps << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void MvnLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    bool acrossChanels, normalizeVariance;
    double eps;
    std::tie(inputShapes, inputPrecision, acrossChanels, normalizeVariance, eps, targetDevice) = this->GetParam();
    auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto param = ngraph::builder::makeParams(inType, {inputShapes});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(param));
    auto mvn = std::dynamic_pointer_cast<ngraph::op::MVN>(ngraph::builder::makeMVN(paramOuts[0], acrossChanels, normalizeVariance, eps));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mvn)};
    function = std::make_shared<ngraph::Function>(results, param, "mvn");
}


std::string Mvn6LayerTest::getTestCaseName(const testing::TestParamInfo<mvn6Params>& obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision dataPrecision, axesPrecision;
    std::vector<int> axes;
    bool normalizeVariance;
    float eps;
    std::string epsMode;
    std::string targetDevice;
    std::tie(inputShapes, dataPrecision, axesPrecision, axes, normalizeVariance, eps, epsMode, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "DataPrc=" << dataPrecision.name() << "_";
    result << "AxPrc=" << axesPrecision.name() << "_";
    result << "Ax=" << CommonTestUtils::vec2str(axes) << "_";
    result << "NormVariance=" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
    result << "Eps=" << eps << "_";
    result << "EM=" << epsMode << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void Mvn6LayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision dataPrecision, axesPrecision;
    std::vector<int> axes;
    bool normalizeVariance;
    float eps;
    std::string epsMode;
    std::tie(inputShapes, dataPrecision, axesPrecision, axes, normalizeVariance, eps, epsMode, targetDevice) = this->GetParam();

    auto dataType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dataPrecision);
    auto axesType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(axesPrecision);

    auto param = ngraph::builder::makeParams(dataType, {inputShapes});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(param));
    auto axesNode = ngraph::builder::makeConstant(axesType, ngraph::Shape{axes.size()}, axes);
    auto mvn = ngraph::builder::makeMVN6(paramOuts[0], axesNode, normalizeVariance, eps, epsMode);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mvn)};
    function = std::make_shared<ngraph::Function>(results, param, "MVN6");
}

}  // namespace LayerTestsDefinitions

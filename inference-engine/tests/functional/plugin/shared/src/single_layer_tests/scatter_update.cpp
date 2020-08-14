// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <ie_core.hpp>
#include <ngraph_functions/builders.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "single_layer_tests/scatter_update.hpp"

using namespace ngraph::opset3;

namespace LayerTestsDefinitions {

std::string ScatterUpdateLayerTest::getTestCaseName(const testing::TestParamInfo<scatterUpdateParamsTuple> &obj) {
    axisShapeInShape shapeDescript;
    std::vector<size_t> inShape;
    std::vector<size_t> indicesShape;
    std::vector<size_t> updateShape;
    int axis;
    std::vector<size_t> indicesValue;
    InferenceEngine::Precision inputPrecision;
    InferenceEngine::Precision indicesPrecision;
    std::string targetName;
    std::tie(shapeDescript, indicesValue, inputPrecision, indicesPrecision, targetName) = obj.param;
    std::tie(inShape, indicesShape, updateShape, axis) = shapeDescript;
    std::ostringstream result;
    result << "InputShape=" << CommonTestUtils::vec2str(inShape) << "_";
    result << "IndicesShape=" << CommonTestUtils::vec2str(indicesShape) << "_";
    result << "IndicesValue=" << CommonTestUtils::vec2str(indicesValue) << "_";
    result << "UpdateShape=" << CommonTestUtils::vec2str(updateShape) << "_";
    result << "Axis=" << axis << "_";
    result << "inPrc=" << inputPrecision.name() << "_";
    result << "idxPrc=" << indicesPrecision.name() << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

std::vector<axisShapeInShape> ScatterUpdateLayerTest::combineShapes(
    const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& inputShapes) {
    std::vector<axisShapeInShape> resVec;
    for (auto& inputShape : inputShapes) {
        auto srcShape = inputShape.first;
        auto srcRank = srcShape.size();
        for (auto& item : inputShape.second) {
            auto indicesShape = item.first;
            auto indicesRank = indicesShape.size();
            for (auto& axis : item.second) {
                auto axisP = axis < 0 ? axis + srcRank : axis;
                std::vector<size_t> updateShape;
                for (size_t rs = 0; rs < srcRank; rs++) {
                    if (rs != axisP) {
                        updateShape.push_back(srcShape[rs]);
                    } else {
                        for (size_t ri = 0; ri < indicesRank; ri++) {
                            updateShape.push_back(indicesShape[ri]);
                        }
                    }
                }
                resVec.push_back(std::make_tuple(srcShape, indicesShape, updateShape, axis));
            }
        }
    }
    return resVec;
}

void ScatterUpdateLayerTest::SetUp() {
    axisShapeInShape shapeDescript;
    InferenceEngine::SizeVector inShape;
    InferenceEngine::SizeVector indicesShape;
    InferenceEngine::SizeVector updateShape;
    int axis;
    InferenceEngine::SizeVector indicesValue;
    InferenceEngine::Precision inputPrecision;
    InferenceEngine::Precision indicesPrecision;
    std::tie(shapeDescript, indicesValue, inputPrecision, indicesPrecision, targetDevice) = this->GetParam();
    std::tie(inShape, indicesShape, updateShape, axis) = shapeDescript;
    auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto idxPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(indicesPrecision);
    ngraph::ParameterVector paramVector;
    auto inputParams = std::make_shared<ngraph::opset1::Parameter>(inPrc, ngraph::Shape(inShape));
    paramVector.push_back(inputParams);
    auto updateParams = std::make_shared<ngraph::opset1::Parameter>(inPrc, ngraph::Shape(updateShape));
    paramVector.push_back(updateParams);
    auto paramVectorOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::v0::Parameter>(paramVector));
    auto s2d = ngraph::builder::makeScatterUpdate(paramVectorOuts[0], idxPrc, indicesShape, indicesValue, paramVectorOuts[1], axis);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(s2d)};
    function = std::make_shared<ngraph::Function>(results, paramVector, "ScatterUpdate");
}

TEST_P(ScatterUpdateLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions

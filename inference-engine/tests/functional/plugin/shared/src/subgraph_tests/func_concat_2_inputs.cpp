// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "subgraph_tests/func_concat_2_inputs.hpp"


namespace LayerTestsDefinitions {

std::string NonTrivialConcat2Inputs::getTestCaseName(testing::TestParamInfo<NonTrivialConcatParams> obj) {
    InferenceEngine::Precision netPrecision = std::get<0>(obj.param);
    InferenceEngine::SizeVector inputShapes = std::get<1>(obj.param);

    std::string targetDevice  = std::get<2>(obj.param);
    std::string configName    = std::get<3>(obj.param);

    std::ostringstream name;
    name << "IS=" << CommonTestUtils::vec2str(inputShapes);
    name << "_netPRC=" << netPrecision.name();
    name << "_targetDevice=" << targetDevice;
    name << "_config=" << configName;

    return name.str();
}

void NonTrivialConcat2Inputs::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::string configName;
    std::tie(netPrecision, inputShape, targetDevice, configName, fp32Threshold, fp32InputRange, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto input0 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, inputShape[0]});
    auto input1 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, inputShape[1]});

    auto relu0 = std::make_shared<ngraph::op::v0::Relu>(input0);
    auto relu1 = std::make_shared<ngraph::op::v0::Relu>(input1);

    auto reshape_pattern00 = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                   ngraph::Shape{4},
                                                                   std::vector<size_t>{1, 1, 1, inputShape[0]});

    auto reshape_pattern01 = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                   ngraph::Shape{4},
                                                                   std::vector<size_t>{1, 1, 1, inputShape[1]});

    auto reshape_pattern0 = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                  ngraph::Shape{2},
                                                                  std::vector<size_t>{1, inputShape[0]});


    auto reshape_pattern1 = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                  ngraph::Shape{2},
                                                                  std::vector<size_t>{1, inputShape[1]});


    auto reshape00 = std::make_shared<ngraph::op::v1::Reshape>(relu0, reshape_pattern00, false);
    auto reshape10 = std::make_shared<ngraph::op::v1::Reshape>(relu1, reshape_pattern01, false);

    auto reshape01 = std::make_shared<ngraph::op::v1::Reshape>(reshape00, reshape_pattern0, false);
    auto reshape11 = std::make_shared<ngraph::op::v1::Reshape>(reshape10, reshape_pattern1, false);


    auto concat = std::make_shared<ngraph::op::Concat>(ngraph::NodeVector{reshape01, reshape11}, 1);

    auto relu3 = std::make_shared<ngraph::op::v0::Relu>(concat);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu3},
        ngraph::ParameterVector{input0, input1},
        "NonTrivialConcat2Inputs");
}

TEST_P(NonTrivialConcat2Inputs, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
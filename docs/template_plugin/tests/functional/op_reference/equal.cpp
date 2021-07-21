// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "comparison.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using ComparisonTypes = ngraph::helpers::ComparisonTypes;

namespace ComparisonOpsRefTestDefinitions {
namespace {

TEST_P(ReferenceComparisonLayerTest, EqualCompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<RefComparisonParams> generateComparisonParams(const ngraph::element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<RefComparisonParams> compParams {
        // 1D // 2D // 3D // 4D
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {2, 2}, ngraph::PartialShape {2, 2}, type, ngraph::element::boolean,
                std::vector<T> {0, 12, 23, 0},
                std::vector<T> {0, 12, 23, 0},
                std::vector<char> {1, 1, 1, 1}),
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {2, 3}, ngraph::PartialShape {2, 3}, type, ngraph::element::boolean,
                std::vector<T> {0, 6, 45, 1, 21, 21},
                std::vector<T> {1, 18, 23, 1, 19, 21},
                std::vector<char> {0, 0, 0, 1, 0, 1}),
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {1}, ngraph::PartialShape {1},  type, ngraph::element::boolean,
                std::vector<T> {53},
                std::vector<T> {53},
                std::vector<char> {1}),
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {2, 4}, ngraph::PartialShape {2, 4}, type, ngraph::element::boolean,
                std::vector<T> {0, 12, 23, 0, 1, 5, 11, 8},
                std::vector<T> {0, 12, 23, 0, 10, 5, 11, 8},
                std::vector<char> {1, 1, 1, 1, 0, 1, 1, 1}),
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {3, 1, 2}, ngraph::PartialShape {1, 2, 1}, type, ngraph::element::boolean,
                std::vector<T> {2, 1, 4, 1, 3, 1},
                std::vector<T> {1, 1},
                std::vector<char> {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}),
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {2, 1, 2, 1}, ngraph::PartialShape {1, 2, 1}, type, ngraph::element::boolean,
                std::vector<T> {2, 1, 4, 1},
                std::vector<T> {1, 1},
                std::vector<char> {0, 1, 0, 1})};
    return compParams;
}

std::vector<RefComparisonParams> generateComparisonCombinedParams() {
    const std::vector<std::vector<RefComparisonParams>> compTypeParams {
        generateComparisonParams<element::Type_t::f32>(ngraph::element::f32),
        generateComparisonParams<element::Type_t::f16>(ngraph::element::f16),
        generateComparisonParams<element::Type_t::i32>(ngraph::element::i32),
        generateComparisonParams<element::Type_t::u32>(ngraph::element::u32),
        generateComparisonParams<element::Type_t::u8>(ngraph::element::boolean)};
    std::vector<RefComparisonParams> combinedParams;

    for (const auto& params : compTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Comparison_With_Hardcoded_Refs, ReferenceComparisonLayerTest, ::testing::ValuesIn(generateComparisonCombinedParams()),
                         ReferenceComparisonLayerTest::getTestCaseName);
} // namespace
} // namespace ComparisonOpsRefTestDefinitions

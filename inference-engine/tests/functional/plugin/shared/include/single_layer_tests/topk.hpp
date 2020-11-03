// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        int64_t,                        // keepK
        int64_t,                        // axis
        ngraph::op::TopKMode,           // mode
        ngraph::op::TopKSortType,       // sort
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::SizeVector,    // inputShape
        std::string                     // Target device name
> TopKParams;

class TopKLayerTest : public testing::WithParamInterface<TopKParams>,
                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TopKParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions

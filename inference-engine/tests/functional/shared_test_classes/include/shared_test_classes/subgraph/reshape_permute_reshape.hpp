// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {
typedef std::tuple<
        std::vector<std::vector<size_t>>, //input shapes and permute shapes
        InferenceEngine::Precision,       //Network precision
        std::string                       //Device name
        > ReshapePermuteReshapeTuple;

class ReshapePermuteReshape : public testing::WithParamInterface<ReshapePermuteReshapeTuple>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReshapePermuteReshapeTuple> &obj);

protected:
    void SetUp() override;
};
} // namespace SubgraphTestsDefinitions

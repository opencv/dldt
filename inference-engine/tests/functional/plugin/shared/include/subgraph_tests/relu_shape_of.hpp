// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/single_layer/shape_of.hpp"

#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

class ReluShapeOfSubgraphTest : public testing::WithParamInterface<shapeOfParams>,
        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<shapeOfParams> obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/convolution_backprop.hpp"

namespace LayerTestsDefinitions {

TEST_P(ConvolutionBackpropLayerTest, CompareWithRefs) {
    Run();
}

}

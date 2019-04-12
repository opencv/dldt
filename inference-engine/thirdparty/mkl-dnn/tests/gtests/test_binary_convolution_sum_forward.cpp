/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"
#include "test_binary_convolution_forward_common.hpp"

namespace mkldnn {

using binary_convolution_test = binary_convolution_forward_test;

TEST_P(binary_convolution_test, TestBinaryConvolutionSum)
{
}

#define BIN
#define WITH_SUM
#define DIRECTION_FORWARD
#include "convolution_common.h"

INST_TEST_CASE(SimpleSmall_Blocked_Padded_Channels,
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 3, 10, 10, 3, 10, 10, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 3, 10, 10, 32, 10, 10, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 3, 10, 10, 41, 10, 10, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 47, 10, 10, 137, 10, 10, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 256, 10, 10, 256, 10, 10, 3, 3, 1, 1, 1, 1)
);

INST_TEST_CASE(SimpleSmall_Blocked_1x1_Padded_Channels,
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 3, 10, 10, 3, 10, 10, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 13, 3, 3, 32, 3, 3, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 13, 3, 3, 41, 3, 3, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 47, 3, 3, 137, 3, 3, 1, 1, 0, 0, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 256, 3, 3, 256, 3, 3, 1, 1, 0, 0, 1, 1)
);

//INST_TEST_CASE(SimpleSmall_Depthwise_Blocked_Padded_Channels,
//    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED_G, FMT_BIAS, FMT_DATA_BLOCKED,
//        2, 32, 32, 10, 10, 32, 10, 10, 3, 3, 1, 1, 1, 1),
//    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED_G, FMT_BIAS, FMT_DATA_BLOCKED,
//        2, 43, 43, 10, 10, 43, 10, 10, 3, 3, 1, 1, 1, 1),
//    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED_G, FMT_BIAS, FMT_DATA_BLOCKED,
//        2, 256, 256, 10, 10, 256, 10, 10, 3, 3, 1, 1, 1, 1)
//);

}

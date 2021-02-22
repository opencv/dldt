//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

template <typename T_IN, typename T_KERN>
static void BinaryConvolutionTest(const std::vector<T_IN>& inputs,
                                  const Shape inputs_shape,
                                  const std::vector<T_KERN>& filters,
                                  const Shape filters_shape,
                                  const std::vector<T_IN>& outputs,
                                  const Shape outputs_shape,
                                  const Strides& strides,
                                  const CoordinateDiff& padding,
                                  const Strides& dilations)
{
    const CoordinateDiff pads_begin{padding};
    const CoordinateDiff pads_end{padding};
    const op::PadType auto_pad{op::PadType::EXPLICIT};
    float pad_value = 0;

    auto inputs_param = make_shared<op::Parameter>(element::from<T_IN>(), inputs_shape);
    auto filters_const = make_shared<op::Constant>(element::u1, filters_shape, &filters[0]);
    auto bin_conv = make_shared<op::v1::BinaryConvolution>(
        inputs_param,
        filters_const,
        strides,
        pads_begin,
        pads_end,
        dilations,
        op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
        pad_value,
        auto_pad);
    auto f = make_shared<Function>(bin_conv, ParameterVector{inputs_param});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto input_tensor = backend->create_tensor(element::from<T_IN>(), inputs_shape);
    copy_data(input_tensor, inputs);
    auto result = backend->create_tensor(element::from<T_IN>(), outputs_shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {input_tensor});
    EXPECT_TRUE(test::all_close_f((outputs), read_vector<T_IN>(result), MIN_FLOAT_TOLERANCE_BITS));
}

template <typename T_IN>
static void ConvolutionTest(const std::vector<T_IN>& inputs,
                            const Shape inputs_shape,
                            const std::vector<T_IN>& filters,
                            const Shape filters_shape,
                            const std::vector<T_IN>& outputs,
                            const Shape outputs_shape,
                            const Strides& strides,
                            const CoordinateDiff& padding,
                            const Strides& dilations)
{
    const CoordinateDiff pads_begin{padding};
    const CoordinateDiff pads_end{padding};
    const op::PadType auto_pad{op::PadType::EXPLICIT};

    auto inputs_param = make_shared<op::Parameter>(element::from<T_IN>(), inputs_shape);
    auto filters_param = make_shared<op::Parameter>(element::from<T_IN>(), filters_shape);
    auto conv = make_shared<op::v1::Convolution>(
        inputs_param, filters_param, strides, pads_begin, pads_end, dilations, auto_pad);
    auto f = make_shared<Function>(conv, ParameterVector{inputs_param, filters_param});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto input_tensor = backend->create_tensor(element::from<T_IN>(), inputs_shape);
    copy_data(input_tensor, inputs);
    auto filters_tensor = backend->create_tensor(element::from<T_IN>(), filters_shape);
    copy_data(filters_tensor, filters);
    auto result = backend->create_tensor(element::from<T_IN>(), outputs_shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {input_tensor, filters_tensor});
    EXPECT_TRUE(test::all_close_f((outputs), read_vector<T_IN>(result), MIN_FLOAT_TOLERANCE_BITS));
}

// clang-format off
// --------------------- 2D convolution ------------------------------------------
NGRAPH_TEST(${BACKEND_NAME}, bin_convolution_2D_1batch_1channel_no_padding)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs_conv{1.0f, -1.0f, -1.0f, 1.0f,
                                         1.0f, 1.0f, -1.0f, -1.0f,
                                         -1.0f, -1.0f, -1.0f, 1.0f,
                                         1.0f, -1.0f, 1.0f, 1.0f};

    const std::vector<float> inputs_bin_conv{1.0f, 0.0f, 0.0f, 1.0f,
                                             1.0f, 1.0f, 0.0f, 0.0f,
                                             0.0f, 0.0f, 0.0f, 1.0f,
                                             1.0f, 0.0f, 1.0f, 1.0f};

    const Shape filters_shape{1, 1, 3, 3};
    const std::vector<float> filters_conv{1.0f, -1.0f, 1.0f,
                                          -1.0f, 1.0f, -1.0f,
                                          1.0f, -1.0f, 1.0f};

    const std::vector<uint8_t> filters_bin_conv{0xAA, 0x80}; // 10101010 10000000

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{1.0f, 1.0f,
                                     3.0f, -1.0f};

    BinaryConvolutionTest(
        inputs_bin_conv,
        inputs_shape,
        filters_bin_conv,
        filters_shape,
        outputs,
        outputs_shape,
        strides,
        padding,
        dilations);

    ConvolutionTest(
        inputs_conv,
        inputs_shape,
        filters_conv,
        filters_shape,
        outputs,
        outputs_shape,
        strides,
        padding,
        dilations);
}
// clang-format on

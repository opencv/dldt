//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, gather_4d_indices_no_axis_uint8)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2, 3, 4};
    Shape out_shape{2, 2, 3, 4, 2};
    auto D = make_shared<op::Parameter>(element::u8, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<uint8_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int32_t>({0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2});
    test_case.add_expected_output<uint8_t>(out_shape,
        {10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
         10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
         10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
         10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
         10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
         10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gather_4d_indices_no_axis_2d_input)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2, 3, 4};
    Shape out_shape{2, 2, 3, 4, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f});
    test_case.add_input<int32_t>({0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2});
    test_case.add_expected_output<float>(out_shape,
        {1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f, 1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,
         1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f, 1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,
         1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f, 1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,
         1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f, 1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,
         1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f, 1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,
         1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f, 1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_3d_indices_no_axis_2d_input)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 3, 4};
    Shape out_shape{2, 3, 4, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f});
    test_case.add_input<int32_t>(
        {0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2});
    test_case.add_expected_output<float>(out_shape,
        {1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f, 1.0f, 1.1f, 2.0f, 2.1f,
         2.0f, 2.1f, 3.0f, 3.1f, 1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f,
         1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f, 1.0f, 1.1f, 2.0f, 2.1f,
         2.0f, 2.1f, 3.0f, 3.1f, 1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_2d_indices_no_axis_2d_input)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f});
    test_case.add_input<int32_t>({0, 1, 1, 2});
    test_case.add_expected_output<float>(out_shape, {1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_2d_negative_and_positive_indices_no_axis_2d_input)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f});
    test_case.add_input<int32_t>({0, -2, 1, 2});
    test_case.add_expected_output<float>(out_shape, {1.0f, 1.1f, 2.0f, 2.1f, 2.0f, 2.1f, 3.0f, 3.1f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_1d_indices_no_axis_1d_input)
{
    Shape data_shape{3};
    Shape indices_shape{2};
    Shape out_shape{2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<int32_t>({1, 0});
    test_case.add_expected_output<float>(out_shape, {2.0f, 1.0f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_scalar_indices_no_axis_2d_input)
{
    Shape data_shape{3, 2};
    Shape indices_shape{};
    Shape out_shape{2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f});
    test_case.add_input<int32_t>({1});
    test_case.add_expected_output<float>(out_shape, {2.0f, 2.1f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_2d_indices_axis_1_2d_input)
{
    Shape data_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{3, 1, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I, 1);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f});
    test_case.add_input<int32_t>({0, 2});
    test_case.add_expected_output<float>(out_shape, {1.0f, 1.2f, 2.0f, 2.2f, 3.0f, 3.2f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_1d_indices_axis_2_4d_input)
{
    Shape data_shape{2, 2, 3, 3};
    Shape indices_shape{2};
    Shape out_shape{2, 2, 2, 3};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I, 2);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f,
                                1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f,
                                1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f,
                                1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f});
    test_case.add_input<int32_t>({0, 2});
    test_case.add_expected_output<float>(out_shape,
        {1.0f, 1.1f, 1.2f, 3.0f, 3.1f, 3.2f, 1.0f, 1.1f, 1.2f, 3.0f, 3.1f, 3.2f,
         1.0f, 1.1f, 1.2f, 3.0f, 3.1f, 3.2f, 1.0f, 1.1f, 1.2f, 3.0f, 3.1f, 3.2f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_scalar_indices_axis_1_2d_input)
{
    Shape data_shape{3, 3};
    Shape indices_shape{};
    Shape out_shape{3};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I, 1);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f});
    test_case.add_input<int32_t>({0});
    test_case.add_expected_output<float>(out_shape, {1.0f, 2.0f, 3.0f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_single_indices)
{
    Shape data_shape{3, 3};
    Shape indices_shape{2};
    Shape out_shape{};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f});
    test_case.add_input<int32_t>({1, 2});
    test_case.add_expected_output<float>(out_shape, {1.5f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_scalar_from_2d)
{
    Shape data_shape{2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 1.3f});
    test_case.add_input<int32_t>({0, 0, 1, 1});
    test_case.add_expected_output<float>(out_shape, {1.0f, 1.3f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_1d_from_2d)
{
    Shape data_shape{2, 2};
    Shape indices_shape{2, 1};
    Shape out_shape{2, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 1.3f});
    test_case.add_input<int32_t>({1, 0});
    test_case.add_expected_output<float>(out_shape, {1.2f, 1.3f, 1.0f, 1.1f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_scalar_from_3d)
{
    Shape data_shape{2, 2, 2};
    Shape indices_shape{2, 3};
    Shape out_shape{2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    test_case.add_input<int32_t>({0, 0, 1, 1, 0, 1});
    test_case.add_expected_output<float>(out_shape, {1.1f, 2.1f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_1d_from_3d)
{
    Shape data_shape{2, 2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    test_case.add_input<int32_t>({0, 1, 1, 0});
    test_case.add_expected_output<float>(out_shape, {1.2f, 1.3f, 2.0f, 2.1f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_2d_from_3d)
{
    Shape data_shape{2, 2, 2};
    Shape indices_shape{1, 1};
    Shape out_shape{1, 2, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    test_case.add_input<int32_t>({1});
    test_case.add_expected_output<float>(out_shape, {2.0f, 2.1f, 2.2f, 2.3f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_scalar_from_2d)
{
    Shape data_shape{2, 2};
    Shape indices_shape{2, 1, 2};
    Shape out_shape{2, 1};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 1.3f});
    test_case.add_input<int32_t>({0, 0, 0, 1});
    test_case.add_expected_output<float>(out_shape, {1.0f, 1.1f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_1d_from_2d)
{
    Shape data_shape{2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 1.3f});
    test_case.add_input<int32_t>({1, 0});
    test_case.add_expected_output<float>(out_shape, {1.2f, 1.3f, 1.0f, 1.1f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_scalar_from_3d)
{
    Shape data_shape{2, 2, 2};
    Shape indices_shape{2, 2, 3};
    Shape out_shape{2, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    test_case.add_input<int32_t>({0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0});
    test_case.add_expected_output<float>(out_shape, {1.1f, 2.1f, 1.3f, 2.2f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_1d_from_3d)
{
    Shape data_shape{2, 2, 2};
    Shape indices_shape{2, 2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    test_case.add_input<int32_t>({0, 1, 1, 0, 0, 0, 1, 1});
    test_case.add_expected_output<float>(out_shape, {1.2f, 1.3f, 2.0f, 2.1f, 1.0f, 1.1f, 2.2f, 2.3f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_2d_from_3d)
{
    Shape data_shape{2, 2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    test_case.add_input<int32_t>({1, 0});
    test_case.add_expected_output<float>(out_shape, {2.0f, 2.1f, 2.2f, 2.3f, 1.0f, 1.1f, 1.2f, 1.3f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_int8)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::i8, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int8_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int32_t>({0, 1, 1, 2});
    test_case.add_expected_output<int8_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_int16)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::i16, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int16_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int64_t>({0, 1, 1, 2});
    test_case.add_expected_output<int16_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_int32)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::i32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int32_t>({0, 1, 1, 2});
    test_case.add_expected_output<int32_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_int64)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::i64, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int64_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int64_t>({0, 1, 1, 2});
    test_case.add_expected_output<int64_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_uint8)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::u8, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<uint8_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int32_t>({0, 1, 1, 2});
    test_case.add_expected_output<uint8_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_uint16)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::u16, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<uint16_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int64_t>({0, 1, 1, 2});
    test_case.add_expected_output<uint16_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_uint32)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::u32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<uint32_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int32_t>({0, 1, 1, 2});
    test_case.add_expected_output<uint32_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_uint64)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::u64, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<uint64_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int64_t>({0, 1, 1, 2});
    test_case.add_expected_output<uint64_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_bool)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto D = make_shared<op::Parameter>(element::boolean, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto G = make_shared<op::Gather>(D, I);
    auto f = make_shared<Function>(G, ParameterVector{D, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<char>({1, 1, 1, 0, 0, 1});
    test_case.add_input<int64_t>({0, 1, 1, 2});
    test_case.add_expected_output<char>(out_shape, {1, 1, 1, 0, 1, 0, 0, 1});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

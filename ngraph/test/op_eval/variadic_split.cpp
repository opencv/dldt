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

#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/variadic_split.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, variadic_split_same_lengths)
{
    const auto data_shape = Shape{3, 8, 3};
    const auto data = make_shared<op::Parameter>(element::i64, data_shape);
    const auto axis = make_shared<op::Parameter>(element::i64, Shape{});
    const auto split_lengths = make_shared<op::Parameter>(element::i64, Shape{4});

    auto split = make_shared<op::v1::VariadicSplit>(data, axis, split_lengths);

    auto f = make_shared<Function>(split, ParameterVector{data, axis, split_lengths});

    std::vector<int64_t> data_vec(shape_size(data_shape));
    std::iota(data_vec.begin(), data_vec.end(), 0);

    std::vector<std::vector<int64_t>> expected_results{
        {0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 48, 49, 50, 51, 52, 53},
        {6, 7, 8, 9, 10, 11, 30, 31, 32, 33, 34, 35, 54, 55, 56, 57, 58, 59},
        {12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41, 60, 61, 62, 63, 64, 65},
        {18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47, 66, 67, 68, 69, 70, 71}};

    const vector<int64_t> split_lengths_vec{2, 2, 2, 2};
    HostTensorVector results(split_lengths_vec.size());
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(
        f->evaluate(results,
                    {make_host_tensor<element::Type_t::i64>(data_shape, data_vec),
                     make_host_tensor<element::Type_t::i64>(Shape{}, std::vector<int64_t>{1}),
                     make_host_tensor<element::Type_t::i64>(Shape{4}, split_lengths_vec)}));

    for (int i = 0; i < split_lengths_vec.size(); ++i)
    {
        EXPECT_EQ(results[i]->get_element_type(), element::i64);
        EXPECT_EQ(results[i]->get_shape(),
                  (Shape{3, static_cast<size_t>(split_lengths_vec[i]), 3}));
        EXPECT_EQ(read_vector<int64_t>(results[i]), expected_results[i]);
    }
}

TEST(op_eval, variadic_split_different_lengths)
{
    const auto data_shape = Shape{6, 2, 3};
    const auto data = make_shared<op::Parameter>(element::i64, data_shape);
    const auto axis = make_shared<op::Parameter>(element::i64, Shape{});
    const auto split_lengths = make_shared<op::Parameter>(element::i64, Shape{3});

    auto split = make_shared<op::v1::VariadicSplit>(data, axis, split_lengths);

    auto f = make_shared<Function>(split, ParameterVector{data, axis, split_lengths});

    std::vector<int64_t> data_vec(shape_size(data_shape));
    std::iota(data_vec.begin(), data_vec.end(), 0);

    std::vector<std::vector<int64_t>> expected_results{
        {0, 1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
        {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}};

    const vector<int64_t> split_lengths_vec{1, 2, 3};
    HostTensorVector results(split_lengths_vec.size());
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(
        f->evaluate(results,
                    {make_host_tensor<element::Type_t::i64>(data_shape, data_vec),
                     make_host_tensor<element::Type_t::i64>(Shape{}, std::vector<int64_t>{0}),
                     make_host_tensor<element::Type_t::i64>(Shape{3}, split_lengths_vec)}));

    for (int i = 0; i < split_lengths_vec.size(); ++i)
    {
        EXPECT_EQ(results[i]->get_element_type(), element::i64);
        EXPECT_EQ(results[i]->get_shape(),
                  (Shape{static_cast<size_t>(split_lengths_vec[i]), 2, 3}));
        EXPECT_EQ(read_vector<int64_t>(results[i]), expected_results[i]);
    }
}

TEST(op_eval, variadic_split_neg_length)
{
    const auto data_shape = Shape{2, 7, 1};
    const auto data = make_shared<op::Parameter>(element::i64, data_shape);
    const auto axis = make_shared<op::Parameter>(element::i64, Shape{});
    const auto split_lengths = make_shared<op::Parameter>(element::i64, Shape{3});

    auto split = make_shared<op::v1::VariadicSplit>(data, axis, split_lengths);

    auto f = make_shared<Function>(split, ParameterVector{data, axis, split_lengths});

    std::vector<int64_t> data_vec(shape_size(data_shape));
    std::iota(data_vec.begin(), data_vec.end(), 0);

    std::vector<std::vector<int64_t>> expected_results{
        {0, 1, 2, 7, 8, 9}, {3, 10}, {4, 5, 6, 11, 12, 13}};

    const vector<int64_t> split_lengths_vec{-1, 1, 3};
    HostTensorVector results(split_lengths_vec.size());
    for (auto& result : results)
    {
        result = make_shared<HostTensor>();
    }
    ASSERT_TRUE(
        f->evaluate(results,
                    {make_host_tensor<element::Type_t::i64>(data_shape, data_vec),
                     make_host_tensor<element::Type_t::i64>(Shape{}, std::vector<int64_t>{1}),
                     make_host_tensor<element::Type_t::i64>(Shape{3}, split_lengths_vec)}));

    const vector<size_t> expected_lengths{3, 1, 3};
    for (int i = 0; i < split_lengths_vec.size(); ++i)
    {
        EXPECT_EQ(results[i]->get_element_type(), element::i64);
        EXPECT_EQ(results[i]->get_shape(), (Shape{2, expected_lengths[i], 1}));
        EXPECT_EQ(read_vector<int64_t>(results[i]), expected_results[i]);
    }
}

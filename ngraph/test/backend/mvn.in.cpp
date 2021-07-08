// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_6_no_variance)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3, 3});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{2, 3});

    auto mvn =
        make_shared<op::v6::MVN>(data, axes, false, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    test_case.add_expected_output<float>({-4, -3, -2, -1, 0, 1, 2, 3, 4,
                                          -4, -3, -2, -1, 0, 1, 2, 3, 4,
                                          -4, -3, -2, -1, 0, 1, 2, 3, 4});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_6)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3, 3});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{2, 3});

    auto mvn =
        make_shared<op::v6::MVN>(data, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    test_case.add_expected_output<float>({-1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_6_inside_sqrt)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3, 3});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{2, 3});

    auto mvn =
        make_shared<op::v6::MVN>(data, axes, true, 1e-9, ngraph::op::MVNEpsMode::INSIDE_SQRT);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    test_case.add_expected_output<float>({-1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_6_across_chanells)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3, 3});
    auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});

    auto mvn =
        make_shared<op::v6::MVN>(data, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    test_case.add_expected_output<float>({-1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_6_across_batch)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 2, 2});
    auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{0, 2, 3});

    auto mvn =
        make_shared<op::v6::MVN>(data, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>(
        {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});

    test_case.add_expected_output<float>(
        {-1.5275252,  -1.0910894, -0.65465367, -0.21821788, 0.21821788,  0.65465367,
          1.0910894,   1.5275252, -1.5275252,  -1.0910894, -0.65465367, -0.21821788,
          0.21821788,  0.65465367, 1.0910894,   1.5275252, -1.5275252,  -1.0910894,
         -0.65465367, -0.21821788, 0.21821788,  0.65465367, 1.0910894,   1.5275252});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_1_no_variance_no_across_channels)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3, 3});
    auto mvn = make_shared<op::v0::MVN>(data, false, false, 1e-9);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    test_case.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    test_case.add_expected_output<float>({-4, -3, -2, -1, 0, 1, 2, 3, 4,
                                          -4, -3, -2, -1, 0, 1, 2, 3, 4,
                                          -4, -3, -2, -1, 0, 1, 2, 3, 4});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_1_across_channels_no_variance)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 2, 2});
    auto mvn = make_shared<op::v0::MVN>(data, true, false, 1e-9);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3});

    test_case.add_expected_output<float>({-3.25, -2.25, -1.25,
                                          -0.25,  0.75, 1.75,
                                           2.75,  3.75, 4.75,
                                          -3.25, -2.25, -1.25});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_1_variance_no_across_channels)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3, 3});
    auto mvn = make_shared<op::v0::MVN>(data, false, true, 1e-9);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    test_case.add_expected_output<float>({-1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_1_across_channels_with_variance)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3, 3});

    auto mvn = make_shared<op::v0::MVN>(data, true, true, 1e-9);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    test_case.add_expected_output<float>({-1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934});
    // clang-format on
    test_case.run();
}

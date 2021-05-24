// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, swish)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto swish_func = make_shared<op::v4::Swish>(data);
    EXPECT_EQ(swish_func->get_element_type(), element::f32);
    EXPECT_EQ(swish_func->get_shape(), data->get_output_shape(0));
}

TEST(type_prop, swish_partial)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto swish_func = make_shared<op::v4::Swish>(data);
    EXPECT_EQ(swish_func->get_element_type(), element::f32);
    ASSERT_TRUE(
        swish_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));

    // rank unknown
    auto swish_partial = make_shared<op::v4::Swish>(
        make_shared<op::Parameter>(element::f32, PartialShape::dynamic()));
    ASSERT_TRUE(swish_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, swish_partial_static_rank)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto swish_func = make_shared<op::v4::Swish>(data);
    EXPECT_EQ(swish_func->get_element_type(), element::f32);
    ASSERT_TRUE(
        swish_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));
    ASSERT_TRUE(swish_func->get_output_partial_shape(0).rank().is_static());
}

TEST(type_prop, swish_incompatible_types)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto beta = make_shared<op::Parameter>(element::f16, Shape{});
    try
    {
        const auto swish_func = make_shared<op::v4::Swish>(data, beta);
        FAIL() << "swish_func node was created with incompatible input data types.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Swish inputs must have the same type"));
    }
}

TEST(type_prop, swish_beta_not_scalar)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{1});
    try
    {
        const auto swish_func = make_shared<op::v4::Swish>(data, beta);
        FAIL() << "swish_func node was created with scalar beta value.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Swish input with beta must be scalar"));
    }
}

TEST(type_prop, swish_2_inputs)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{});
    const auto swish_func = make_shared<op::v4::Swish>(data, beta);

    EXPECT_EQ(swish_func->get_element_type(), element::f32);
    ASSERT_TRUE(swish_func->get_output_partial_shape(0).same_scheme(data->get_output_shape(0)));
    ASSERT_TRUE(swish_func->get_output_partial_shape(0).rank().is_static());
}

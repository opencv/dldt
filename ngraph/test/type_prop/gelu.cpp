// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, gelu_default_mode_inference_f32)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 32, 32});
    auto gelu = make_shared<op::v7::Gelu>(param);

    ASSERT_EQ(gelu->get_element_type(), element::f32);
    ASSERT_EQ(gelu->get_shape(), (Shape{1, 32, 32}));
    ASSERT_EQ(gelu->get_approximation_mode(), op::GeluApproximationMode::ERF);
}

TEST(type_prop, gelu_default_mode_inference_f16)
{
    auto param = make_shared<op::Parameter>(element::f16, Shape{1, 32, 32});
    auto gelu = make_shared<op::v7::Gelu>(param);

    ASSERT_EQ(gelu->get_element_type(), element::f16);
    ASSERT_EQ(gelu->get_shape(), (Shape{1, 32, 32}));
    ASSERT_EQ(gelu->get_approximation_mode(), op::GeluApproximationMode::ERF);
}

TEST(type_prop, gelu_tanh_mode_inference_f32)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 32, 32});
    auto gelu = make_shared<op::v7::Gelu>(param, op::GeluApproximationMode::TANH);

    ASSERT_EQ(gelu->get_element_type(), element::f32);
    ASSERT_EQ(gelu->get_shape(), (Shape{1, 32, 32}));
    ASSERT_EQ(gelu->get_approximation_mode(), op::GeluApproximationMode::TANH);
}

TEST(type_prop, gelu_tanh_mode_inference_f16)
{
    auto param = make_shared<op::Parameter>(element::f16, Shape{1, 32, 32});
    auto gelu = make_shared<op::v7::Gelu>(param, op::GeluApproximationMode::TANH);

    ASSERT_EQ(gelu->get_element_type(), element::f16);
    ASSERT_EQ(gelu->get_shape(), (Shape{1, 32, 32}));
    ASSERT_EQ(gelu->get_approximation_mode(), op::GeluApproximationMode::TANH);
}  

TEST(type_prop, gelu_incompatible_input_type_boolean)
{
    auto param = make_shared<op::Parameter>(element::boolean, Shape{1, 32, 32});
    ASSERT_THROW(std::make_shared<op::v7::Gelu>(param), ngraph::NodeValidationFailure);
}

TEST(type_prop, gelu_incompatible_input_type_u16)
{
    auto param = make_shared<op::Parameter>(element::u16, Shape{1, 32, 32});
    ASSERT_THROW(std::make_shared<op::v7::Gelu>(param), ngraph::NodeValidationFailure);
}

TEST(type_prop, gelu_dynamic_rank_input_shape_2D)
{
    const PartialShape param_shape{Dimension::dynamic(), 10};
    const auto param = std::make_shared<op::Parameter>(element::f32, param_shape);
    const auto op = std::make_shared<op::v7::Gelu>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{Dimension(), 10}));
}

TEST(type_prop, gelu_dynamic_rank_input_shape_3D)
{
    const PartialShape param_shape{100, Dimension::dynamic(), 58};
    const auto param = std::make_shared<op::Parameter>(element::f32, param_shape);
    const auto op = std::make_shared<op::v7::Gelu>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{100, Dimension(), 58}));
}

TEST(type_prop, gelu_dynamic_rank_input_shape_full)
{
    const auto param = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto op = std::make_shared<op::v7::Gelu>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unary_ops.hpp"
using Type = ::testing::Types<UnaryOperatorType<ngraph::op::v0::Sinh, element::f32>>;

INSTANTIATE_TYPED_TEST_CASE_P(visitor_without_attribute,
                              UnaryOperatorVisitor,
                              Type,
                              UnaryOperatorTypeName);

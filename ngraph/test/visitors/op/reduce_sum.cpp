// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, reduce_sum_op)
{
    // ReduceSum derives visit_attributes from op::util::ArithmeticReductionKeepDims
    NodeBuilder::get_ops().register_factory<opset1::ReduceSum>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_sum = make_shared<opset1::ReduceSum>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_sum);
    auto g_reduce_sum = as_type_ptr<opset1::ReduceSum>(builder.create());

    EXPECT_EQ(g_reduce_sum->get_keep_dims(), reduce_sum->get_keep_dims());
}

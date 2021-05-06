// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/visitor.hpp"

using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, convert_op)
{
    using Convert = op::v0::Convert;

    NodeBuilder::get_ops().register_factory<Convert>();
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{2, 4});
    const element::Type destination_type = element::Type_t::i32;

    const auto convert = std::make_shared<Convert>(data, destination_type);
    NodeBuilder builder(convert);
    auto g_convert = as_type_ptr<Convert>(builder.create());

    EXPECT_EQ(g_convert->get_destination_type(), convert->get_destination_type());
}

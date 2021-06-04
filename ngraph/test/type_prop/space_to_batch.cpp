// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, space_to_batch_output_shape_2D)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 0});

    auto space_to_batch =
        make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_shape(), (Shape{2 * 5, (128 + 2) / 5}));
}

TEST(type_prop, space_to_batch_output_shape_4D)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 64, 64, 3});
    auto block_shape =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 10, 5, 1});
    auto pads_begin =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 0, 0});

    auto space_to_batch =
        make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_shape(), (Shape{2 * 10 * 5, (64 + 3 + 3) / 10, (64 + 1) / 5, 3}));
}

TEST(type_prop, space_to_batch_output_shape_5D)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 32, 64, 128, 256});
    auto block_shape =
        make_shared<op::Constant>(element::i32, Shape{5}, vector<int64_t>{1, 6, 5, 1, 16});
    auto pads_begin =
        make_shared<op::Constant>(element::i32, Shape{5}, vector<int64_t>{0, 2, 0, 0, 0});
    auto pads_end =
        make_shared<op::Constant>(element::i32, Shape{5}, vector<int64_t>{0, 2, 1, 0, 0});

    auto space_to_batch =
        make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_shape(),
              (Shape{2 * 6 * 5 * 16, (32 + 2 + 2) / 6, (64 + 1) / 5, 128, 256 / 16}));
}

TEST(type_prop, space_to_batch_and_batch_to_space)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 100, 1024, 3});
    auto block_shape =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 12, 100, 2});
    auto pads_begin =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 38, 1});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 5, 38, 0});

    auto space_to_batch =
        make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_shape(),
              (Shape{2 * 12 * 100 * 2, (100 + 3 + 5) / 12, (1024 + 38 + 38) / 100, (3 + 1) / 2}));

    auto batch_to_space =
        make_shared<op::v1::BatchToSpace>(space_to_batch, block_shape, pads_begin, pads_end);
    ASSERT_EQ(batch_to_space->get_element_type(), element::f32);
    ASSERT_EQ(batch_to_space->get_shape(), (Shape{2, 100, 1024, 3}));
}

TEST(type_prop, space_to_batch_dynamic_shape_static_rank)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));
    auto block_shape =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 10, 5, 1});
    auto pads_begin =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 0, 0});

    auto space_to_batch =
        make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, space_to_batch_dynamic_shape_dynamic_rank)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto block_shape =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 10, 5, 1});
    auto pads_begin =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 0, 0});

    auto space_to_batch =
        make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, space_to_batch_invalid_element_type_block_shape)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<op::Constant>(element::f32, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin = make_shared<op::Constant>(element::i64, Shape{2}, vector<float>{0, 2});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 0});

    try
    {
        auto space_to_batch =
            make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);
        // Input element type is float32
        FAIL() << "Invalid f32 element type for block_shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "block_shape must be an integral number");
    }
    catch(...)
    {
        FAIL() << "Integral element type node validation check failed for unexpected reason";
    }

}

TEST(type_prop, space_to_batch_invalid_element_type_pads_begin)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin = make_shared<op::Constant>(element::f32, Shape{2}, vector<float>{0, 2});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 0});

    try
    {
        auto space_to_batch =
            make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);
        // Input element type is float32
        FAIL() << "Invalid f32 element type for pads_begin not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "crops_begin must be an integral number but got");
    }
    catch(...)
    {
        FAIL() << "Integral element type node validation check failed for unexpected reason";
    }

}

TEST(type_prop, space_to_batch_invalid_element_type_pads_end)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<op::Constant>(element::i16, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    auto pads_end = make_shared<op::Constant>(element::f32, Shape{2}, vector<float>{0, 0});

    try
    {
        auto space_to_batch =
            make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);
        // Input element type is float32
        FAIL() << "Invalid f32 element type for pads_end not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "crops_end must be an integral number but got");
    }
    catch(...)
    {
        FAIL() << "Integral element type node validation check failed for unexpected reason";
    }

}

TEST(type_prop, space_to_batch_invalid_value_block_shape)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{-1, -5});
    auto pads_begin = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{2}, vector<float>{0, 0});

    try
    {
        auto space_to_batch =
            make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);
        // Input element type is float32
        FAIL() << "Invalid block_shape value not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "block_shape values must be greater than 0");
    }
    catch(...)
    {
        FAIL() << "block_shape value node validation check failed for unexpected reason";
    }

}


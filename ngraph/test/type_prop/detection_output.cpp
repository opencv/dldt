//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "ngraph/op/detection_output.hpp"
#include "util/type_prop.hpp"

#include <memory>

using namespace std;
using namespace ngraph;

std::shared_ptr<op::DetectionOutput>
    create_detection_output_with_shape(const PartialShape& box_logits_shape,
                                       const PartialShape& class_preds_shape,
                                       const PartialShape& proposals_shape,
                                       const PartialShape& aux_class_preds_shape,
                                       const PartialShape& aux_box_preds_shape,
                                       const op::DetectionOutputAttrs& attrs)
{
    auto box_logits = make_shared<op::Parameter>(element::f32, box_logits_shape);
    auto class_preds = make_shared<op::Parameter>(element::f32, class_preds_shape);
    auto proposals = make_shared<op::Parameter>(element::f32, proposals_shape);
    auto aux_class_preds = make_shared<op::Parameter>(element::f32, aux_class_preds_shape);
    auto aux_box_preds = make_shared<op::Parameter>(element::f32, aux_box_preds_shape);
    return make_shared<op::DetectionOutput>(
        box_logits, class_preds, proposals, aux_class_preds, aux_box_preds, attrs);
}

TEST(type_prop_layers, detection_output)
{
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 2;
    attrs.normalized = true;
    auto op = create_detection_output_with_shape(
        Shape{4, 20}, Shape{4, 10}, Shape{4, 2, 20}, Shape{4, 10}, Shape{4, 20}, attrs);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 800, 7}));
}

TEST(type_prop_layers, detection_output_not_normalized)
{
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 2;
    attrs.normalized = false;
    auto op = create_detection_output_with_shape(
        Shape{4, 20}, Shape{4, 10}, Shape{4, 2, 25}, Shape{4, 10}, Shape{4, 20}, attrs);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 800, 7}));
}

TEST(type_prop_layers, detection_output_negative_keep_top_k)
{
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = -1;
    attrs.normalized = true;
    attrs.num_classes = 2;
    auto op = create_detection_output_with_shape(
        Shape{4, 20}, Shape{4, 10}, Shape{4, 2, 20}, Shape{4, 10}, Shape{4, 20}, attrs);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 40, 7}));
}

TEST(type_prop_layers, detection_output_no_share_location)
{
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = -1;
    attrs.normalized = true;
    attrs.num_classes = 2;
    attrs.share_location = false;
    auto op = create_detection_output_with_shape(
        Shape{4, 40}, Shape{4, 40}, Shape{4, 2, 20}, Shape{4, 40}, Shape{4, 40}, attrs);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 40, 7}));
}

TEST(type_prop_layers, detection_output_top_k)
{
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = 7;
    attrs.normalized = true;
    attrs.num_classes = 2;
    auto op = create_detection_output_with_shape(
        Shape{4, 20}, Shape{4, 40}, Shape{4, 2, 20}, Shape{4, 40}, Shape{4, 20}, attrs);
    ASSERT_EQ(op->get_shape(), (Shape{1, 1, 56, 7}));
}

TEST(type_prop_layers, detection_output_all_dynamic_shapes)
{
    PartialShape dyn_shape = PartialShape::dynamic();
    op::DetectionOutputAttrs attrs;
    attrs.num_classes = 1;
    auto op = create_detection_output_with_shape(
        dyn_shape, dyn_shape, dyn_shape, dyn_shape, dyn_shape, attrs);
    ASSERT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 1, Dimension::dynamic(), 7}));
}

TEST(type_prop_layers, detection_output_dynamic_batch)
{
    op::DetectionOutputAttrs attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 2;
    attrs.normalized = true;
    auto op = create_detection_output_with_shape(PartialShape{Dimension::dynamic(), 20},
                                                 PartialShape{Dimension::dynamic(), 40},
                                                 PartialShape{Dimension::dynamic(), 2, 20},
                                                 PartialShape{Dimension::dynamic(), 40},
                                                 PartialShape{Dimension::dynamic(), 20},
                                                 attrs);
    ASSERT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 1, Dimension::dynamic(), 7}}));
}

void detection_output_invalid_data_type_test(element::Type box_logits_et,
                                             element::Type class_preds_et,
                                             element::Type proposals_et,
                                             element::Type aux_class_preds_et,
                                             element::Type aux_box_preds_et,
                                             const std::string& expected_msg)
{
    try
    {
        auto box_logits = make_shared<op::Parameter>(box_logits_et, Shape{4, 20});
        auto class_preds = make_shared<op::Parameter>(class_preds_et, Shape{4, 40});
        auto proposals = make_shared<op::Parameter>(proposals_et, Shape{4, 2, 20});
        auto aux_class_preds = make_shared<op::Parameter>(aux_class_preds_et, Shape{4, 40});
        auto aux_box_preds = make_shared<op::Parameter>(aux_box_preds_et, Shape{4, 20});
        op::DetectionOutputAttrs attrs;
        attrs.keep_top_k = {200};
        attrs.num_classes = 2;
        attrs.normalized = true;
        auto op = make_shared<op::DetectionOutput>(
            box_logits, class_preds, proposals, aux_class_preds, aux_box_preds, attrs);
        FAIL() << "Exception expected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), expected_msg);
    }
    catch (...)
    {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop_layers, detection_output_invalid_data_type)
{
    detection_output_invalid_data_type_test(
        element::i32,
        element::f32,
        element::f32,
        element::f32,
        element::f32,
        "Box logits' data type must be floating point. Got i32");
    detection_output_invalid_data_type_test(
        element::f32,
        element::i32,
        element::f32,
        element::f32,
        element::f32,
        "Class predictions' data type must be floating point. Got i32");
    detection_output_invalid_data_type_test(element::f32,
                                            element::f32,
                                            element::i32,
                                            element::f32,
                                            element::f32,
                                            "Proposals' data type must be floating point. Got i32");
    detection_output_invalid_data_type_test(
        element::f32,
        element::f32,
        element::f32,
        element::i32,
        element::f32,
        "Additional class predictions' data type must be floating point. Got i32");
    detection_output_invalid_data_type_test(
        element::f32,
        element::f32,
        element::f32,
        element::f32,
        element::i32,
        "Additional box predictions' data type must be floating point. Got i32");
}

TEST(type_prop_layers, detection_output_mismatched_batch_size)
{
    {
        try
        {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {200};
            attrs.num_classes = 2;
            attrs.normalized = true;
            auto op = create_detection_output_with_shape(
                Shape{4, 20}, Shape{5, 20}, Shape{4, 2, 20}, Shape{4, 40}, Shape{4, 20}, attrs);
            FAIL() << "Exception expected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string(
                    "Class predictions' first dimension is not compatible with batch size."));
        }
        catch (...)
        {
            FAIL() << "Unknown exception was thrown";
        }
    }
    {
        try
        {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {200};
            attrs.num_classes = 2;
            attrs.normalized = true;
            auto op = create_detection_output_with_shape(
                Shape{4, 20}, Shape{4, 20}, Shape{5, 2, 20}, Shape{4, 40}, Shape{4, 20}, attrs);
            FAIL() << "Exception expected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Proposals' first dimension is not compatible with batch size."));
        }
        catch (...)
        {
            FAIL() << "Unknown exception was thrown";
        }
    }
}

TEST(type_prop_layers, detection_output_invalid_ranks)
{
    {
        try
        {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {200};
            attrs.num_classes = 2;
            attrs.normalized = true;
            auto op = create_detection_output_with_shape(
                Shape{4, 20, 1}, Shape{4, 40}, Shape{4, 2, 20}, Shape{4, 40}, Shape{4, 20}, attrs);
            FAIL() << "Exception expected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Box logits rank must be 2. Got 3"));
        }
        catch (...)
        {
            FAIL() << "Unknown exception was thrown";
        }
    }
    {
        try
        {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {200};
            attrs.num_classes = 2;
            attrs.normalized = true;
            auto op = create_detection_output_with_shape(
                Shape{4, 20}, Shape{4, 40, 1}, Shape{4, 2, 20}, Shape{4, 40}, Shape{4, 20}, attrs);
            FAIL() << "Exception expected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Class predictions rank must be 2. Got 3"));
        }
        catch (...)
        {
            FAIL() << "Unknown exception was thrown";
        }
    }
    {
        try
        {
            op::DetectionOutputAttrs attrs;
            attrs.keep_top_k = {200};
            attrs.num_classes = 2;
            attrs.normalized = true;
            auto op = create_detection_output_with_shape(
                Shape{4, 20}, Shape{4, 40}, Shape{4, 2}, Shape{4, 40}, Shape{4, 20}, attrs);
            FAIL() << "Exception expected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Proposals rank must be 3. Got 2"));
        }
        catch (...)
        {
            FAIL() << "Unknown exception was thrown";
        }
    }
}

TEST(type_prop_layers, detection_output_invalid_box_logits_shape)
{
    // share_location = true
    {
        try
        {
            op::DetectionOutputAttrs attrs;
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output_with_shape(
                Shape{4, 13}, Shape{4, 9}, Shape{4, 2, 12}, Shape{4, 9}, Shape{4, 12}, attrs);
            FAIL() << "Exception expected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string(
                    "Box logits' second dimension must be a multiply of num_loc_classes * 4 (4)"));
        }
        catch (...)
        {
            FAIL() << "Unknown exception was thrown";
        }
    }
    // share_location = false
    {
        try
        {
            op::DetectionOutputAttrs attrs;
            attrs.num_classes = 3;
            attrs.share_location = false;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output_with_shape(
                Shape{4, 37}, Shape{4, 9}, Shape{4, 2, 12}, Shape{4, 9}, Shape{4, 12}, attrs);
            FAIL() << "Exception expected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string(
                    "Box logits' second dimension must be a multiply of num_loc_classes * 4 (12)"));
        }
        catch (...)
        {
            FAIL() << "Unknown exception was thrown";
        }
    }
}

TEST(type_prop_layers, detection_output_invalid_class_preds_shape)
{
    try
    {
        op::DetectionOutputAttrs attrs;
        attrs.num_classes = 3;
        auto op = create_detection_output_with_shape(
            Shape{4, 12}, Shape{4, 10}, Shape{4, 2, 12}, Shape{4, 9}, Shape{4, 12}, attrs);
        FAIL() << "Exception expected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Class predictions' second dimension must be a multiply of num_classes (3)"));
    }
    catch (...)
    {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop_layers, detection_output_invalid_proposals_shape)
{
    // variance_encoded_in_target = false
    {
        try
        {
            op::DetectionOutputAttrs attrs;
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output_with_shape(
                Shape{4, 12}, Shape{4, 9}, Shape{4, 1, 12}, Shape{4, 9}, Shape{4, 12}, attrs);
            FAIL() << "Exception expected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string(
                    "Proposals' second dimension is mismatched. Current value is: 1, expected: 2"));
        }
        catch (...)
        {
            FAIL() << "Unknown exception was thrown";
        }
    }
    // variance_encoded_in_target = true
    {
        try
        {
            op::DetectionOutputAttrs attrs;
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = true;
            attrs.normalized = true;
            auto op = create_detection_output_with_shape(
                Shape{4, 12}, Shape{4, 9}, Shape{4, 2, 12}, Shape{4, 9}, Shape{4, 12}, attrs);
            FAIL() << "Exception expected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string(
                    "Proposals' second dimension is mismatched. Current value is: 2, expected: 1"));
        }
        catch (...)
        {
            FAIL() << "Unknown exception was thrown";
        }
    }
    // normalized = false
    {
        try
        {
            op::DetectionOutputAttrs attrs;
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = false;
            auto op = create_detection_output_with_shape(
                Shape{4, 12}, Shape{4, 9}, Shape{4, 2, 16}, Shape{4, 9}, Shape{4, 12}, attrs);
            FAIL() << "Exception expected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Proposals' third dimension must be a multiply of prior_box_size (5)"));
        }
        catch (...)
        {
            FAIL() << "Unknown exception was thrown";
        }
    }
    // normalized = true
    {
        try
        {
            op::DetectionOutputAttrs attrs;
            attrs.num_classes = 3;
            attrs.share_location = true;
            attrs.variance_encoded_in_target = false;
            attrs.normalized = true;
            auto op = create_detection_output_with_shape(
                Shape{4, 12}, Shape{4, 9}, Shape{4, 2, 13}, Shape{4, 9}, Shape{4, 12}, attrs);
            FAIL() << "Exception expected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("Proposals' third dimension must be a multiply of prior_box_size (4)"));
        }
        catch (...)
        {
            FAIL() << "Unknown exception was thrown";
        }
    }
}

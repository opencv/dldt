// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <legacy/ngraph_ops/nms_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_nms_5_to_legacy.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertNMS5ToNMSIEStatic) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_nms_sigma = ngraph::opset5::Constant::create(element::f32, Shape{}, {0.25});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               soft_nms_sigma, opset5::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        const auto &orig_selected_indices_shape = f->get_output_partial_shape(0);
        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_TRUE(f->get_output_partial_shape(0).is_static()) << "Shape " << f->get_output_partial_shape(0) << " should be static";
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_nms_sigma = ngraph::opset5::Constant::create(element::f32, Shape{}, {0.25});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                       opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                                one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto new_soft_nms_sigma = std::make_shared<opset5::Reshape>(soft_nms_sigma,
                                                                    opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                             one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              new_soft_nms_sigma, 0, true);
        nms->set_friendly_name("nms");

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
        ASSERT_TRUE(f_ref->get_output_partial_shape(0).is_static()) << "Shape " << f_ref->get_output_partial_shape(0) << " should be static";
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

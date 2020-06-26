// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include <ngraph/graph_util.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph_ops/nms_ie.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>

#include "transformations/convert_nms_4_to_legacy.hpp"

bool ngraph::pass::ConvertNMS4ToLegacy::run_on_function(std::shared_ptr<ngraph::Function> f) {
    bool rewritten = false;
    for (auto &node : f->get_ops()) {
        auto nms_4 = ngraph::as_type_ptr<opset4::NonMaxSuppression>(node);
        if (!nms_4)
            continue;

        const auto box_encoding = static_cast<const op::v3::NonMaxSuppression::BoxEncodingType>(nms_4->get_box_encoding());
        const auto new_args = nms_4->input_values();
        NODE_VALIDATION_CHECK(nms_4.get(),
                              new_args.size() >= 2 && new_args.size() <= 5,
                              "Number of inputs must be 2, 3, 4 or 5");

        const auto& arg2 = new_args.size() > 2 ? new_args.at(2) : ngraph::opset4::Constant::create(element::i32, Shape{}, {0});
        const auto& arg3 = new_args.size() > 3 ? new_args.at(3) : ngraph::opset4::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg4 = new_args.size() > 4 ? new_args.at(4) : ngraph::opset4::Constant::create(element::f32, Shape{}, {.0f});

        const auto max_output_boxes_per_class_rank = arg2.get_partial_shape().rank();
        const auto iou_threshold_rank = arg3.get_partial_shape().rank();
        const auto score_threshold_rank = arg4.get_partial_shape().rank();

        // Check that required ranks are not dynamic
        if (max_output_boxes_per_class_rank.is_dynamic() ||
            iou_threshold_rank.is_dynamic() ||
            score_threshold_rank.is_dynamic()) {
            return false;
        }

        if (max_output_boxes_per_class_rank.get_length() == 1 &&
            iou_threshold_rank.get_length() == 1 &&
            score_threshold_rank.get_length() == 1) {
            return false;
        }

        // vector of new nGraph operations
        NodeVector new_ops;

        auto new_max_per_class = arg2;
        if (max_output_boxes_per_class_rank.get_length() == 0) {
            // WA: we need to create Constant manually because it requires by NMS shape inference
            //     otherwise we will get dynamic shape until first CF is executed. It can be resolved
            //     if CF will be executed right after transformation and before Validate pass.
            if (auto new_max_per_class_const = std::dynamic_pointer_cast<opset1::Constant>(new_max_per_class.get_node_shared_ptr())) {
                new_max_per_class = opset1::Constant::create(element::i64, Shape{1}, new_max_per_class_const->cast_vector<int64_t>());
            } else {
                new_max_per_class = std::make_shared<ngraph::op::Unsqueeze>(arg2, opset1::Constant::create(element::i64, Shape{1}, {0}));
                new_ops.push_back(new_max_per_class.get_node_shared_ptr());
            }
        }
        auto new_iou_threshold = arg3;
        if (iou_threshold_rank.get_length() == 0) {
            new_iou_threshold = std::make_shared<ngraph::op::Unsqueeze>(arg3, opset1::Constant::create(element::i64, Shape{1}, {0}));
            new_ops.push_back(new_iou_threshold.get_node_shared_ptr());
        }
        auto new_score_threshold = arg4;
        if (score_threshold_rank.get_length() == 0) {
            new_score_threshold = std::make_shared<ngraph::op::Unsqueeze>(arg4, opset1::Constant::create(element::i64, Shape{1}, {0}));
            new_ops.push_back(new_score_threshold.get_node_shared_ptr());
        }

        int center_point_box = 0;
        switch (nms_4->get_box_encoding()) {
            case ::ngraph::opset3::NonMaxSuppression::BoxEncodingType::CENTER:
                center_point_box = 1;
                break;
            case ::ngraph::opset3::NonMaxSuppression::BoxEncodingType::CORNER:
                center_point_box = 0;
                break;
            default:
                throw ngraph_error("NonMaxSuppression layer " + nms_4->get_friendly_name() +
                                   " has unsupported box encoding");
        }
        const auto nms_legacy = std::make_shared<op::NonMaxSuppressionIE2>(
                new_args.at(0),
                new_args.at(1),
                new_max_per_class,
                new_iou_threshold,
                new_score_threshold,
                center_point_box,
                nms_4->get_sort_result_descending());
        new_ops.push_back(nms_legacy);

        Output<Node> last;
        // if the output is the i32 then it matches behavior of the v1::NonMaxSuppression otherwise need to insert Convert
        if (nms_4->get_output_type() == element::i32) {
            last = nms_legacy;
        } else {
            last = std::make_shared<ngraph::opset4::Convert>(nms_legacy, nms_4->get_output_type());
            new_ops.push_back(last.get_node_shared_ptr());
        }

        last.get_node_shared_ptr()->set_friendly_name(nms_4->get_friendly_name());
        ngraph::copy_runtime_info(nms_4, last.get_node_shared_ptr());
        ngraph::replace_node(nms_4, last.get_node_shared_ptr());
        rewritten = true;
    }
    return rewritten;
}

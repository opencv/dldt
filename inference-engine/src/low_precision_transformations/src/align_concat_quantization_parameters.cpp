// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/align_concat_quantization_parameters.hpp"

#include <algorithm>
#include <assert.h>
#include <memory>
#include <unordered_set>
#include <set>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/layer_transformation.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/quantization_alignment_intervals_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_value_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

//void replaceAttributeInNodes(
//    std::shared_ptr<ngraph::Function> f,
//    const std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<QuantizationAlignmentIntervalsAttribute>>> newAttribute,
//    const std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<QuantizationAlignmentIntervalsAttribute>>> oldAttribute,
//    const std::shared_ptr<ngraph::Node>& initialNode) {
//    const std::string name = ngraph::VariantWrapper<std::shared_ptr<QuantizationAlignmentIntervalsAttribute>>::type_info.name;
//
//    std::set<std::shared_ptr<Node>> visited;
//    std::deque<std::shared_ptr<Node>> nodes;
//    nodes.emplace_back(initialNode);
//
//    bool initialNodeIsNotInitialized = true;
//
//    while (!nodes.empty()) {
//        auto node = nodes.front();
//        nodes.pop_front();
//
//        if (visited.count(node) || is_type<op::Constant>(node)) {
//            continue;
//        }
//
//        visited.insert(node);
//
//        bool handleConnectedNodes = false;
//        if (NetworkHelper::isPrecisionPreserved(node) || is_type<opset1::FakeQuantize>(node)) {
//            auto& rt = node->get_rt_info();
//
//            if (node == initialNode) {
//                rt[name] = newAttribute;
//                handleConnectedNodes = true;
//            } else {
//                auto it = rt.find(name);
//                if (it != rt.end()) {
//                    const auto currentAttribute = it->second;
//                    if (oldAttribute.get() == currentAttribute.get()) {
//                        rt[name] = newAttribute;
//                    }
//                    handleConnectedNodes = true;
//                }
//            }
//        }
//
//        if (!handleConnectedNodes) {
//            continue;
//        }
//
//        if (!is_type<opset1::FakeQuantize>(node)) {
//            for (auto& input : node->inputs()) {
//                const auto& input_node = input.get_source_output().get_node_shared_ptr();
//                if (visited.count(input_node) || is_type<op::Constant>(input_node)) {
//                    continue;
//                }
//
//                nodes.push_front(input_node);
//            }
//        }
//
//        for (auto& output : node->outputs()) {
//            for (auto& input_value : output.get_target_inputs()) {
//                const auto& output_node = input_value.get_node()->shared_from_this();
//                if (visited.count(output_node) || is_type<op::Constant>(output_node)) {
//                    continue;
//                }
//
//                nodes.push_front(output_node);
//            }
//        }
//    }
//}

void replaceAttributeInNodes(
    std::shared_ptr<ngraph::Function> f,
    const std::string& name,
    const std::shared_ptr<ngraph::Variant> newAttribute,
    const std::shared_ptr<ngraph::Variant> oldAttribute,
    const std::shared_ptr<ngraph::Node>& initialNode) {
    std::set<std::shared_ptr<Node>> visited;
    std::deque<std::shared_ptr<Node>> nodes;
    nodes.emplace_back(initialNode);

    bool initialNodeIsNotInitialized = true;

    while (!nodes.empty()) {
        auto node = nodes.front();
        nodes.pop_front();

        if (visited.count(node) || is_type<op::Constant>(node)) {
            continue;
        }

        visited.insert(node);

        bool handleConnectedNodes = false;
        if (NetworkHelper::isPrecisionPreserved(node) || is_type<opset1::FakeQuantize>(node)) {
            auto& rt = node->get_rt_info();

            if (node == initialNode) {
                rt[name] = newAttribute;
                handleConnectedNodes = true;
            } else {
                auto it = rt.find(name);
                if (it != rt.end()) {
                    const auto currentAttribute = it->second;
                    if (oldAttribute.get() == currentAttribute.get()) {
                        rt[name] = newAttribute;
                    }
                    handleConnectedNodes = true;
                }
            }
        }

        if (!handleConnectedNodes) {
            continue;
        }

        if (!is_type<opset1::FakeQuantize>(node)) {
            for (auto& input : node->inputs()) {
                const auto& input_node = input.get_source_output().get_node_shared_ptr();
                if (visited.count(input_node) || is_type<op::Constant>(input_node)) {
                    continue;
                }

                nodes.push_front(input_node);
            }
        }

        for (auto& output : node->outputs()) {
            for (auto& input_value : output.get_target_inputs()) {
                const auto& output_node = input_value.get_node()->shared_from_this();
                if (visited.count(output_node) || is_type<op::Constant>(output_node)) {
                    continue;
                }

                nodes.push_front(output_node);
            }
        }
    }
}

// merge: share between other operations - implicit backward propagation
template <typename T>
void mergeAndReplace(
    std::shared_ptr<ngraph::Function> f,
    const std::shared_ptr<ngraph::Node>& node,
    std::shared_ptr<ngraph::VariantWrapper<T>> firstExistingIntervalsAttribute,
    const std::vector<std::shared_ptr<ngraph::Node>>& inputNodes) {
    if (firstExistingIntervalsAttribute != nullptr) {
        auto attribute = firstExistingIntervalsAttribute->merge(inputNodes);
        auto newAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<T>>(attribute);
        assert(newAttribute != nullptr);

        bool wasReplaced = false;
        for (size_t i = 1ul; i < inputNodes.size(); i++) {
            auto oldAttribute = ngraph::pass::low_precision::getAttribute<T>(inputNodes[i]);
            if (oldAttribute != nullptr) {
                const std::string name = ngraph::VariantWrapper<T>::type_info.name;
                replaceAttributeInNodes(f, name, newAttribute, oldAttribute, node);
                wasReplaced = true;
            }
        }
        if (!wasReplaced) {
            node->get_rt_info()[ngraph::VariantWrapper<T>::type_info.name] = newAttribute;
        }
    }
}

bool ngraph::pass::low_precision::AlignConcatQuantizationParamters::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        // create new
        if (ngraph::is_type<opset1::FakeQuantize>(node)) {
            // TODO: FakeQuantize validation is skipped: if FakeQuantize will be not handled then ignore it
            const std::vector<float> lowIntervals = as_type<opset1::Constant>(node->get_input_node_ptr(3))->cast_vector<float>();
            const float lowInterval = *std::min_element(lowIntervals.begin(), lowIntervals.end());

            const auto& highIntervals = as_type<opset1::Constant>(node->get_input_node_ptr(4))->cast_vector<float>();
            const float highInterval = *std::max_element(highIntervals.begin(), highIntervals.end());

            auto& rtInfo = node->get_rt_info();

            const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(ngraph::as_type_ptr<opset1::FakeQuantize>(node));
            const LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(quantizationDetails);

            const auto attribute = std::make_shared<::ngraph::VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>>(
                std::make_shared<QuantizationAlignmentIntervalsAttribute>(lowInterval, highInterval));
            rtInfo[ngraph::VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>::type_info.name] = attribute;
            continue;
        }

        if (is_type<opset1::Convolution>(node)) {
            auto& rtInfo = node->get_input_node_shared_ptr(0)->get_rt_info();
            auto it = rtInfo.find(ngraph::VariantWrapper<QuantizationAlignmentValueAttributePtr>::type_info.name);
            if (it != rtInfo.end()) {
                auto attributeWrapper = std::dynamic_pointer_cast<ngraph::VariantWrapper<QuantizationAlignmentValueAttributePtr>>(it->second);
                assert(attributeWrapper != nullptr);
                attributeWrapper->get()->hasToBeAligned = true;
                continue;
            }
        }

        if (!NetworkHelper::isPrecisionPreserved(node)) {
            continue;
        }

        // TODO: limitation: one operation type is used
        std::shared_ptr<ngraph::VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>> firstExistingIntervalsAttribute;
        std::shared_ptr<ngraph::VariantWrapper<QuantizationAlignmentValueAttributePtr>> firstExistingValueAttribute;

        //auto getAttribute = [](std::shared_ptr<Node>& inputNode) -> std::shared_ptr<ngraph::VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>> {
        //    auto& rtInfo = inputNode->get_rt_info();
        //    auto it = rtInfo.find(ngraph::VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>::type_info.name);
        //    if (it == rtInfo.end()) {
        //        return nullptr;
        //    }

        //    auto tmpAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>>(it->second);
        //    assert(tmpAttribute != nullptr);
        //    return tmpAttribute;
        //};

        // get nodes
        std::vector<std::shared_ptr<ngraph::Node>> inputNodes;
        for (const auto& input : node->inputs()) {
            auto inputNode = input.get_source_output().get_node_shared_ptr();

            auto existingIntervalsAttribute = getAttribute<QuantizationAlignmentIntervalsAttributePtr>(inputNode);
            if (existingIntervalsAttribute != nullptr) {
                if (firstExistingIntervalsAttribute == nullptr) {
                    firstExistingIntervalsAttribute = existingIntervalsAttribute;
                }
            }

            auto existingValueAttribute = getAttribute<QuantizationAlignmentValueAttributePtr>(inputNode);
            if (existingValueAttribute != nullptr) {
                if (firstExistingValueAttribute == nullptr) {
                    firstExistingValueAttribute = existingValueAttribute;
                }
            }

            if (is_type<opset1::FakeQuantize>(inputNode)) {
                auto& rt = node->get_rt_info();
                const std::string& name = ngraph::VariantWrapper<QuantizationAlignmentValueAttributePtr>::type_info.name;
                if (rt.find(name) == rt.end()) {
                    const auto attribute = std::make_shared<ngraph::VariantWrapper<QuantizationAlignmentValueAttributePtr>>(
                        std::make_shared<QuantizationAlignmentValueAttribute>());
                    rt[name] = attribute;
                }
            }

            inputNodes.push_back(inputNode);
        }

        //// merge: share between other operations - implicit backward propagation
        //if (firstExistingIntervalsAttribute != nullptr) {
        //    auto attribute = firstExistingIntervalsAttribute->merge(inputNodes);
        //    auto newAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>>(attribute);
        //    assert(newAttribute != nullptr);

        //    bool wasReplaced = false;
        //    for (size_t i = 1ul; i < inputNodes.size(); i++) {
        //        auto oldAttribute = getAttribute<QuantizationAlignmentIntervalsAttributePtr>(inputNodes[i]);
        //        if (oldAttribute != nullptr) {
        //            const std::string name = ngraph::VariantWrapper<std::shared_ptr<QuantizationAlignmentIntervalsAttribute>>::type_info.name;
        //            replaceAttributeInNodes(f, name, newAttribute, oldAttribute, node);
        //            wasReplaced = true;
        //        }
        //    }
        //    if (!wasReplaced) {
        //        node->get_rt_info()[ngraph::VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>::type_info.name] = newAttribute;
        //    }
        //}

        mergeAndReplace<QuantizationAlignmentIntervalsAttributePtr>(f, node, firstExistingIntervalsAttribute, inputNodes);

        mergeAndReplace<QuantizationAlignmentValueAttributePtr>(f, node, firstExistingValueAttribute, inputNodes);
    }
    return true;
}

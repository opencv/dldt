// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/intervals_alignment_attribute.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

IntervalsAlignmentAttribute::IntervalsAlignmentAttribute(
    const IntervalsAlignmentSharedValue::Interval combinedInterval,
    size_t levels) : levels(levels) {
    sharedValue = std::make_shared<IntervalsAlignmentSharedValue>(combinedInterval, combinedInterval, levels);
}

IntervalsAlignmentAttribute::IntervalsAlignmentAttribute(
    const IntervalsAlignmentSharedValue::Interval combinedInterval,
    size_t levels,
    const IntervalsAlignmentSharedValue::Interval minInterval,
    size_t minLevels) : levels(levels) {
    sharedValue = std::make_shared<IntervalsAlignmentSharedValue>(combinedInterval, minInterval, minLevels);
}

template class ngraph::VariantImpl<IntervalsAlignmentAttributePtr>;

constexpr VariantTypeInfo VariantWrapper<IntervalsAlignmentAttributePtr>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<IntervalsAlignmentAttributePtr>::merge(const ngraph::NodeVector& nodes) {
    std::shared_ptr<::ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>> resultAttributeWrapper;
    std::shared_ptr<IntervalsAlignmentAttribute> resultAttribute;

    // update
    for (const std::shared_ptr<ngraph::Node>& node : nodes) {
        auto& rt = node->get_rt_info();
        auto rtIt = rt.find(VariantWrapper<IntervalsAlignmentAttributePtr>::type_info.name);
        if (rtIt == rt.end()) {
            continue;
        }

        auto attributeWrapper = std::dynamic_pointer_cast<VariantWrapper<IntervalsAlignmentAttributePtr>>(rtIt->second);
        auto attribute = attributeWrapper->get();

        if (resultAttributeWrapper == nullptr) {
            resultAttributeWrapper = attributeWrapper;
            resultAttribute = attribute;
            continue;
        }

        // TODO: LPT: copy/past: merge()
        const auto& resultSharedValue = resultAttribute->sharedValue;
        const auto& sharedValue = attribute->sharedValue;
        if (resultAttribute->levels != attribute->levels) {
            // TODO: LPT: not supported right now
            resultAttribute->levels = 0ul;
            resultSharedValue->minLevels = 0ul;
        }

        if (resultSharedValue->combinedInterval.low > sharedValue->combinedInterval.low) {
            resultSharedValue->combinedInterval.low = sharedValue->combinedInterval.low;
        }

        if (resultSharedValue->combinedInterval.high < sharedValue->combinedInterval.high) {
            resultSharedValue->combinedInterval.high = sharedValue->combinedInterval.high;
        }

        assert(!std::isinf(resultSharedValue->combinedInterval.low));
        assert(!std::isinf(resultSharedValue->combinedInterval.high));

        resultSharedValue->preferablePrecisions.insert(sharedValue->preferablePrecisions.begin(), sharedValue->preferablePrecisions.end());

        const auto resultSize = abs(resultSharedValue->minInterval.high - resultSharedValue->minInterval.low);
        const auto size = abs(sharedValue->minInterval.high - sharedValue->minInterval.low);
        if (resultSize > size) {
            resultSharedValue->minInterval = sharedValue->minInterval;

            float dequantizationMul;
            float dequantizationSub;
            float updatedOutputLowValue;
            float updatedOutputHighValue;

            const size_t minLevels = NetworkHelper::calculateLevels(
                0.f,
                DataPrecision::getMaxValue(resultAttribute->levels),
                resultSharedValue->combinedInterval.low,
                resultSharedValue->combinedInterval.high,
                resultSharedValue->minInterval.low,
                resultSharedValue->minInterval.high,
                dequantizationMul,
                dequantizationSub,
                updatedOutputLowValue,
                updatedOutputHighValue);

            resultSharedValue->minLevels = minLevels;
            resultSharedValue->minLevelsOperation = sharedValue->minLevelsOperation;
        }
    }

    return resultAttributeWrapper;
}

std::shared_ptr<VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>>> VariantWrapper<IntervalsAlignmentAttributePtr>::create(
    const std::shared_ptr<ngraph::Node>& node,
    const AttributeParameters& params) {
    if (!is_type<opset1::FakeQuantize>(node)) {
        return nullptr;
    }

    auto fakeQuantize = as_type_ptr<opset1::FakeQuantize>(node);
    if (!QuantizationDetails::outputLayoutIsSupported(fakeQuantize) || !QuantizationDetails::isSupportedLevel(fakeQuantize->get_levels())) {
        return nullptr;
    }

    float lowInterval;
    float highInterval;
    FakeQuantizeDequantization dequantization;
    {
        const auto targetInputs = node->output(0).get_target_inputs();
        if (targetInputs.size() == 1ul) {
            dequantization = NetworkHelper::getDequantizationBelow(node);
        }
    }

    const auto outLow = as_type_ptr<opset1::Constant>(node->get_input_node_shared_ptr(3));
    const auto outHigh = as_type_ptr<opset1::Constant>(node->get_input_node_shared_ptr(4));
    if (!NetworkHelper::isScalarLike(outLow) || !NetworkHelper::isScalarLike(outHigh)) {
        return nullptr;
    }

    if (dequantization.empty()) {
        const std::vector<float> lowIntervals = outLow->cast_vector<float>();
        lowInterval = *std::min_element(lowIntervals.begin(), lowIntervals.end());

        const std::vector<float> highIntervals = outHigh->cast_vector<float>();
        highInterval = *std::max_element(highIntervals.begin(), highIntervals.end());
    } else {
        {
            auto multiplyResult = dequantization.multiplyConstant == nullptr ?
                                  node->get_input_node_ptr(3)->shared_from_this() :
                                  fold<opset1::Multiply>(
                                              foldConvert(node->get_input_node_ptr(3)->shared_from_this(), params.deqPrecision),
                                          dequantization.multiplyConstant);

            auto multiplyResultConstant = as_type_ptr<opset1::Constant>(multiplyResult);
            auto intervals = multiplyResultConstant->cast_vector<float>();
            lowInterval = *std::min_element(intervals.begin(), intervals.end());
        }

        {
            auto multiplyResult = dequantization.multiplyConstant == nullptr ?
                                  node->get_input_node_ptr(4)->shared_from_this() :
                                  fold<opset1::Multiply>(
                                          foldConvert(node->get_input_node_ptr(4)->shared_from_this(), params.deqPrecision),
                                          dequantization.multiplyConstant);

            auto multiplyResultConstant = as_type_ptr<opset1::Constant>(multiplyResult);
            auto intervals = multiplyResultConstant->cast_vector<float>();
            highInterval = *std::max_element(intervals.begin(), intervals.end());
        }
    }

    if (std::isinf(lowInterval) || std::isinf(highInterval)) {
        // ticket: 57172
        return nullptr;
    }

    assert(!std::isinf(lowInterval));
    assert(!std::isinf(highInterval));

    auto& rtInfo = node->get_rt_info();
    const IntervalsAlignmentSharedValue::Interval interval{ lowInterval, highInterval };
    const auto attribute = std::make_shared<::ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>>(
        ngraph::pass::low_precision::make_shared_attribute<IntervalsAlignmentAttribute>(
            interval,
            fakeQuantize->get_levels()));
    rtInfo[ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>::type_info.name] = attribute;

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantize);
    LayerTransformation::PrecisionDetails preferablePrecision = LayerTransformation::getPrecisionDetails(quantizationDetails);
    if (preferablePrecision.precision != element::undefined) {
        attribute->get()->sharedValue->preferablePrecisions.insert(preferablePrecision.precision);
    }

    attribute->get()->sharedValue->minLevelsOperation = node->get_friendly_name();

    return attribute;
}

void VariantWrapper<IntervalsAlignmentAttributePtr>::merge(
    std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>>>>& attributes) {
    std::shared_ptr<IntervalsAlignmentAttribute> resultAttribute = get();
    for (const auto& attributeWrapper : attributes) {
        auto attribute = attributeWrapper->get();

        // TODO: LPT: copy/past: merge()
        const auto& resultSharedValue = resultAttribute->sharedValue;
        const auto& sharedValue = attribute->sharedValue;
        if (resultAttribute->levels != attribute->levels) {
            // TODO: LPT: not supported right now
            resultAttribute->levels = 0ul;
            resultSharedValue->minLevels = 0ul;
        }

        if (resultSharedValue->combinedInterval.low > sharedValue->combinedInterval.low) {
            resultSharedValue->combinedInterval.low = sharedValue->combinedInterval.low;
        }

        if (resultSharedValue->combinedInterval.high < sharedValue->combinedInterval.high) {
            resultSharedValue->combinedInterval.high = sharedValue->combinedInterval.high;
        }

        assert(!std::isinf(resultSharedValue->combinedInterval.low));
        assert(!std::isinf(resultSharedValue->combinedInterval.high));

        resultSharedValue->preferablePrecisions.insert(sharedValue->preferablePrecisions.begin(), sharedValue->preferablePrecisions.end());

        const auto resultSize = abs(resultSharedValue->minInterval.high - resultSharedValue->minInterval.low);
        const auto size = abs(sharedValue->minInterval.high - sharedValue->minInterval.low);
        if (resultSize > size) {
            resultSharedValue->minInterval = sharedValue->minInterval;

            float dequantizationMul;
            float dequantizationSub;
            float updatedOutputLowValue;
            float updatedOutputHighValue;

            const size_t minLevels = NetworkHelper::calculateLevels(
                0.f,
                DataPrecision::getMaxValue(resultAttribute->levels),
                resultSharedValue->combinedInterval.low,
                resultSharedValue->combinedInterval.high,
                resultSharedValue->minInterval.low,
                resultSharedValue->minInterval.high,
                dequantizationMul,
                dequantizationSub,
                updatedOutputLowValue,
                updatedOutputHighValue);

            resultSharedValue->minLevels = minLevels;
            resultSharedValue->minLevelsOperation = sharedValue->minLevelsOperation;
        }
    }
}

std::string VariantWrapper<IntervalsAlignmentAttributePtr>::get_string() {
    std::stringstream preferablePrecisions;
    preferablePrecisions << "{";
    size_t index = 0;
    for (const auto& precision : m_value->sharedValue->preferablePrecisions) {
        preferablePrecisions << (index > 0 ? ", " : "") << precision;
        ++index;
    }
    preferablePrecisions << "}";

    std::stringstream ss;
    ss << m_value->get_string();
    ss << "levels: " + std::to_string(m_value->levels) << ", " <<
        "combined: { " << m_value->sharedValue->combinedInterval.low << ", " << m_value->sharedValue->combinedInterval.high << " }, " <<
        "min: { " << m_value->sharedValue->minInterval.low << ", " << m_value->sharedValue->minInterval.high << " }, "
        "minLevels: " << m_value->sharedValue->minLevels << ", minLevelsOperation: " << m_value->sharedValue->minLevelsOperation << ", " <<
        "preferablePrecisions: " << preferablePrecisions.str();
    return ss.str();
}

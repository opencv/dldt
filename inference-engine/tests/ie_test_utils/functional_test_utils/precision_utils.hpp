// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <ngraph/type/element_type.hpp>

#include "ie_precision.hpp"

namespace FuncTestUtils {
namespace PrecisionUtils {

// Copied from inference-engine/src/inference_engine/ie_ngraph_utils.hpp
inline ::ngraph::element::Type convertIE2nGraphPrc(const InferenceEngine::Precision& precision) {
    InferenceEngine::Precision::ePrecision pType = precision;
    switch (pType) {
        case InferenceEngine::Precision::UNSPECIFIED:
            return ::ngraph::element::Type(::ngraph::element::Type_t::undefined);
        case InferenceEngine::Precision::FP32:
            return ::ngraph::element::Type(::ngraph::element::Type_t::f32);
        case InferenceEngine::Precision::FP16:
            return ::ngraph::element::Type(::ngraph::element::Type_t::f16);
        case InferenceEngine::Precision::U8:
            return ::ngraph::element::Type(::ngraph::element::Type_t::u8);
        case InferenceEngine::Precision::I8:
            return ::ngraph::element::Type(::ngraph::element::Type_t::i8);
        case InferenceEngine::Precision::U16:
            return ::ngraph::element::Type(::ngraph::element::Type_t::u16);
        case InferenceEngine::Precision::I16:
            return ::ngraph::element::Type(::ngraph::element::Type_t::i16);
        case InferenceEngine::Precision::I32:
            return ::ngraph::element::Type(::ngraph::element::Type_t::i32);
        case InferenceEngine::Precision::I64:
            return ::ngraph::element::Type(::ngraph::element::Type_t::i64);
        case InferenceEngine::Precision::BOOL:
            return ::ngraph::element::Type(::ngraph::element::Type_t::boolean);
        case InferenceEngine::Precision::BIN:
            return ::ngraph::element::Type(::ngraph::element::Type_t::u1);
        case InferenceEngine::Precision::Q78:
        case InferenceEngine::Precision::MIXED:
        case InferenceEngine::Precision::CUSTOM:
        default:
            THROW_IE_EXCEPTION << "Incorrect precision!";
    }
}

}  // namespace PrecisionUtils
}  // namespace FuncTestUtils
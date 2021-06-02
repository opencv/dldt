// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <random>
#include <chrono>

#include <ngraph/type/element_type.hpp>
#include "ngraph_helpers.hpp"

namespace NGraphFunctions {
namespace Utils {


template<ngraph::element::Type_t dType>
std::vector<typename ngraph::helpers::nGraphTypesTrait<dType>::value_type> inline
generateVector(size_t vec_len, typename ngraph::helpers::nGraphTypesTrait<dType>::value_type upTo = 10,
               typename ngraph::helpers::nGraphTypesTrait<dType>::value_type startFrom = 1, int32_t seed = 1) {
    using DataType = typename ngraph::helpers::nGraphTypesTrait<dType>::value_type;
    // chose values between int and bool
    using cast_bool_to_int = typename std::conditional<
            std::is_same<DataType, bool>::value,
            unsigned int,
            DataType> :: type;

    std::vector<DataType> res;

    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    typename std::conditional<
            std::is_integral<DataType>::value,
            std::uniform_int_distribution<cast_bool_to_int>,
            std::uniform_real_distribution<cast_bool_to_int>>::type dist(
                    static_cast<cast_bool_to_int>(startFrom), static_cast<cast_bool_to_int>(upTo));

    for (size_t i = 0; i < vec_len; i++) {
        res.push_back(
                static_cast<typename ngraph::helpers::nGraphTypesTrait<dType>::value_type>(dist(gen)));
    }
    return res;
}

std::vector<ngraph::float16> inline generateF16Vector(size_t vec_len, uint32_t upTo = 10, uint32_t startFrom = 1, int32_t seed = 1) {
    std::vector<ngraph::float16> res;

    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_int_distribution<unsigned long> dist(startFrom, upTo);

    for (size_t i = 0; i < vec_len; i++) {
        res.emplace_back(ngraph::float16(static_cast<float>(dist(gen))));
    }
    return res;
}

std::vector<ngraph::bfloat16> inline generateBF16Vector(size_t vec_len, uint32_t upTo = 10, uint32_t startFrom = 1, int32_t seed = 1) {
    std::vector<ngraph::bfloat16> res;

    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_int_distribution<unsigned long> dist(startFrom, upTo);

    for (size_t i = 0; i < vec_len; i++) {
        res.emplace_back(ngraph::bfloat16(static_cast<float>(dist(gen))));
    }
    return res;
}

template<typename fromType, typename toType>
std::vector<toType> castVector(const std::vector<fromType> &vec) {
    std::vector<toType> resVec;
    resVec.reserve(vec.size());
    for (auto& el : vec) {
        resVec.push_back(static_cast<toType>(el));
    }
    return resVec;
}

}  // namespace Utils
}  // namespace NGraphFunctions

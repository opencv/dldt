// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

#include <low_precision/reduce_sum.hpp>
#include "lpt_ngraph_functions/reduce_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/constant.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class ReduceSumTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type preicsionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::vector<int64_t> constantValues;
    bool keepDims;
    Actual actual;
    Expected expected;
};

typedef std::tuple <
    ngraph::Shape,
    ReduceSumTransformationTestValues
> ReduceSumTransformationParams;

class ReduceSumTransformation : public LayerTransformation, public testing::WithParamInterface<ReduceSumTransformationParams> {
public:
    void SetUp() override {
        const ngraph::Shape inputShape = std::get<0>(GetParam());
        const ReduceSumTransformationTestValues& testValues = std::get<1>(GetParam());
        const std::string reduceType = "Sum";

        actualFunction = ngraph::builder::subgraph::ReduceFunction::getOriginal(
            testValues.actual.inputPrecision,
            inputShape,
            testValues.actual.dequantization,
            testValues.constantValues,
            reduceType,
            testValues.keepDims);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ReduceSumTransformation, ngraph::opset1::ReduceSum>(
                low_precision::LayerTransformation::Params(testValues.params));
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ReduceFunction::getReference(
            testValues.expected.inputPrecision,
            inputShape,
            testValues.expected.dequantizationBefore,
            testValues.constantValues,
            reduceType,
            testValues.keepDims,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ReduceSumTransformationParams> obj) {
        const ngraph::Shape inputShape = std::get<0>(obj.param);
        const ReduceSumTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            testValues.actual.inputPrecision << "_" <<
            LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision, inputShape, testValues.params) << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            testValues.expected.preicsionAfterOperation << "_" <<
            testValues.expected.dequantizationAfter << "_" <<
            (testValues.keepDims ? "_keep_dims_" : "_") <<
            "reduction_axes_";
        for (const auto& elem : testValues.constantValues) {
            result << "_" << elem << "_";
        }

        return result.str();
    }
};

TEST_P(ReduceSumTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::Shape> inputShapes = {
    {1, 3, 16, 16},
    {4, 3, 16, 16}
};

const std::vector<ReduceSumTransformationTestValues> addTransformationTestValues = {
    // U8: keep dims, per-channel quantization, reduction by batch
    {
        LayerTransformation::createParamsU8I8(),
        {0},
        true,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::f32,
            {{}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        }
    },
    // U8: keep dims, per-channel quantization with subtract, reduction by channel
    {
        LayerTransformation::createParamsU8I8(),
        {1},
        true,
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{64.f, 128.f, 32.f}, ngraph::element::f32, {1, 3, 1, 1}},
                {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}
            }
        },
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{64.f, 128.f, 32.f}, ngraph::element::f32, {1, 3, 1, 1}},
                {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}
            },
            ngraph::element::f32,
            {}
        }
    },
    // U8: don't keep dims, per-channel quantization, reduction by channel
    {
        LayerTransformation::createParamsU8I8(),
        {1},
        false,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}},
            ngraph::element::f32,
            {}
        }
    },
    // U8: don't keep dims, per-tensor quantization, reduction by channel (reduction constant with negative values)
    {
        LayerTransformation::createParamsU8I8(),
        {-2},
        false,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::f32,
            {{}, {128.f * 16.f}, {0.1f}}
        }
    },
    // U8: keep dims, per-channel quantization, reduction by special dimensions
    {
        LayerTransformation::createParamsU8I8(),
        {2, 3},
        true,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::f32,
            {{}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        }
    },
    // U8: don't keep dims, per-channel quantization, reduction by special dimensions
    {
        LayerTransformation::createParamsU8I8(),
        {2, 3},
        false,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::f32,
            {{}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3}}}
        }
    },
    // U8: don't keep dims, per-channel quantization with subtract, reduction by special dimensions
    {
        LayerTransformation::createParamsU8I8(),
        {2, 3},
        false,
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f, 12.8f, 1.28f}, ngraph::element::f32, {1, 3, 1, 1}},
                {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}
            }
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::f32,
            {
                {},
                {{128.f * 256.f, 12.8f * 256.f, 1.28f * 256.f}, ngraph::element::f32, {1, 3}},
                {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3}}
            }
        }
    },
    // I8: keep dims, per-channel quantization, reduction by batch
    {
        LayerTransformation::createParamsI8I8(),
        {0},
        true,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::f32,
            {{}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        }
    },
    // I8: don't keep dims, per-channel quantization, reduction by channel
    {
        LayerTransformation::createParamsI8I8(),
        {1},
        false,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}},
            ngraph::element::f32,
            {}
        }
    },
    // I8: don't keep dims, per-tensor quantization, reduction by channel (reduction constant with negative values)
    {
        LayerTransformation::createParamsI8I8(),
        {-2},
        false,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {64.f}, {0.1f}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::f32,
            {{}, {64.f * 16.f}, {0.1f}}
        }
    },
    // I8: don't keep dims, per-channel quantization, reduction by special dimensions
    {
        LayerTransformation::createParamsI8I8(),
        {2, 3},
        false,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::f32,
            {{}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3}}}
        }
    },
    // I8: keep dims, per-channel quantization, reduction by special dimensions
    {
        LayerTransformation::createParamsI8I8(),
        {2, 3},
        true,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::f32,
            {{}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        }
    },
    // not update precisions, keep dims, per-channel quantization, reduction by special dimensions
    {
        LayerTransformation::createParamsI8I8().setUpdatePrecisions(false),
        {2, 3},
        true,
        {
            ngraph::element::f32,
            {{}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{}, {}, {{0.1f, 1.f, 10.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        }
    },
    // I8: keep dims, no dequantization, reduction by special dimensions
    {
        LayerTransformation::createParamsI8I8(),
        {2, 3},
        true,
        {
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    ReduceSumTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(addTransformationTestValues)),
    ReduceSumTransformation::getTestCaseName);

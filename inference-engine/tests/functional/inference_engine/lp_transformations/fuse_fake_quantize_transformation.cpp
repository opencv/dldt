// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/transformer.hpp>
#include <transformations/low_precision/fake_quantize.hpp>
#include "ngraph_functions/low_precision_transformations/common/add.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/fuse_fake_quantize_function.hpp"

#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FuseFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeAdd;
        ngraph::builder::subgraph::Add add;
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ngraph::element::Type precisionAfterDequantization;
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeAdd;
        ngraph::builder::subgraph::Add add;
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ngraph::element::Type precisionAfterDequantization;
        ngraph::element::Type precisionFakeQuantizeOnData;
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    };

    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

class FuseFakeQuantizeTransformation : public LayerTransformation, public testing::WithParamInterface<FuseFakeQuantizeTransformationTestValues> {
public:
    void SetUp() override {
        const FuseFakeQuantizeTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::FuseFakeQuantizeFunction::getOriginal(
            testValues.inputShape,
            testValues.actual.precisionBeforeAdd,
            testValues.actual.add,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            testValues.actual.precisionAfterDequantization,
            testValues.actual.precisionAfterDequantization,
            testValues.actual.fakeQuantizeOnData);
        VisualizeTree("/home/vzinoviev/work/model_dumps/model.actual.dot").run_on_function(actualFunction);
        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::FakeQuantizeTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transformer.transform(actualFunction);
        VisualizeTree("/home/vzinoviev/work/model_dumps/model.transformed.dot").run_on_function(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FuseFakeQuantizeFunction::getReference(
            testValues.inputShape,
            testValues.expected.precisionBeforeAdd,
            testValues.expected.add,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantization,
            testValues.expected.precisionAfterDequantization,
            testValues.expected.precisionFakeQuantizeOnData,
            testValues.expected.fakeQuantizeOnData);
        VisualizeTree("/home/vzinoviev/work/model_dumps/model.reference.dot").run_on_function(referenceFunction);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FuseFakeQuantizeTransformationTestValues> obj) {
        const FuseFakeQuantizeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result << testValues.params.updatePrecisions << "_" <<
            testValues.actual.precisionBeforeAdd << "_" <<
            testValues.actual.add.values.size() << "_" <<
            testValues.actual.add.outPrecision << "_" <<
            testValues.actual.add.constantShape << "_" <<
            testValues.actual.precisionBeforeDequantization <<
            testValues.actual.dequantization << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.fakeQuantizeOnData << "_" <<
            testValues.expected.dequantization << "_" <<
            testValues.expected.add.values.size() << "_" <<
            testValues.expected.add.outPrecision << "_" <<
            testValues.expected.add.constantShape;
        return result.str();
    }
};

TEST_P(FuseFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<FuseFakeQuantizeTransformationTestValues> testValues = {
    // 1) Multiply
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {} },
            element::f32,
            element::f32,
            { 256ul, {}, { 0.f }, { 255.f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Multiply + 2) Add
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            { {128}, element::f32 },
            element::f32,
            { {}, {}, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {} },
            element::f32,
            element::f32,
            { 256ul, {}, { -128.f }, { 127.f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Subtract + Multiply
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::f32,
            { {}, { -128 }, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {} },
            element::f32,
            element::f32,
            { 256ul, {}, { -128.f }, { 127.f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Convert + Subtract + Multiply
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::u8,
            { {element::f32}, { -128 }, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::u8,
            { {}, {}, {} },
            element::u8,
            element::f32,
            { 256ul, {}, { -128.f }, { 127.f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Convert + Subtract + Multiply 2) Add
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            { {127}, element::f32 },
            element::f32,
            { {element::f32}, { -128 }, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {} },
            element::f32,
            element::f32,
            { 256ul, {}, { -255.f }, { 0.f }, { 0.f }, { 2.55f } }
        }
    },
    // negative multiply
    {
            Shape{1, 3, 16, 16},
            LayerTransformation::createParamsU8I8(),
            {
                    element::f32,
                    {},
                    element::f32,
                    { {}, { -128 }, { -0.01f } },
                    element::f32,
                    { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
            },
            {
                    element::f32,
                    {},
                    element::f32,
                    { {}, { -128 }, { -0.01f } },
                    element::f32,
                    element::f32,
                    { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
            }
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    FuseFakeQuantizeTransformation,
    ::testing::ValuesIn(testValues),
    FuseFakeQuantizeTransformation::getTestCaseName);

} // namespace

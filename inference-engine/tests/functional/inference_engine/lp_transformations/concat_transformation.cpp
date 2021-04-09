// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/rt_info/intervals_alignment_attribute.hpp>
#include <low_precision/rt_info/quantization_alignment_attribute.hpp>
#include <low_precision/rt_info/precisions_attribute.hpp>
#include <low_precision/rt_info/precision_preserved_attribute.hpp>

#include <low_precision/align_concat_quantization_parameters.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/markup_precisions.hpp>
#include <low_precision/markup_avg_pool_precisions.hpp>
#include <low_precision/propagate_precisions.hpp>

#include <low_precision/transformer.hpp>
#include <low_precision/concat.hpp>
#include <low_precision/concat_multi_channels.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

namespace {

class ConcatTransformationActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert1;
    ngraph::builder::subgraph::DequantizationOperations dequantization1;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert2;
    ngraph::builder::subgraph::DequantizationOperations dequantization2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.convert1.outPrecision << "_" <<
        values.dequantization1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.convert2.outPrecision << "_" <<
        values.dequantization2;
}

class ConcatTransformationResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert1;
    ngraph::builder::subgraph::DequantizationOperations dequantization1;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert2;
    ngraph::builder::subgraph::DequantizationOperations dequantization2;
    ngraph::element::Type precisionAfterOperation;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.convert1.outPrecision << "_" <<
        values.dequantization1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.convert2.outPrecision << "_" <<
        values.dequantization2 << "_" <<
        values.dequantizationAfter;
}

class ConcatTransformationTestValues {
public:
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannels;
    ConcatTransformationActualValues actual;
    ConcatTransformationResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    ngraph::Shape,
    ConcatTransformationTestValues
> ConcatTransformationParams;

class ConcatTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        ConcatTransformationTestValues testValues = std::get<2>(GetParam());

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.actual.dequantization1.multiply.empty()) {
            testValues.actual.dequantization1.multiply.outPrecision = precision;
        }
        if (!testValues.actual.dequantization2.multiply.empty()) {
            testValues.actual.dequantization2.multiply.outPrecision = precision;
        }

        actualFunction = ngraph::builder::subgraph::ConcatFunction::get(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.convert1,
            testValues.actual.dequantization1,
            testValues.actual.fakeQuantize2,
            testValues.actual.convert2,
            testValues.actual.dequantization2,
            {},
            ngraph::element::undefined,
            {});

        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.actual").run_on_function(actualFunction);

        //SimpleLowPrecisionTransformer transform;
        //transform.register_pass<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>();
        //transform.register_pass<ngraph::pass::low_precision::ConcatTransformation>();
        //transform.transform(actualFunction);

        ngraph::pass::Manager manager1;
        manager1.register_pass<ngraph::pass::low_precision::MarkupPrecisions>();
        manager1.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming1").run_on_function(actualFunction);

        ngraph::pass::Manager manager2;
        manager2.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisions>();
        manager2.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming2").run_on_function(actualFunction);

        ngraph::pass::Manager manager3;
        manager3.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
        manager3.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming3").run_on_function(actualFunction);

        ngraph::pass::Manager manager4;
        manager4.register_pass<ngraph::pass::low_precision::AlignConcatQuantizationParamters>();
        manager4.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming4").run_on_function(actualFunction);

        // TODO: debug only
        ngraph::pass::low_precision::TransformationContext context(actualFunction);
        {
            ngraph::pass::Manager manager;
            std::shared_ptr<ngraph::pass::GraphRewrite> common = manager.register_pass<ngraph::pass::GraphRewrite>();
            common->add_matcher<ngraph::pass::low_precision::ConcatTransformation>();
            common->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>(ngraph::pass::low_precision::LayerTransformation::Params(), context);
            manager.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transformed").run_on_function(actualFunction);
        }

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.result.dequantizationAfter.multiply.empty()) {
            testValues.result.dequantizationAfter.multiply.outPrecision = precision;
        }

        if (!testValues.params.updatePrecisions &&
            (precision == ngraph::element::f32) &&
            !testValues.result.dequantizationAfter.convert.empty()) {
            testValues.result.dequantizationAfter.convert = {};
        }

        referenceFunction = ngraph::builder::subgraph::ConcatFunction::get(
            precision,
            shape,
            testValues.result.fakeQuantize1,
            testValues.result.convert1,
            testValues.result.dequantization1,
            testValues.result.fakeQuantize2,
            testValues.result.convert2,
            testValues.result.dequantization2,
            {
                make_shared_attribute<PrecisionPreservedAttribute>(true),
                make_shared_attribute_ptr<IntervalsAlignmentAttribute>(-1.28f, 2.55f),
                make_shared_attribute_ptr<QuantizationAlignmentAttribute>(false)
            },
            testValues.result.precisionAfterOperation,
            testValues.result.dequantizationAfter);

        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.reference").run_on_function(referenceFunction);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const ConcatTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
            (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;

    const auto fakeQuantizes = LayerTransformation::get<opset1::FakeQuantize>(actualFunction);
    ASSERT_TRUE(checkIfOutputAttributesAreEqual<std::shared_ptr<PrecisionsAttribute>>(fakeQuantizes)) << "PrecisionsAttribute are not the same";
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<ConcatTransformationTestValues> testValues = {
    //// U8: concat
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    false,
    //    {
    //        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        {},
    //        {},
    //        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} }
    //    },
    //    {
    //        {
    //            256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        {
    //            256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        ngraph::element::u8,
    //        { ngraph::element::f32, {}, { 0.01f } },
    //    }
    //},
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
        },
        {
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    ////// U8: concat
    ////{
    ////    LayerTransformation::createParamsU8I8(),
    ////    true,
    ////    {
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
    ////        { ngraph::element::u8 },
    ////        {
    ////            { element::f32 },
    ////            {},
    ////            { 0.01f }
    ////        },
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
    ////        { ngraph::element::u8 },
    ////        {
    ////            { element::f32 },
    ////            {},
    ////            { 0.01f }
    ////        },
    ////    },
    ////    {
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        ngraph::element::u8,
    ////        { ngraph::element::f32, {}, { 0.01f } }
    ////    }
    ////},
    ////// U8: concat
    ////{
    ////    LayerTransformation::createParamsU8I8(),
    ////    false,
    ////    {
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
    ////        { ngraph::element::u8 },
    ////        {
    ////            { element::f32 },
    ////            {},
    ////            { 0.01f }
    ////        },
    ////    },
    ////    {
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        ngraph::element::u8,
    ////        { ngraph::element::f32, {}, { 0.01f } }
    ////    }
    ////},
    ////// U8: concat
    ////{
    ////    LayerTransformation::createParamsU8I8(),
    ////    true,
    ////    {
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
    ////        { ngraph::element::u8 },
    ////        {
    ////            { element::f32 },
    ////            {},
    ////            { 0.01f }
    ////        },
    ////    },
    ////    {
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        ngraph::element::u8,
    ////        { ngraph::element::f32, {}, { 0.01f } }
    ////    }
    ////},
    //// U8: concat
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    false,
    //    {
    //        { 256ul, {{1}, {1}, {1}, {1}}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        {},
    //        {},
    //        { 256ul, {{1}, {1}, {1}, {1}}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        {},
    //        {}
    //    },
    //    {
    //        {
    //            256ul, {{1}, {1}, {}, {}}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        {
    //            256ul, {{1}, {1}, {}, {}}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        ngraph::element::u8,
    //        { ngraph::element::f32, {}, { 0.01f } }
    //    }
    //},
    //// U8: concat
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    false,
    //    {
    //        { 256ul, {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        {},
    //        {},
    //        { 256ul, {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        {},
    //        {}
    //    },
    //    {
    //        {
    //            256ul, {{1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        {
    //            256ul, {{1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        ngraph::element::u8,
    //        { ngraph::element::f32, {}, { 0.01f } }
    //    }
    //},
    //// U8: concat multi channels
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    true,
    //    {
    //        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        {},
    //        {},
    //        { 256ul, {}, {0.f}, {1.275f}, {0.f}, {1.275f} },
    //        {},
    //        {}
    //    },
    //    {
    //        {
    //            256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        {
    //            256ul, {}, {0.f}, {1.275f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        ngraph::element::u8,
    //        { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} }
    //    }
    //},
    //// U8: concat multi channels
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    true,
    //    {
    //        { 256ul, {{1}, {1}, {1}, {1}}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        {},
    //        {},
    //        { 256ul, {{1}, {1}, {1}, {1}}, {0.f}, {1.275f}, {0.f}, {1.275f} },
    //        {},
    //        {}
    //    },
    //    {
    //        {
    //            256ul, {{1}, {1}, {}, {}}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        {
    //            256ul, {{1}, {1}, {}, {}}, {0.f}, {1.275f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        ngraph::element::u8,
    //        { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} }
    //    }
    //},
    //// U8: concat multi channels
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    true,
    //    {
    //        {
    //            256ul,
    //            {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
    //            {0.f, 0.f, 0.f}, {2.55f, 2.55f, 2.55f}, {0.f, 0.f, 0.f}, {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}
    //        },
    //        {},
    //        {},
    //        {
    //            256ul,
    //            {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
    //            {0.f, 0.f, 0.f}, {1.275f, 1.275f, 1.275f}, {0.f, 0.f, 0.f}, {1.275f / 1.f, 1.275f / 2.f, 1.275f / 3.f}
    //        },
    //        {},
    //        {}
    //    },
    //    {
    //        {
    //            256ul,
    //            {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
    //            {0.f, 0.f, 0.f}, {2.55f, 2.55f, 2.55f}, {0.f}, {255.f},
    //            ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        {
    //            256ul,
    //            {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
    //            {0.f, 0.f, 0.f}, {1.275f, 1.275f, 1.275f}, {0.f}, {255.f},
    //            ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        ngraph::element::u8,
    //        { ngraph::element::f32, {}, {{ 0.01f / 1.f, 0.01f / 2.f, 0.01f / 3.f, 0.005f / 1.f, 0.005f / 2.f, 0.005f / 3.f }} }
    //    }
    //},
    //// U8: concat multi channels with subtract
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    true,
    //    {
    //        { 256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f} },
    //        {},
    //        {},
    //        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        {},
    //        {}
    //    },
    //    {
    //        {
    //            256ul, {}, {1.275f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        {
    //            256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        ngraph::element::u8,
    //        {
    //            ngraph::element::f32,
    //            {{ -255.f, -255.f, -255.f, 0.f, 0.f, 0.f }},
    //            {{ 0.005f, 0.005f, 0.005f, 0.01f, 0.01f, 0.01f }}
    //        }
    //    }
    //},
    //// I8
    //{
    //    LayerTransformation::createParamsI8I8(),
    //    false,
    //    {
    //        { 256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
    //        {},
    //        {},
    //        { 256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
    //        {},
    //        {}
    //    },
    //    {
    //        {
    //            256ul, {}, {-1.28f}, {1.27f}, {-128.f}, {127.f}, ngraph::element::i8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        {
    //            256ul, {}, {-1.28f}, {1.27f}, {-128.f}, {127.f}, ngraph::element::i8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        ngraph::element::i8,
    //        { ngraph::element::f32, {}, { 0.01f } }
    //    }
    //},
    //// mixed: U8 + I8: concat (check constant values here)
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    false,
    //    {
    //        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        {},
    //        {},
    //        { 256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
    //        {},
    //        {}
    //    },
    //    {
    //        {
    //            256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(-1.28f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        {
    //            256ul, {}, {-1.28f}, {1.27f}, {0.f}, {255.f}, ngraph::element::u8,
    //            { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(-1.28f, 2.55f) }
    //        },
    //        {},
    //        {},
    //        ngraph::element::u8,
    //        { ngraph::element::f32, { {0.f, 0.f, 0.f, 128.f, 128.f, 128.f } }, { 0.01f } }
    //    }
    //},

    ////// mixed: U8 + I8: concat multi channels
    ////{
    ////    LayerTransformation::createParamsU8I8(),
    ////    true,
    ////    {
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
    ////        {},
    ////        {}
    ////    },
    ////    {
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {-1.28f}, {1.27f}, {0.f}, {255.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        ngraph::element::u8,
    ////        { ngraph::element::f32, {{ 0.f, 0.f, 0.f, 128.f, 128.f, 128.f }}, { 0.01f } }
    ////    }
    ////},
    ////// mixed: I8 + U8: concat (check constant values here)
    ////{
    ////    LayerTransformation::createParamsU8I8(),
    ////    false,
    ////    {
    ////        { 256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    ////        {},
    ////        {}
    ////    },
    ////    {
    ////        { 256ul, {}, {-1.28f}, {1.27f}, {0.f}, {170.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {0.f}, {2.55f}, {85.f}, {255.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        ngraph::element::u8,
    ////        { ngraph::element::f32, { 85 }, { 0.015f } }
    ////    }
    ////},
    ////// real case from ctdet_coco_dlav0_384 model, coverage bad rounding
    ////{
    ////    LayerTransformation::createParamsU8I8(),
    ////    false,
    ////    {
    ////        { 256ul, {}, {-1.28f}, {1.27f}, {0.f}, {2.3007815f} },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {0.f}, {2.55f}, {-3.873046875f}, {3.84375} },
    ////        {},
    ////        {}
    ////    },
    ////    {
    ////        { 256ul, {}, {-1.28f}, {1.27f}, {128.f}, {204.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
    ////        {},
    ////        {},
    ////        ngraph::element::u8,
    ////        { ngraph::element::f32, { 128 }, { 0.0302619f } }
    ////    }
    ////},
    ////// not update precisions
    ////{
    ////    LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
    ////    false,
    ////    {
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    ////        {},
    ////        {}
    ////    },
    ////    {
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
    ////        {},
    ////        {},
    ////        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
    ////        {},
    ////        {},
    ////        ngraph::element::f32,
    ////        { {element::f32}, {}, { 0.01f } },
    ////    }
    ////}
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 9, 9 },
    //{ 4, 3, 9, 9 }
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    ConcatTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConcatTransformation::getTestCaseName);
}  // namespace

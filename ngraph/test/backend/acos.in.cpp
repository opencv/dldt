// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, acos)
{
    test::make_unary_test<TestEngine, op::Acos, element::f32>(Shape{11}).test(
        {-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f},
        {3.14159265f,
         2.41885841f,
         2.09439510f,
         1.82347658f,
         1.69612416f,
         1.57079633f,
         1.44546850f,
         1.31811607f,
         1.04719755f,
         0.72273425f,
         0.00000000f},
        MIN_FLOAT_TOLERANCE_BITS);
}

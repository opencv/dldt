// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, log)
{
    test_unary<TestEngine, element::f32>(unary_func<op::Log>(),
                                         {0.125f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f},
                                         {-2.07944154f,
                                          -1.38629436f,
                                          -0.69314718f,
                                          0.00000000f,
                                          0.69314718f,
                                          1.38629436f,
                                          2.07944154f,
                                          2.77258872f},
                                         Shape{2, 2, 2},
                                         Shape{2, 2, 2});
}

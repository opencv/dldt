//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, maximum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Maximum>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 8, 4, 17, 0, 0.5, 2, 1.5}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum_int32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::i32, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Maximum>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::i32, shape);
    copy_data(a, vector<int32_t>{0x40000140, 0x40000001, -8, 17});
    auto b = backend->create_tensor(element::Type_t::i32, shape);
    copy_data(b, vector<int32_t>{0x40000170, 0x40000000, 4, 8});
    auto result = backend->create_tensor(element::Type_t::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{0x40000170, 0x40000001, 4, 17}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum_int64)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::i64, shape);
    auto B = make_shared<op::Parameter>(element::Type_t::i64, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Maximum>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::i64, shape);
    copy_data(a, vector<int64_t>{1, 8, -8, 17, -5, 67635216, 2, 17179887632});
    auto b = backend->create_tensor(element::Type_t::i64, shape);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 0, 18448, 1, 280592});
    auto result = backend->create_tensor(element::Type_t::i64, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int64_t>{1, 8, 4, 17, 0, 67635216, 2, 17179887632}),
              read_vector<int64_t>(result));
}

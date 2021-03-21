// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/nop.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(snippets::op::Nop, "Nop", 0);

snippets::op::Nop::Nop(const OutputVector& arguments, const OutputVector& results) : Op([arguments, results]() -> OutputVector {
    OutputVector x;
    x.insert(x.end(), arguments.begin(), arguments.end());
    x.insert(x.end(), results.begin(), results.end());
    return x;
    }()) {
}

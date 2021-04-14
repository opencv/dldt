// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "split.hpp"
#include "ngraph/node.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API VariadicSplitTransformation : public SplitTransformation {
public:
    VariadicSplitTransformation(const Params& params = Params());
};
} // namespace low_precision
} // namespace pass
} // namespace ngraph

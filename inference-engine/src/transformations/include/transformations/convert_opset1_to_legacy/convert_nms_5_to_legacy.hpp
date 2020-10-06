// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API ConvertNMS4ToLegacyMatcher;

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *      Convert NMS-5 directly to inner NMS.
 */


class ngraph::pass::ConvertNMS5ToLegacyMatcher: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNMS4ToLegacyMatcher();
};


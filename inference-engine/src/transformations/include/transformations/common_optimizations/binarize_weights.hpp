// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API BinarizeWeights;

} // namespace pass
} // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief This transformation converts weights to -1/+1 form
 * and applies normalization factors to output low/high and after Convolution.
 * For example, following graph
 *
 *         .... ....  out_low  out_high           weights ..    ..  out_low out_high
 *           |    |      |        |                  |     |    |      |     |
 *          +--------------------------+           +--------------------------+
 *          | FakeQuantize (levels==2) |           | FakeQuantize (levels==2) |
 *          |     (on activations)     |           |       (on weights)       |
 *          +--------------------------+           +--------------------------+
 *                        |                                      |
 *                        |                                      |
 *                        -----------------    -------------------
 *                                        |    |
 *                                        v    v
 *                                   +-------------+
 *                                   | Convolution |
 *                                   +-------------+
 *                                          |
 *                                          v
 *
 * is transformed to:
 *
 *                  normalized normalized
 *         .... ....  out_low   out_high
 *           |    |      |         |
 *          +--------------------------+           +--------------------------+
 *          | FakeQuantize (levels==2) |           |         Constant         |
 *          |     (on activations)     |           | (with converted weights) |
 *          +--------------------------+           +--------------------------+
 *                        |                                      |
 *                        |                                      |
 *                        -----------------    -------------------
 *                                        |    |
 *                                        v    v
 *                                   +-------------+
 *                                   | Convolution |
 *                                   +-------------+
 *                                          |
 *                                          v
 *                                   +------------+     +---------------------------------------------------------------+
 *                                   |  Multiply  | <---| Constant (normalization factor coming from FQ on activations) |
 *                                   +------------+     +---------------------------------------------------------------+
 *                                          |
 *                                          v
 *                                   +------------+     +-----------------------------------------------------------+
 *                                   |  Multiply  | <---| Constant (normalization factor coming from FQ on weights) |
 *                                   +------------+     +------------------------------------------------------------
 *                                          |
 *                                          v
 *
 * Normalization factors are chosen based output_high value.
 * If it's zero - norm factor is equal to output_low and output_high otherwise
 */

class ngraph::pass::BinarizeWeights : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BinarizeWeights();
};

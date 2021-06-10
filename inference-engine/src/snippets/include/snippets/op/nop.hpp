// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"
#include "snippets/emitter.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Nop
 * @brief Generated by Canonicalization and represents not-an-operation
 * @ingroup snippets
 */
class TRANSFORMATIONS_API Nop : public ngraph::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;

    Nop(const OutputVector& arguments, const OutputVector& results);
    Nop() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<Nop>();
    }
};

class TRANSFORMATIONS_API Tile : public ngraph::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;

    Tile(const std::vector<std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo>>& region);
    Tile() = default;
    std::vector<std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo>> region;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<Tile>(region);
    }
};

class TRANSFORMATIONS_API Kernel : public ngraph::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;

    Kernel(const std::vector<std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo>>& region);
    Kernel() = default;

    std::vector<std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo>> region;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<Kernel>(region);
    }
};

} // namespace op
} // namespace snippets
} // namespace ngraph
// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

#include <vector>
#include <memory>

namespace vpu {

using Transformations = std::unordered_map<ngraph::NodeTypeInfo, std::function<void(std::shared_ptr<ngraph::Node>)>>;

class DynamicToStaticShape: public ngraph::pass::FunctionPass {
public:
    explicit DynamicToStaticShape(const Transformations& specificTransformations = {});
    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;

private:
    Transformations transformations;
};

void printTo(std::ostream& stream, const ngraph::NodeTypeInfo& object);

}  // namespace vpu

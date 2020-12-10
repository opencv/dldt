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

#include "constant_folding.hpp"
#include <ngraph/rt_info.hpp>
#include "ngraph/op/util/sub_graph_base.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConstantFolding, "ConstantFolding", 0);

bool ngraph::pass::ConstantFolding::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    bool rewritten = false;

    auto it = f->get_iterator();
    while(it.get())
    {
        auto node = it.get();
        it.next();
        if (rewritten)
        {
            node->revalidate_and_infer_types();
        }

        // recursively constant fold operators containing subgraphs (ie: TensorIterator)
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node))
        {
            if (auto sub_graph = sub_graph_node->get_function())
            {
                rewritten |= run_on_function(sub_graph);
                continue;
            }
        }

        OutputVector replacements(node->get_output_size());
        if (node->constant_fold(replacements, node->input_values()))
        {
            NGRAPH_CHECK(replacements.size() == node->get_output_size(),
                         "constant_fold_default returned incorrect number of replacements for ",
                         node);

            for (size_t i = 0; i < replacements.size(); ++i)
            {
                auto node_output = node->output(i);
                auto replacement = replacements.at(i);
                if (replacement.get_node_shared_ptr() && (node_output != replacement))
                {
                    if (replacements.size() == 1)
                    {
                        replacement.get_node_shared_ptr()->set_friendly_name(
                            node->get_friendly_name());
                    }
                    else
                    {
                        replacement.get_node_shared_ptr()->set_friendly_name(
                            node->get_friendly_name() + "." + std::to_string(i));
                    }
                    node_output.replace(replacement);
                    // Propagate runtime info attributes to replacement consumer nodes
                    copy_runtime_info_to_target_inputs(node, replacement);

                    rewritten = true;
                }
            }
        }
    }

    return rewritten;
}

void ngraph::pass::ConstantFolding::copy_runtime_info_to_target_inputs(
    const std::shared_ptr<Node>& node, const Output<Node>& replacement)
{
    for (auto& input : replacement.get_target_inputs())
    {
        auto consumer = input.get_node()->shared_from_this();
        copy_runtime_info({node, consumer}, consumer);
    }
}

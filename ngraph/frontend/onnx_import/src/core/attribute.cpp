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

#include "attribute.hpp"
#include "graph.hpp"
#include "model.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        std::vector<Graph> Attribute::get_graph_array(Model& model) const
        {
            std::vector<Graph> result;
            for (const auto& graph : m_attribute_proto->graphs())
            {
                result.emplace_back(graph, model);
            }
            return result;
        }

        Subgraph Attribute::get_subgraph(const Graph& parent_graph) const
        {
            if (m_attribute_proto->type() != ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH)
            {
                throw error::attribute::InvalidData{m_attribute_proto->type()};
            }

            ONNX_NAMESPACE::ModelProto model_proto;
            const auto& graph = m_attribute_proto->g();
            *(model_proto.mutable_graph()) = graph;
            const auto parent_graph_opset = parent_graph.get_opset_versions();
            // set the opset version from parent graph
            model_proto.add_opset_import();
            for (size_t i = 0; i < parent_graph_opset.size(); ++i)
            {
                model_proto.mutable_opset_import(i)->set_version(parent_graph_opset[i]);
            }
            Model model{model_proto};
            return Subgraph{graph, model, parent_graph};
        }

    } // namespace onnx_import

} // namespace ngraph

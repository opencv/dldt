//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <fstream>
#include <memory>
#include <onnx_import/onnx_node.hpp>

#include "core/graph.hpp"
#include "core/model.hpp"
#include "core/transform.hpp"
#include "ngraph/except.hpp"
#include "onnx_import/onnx.hpp"
#include "ops_bridge.hpp"
#include "utils/parser.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            std::shared_ptr<Function>
                convert_to_ng_function(std::shared_ptr<const ONNX_NAMESPACE::ModelProto> model_proto)
            {
                auto model = std::make_shared<Model>(*model_proto);
                auto graph = std::make_shared<Graph>(model_proto->graph(), *model);
                auto function = std::make_shared<Function>(
                    graph->get_ng_outputs(), graph->get_ng_parameters(), graph->get_name());

                for(auto node: function->get_ops())
                {
                    if(auto raw_node = std::dynamic_pointer_cast<frontend::ONNXNode>(node))
                    {
                        raw_node->set_onnx_graph(graph);
                        raw_node->set_onnx_model(model);
                        raw_node->set_onnx_model_proto(model_proto);
                    }
                }

                for (std::size_t i{0}; i < function->get_output_size(); ++i)
                {
                    function->get_output_op(i)->set_friendly_name(
                        graph->get_outputs().at(i).get_name());
                }
                return function;
            }

            std::shared_ptr<Function> import_onnx_model(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto,
                                                        const std::string& model_path)
            {
                transform::expand_onnx_functions(*model_proto);
                transform::fixup_legacy_operators(*model_proto);
                transform::update_external_data_paths(*model_proto, model_path);

                return detail::convert_to_ng_function(model_proto);
            }
        } // namespace detail

        std::shared_ptr<Function> import_onnx_model(std::istream& stream,
                                                    const std::string& model_path)
        {
            auto model_proto = std::make_shared<ONNX_NAMESPACE::ModelProto>(parse_from_istream(stream));

            return detail::import_onnx_model(model_proto, model_path);
        }

        std::shared_ptr<Function> import_onnx_model(const std::string& file_path)
        {
            std::ifstream model_stream{file_path, std::ios::in | std::ios::binary};

            if (!model_stream.is_open())
            {
                throw ngraph_error("Error during import of ONNX model expected to be in file: " +
                                   file_path + ". Could not open the file.");
            };

            return import_onnx_model(model_stream, file_path);
        }

        std::shared_ptr<Function> import_onnx_model(const ONNXModelEditor& model_editor)
        {
            //throw false; // TODO: Migrate ONNX Model Editor to shared_ptr's to keep model
            return detail::import_onnx_model(std::make_shared<ONNX_NAMESPACE::ModelProto>(model_editor.model()), model_editor.model_path());
        }

        std::set<std::string> get_supported_operators(std::int64_t version,
                                                      const std::string& domain)
        {
            OperatorSet op_set{
                OperatorsBridge::get_operator_set(domain == "ai.onnx" ? "" : domain, version)};
            std::set<std::string> op_list{};
            for (const auto& op : op_set)
            {
                op_list.emplace(op.first);
            }
            return op_list;
        }

        bool is_operator_supported(const std::string& op_name,
                                   std::int64_t version,
                                   const std::string& domain)
        {
            return OperatorsBridge::is_operator_registered(
                op_name, version, domain == "ai.onnx" ? "" : domain);
        }

    } // namespace onnx_import

} // namespace ngraph

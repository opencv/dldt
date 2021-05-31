// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/low_latency.hpp"

#include <memory>

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::LowLatency2, "LowLatency2", 0);

NGRAPH_SUPPRESS_DEPRECATED_START
NGRAPH_RTTI_DEFINITION(ngraph::pass::LowLatency, "LowLatency", 0);

using namespace std;
using namespace ngraph;

string generate_variable_name(const string& op_name, const string& param_name, int variable_idx)
{
    return op_name + "/" + param_name + "/" + "variable_" + to_string(variable_idx);
}

ngraph::pass::LowLatency::LowLatency()
{
    auto tensor_iterator = ngraph::pattern::wrap_type<opset6::TensorIterator, opset6::Loop>();
    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        const auto& sub_graph_op =
            std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(m.get_match_root());
        if (!sub_graph_op)
        {
            return false;
        }

        if (const auto& loop = std::dynamic_pointer_cast<opset6::Loop>(sub_graph_op))
        {
            const auto& trip_count =
                std::dynamic_pointer_cast<opset6::Constant>(loop->get_input_node_shared_ptr(0));
            const auto& num_iter = loop->get_num_iterations();
            if (trip_count && num_iter > 0 && trip_count->get_output_target_inputs(0).size() == 1)
            {
                auto single_iter =
                    std::make_shared<opset6::Constant>(ngraph::element::i64, Shape{}, 1);
                replace_node(trip_count, single_iter);
            }
            else
            {
                // count of iterations is dynamic;
                return false;
            }
        }
        // Mark the TI layer to be unrolled. Enable unconditional ti unrolling for all plugins.
        auto& rt_info = sub_graph_op->get_rt_info();
        rt_info["UNROLL_TI"] = std::make_shared<ngraph::VariantWrapper<int64_t>>(1);

        int64_t variable_id = 0;
        std::vector<std::shared_ptr<ngraph::op::Sink>> assigns;
        const auto& func = sub_graph_op->get_function();
        for (const auto& in : sub_graph_op->get_input_descriptions())
        {
            // Process all back edges
            if (const auto& merged_in =
                    std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::MergedInputDescription>(
                        in))
            {
                // Insert ReadValue nodes: Parameter -> (new ReadValue) -> consumers
                const auto& inputs_to = func->get_parameters()
                                            .at(merged_in->m_body_parameter_index)
                                            ->get_output_target_inputs(0);
                const std::string variable_name(
                    generate_variable_name(sub_graph_op->get_friendly_name(),
                                           func->get_parameters()
                                               .at(merged_in->m_body_parameter_index)
                                               ->get_friendly_name(),
                                           variable_id));
                auto variable = std::make_shared<Variable>(
                    VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name});
                auto read_value = std::make_shared<opset6::ReadValue>(
                    func->get_parameters().at(merged_in->m_body_parameter_index), variable);
                read_value->set_friendly_name(variable_name);
                for (const auto& input_to : inputs_to)
                {
                    input_to.replace_source_output(read_value->output(0));
                }

                // insert Assign nodes: provider -> (new Assign) -> Result
                const auto res = func->get_results().at(merged_in->m_body_value_index);
                auto assign = std::make_shared<opset6::Assign>(res->input_value(0), variable);
                // control dependency so that ReadValue is processed before Assign
                assign->add_control_dependency(read_value);
                assigns.emplace_back(assign);
            }
            variable_id++;
        }
        // save Assign in the func so that it gets into graph traversals and isn't deleted.
        func->add_sinks(assigns);
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "LowLatency");
    register_matcher(m, callback);
}
NGRAPH_SUPPRESS_DEPRECATED_END

void UnrollSingleIteration(const shared_ptr<op::util::SubGraphOp>& sub_graph_op,
                           const shared_ptr<Function>& outer_f)
{
    using namespace opset7;

    const auto& params = sub_graph_op->get_function()->get_parameters();
    const auto& results = sub_graph_op->get_function()->get_results();

    // before: Layer1 -> TI [input -> bodyParameter -> Layer2 -> ...]
    // after:  Layer1 -> Layer2 ->...
    for (const auto& in : sub_graph_op->get_input_descriptions())
    {
        const auto& connect_to = sub_graph_op->get_input_source_output(in->m_input_index);
        for (auto& output : params.at(in->m_body_parameter_index)->outputs())
        {
            output.replace(connect_to);
        }
    }

    // before: TI [...-> Layer1 -> Result -> output] -> Layer2 -> ...
    // after:  ...-> Layer1 -> Layer2 -> ...
    NodeVector new_ops;
    for (const auto& out : sub_graph_op->get_output_descriptions())
    {
        const auto& connect_to = results.at(out->m_body_value_index)->get_input_source_output(0);
        for (auto& input_to : sub_graph_op->output(out->m_output_index).get_target_inputs())
        {
            // create IE output name
            std::string out_name = sub_graph_op->get_friendly_name();
            if (sub_graph_op->get_output_size() != 1)
                out_name += "." + std::to_string(out->m_output_index);

            // IECompatibility: insert identity (Unsqueeze + Squeeze) to store the TensorIterator
            // output names
            auto axis_1 = Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
            auto identity_1 = std::make_shared<Unsqueeze>(connect_to, axis_1);
            auto identity_2 = std::make_shared<Squeeze>(identity_1, axis_1);
            identity_2->set_friendly_name(out_name);
            new_ops.push_back(identity_1);
            new_ops.push_back(identity_2);

            input_to.replace_source_output(identity_2);
        }
    }
    outer_f->add_sinks(sub_graph_op->get_function()->get_sinks());
    ngraph::copy_runtime_info(sub_graph_op, sub_graph_op->get_function()->get_ops());
    ngraph::copy_runtime_info(sub_graph_op, new_ops);
}

Output<Node> create_init_subgraph(const shared_ptr<op::util::SubGraphOp>& sub_graph_op,
                                  const Output<Node>& in_node)
{
    using namespace opset7;

    auto const_zero = make_shared<Constant>(in_node.get_element_type(), Shape{1}, 0);
    auto shape_of = make_shared<ShapeOf>(in_node);
    auto broadcast = make_shared<Broadcast>(const_zero, shape_of);
    copy_runtime_info(sub_graph_op, {const_zero, shape_of, broadcast});
    return broadcast->output(0);
}

bool pass::LowLatency2::run_on_function(shared_ptr<Function> f)
{
    using namespace opset7;

    SinkVector assigns;
    for (const auto& op : f->get_ops())
    {
        if (const auto& sub_graph_op = dynamic_pointer_cast<op::util::SubGraphOp>(op))
        {
            int64_t iterations = 1;
            auto subgraph_iters = m_sub_graph_iterations.find(sub_graph_op->get_friendly_name());
            if (subgraph_iters != m_sub_graph_iterations.end())
            {
                iterations = subgraph_iters->second;
                if (iterations <= 0)
                {
                    continue;
                }
            }
            if (const auto& loop = dynamic_pointer_cast<Loop>(sub_graph_op))
            {
                const auto& trip_count =
                    dynamic_pointer_cast<Constant>(loop->get_input_node_shared_ptr(0));
                const auto& num_iter = loop->get_num_iterations();
                if (trip_count && num_iter > 0 &&
                    trip_count->get_output_target_inputs(0).size() == 1)
                {
                    auto iter_const = std::make_shared<Constant>(element::i64, Shape{}, iterations);
                    replace_node(trip_count, iter_const);
                    loop->validate_and_infer_types();
                }
                else
                {
                    // this loop is not supported
                    continue;
                }
            }

            int64_t variable_id = 0;
            const auto& func = sub_graph_op->get_function();
            const auto& params = func->get_parameters();
            for (const auto& in : sub_graph_op->get_input_descriptions())
            {
                // Process all back edges
                if (const auto& merged_in =
                        dynamic_pointer_cast<op::util::SubGraphOp::MergedInputDescription>(in))
                {
                    // create new Variable
                    const string& param_name =
                        params.at(merged_in->m_body_parameter_index)->get_friendly_name();
                    const string& var_name = generate_variable_name(
                        sub_graph_op->get_friendly_name(), param_name, variable_id);

                    if (func->get_variable_by_id(var_name) != nullptr ||
                        f->get_variable_by_id(var_name) != nullptr)
                    {
                        // ReadValue/Assign with this Variable id already exist in the graph
                        // Most likely LowLatency v1/v2 transformation has already been applied
                        variable_id++;
                        continue;
                    }

                    VariableInfo var_info{PartialShape::dynamic(), element::dynamic, var_name};
                    auto variable = make_shared<Variable>(var_info);

                    // insert ReadValue
                    // Layers -> [new op: ReadValue] -> Subgraph operation
                    const auto& input = sub_graph_op->input(merged_in->m_input_index);
                    Output<Node> read_value_in = input.get_source_output();
                    if (m_use_const_initializer)
                    {
                        read_value_in = create_init_subgraph(sub_graph_op, read_value_in);
                    }
                    auto read_value = make_shared<ReadValue>(read_value_in, variable);
                    input.replace_source_output(read_value->output(0));
                    read_value->set_friendly_name(var_name);
                    ngraph::copy_runtime_info(sub_graph_op, read_value);

                    /* insert Assign
                    // Subgraph operation -> [new op: Assign]
                    //                    \
                    //                      ---> Layers -> ...
                    */
                    const auto& out_desc = sub_graph_op->get_output_descriptions();
                    bool is_output_exist = std::any_of(
                        out_desc.begin(),
                        out_desc.end(),
                        [&merged_in](
                            const std::shared_ptr<op::util::SubGraphOp::OutputDescription>& out) {
                            return out->m_body_value_index == merged_in->m_body_value_index;
                        });
                    // Create new output if it doesn't exist.
                    if (!is_output_exist)
                    {
                        sub_graph_op->get_iter_value(
                            func->get_results().at(merged_in->m_body_value_index));
                    }
                    for (const auto& out : sub_graph_op->get_output_descriptions())
                    {
                        if (out->m_body_value_index == merged_in->m_body_value_index)
                        {
                            auto assign = make_shared<Assign>(
                                sub_graph_op->output(out->m_output_index), variable);
                            ngraph::copy_runtime_info(sub_graph_op, assign);
                            // control dependency so that ReadValue is processed before Assign
                            assign->add_control_dependency(read_value);
                            assigns.emplace_back(assign);
                            break;
                        }
                    }
                }

                variable_id++;
            }

            if (sub_graph_op->get_num_iterations() == 1)
            {
                UnrollSingleIteration(sub_graph_op, f);
            }
        }
    }
    f->add_sinks(assigns);
    return true;
}

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cctype>
#include <ngraph/validation_util.hpp>
#include <regex>
#include <string>
#include <unordered_map>

#include "itt.hpp"
#include "ngraph/op/einsum.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v7::Einsum, "Einsum", 7);

op::v7::Einsum::Einsum(const OutputVector& inputs, const std::string& equation)
    : Op(inputs)
    , m_equation(equation)
{
    constructor_validate_and_infer_types();
}

bool op::v7::Einsum::parse_equation(std::vector<std::string>& input_subscripts,
                                    std::string& output_subscript)
{
    NGRAPH_OP_SCOPE(v7_Einsum_parse_equation);

    // create regular expression to match subscript that consists of either alphabetical letters
    // only or alphabetical letters with one ellisis
    const std::regex subscript_regex("[A-Za-z]*|[A-Za-z]*\\.\\.\\.[A-Za-z]*");

    // create regular expression to match ellipsis label
    const std::regex ellipsis_regex("\\.\\.\\.");

    // remove extra white-spaces from the equation
    m_equation.erase(std::remove_if(m_equation.begin(), m_equation.end(), ::isspace),
                     m_equation.end());

    // split equation to input subscripts and an output subscript
    auto pos_output_delimeter = m_equation.find("->");
    auto input_subscripts_str = m_equation.substr(0, pos_output_delimeter);

    // split the input subscripts into a vector of input subscripts
    bool is_ellipsis_met = false;
    std::smatch ellipsis_match;
    input_subscripts.clear();
    std::istringstream input;
    input.str(input_subscripts_str);
    for (std::string input_subscript; std::getline(input, input_subscript, ',');)
    {
        // check that input subscript contains only alphabetic letter or ellipsis
        if (std::regex_match(input_subscript, subscript_regex) == false)
        {
            return false;
        }

        // figure out if ellipsis is met in input subscript
        if (std::regex_search(input_subscript, ellipsis_match, ellipsis_regex) == true)
        {
            is_ellipsis_met = true;
        }
        input_subscripts.push_back(input_subscript);
    }

    if (pos_output_delimeter == std::string::npos)
    {
        // recover output subscript
        output_subscript = "";
        for (auto const& input_subscript : input_subscripts)
        {
            for (auto const& label : input_subscript)
            {
                if (std::isalpha(label) && output_subscript.find(label) == std::string::npos)
                {
                    output_subscript += label;
                }
            }
        }
        std::sort(output_subscript.begin(), output_subscript.end());
        if (is_ellipsis_met)
        {
            output_subscript = "..." + output_subscript;
        }
    }
    else
    {
        output_subscript = m_equation.substr(pos_output_delimeter + 2);

        // check that the output subscript has the correct format
        if (std::regex_match(output_subscript, subscript_regex) == false)
        {
            return false;
        }

        // if the ellipsis is met in input subscripts, one ellipsis must be in the output subscript
        std::regex_search(output_subscript, ellipsis_match, ellipsis_regex);
        if (is_ellipsis_met && ellipsis_match.size() != 1)
        {
            return false;
        }
    }

    return true;
}

void op::v7::Einsum::extract_labels(std::string const& subscript, std::vector<std::string>& labels)
{
    NGRAPH_OP_SCOPE(v7_Einsum_extract_labels);

    labels.clear();
    auto subscript_length = subscript.length();
    for (size_t ch_idx = 0; ch_idx < subscript_length; ++ch_idx)
    {
        if (std::isalpha(subscript[ch_idx]))
        {
            labels.push_back(subscript.substr(ch_idx, 1));
        }
        else if (((subscript_length - ch_idx) > 2) &&
                 (subscript.substr(ch_idx, 3).compare("...") == 0))
        {
            labels.push_back("...");
            // make additional increment since ellipsis consists of three dots.
            ch_idx += 2;
        }
    }
}

void op::v7::Einsum::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v7_Einsum_validate_and_infer_types);

    // check that Einsum operation has at least one input
    auto num_inputs = get_input_size();
    NODE_VALIDATION_CHECK(this, num_inputs > 0, "Einsum must have at least one input.");

    // check that all inputs have the same type and the type is numeric
    const auto& input_type_0 = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_type_0.is_real() || input_type_0.is_integral_number(),
                          "The input type for Einsum operation must be numeric.");
    for (size_t input_idx = 1; input_idx < num_inputs; ++input_idx)
    {
        const auto& input_type_i = get_input_element_type(input_idx);
        NODE_VALIDATION_CHECK(this,
                              input_type_0 == input_type_i,
                              "Inputs to Einsum operation must have the same type.");
    }

    // check that equation has correct format and extract input and output subscripts
    std::vector<std::string> input_subscripts;
    std::string output_subscript;
    NODE_VALIDATION_CHECK(this,
                          parse_equation(input_subscripts, output_subscript) == true,
                          "Equation must be strictly in explicit or implicit mode.");
    NODE_VALIDATION_CHECK(
        this,
        input_subscripts.size() == num_inputs,
        "Equation must contain a number of subscripts equal to a number of Einsum inputs.");

    // create a dictionary with dimension sizes (or ranges in case dynamic shapes) for each label
    // and check their compatibility in case repeating labels
    unordered_map<string, PartialShape> label_to_shape;
    label_to_shape.clear();

    for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx)
    {
        const auto& pshape = get_input_partial_shape(input_idx);
        std::vector<std::string> labels;
        extract_labels(input_subscripts[input_idx], labels);

        if (pshape.rank().is_static())
        {
            size_t input_rank = pshape.rank().get_length();
            // check that a rank is equal to a number of labels
            NODE_VALIDATION_CHECK(this,
                                  input_rank >= labels.size(),
                                  "Input rank must be equal to a number of labels in the "
                                  "corresponding input subscript.");

            for (size_t label_ind = 0, dim_ind = 0;
                 label_ind < labels.size() && dim_ind < input_rank;
                 ++label_ind)
            {
                auto const& label = labels[label_ind];
                if (label.compare("...") == 0)
                {
                    size_t num_broadcasted_dims = input_rank - labels.size() + 1;
                    auto current_sub_pshape = PartialShape(std::vector<Dimension>(
                        pshape.begin() + dim_ind, pshape.begin() + dim_ind + num_broadcasted_dims));
                    if (label_to_shape.find(label) == label_to_shape.end())
                    {
                        label_to_shape[label] = current_sub_pshape;
                    }
                    else
                    {
                        bool is_broadcast_success =
                            PartialShape::broadcast_merge_into(label_to_shape[label],
                                                               current_sub_pshape,
                                                               op::AutoBroadcastType::NUMPY);
                        NODE_VALIDATION_CHECK(this,
                                              is_broadcast_success == true,
                                              "Input dimensions labeled with ellipsis for Einsum "
                                              "must be broadcastable.");
                    }
                    dim_ind += num_broadcasted_dims;
                }
                else
                {
                    if (label_to_shape.find(label) == label_to_shape.end())
                    {
                        label_to_shape[label] = PartialShape{pshape[dim_ind]};
                    }
                    else
                    {
                        NODE_VALIDATION_CHECK(
                            this,
                            label_to_shape[label].compatible(PartialShape{pshape[label_ind]}),
                            "Different input dimensions indicated by the same labels for Einsum "
                            "must be compatible.");
                        PartialShape::merge_into(label_to_shape[label],
                                                 PartialShape{pshape[dim_ind]});
                    }
                    ++dim_ind;
                }
            }
        }
        else
        {
            for (auto const& label : labels)
            {
                NODE_VALIDATION_CHECK(this,
                                      label != "...",
                                      "The subscript corresponding to a dynamic rank input must "
                                      "not contain ellipsis.");

                if (label_to_shape.find(label) == label_to_shape.end())
                {
                    label_to_shape[label] = PartialShape{Dimension::dynamic()};
                }
            }
        }
    }

    // compute the output shape
    std::vector<std::string> output_labels;
    extract_labels(output_subscript, output_labels);
    std::vector<Dimension> output_pshape_vector;

    for (auto const& output_label : output_labels)
    {
        NODE_VALIDATION_CHECK(this,
                              label_to_shape.find(output_label) != label_to_shape.end(),
                              "Label in output subscript of Einsum equation must enter at least "
                              "one input subscript.");
        output_pshape_vector.insert(output_pshape_vector.end(),
                                    label_to_shape[output_label].begin(),
                                    label_to_shape[output_label].end());
    }
    set_output_type(0, input_type_0, PartialShape(output_pshape_vector));
}

bool op::v7::Einsum::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v7_Einsum_visit_attributes);
    visitor.on_attribute("equation", m_equation);
    return true;
}

shared_ptr<Node> op::v7::Einsum::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v7_Einsum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v7::Einsum>(new_args, m_equation);
}

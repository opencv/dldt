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

#include "ngraph/op/read_value.hpp"
#include "itt.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ReadValue::type_info;

op::ReadValue::ReadValue(const Output<Node>& init_value, const std::string& variable_id)
    : Op({init_value})
    , m_variable_id(variable_id)
{
    constructor_validate_and_infer_types();
}

void op::ReadValue::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v3_ReadValue_validate_and_infer_types);
    auto arg_t = get_input_element_type(0);
    auto output_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(
        this, m_variable_id != "", "Variable identifier may not be an empty string.");

    VariableInfo info = {output_pshape, arg_t, m_variable_id};
    if (m_variable == nullptr)
    {
        m_variable = std::make_shared<Variable>(info);
    }
    else
    {
        m_variable->update(info);
    }
    set_output_type(0, arg_t, output_pshape);
}

shared_ptr<Node> op::ReadValue::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v3_ReadValue_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ReadValue>(new_args.at(0), m_variable_id);
}

bool op::v3::ReadValue::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v3_ReadValue_visit_attributes);
    visitor.on_attribute("variable_id", m_variable_id);
    return true;
}

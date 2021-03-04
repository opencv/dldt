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

#include <memory>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/dft.hpp"
#include "ngraph/runtime/host_tensor.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v7::DFT, "DFT", 7);

op::v7::DFT::DFT(const Output<Node>& data, const Output<Node>& axes)
    : Op({data, axes})
{
    constructor_validate_and_infer_types();
}

op::v7::DFT::DFT(const Output<Node>& data,
                 const Output<Node>& axes,
                 const Output<Node>& signal_size)
    : Op({data, axes, signal_size})
{
    constructor_validate_and_infer_types();
}

bool op::v7::DFT::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v7_DFT_visit_attributes);
    return true;
}

std::shared_ptr<Node> op::v7::DFT::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v7_DFT_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 2 || new_args.size() == 6,
                          "Number of inputs must be 2 or 3");

    if (new_args.size() == 2)
    {
        return std::make_shared<op::v7::DFT>(new_args.at(0), new_args.at(1));
    }

    return std::make_shared<op::v7::DFT>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void op::v7::DFT::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v7_DFT_validate_and_infer_types);
    element::Type input_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_et == element::f32 || input_et == element::f16 ||
                              input_et == element::bf16,
                          "Input element type must be f32, f16, or bf16");

    element::Type axes_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          axes_et == element::i64 || axes_et == element::i32 ||
                              sizes_et == element::u32 || sizes_et == element::u64,
                          "Axes element type must be i32, i64, u32 or u64");

    if (input_values().size() == 3)
    {
        element::Type signal_sizes_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              signal_sizes_et == element::i64 || signal_sizes_et == element::i32 ||
                                  signal_sizes_et == element::u32 || signal_sizes_et == element::u64,
                              "Axes element type must be i32, i64, u32 or u64");
    }

    PartialShape input_shape = PartialShape(get_input_partial_shape(0));

    if (!input_shape.rank().is_static())
    {
        set_output_type(0, get_input_element_type(0), input_shape);
        return;
    }
}

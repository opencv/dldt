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

#include "ngraph/op/equal.hpp"
#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/equal.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

namespace equal
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  const op::AutoBroadcastSpec& broadcast_spec)
    {
        runtime::reference::equal(arg0->get_data_ptr<ET>(),
                                  arg1->get_data_ptr<ET>(),
                                  out->get_data_ptr<element::Type_t::boolean>(),
                                  arg0->get_shape(),
                                  arg1->get_shape(),
                                  broadcast_spec);
        return true;
    }

    bool evaluate_equal(const HostTensorPtr& arg0,
                        const HostTensorPtr& arg1,
                        const HostTensorPtr& out,
                        const op::AutoBroadcastSpec& broadcast_spec)
    {
        bool rc = true;
        out->set_broadcast(broadcast_spec, arg0, arg1, element::boolean);
        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_equal, boolean, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_equal, i32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_equal, i64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_equal, u32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_equal, u64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_equal, f16, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_equal, f32, arg0, arg1, out, broadcast_spec);
        default: rc = false; break;
        }
        return rc;
    }
}

//------------------------------- v1 -------------------------------------------

NGRAPH_RTTI_DEFINITION(op::v1::Equal, "Equal", 1);

op::v1::Equal::Equal(const Output<Node>& arg0,
                     const Output<Node>& arg1,
                     const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseComparison(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::Equal::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Equal>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool op::v1::Equal::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Equal_evaluate) {
        return equal::evaluate_equal(inputs[0], inputs[1], outputs[0], get_autob());
    }
    return false;
}

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

#include "ngraph/op/and.hpp"
#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/and.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v1::LogicalAnd, "LogicalAnd", 1, util::BinaryElementwiseLogical);

op::v1::LogicalAnd::LogicalAnd(const Output<Node>& arg0,
                               const Output<Node>& arg1,
                               const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseLogical(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

bool op::v1::LogicalAnd::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_LogicalAnd_visit_attributes)
    {
        BinaryElementwiseLogical::visit_attributes(visitor);
        return true;
    }
    return false;
}

shared_ptr<Node> op::v1::LogicalAnd::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_LogicalAnd_clone_with_new_inputs)
    {
        check_new_args_count(this, new_args);
        return make_shared<v1::LogicalAnd>(new_args.at(0), new_args.at(1), this->get_autob());
    }
    return nullptr;
}

namespace logand
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  const op::AutoBroadcastSpec& broadcast_spec)
    {
        runtime::reference::logical_and(arg0->get_data_ptr<ET>(),
                                        arg1->get_data_ptr<ET>(),
                                        out->get_data_ptr<ET>(),
                                        arg0->get_shape(),
                                        arg1->get_shape(),
                                        broadcast_spec);
        return true;
    }

    bool evaluate_logand(const HostTensorPtr& arg0,
                         const HostTensorPtr& arg1,
                         const HostTensorPtr& out,
                         const op::AutoBroadcastSpec& broadcast_spec)
    {
        bool rc = true;
        out->set_broadcast(broadcast_spec, arg0, arg1);
        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_logand, boolean, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_logand, i32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_logand, i64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_logand, u32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_logand, u64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_logand, f16, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_logand, f32, arg0, arg1, out, broadcast_spec);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v1::LogicalAnd::evaluate(const HostTensorVector& outputs,
                                  const HostTensorVector& inputs) const
{
    bool rc = false;
    NGRAPH_OP_SCOPE(v1_LogicalAnd_evaluate)
    {
        rc = logand::evaluate_logand(inputs[0], inputs[1], outputs[0], get_autob());
    }
    return rc;
}

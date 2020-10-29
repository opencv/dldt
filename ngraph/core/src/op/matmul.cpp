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
#include <memory>
#include <numeric>

#include "itt.hpp"
#include "matmul.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/reference/matmul.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::MatMul, "MatMul", 0);

op::MatMul::MatMul(const Output<Node>& A,
                   const Output<Node>& B,
                   const bool& transpose_a,
                   const bool& transpose_b)
    : Op(OutputVector{A, B})
    , m_transpose_a{transpose_a}
    , m_transpose_b{transpose_b}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::MatMul::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("transpose_a", m_transpose_a);
    visitor.on_attribute("transpose_b", m_transpose_b);
    return true;
}

shared_ptr<Node> op::MatMul::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<MatMul>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}

namespace matmul
{
    PartialShape validate_matmul_output_shape(const PartialShape& arg0_shape,
                                              const PartialShape& arg1_shape,
                                              bool transpose_a,
                                              bool transpose_b)
    {
        auto arg0_rank = arg0_shape.rank().get_length();
        auto arg1_rank = arg1_shape.rank().get_length();

        NGRAPH_CHECK((arg0_rank != 0 && arg1_rank != 0),
                     "Scalars are not supported as MatMul inputs.");

        // Temporary Dimension vectors to calculate output shape
        std::vector<Dimension> arg0_shape_tmp(arg0_shape);
        std::vector<Dimension> arg1_shape_tmp(arg1_shape);

        // Result of merging compatible dimensions for validation usage
        auto merged_dimension = Dimension::dynamic();

        // Transpose attributes are ignored for 1D tensors.
        if (arg0_rank == 1 && arg1_rank == 1)
        {
            NGRAPH_CHECK(Dimension::merge(merged_dimension, arg0_shape_tmp[0], arg1_shape_tmp[0]),
                         "Incompatible matrix dimensions.");
            return PartialShape(Shape{});
        }

        if (transpose_a && arg0_rank > 1)
        {
            swap(arg0_shape_tmp[arg0_rank - 2], arg0_shape_tmp[arg0_rank - 1]);
        }

        if (transpose_b && arg1_rank > 1)
        {
            swap(arg1_shape_tmp[arg1_rank - 2], arg1_shape_tmp[arg1_rank - 1]);
        }

        if (arg0_rank == 1)
        {
            // i.e. arg0 shape {3}, arg1 shape{2, 3, 2}, output shape {2, 2}
            NGRAPH_CHECK(Dimension::merge(merged_dimension,
                                          arg0_shape_tmp[0],
                                          arg1_shape_tmp[arg1_shape_tmp.size() - 2]),
                         "Incompatible matrix dimensions.");
            arg1_shape_tmp.erase(arg1_shape_tmp.begin() + arg1_rank - 2);
            return PartialShape(arg1_shape_tmp);
        }
        else if (arg1_rank == 1)
        {
            // i.e. arg0 shape {2, 2, 3}, arg1 shape{3}, output shape {2, 2}
            NGRAPH_CHECK(Dimension::merge(
                             merged_dimension, arg1_shape_tmp[0], arg0_shape_tmp[arg0_rank - 1]),
                         "Incompatible matrix dimensions.");
            arg0_shape_tmp.erase(arg0_shape_tmp.begin() + arg0_rank - 1);
            return PartialShape(arg0_shape_tmp);
        }

        // 2D and bigger tensors cases.
        NGRAPH_CHECK(Dimension::merge(merged_dimension,
                                      arg0_shape_tmp[arg0_rank - 1],
                                      arg1_shape_tmp[arg1_rank - 2]),
                     "Incompatible matrix dimensions.");

        auto max_rank = std::max(arg0_rank, arg1_rank);
        std::vector<Dimension> output_shape(max_rank);

        // Handle batch size
        if (max_rank > 2)
        {
            std::vector<Dimension> small_batch_matrix =
                arg0_rank > arg1_rank ? arg1_shape_tmp : arg0_shape_tmp;
            std::vector<Dimension> big_batch_matrix =
                arg0_rank > arg1_rank ? arg0_shape_tmp : arg1_shape_tmp;

            if (arg0_rank != arg1_rank)
            {
                // Expand small_batch_matrix (with 1) to have the same rank as big_batch_matrix
                size_t delta_rank = big_batch_matrix.size() - small_batch_matrix.size();
                small_batch_matrix.insert(small_batch_matrix.begin(), delta_rank, 1);
            }

            // Broadcast all batches (max_rank - 2),
            // expand dim with value 1 to bigger dim if dimensions are not equal
            for (auto i = 0; i < max_rank - 2; i++)
            {
                // Both dimensions are static
                if (small_batch_matrix[i].is_static() && big_batch_matrix[i].is_static())
                {
                    // Check if batch dimension broadcast is possible
                    NGRAPH_CHECK(
                        (small_batch_matrix[i].get_length() == big_batch_matrix[i].get_length()) ||
                            (std::min(small_batch_matrix[i].get_length(),
                                      big_batch_matrix[i].get_length()) == 1),
                        "Incompatible batch dimensions.");

                    output_shape[i] = std::max(small_batch_matrix[i].get_length(),
                                               big_batch_matrix[i].get_length());
                }
                // At least one dimension is dynamic
                else
                {
                    // Dimension with value 1 can be expanded to any bigger
                    auto min_dim_val = std::min(small_batch_matrix[i].get_min_length(),
                                                big_batch_matrix[i].get_min_length());
                    if (min_dim_val > 1)
                    {
                        NGRAPH_CHECK(Dimension::merge(merged_dimension,
                                                      small_batch_matrix[i],
                                                      big_batch_matrix[i]),
                                     "Incompatible batch dimensions bounds.");
                        output_shape[i] = merged_dimension;
                        continue;
                    }

                    // Both dimensions are dynamic and at least one of them has value <= 1
                    if (small_batch_matrix[i].is_dynamic() && big_batch_matrix[i].is_dynamic())
                    {
                        auto upper_bound_limit = -1;
                        auto upper_bound = std::min(small_batch_matrix[i].get_max_length(),
                                                    big_batch_matrix[i].get_max_length());

                        if (upper_bound != upper_bound_limit)
                        {
                            upper_bound = std::max(small_batch_matrix[i].get_max_length(),
                                                   big_batch_matrix[i].get_max_length());
                        }

                        auto lower_bound = std::max(small_batch_matrix[i].get_min_length(),
                                                    big_batch_matrix[i].get_min_length());
                        output_shape[i] = Dimension(lower_bound, upper_bound);
                    }
                    // One dimension is static and one dynamic
                    else
                    {
                        Dimension dim_static, dim_dynamic;
                        if (small_batch_matrix[i].is_static())
                        {
                            dim_static = small_batch_matrix[i];
                            dim_dynamic = big_batch_matrix[i];
                        }
                        else
                        {
                            dim_static = big_batch_matrix[i];
                            dim_dynamic = small_batch_matrix[i];
                        }

                        if (dim_static.get_length() <= 1 && dim_dynamic.get_min_length() <= 1)
                        {
                            output_shape[i] = Dimension(1, dim_dynamic.get_max_length());
                        }
                        else
                        {
                            output_shape[i] = dim_static;
                        }
                    }
                }
            }
        }

        // in output_shape replace 2 last axes with ROW_INDEX_DIM from arg0 matrix
        // and COL_INDEX_DIM from arg1 matrix
        output_shape.at(output_shape.size() - 2) = arg0_shape_tmp.at(arg0_rank - 2);
        output_shape.at(output_shape.size() - 1) = arg1_shape_tmp.at(arg1_rank - 1);

        return PartialShape(output_shape);
    }

    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& output,
                  bool transpose_a,
                  bool transpose_b)
    {
        using T = typename element_type_traits<ET>::value_type;

        Shape arg0_shape = arg0->get_shape();
        Shape arg1_shape = arg1->get_shape();

        PartialShape output_partial_shape = validate_matmul_output_shape(
            PartialShape(arg0_shape), PartialShape(arg1_shape), transpose_a, transpose_b);
        Shape output_shape = output_partial_shape.to_shape();
        output->set_element_type(arg0->get_element_type());
        output->set_shape(output_shape);

        runtime::reference::matmul<T>(arg0->get_data_ptr<ET>(),
                                      arg1->get_data_ptr<ET>(),
                                      output->get_data_ptr<ET>(),
                                      arg0_shape,
                                      arg1_shape,
                                      output_shape,
                                      transpose_a,
                                      transpose_b);
        return true;
    }

    bool evaluate_matmul(const HostTensorPtr& arg0,
                         const HostTensorPtr& arg1,
                         const HostTensorPtr& output,
                         bool transpose_a,
                         bool transpose_b)
    {
        bool rc = true;

        switch (arg0->get_element_type())
        {
            TYPE_CASE(i32)(arg0, arg1, output, transpose_a, transpose_b);
            break;
            TYPE_CASE(i64)(arg0, arg1, output, transpose_a, transpose_b);
            break;
            TYPE_CASE(u32)(arg0, arg1, output, transpose_a, transpose_b);
            break;
            TYPE_CASE(u64)(arg0, arg1, output, transpose_a, transpose_b);
            break;
            TYPE_CASE(f16)(arg0, arg1, output, transpose_a, transpose_b);
            break;
            TYPE_CASE(f32)(arg0, arg1, output, transpose_a, transpose_b);
            break;
        default: rc = false; break;
        }
        return rc;
    }
} // namespace

bool op::MatMul::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::MatMul::evaluate");
    return matmul::evaluate_matmul(
        inputs[0], inputs[1], outputs[0], get_transpose_a(), get_transpose_b());
}

void ngraph::op::v0::MatMul::validate_and_infer_types()
{
    element::Type result_et;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
        "Arguments do not have the same element type (arg0 element type: ",
        get_input_element_type(0),
        ", arg1 element type: ",
        get_input_element_type(1),
        ").");

    const auto& A_partial_shape = get_input_partial_shape(0);
    const auto& B_partial_shape = get_input_partial_shape(1);

    if (A_partial_shape.rank().is_static() && B_partial_shape.rank().is_static())
    {
        PartialShape output_shape;

        const bool transpose_a = get_transpose_a();
        const bool transpose_b = get_transpose_b();

        output_shape = matmul::validate_matmul_output_shape(
            A_partial_shape, B_partial_shape, transpose_a, transpose_b);

        set_output_type(0, result_et, output_shape);
    }
    else
    {
        set_output_type(0, result_et, PartialShape::dynamic());
    }
}

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

#include "recurrent_sequence.hpp"

using namespace ngraph;

void ngraph::op::util::validate_seq_input_rank_dimension(
    const std::vector<ngraph::PartialShape>& input)
{
    // Verify static ranks for all inputs
    for (size_t i = 0; i < input.size(); i++)
    {
        NGRAPH_CHECK((input[i].rank().is_static()),
                     "RNN Sequence supports only static rank for input tensors.");
    }

    for (size_t i = 0; i < input.size(); i++)
    {
        if (i == input.size() - 1)
        {
            // verify B input dimension which is 2D
            NGRAPH_CHECK((input[i].rank().get_length() == 2),
                         "RNN Sequence B input tensor dimension is not correct.");
        }
        else if (i == input.size() - 4)
        {
            // verify sequence_length input dimension which is 1D
            NGRAPH_CHECK((input[i].rank().get_length() == 1),
                         "RNN Sequence sequence_lengths input tensor dimension is not correct.");
        }
        else
        {
            // Verify all other input dimensions which are 3D tensor types
            NGRAPH_CHECK((input[i].rank().get_length() == 3),
                         "RNN Sequence input tensor dimension is not correct for ",
                         i,
                         " input parameter. Current input length: ",
                         input[i].rank().get_length());
        }
    }

    // Compare input_size dimension for X and W inputs
    const auto& x_pshape = input.at(0);
    const auto& w_pshape = input.at(input.size() - 3);

    NGRAPH_CHECK((x_pshape[2].compatible(w_pshape[2])),
                 "RNN Sequence mismatched input_size dimension.");
}
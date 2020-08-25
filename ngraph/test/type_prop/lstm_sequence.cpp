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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

//
// RNN sequence parameters
//
struct recurrent_sequence_parameters
{
    Dimension batch_size = 24;
    Dimension num_directions = 2;
    Dimension seq_length = 12;
    Dimension input_size = 8;
    Dimension hidden_size = 256;
    ngraph::element::Type et;
};

//
// Create and initialize default input test tensors.
//
shared_ptr<op::LSTMSequence>
    lstm_seq_tensor_initialization(const recurrent_sequence_parameters& param)
{
    auto batch_size = param.batch_size;
    auto seq_length = param.seq_length;
    auto input_size = param.input_size;
    auto num_directions = param.num_directions;
    auto hidden_size = param.hidden_size;
    auto et = param.et;

    const auto X = make_shared<op::Parameter>(et, PartialShape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<op::Parameter>(et, PartialShape{batch_size, num_directions, hidden_size});
    const auto initial_cell_state =
        make_shared<op::Parameter>(et, PartialShape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<op::Parameter>(et, PartialShape{batch_size});
    const auto W =
        make_shared<op::Parameter>(et, PartialShape{num_directions, hidden_size * 4, input_size});
    const auto R =
        make_shared<op::Parameter>(et, PartialShape{num_directions, hidden_size * 4, hidden_size});
    const auto B = make_shared<op::Parameter>(et, PartialShape{num_directions, hidden_size * 4});
    const auto P = make_shared<op::Parameter>(et, PartialShape{num_directions, hidden_size * 3});

    const auto lstm_sequence = make_shared<op::LSTMSequence>();

    lstm_sequence->set_argument(0, X);
    lstm_sequence->set_argument(1, initial_hidden_state);
    lstm_sequence->set_argument(2, initial_cell_state);
    lstm_sequence->set_argument(3, sequence_lengths);
    lstm_sequence->set_argument(4, W);
    lstm_sequence->set_argument(5, R);
    lstm_sequence->set_argument(6, B);
    lstm_sequence->set_argument(7, P);

    return lstm_sequence;
}

TEST(type_prop, lstm_sequence_forward)
{
    const size_t batch_size = 8;
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const size_t input_size = 4;
    const size_t hidden_size = 128;

    const auto X =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto initial_cell_state =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{batch_size});
    const auto W = make_shared<op::Parameter>(element::f32,
                                              Shape{num_directions, 4 * hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32,
                                              Shape{num_directions, 4 * hidden_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{num_directions, 4 * hidden_size});

    const auto lstm_direction = op::RecurrentSequenceDirection::FORWARD;

    const auto lstm_sequence = make_shared<op::LSTMSequence>(X,
                                                             initial_hidden_state,
                                                             initial_cell_state,
                                                             sequence_lengths,
                                                             W,
                                                             R,
                                                             B,
                                                             hidden_size,
                                                             lstm_direction);

    EXPECT_EQ(lstm_sequence->get_hidden_size(), hidden_size);
    EXPECT_EQ(lstm_sequence->get_direction(), op::RecurrentSequenceDirection::FORWARD);
    EXPECT_EQ(lstm_sequence->get_weights_format(), op::LSTMWeightsFormat::IFCO);
    EXPECT_TRUE(lstm_sequence->get_activations_alpha().empty());
    EXPECT_TRUE(lstm_sequence->get_activations_beta().empty());
    EXPECT_EQ(lstm_sequence->get_activations()[0], "sigmoid");
    EXPECT_EQ(lstm_sequence->get_activations()[1], "tanh");
    EXPECT_EQ(lstm_sequence->get_activations()[2], "tanh");
    EXPECT_EQ(lstm_sequence->get_clip_threshold(), 0.f);
    EXPECT_FALSE(lstm_sequence->get_input_forget());
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(0),
              (Shape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(1), (Shape{batch_size, num_directions, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(2), (Shape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, lstm_sequence_bidirectional)
{
    const size_t batch_size = 24;
    const size_t num_directions = 2;
    const size_t seq_length = 12;
    const size_t input_size = 8;
    const size_t hidden_size = 256;

    const auto X =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto initial_cell_state =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{batch_size});
    const auto W = make_shared<op::Parameter>(element::f32,
                                              Shape{num_directions, 4 * hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32,
                                              Shape{num_directions, 4 * hidden_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{num_directions, 4 * hidden_size});

    const auto weights_format = op::LSTMWeightsFormat::FICO;
    const auto lstm_direction = op::LSTMSequence::direction::BIDIRECTIONAL;
    const std::vector<float> activations_alpha = {2.7, 7.0, 32.367};
    const std::vector<float> activations_beta = {0.0, 5.49, 6.0};
    const std::vector<std::string> activations = {"tanh", "sigmoid", "sigmoid"};

    const auto lstm_sequence = make_shared<op::LSTMSequence>(X,
                                                             initial_hidden_state,
                                                             initial_cell_state,
                                                             sequence_lengths,
                                                             W,
                                                             R,
                                                             B,
                                                             hidden_size,
                                                             lstm_direction,
                                                             weights_format,
                                                             activations_alpha,
                                                             activations_beta,
                                                             activations);
    EXPECT_EQ(lstm_sequence->get_hidden_size(), hidden_size);
    EXPECT_EQ(lstm_sequence->get_direction(), op::LSTMSequence::direction::BIDIRECTIONAL);
    EXPECT_EQ(lstm_sequence->get_weights_format(), op::LSTMWeightsFormat::FICO);
    EXPECT_EQ(lstm_sequence->get_activations_alpha(), activations_alpha);
    EXPECT_EQ(lstm_sequence->get_activations_beta(), activations_beta);
    EXPECT_EQ(lstm_sequence->get_activations()[0], "tanh");
    EXPECT_EQ(lstm_sequence->get_activations()[1], "sigmoid");
    EXPECT_EQ(lstm_sequence->get_activations()[2], "sigmoid");
    EXPECT_EQ(lstm_sequence->get_clip_threshold(), 0.f);
    EXPECT_FALSE(lstm_sequence->get_input_forget());
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(0),
              (Shape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(1), (Shape{batch_size, num_directions, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(2), (Shape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, lstm_sequence_dynamic_batch_size)
{
    recurrent_sequence_parameters param;

    param.batch_size = Dimension::dynamic();
    param.num_directions = 2;
    param.seq_length = 12;
    param.input_size = 8;
    param.hidden_size = 256;
    param.et = element::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    lstm_sequence->validate_and_infer_types();

    EXPECT_EQ(lstm_sequence->get_output_partial_shape(0),
              (PartialShape{
                  param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(2),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), param.et);
}

TEST(type_prop, lstm_sequence_dynamic_num_directions)
{
    recurrent_sequence_parameters param;

    param.batch_size = 24;
    param.num_directions = Dimension::dynamic();
    param.seq_length = 12;
    param.input_size = 8;
    param.hidden_size = 256;
    param.et = element::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    lstm_sequence->validate_and_infer_types();

    EXPECT_EQ(lstm_sequence->get_output_partial_shape(0),
              (PartialShape{
                  param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(2),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), param.et);
}

TEST(type_prop, lstm_sequence_dynamic_seq_length)
{
    recurrent_sequence_parameters param;

    param.batch_size = 24;
    param.num_directions = 2;
    param.seq_length = Dimension::dynamic();
    param.input_size = 8;
    param.hidden_size = 256;
    param.et = element::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    lstm_sequence->validate_and_infer_types();

    EXPECT_EQ(lstm_sequence->get_output_partial_shape(0),
              (PartialShape{
                  param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(2),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), param.et);
}

TEST(type_prop, lstm_sequence_dynamic_hidden_size)
{
    recurrent_sequence_parameters param;

    param.batch_size = 24;
    param.num_directions = 2;
    param.seq_length = 12;
    param.input_size = 8;
    param.hidden_size = Dimension::dynamic();
    param.et = element::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    lstm_sequence->validate_and_infer_types();

    EXPECT_EQ(lstm_sequence->get_output_partial_shape(0),
              (PartialShape{
                  param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(2),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), param.et);
}

TEST(type_prop, lstm_sequence_dynamic_inputs)
{
    recurrent_sequence_parameters param;

    param.batch_size = Dimension::dynamic();
    param.input_size = Dimension::dynamic();
    param.hidden_size = Dimension::dynamic();
    param.num_directions = Dimension::dynamic();
    param.seq_length = Dimension::dynamic();
    param.et = element::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    lstm_sequence->validate_and_infer_types();

    EXPECT_EQ(lstm_sequence->get_output_partial_shape(0),
              (PartialShape{
                  param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(2),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), param.et);
}

TEST(type_prop, lstm_sequence_invalid_input_dimension)
{
    recurrent_sequence_parameters param;

    param.batch_size = 24;
    param.num_directions = 2;
    param.seq_length = 12;
    param.input_size = 8;
    param.hidden_size = 256;
    param.et = element::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);

    // Invalid rank0 for W tensor.
    const auto W = make_shared<op::Parameter>(param.et, PartialShape{});
    try
    {
        lstm_sequence->set_argument(4, W);
        lstm_sequence->validate_and_infer_types();
        FAIL() << "LSTMSequence node was created with invalid data.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("RNN Sequence input tensor dimension is not correct "));
    }

    // Invalid rank0 for X tensor.
    const auto X = make_shared<op::Parameter>(param.et, PartialShape{});
    try
    {
        lstm_sequence = lstm_seq_tensor_initialization(param);

        lstm_sequence->set_argument(0, X);
        lstm_sequence->validate_and_infer_types();
        FAIL() << "LSTMSequence node was created with invalid data.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("RNN Sequence input tensor dimension is not correct "));
    }

    // Invalid rank0 for initial_hidden_state tensor.
    const auto initial_hidden_state = make_shared<op::Parameter>(param.et, PartialShape{});
    try
    {
        lstm_sequence = lstm_seq_tensor_initialization(param);

        lstm_sequence->set_argument(1, initial_hidden_state);
        lstm_sequence->validate_and_infer_types();
        FAIL() << "LSTMSequence node was created with invalid data.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("RNN Sequence input tensor dimension is not correct "));
    }

    // Invalid rank0 for initial_cell_state tensor.
    const auto initial_cell_state = make_shared<op::Parameter>(param.et, PartialShape{});
    try
    {
        lstm_sequence = lstm_seq_tensor_initialization(param);

        lstm_sequence->set_argument(2, initial_cell_state);
        lstm_sequence->validate_and_infer_types();
        FAIL() << "LSTMSequence node was created with invalid data.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("RNN Sequence input tensor dimension is not correct "));
    }

    // Invalid rank0 for sequence_lengths tensor.
    const auto sequence_lengths = make_shared<op::Parameter>(param.et, PartialShape{});
    try
    {
        lstm_sequence = lstm_seq_tensor_initialization(param);

        lstm_sequence->set_argument(3, sequence_lengths);
        lstm_sequence->validate_and_infer_types();
        FAIL() << "LSTMSequence node was created with invalid data.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("RNN Sequence sequence_lengths input tensor dimension is not correct"));
    }

    // Invalid rank0 for R tensor.
    const auto R = make_shared<op::Parameter>(param.et, PartialShape{});
    try
    {
        lstm_sequence = lstm_seq_tensor_initialization(param);

        lstm_sequence->set_argument(5, R);
        lstm_sequence->validate_and_infer_types();
        FAIL() << "LSTMSequence node was created with invalid data.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("RNN Sequence input tensor dimension is not correct "));
    }

    // Invalid rank0 for B tensor.
    auto B = make_shared<op::Parameter>(param.et, PartialShape{});
    try
    {
        lstm_sequence = lstm_seq_tensor_initialization(param);

        lstm_sequence->set_argument(6, B);
        lstm_sequence->validate_and_infer_types();
        FAIL() << "LSTMSequence node was created with invalid data.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("RNN Sequence B input tensor dimension is not correct."));
    }

    // Invalid rank0 for P tensor.
    auto P = make_shared<op::Parameter>(param.et, PartialShape{});
    try
    {
        lstm_sequence = lstm_seq_tensor_initialization(param);

        lstm_sequence->set_argument(7, P);
        lstm_sequence->validate_and_infer_types();
        FAIL() << "LSTMSequence node was created with invalid data.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("LSTMSequence input tensor P shall have dimension 2D."));
    }
}

TEST(type_prop, lstm_sequence_invalid_input_dynamic_rank)
{
    recurrent_sequence_parameters param;

    param.batch_size = 24;
    param.num_directions = 2;
    param.seq_length = 12;
    param.input_size = 8;
    param.hidden_size = 256;
    param.et = element::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);

    // Validate invalid dynamic tensors for inputs from X to B. P input is validated separetely
    for (auto i = 0; i < lstm_sequence->get_input_size() - 1; i++)
    {
        auto dynamic_tensor =
            make_shared<op::Parameter>(param.et, PartialShape::dynamic(Rank::dynamic()));
        try
        {
            lstm_sequence = lstm_seq_tensor_initialization(param);

            lstm_sequence->set_argument(i, dynamic_tensor);
            lstm_sequence->validate_and_infer_types();
            FAIL() << "LSTMSequence node was created with invalid data.";
        }
        catch (const CheckFailure& error)
        {
            EXPECT_HAS_SUBSTRING(
                error.what(),
                std::string("RNN Sequence supports only static rank for input tensors."));
        }
    }

    // Invalid dynamic rank for P tensor.
    auto P = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    try
    {
        lstm_sequence = lstm_seq_tensor_initialization(param);

        lstm_sequence->set_argument(7, P);
        lstm_sequence->validate_and_infer_types();

        FAIL() << "LSTMSequence node was created with invalid data.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("LSTMSequence input tensor P shall have static rank."));
    }
}

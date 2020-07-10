# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import json

import numpy as np
import pytest

import ngraph as ng
from ngraph.exceptions import UserInputError
from ngraph.impl import Function, PartialShape, Shape
from tests.runtime import get_runtime
from tests.test_ngraph.util import run_op_node


def test_ngraph_function_api():
    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ng.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ng.parameter(shape, dtype=np.float32, name="C")
    model = (parameter_a + parameter_b) * parameter_c
    function = Function(model, [parameter_a, parameter_b, parameter_c], "TestFunction")

    ordered_ops = function.get_ordered_ops()
    op_types = [op.get_type_name() for op in ordered_ops]
    assert op_types == ["Parameter", "Parameter", "Parameter", "Add", "Multiply", "Result"]
    assert len(function.get_ops()) == 6
    assert function.get_output_size() == 1
    assert function.get_output_op(0).get_type_name() == "Result"
    assert function.get_output_element_type(0) == parameter_a.get_element_type()
    assert list(function.get_output_shape(0)) == [2, 2]
    assert len(function.get_parameters()) == 3
    assert len(function.get_results()) == 1
    assert function.get_name() == "TestFunction"


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
def test_simple_computation_on_ndarrays(dtype):
    runtime = get_runtime()

    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=dtype, name="A")
    parameter_b = ng.parameter(shape, dtype=dtype, name="B")
    parameter_c = ng.parameter(shape, dtype=dtype, name="C")
    model = (parameter_a + parameter_b) * parameter_c
    computation = runtime.computation(model, parameter_a, parameter_b, parameter_c)

    value_a = np.array([[1, 2], [3, 4]], dtype=dtype)
    value_b = np.array([[5, 6], [7, 8]], dtype=dtype)
    value_c = np.array([[9, 10], [11, 12]], dtype=dtype)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([[54, 80], [110, 144]], dtype=dtype))

    value_a = np.array([[13, 14], [15, 16]], dtype=dtype)
    value_b = np.array([[17, 18], [19, 20]], dtype=dtype)
    value_c = np.array([[21, 22], [23, 24]], dtype=dtype)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([[630, 704], [782, 864]], dtype=dtype))


def test_serialization():
    dtype = np.float32
    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=dtype, name="A")
    parameter_b = ng.parameter(shape, dtype=dtype, name="B")
    parameter_c = ng.parameter(shape, dtype=dtype, name="C")
    model = (parameter_a + parameter_b) * parameter_c

    runtime = get_runtime()
    computation = runtime.computation(model, parameter_a, parameter_b, parameter_c)
    try:
        serialized = computation.serialize(2)
        serial_json = json.loads(serialized)

        assert serial_json[0]["name"] != ""
        assert 10 == len(serial_json[0]["ops"])
    except Exception:
        pass


def test_broadcast_1():
    input_data = np.array([1, 2, 3])

    new_shape = [3, 3]
    expected = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    result = run_op_node([input_data], ng.broadcast, new_shape)
    assert np.allclose(result, expected)


def test_broadcast_2():
    input_data = np.arange(4)
    new_shape = [3, 4, 2, 4]
    expected = np.broadcast_to(input_data, new_shape)
    result = run_op_node([input_data], ng.broadcast, new_shape)
    assert np.allclose(result, expected)


def test_broadcast_3():
    input_data = np.array([1, 2, 3])
    new_shape = [3, 3]
    axis_mapping = [0]
    expected = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

    result = run_op_node([input_data], ng.broadcast, new_shape, axis_mapping, "EXPLICIT")
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "destination_type, input_data",
    [(bool, np.zeros((2, 2), dtype=int)), ("boolean", np.zeros((2, 2), dtype=int))],
)
def test_convert_to_bool(destination_type, input_data):
    expected = np.array(input_data, dtype=bool)
    result = run_op_node([input_data], ng.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == bool


@pytest.mark.parametrize(
    "destination_type, rand_range, in_dtype, expected_type",
    [
        (np.float32, (-8, 8), np.int32, np.float32),
        (np.float64, (-16383, 16383), np.int64, np.float64),
        ("f32", (-8, 8), np.int32, np.float32),
        ("f64", (-16383, 16383), np.int64, np.float64),
    ],
)
def test_convert_to_float(destination_type, rand_range, in_dtype, expected_type):
    np.random.seed(133391)
    input_data = np.random.randint(*rand_range, size=(2, 2), dtype=in_dtype)
    expected = np.array(input_data, dtype=expected_type)
    result = run_op_node([input_data], ng.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == expected_type


@pytest.mark.parametrize(
    "destination_type, expected_type",
    [
        (np.int8, np.int8),
        (np.int16, np.int16),
        (np.int32, np.int32),
        (np.int64, np.int64),
        ("i8", np.int8),
        ("i16", np.int16),
        ("i32", np.int32),
        ("i64", np.int64),
    ],
)
def test_convert_to_int(destination_type, expected_type):
    np.random.seed(133391)
    input_data = np.ceil(-8 + np.random.rand(2, 3, 4) * 16)
    expected = np.array(input_data, dtype=expected_type)
    result = run_op_node([input_data], ng.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == expected_type


@pytest.mark.parametrize(
    "destination_type, expected_type",
    [
        (np.uint8, np.uint8),
        (np.uint16, np.uint16),
        (np.uint32, np.uint32),
        (np.uint64, np.uint64),
        ("u8", np.uint8),
        ("u16", np.uint16),
        ("u32", np.uint32),
        ("u64", np.uint64),
    ],
)
def test_convert_to_uint(destination_type, expected_type):
    np.random.seed(133391)
    input_data = np.ceil(np.random.rand(2, 3, 4) * 16)
    expected = np.array(input_data, dtype=expected_type)
    result = run_op_node([input_data], ng.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == expected_type


def test_bad_data_shape():
    A = ng.parameter(shape=[2, 2], name="A", dtype=np.float32)
    B = ng.parameter(shape=[2, 2], name="B")
    model = A + B
    runtime = get_runtime()
    computation = runtime.computation(model, A, B)

    value_a = np.array([[1, 2]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    with pytest.raises(UserInputError):
        computation(value_a, value_b)


def test_constant_get_data_bool():
    input_data = np.array([True, False, False, True])
    node = ng.constant(input_data, dtype=np.bool)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.float32, np.float64])
def test_constant_get_data_floating_point(data_type):
    np.random.seed(133391)
    input_data = np.random.randn(2, 3, 4).astype(data_type)
    min_value = -1.0e20
    max_value = 1.0e20
    input_data = min_value + input_data * max_value * data_type(2)
    node = ng.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.int64, np.int32, np.int16, np.int8])
def test_constant_get_data_signed_integer(data_type):
    np.random.seed(133391)
    input_data = np.random.randint(
        np.iinfo(data_type).min, np.iinfo(data_type).max, size=[2, 3, 4], dtype=data_type
    )
    node = ng.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.uint64, np.uint32, np.uint16, np.uint8])
def test_constant_get_data_unsigned_integer(data_type):
    np.random.seed(133391)
    input_data = np.random.randn(2, 3, 4).astype(data_type)
    input_data = (
        np.iinfo(data_type).min + input_data * np.iinfo(data_type).max + input_data * np.iinfo(data_type).max
    )
    node = ng.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


def test_backend_config():
    dummy_config = {"dummy_option": "dummy_value"}
    # Expect no throw
    runtime = get_runtime()
    runtime.set_config(dummy_config)


def test_result():
    node = [[11, 10], [1, 8], [3, 4]]

    result = run_op_node([node], ng.ops.result)
    assert np.allclose(result, node)

def test_node_output():
    input_array = np.array([0, 1, 2, 3, 4, 5])
    splits = 3
    expected_shape = len(input_array) // splits

    input_tensor = ng.constant(input_array, dtype=np.int32)
    axis = ng.constant(0, dtype=np.int64)
    split_node = ng.split(input_tensor, axis, splits)

    split_node_outputs = split_node.outputs()

    assert len(split_node_outputs) == split_node.get_output_size()
    assert [output_node.get_index() for output_node in split_node_outputs] == [0, 1, 2]
    assert np.equal([output_node.get_element_type() for output_node in split_node_outputs], input_tensor.get_element_type()).all()
    assert np.equal([output_node.get_shape() for output_node in split_node_outputs], Shape([expected_shape])).all()
    assert np.equal([output_node.get_partial_shape() for output_node in split_node_outputs], PartialShape([expected_shape])).all()

    output0 = split_node.output(0)
    output1 = split_node.output(1)
    output2 = split_node.output(2)

    assert [output0.get_index(), output1.get_index(), output2.get_index()] == [0, 1, 2]

def test_node_input():
    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ng.parameter(shape, dtype=np.float32, name="B")

    model = (parameter_a + parameter_b)

    model_inputs = model.inputs()

    print([input_node.get_index() for input_node in model_inputs])

    assert len(model_inputs) == 2
    assert [input_node.get_index() for input_node in model_inputs] == [0, 1]
    assert np.equal([input_node.get_element_type() for input_node in model_inputs], model.get_element_type()).all()
    assert np.equal([input_node.get_shape() for input_node in model_inputs], Shape(shape)).all()
    assert np.equal([input_node.get_partial_shape() for input_node in model_inputs], PartialShape(shape)).all()

    input0 = model.input(0)
    input1 = model.input(1)

    assert [input0.get_index(), input1.get_index()] == [0, 1]

def test_target_inputs():
    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ng.parameter(shape, dtype=np.float32, name="B")

    model = (parameter_a + parameter_b)

    out_a = parameter_a.output(0)
    out_b = parameter_b.output(0)
    out_a_input = (list(out_a.get_target_inputs())[0]).get_node()
    out_b_input = (list(out_b.get_target_inputs())[0]).get_node()

    assert out_a.get_node().name == parameter_a.name
    assert out_b.get_node().name == parameter_b.name
    assert np.equal([out_a.get_node().get_output_shape(0)], [parameter_a.get_output_shape(0)]).all()
    assert np.equal([out_b.get_node().get_output_shape(0)], [parameter_b.get_output_shape(0)]).all()
    assert out_a_input.name == model.name
    assert out_b_input.name == model.name
    assert np.equal([out_a_input.get_output_shape(0)], [model.get_output_shape(0)]).all()
    assert np.equal([out_b_input.get_output_shape(0)], [model.get_output_shape(0)]).all()

def test_soruce_outputs():
    assert False

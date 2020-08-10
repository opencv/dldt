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
import ngraph as ng
from ngraph.impl import Shape, Type


def test_swish_props_with_beta():
    data = ng.parameter(Shape([3, 10]), dtype=Type.f32, name="data")
    beta = ng.parameter(Shape([]), dtype=Type.f32, name="beta")

    node = ng.swish(data, beta)
    assert node.get_type_name() == "Swish"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_swish_props_without_beta():
    data = ng.parameter(Shape([3, 10]), dtype=Type.f32, name="data")

    node = ng.swish(data)
    assert node.get_type_name() == "Swish"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32

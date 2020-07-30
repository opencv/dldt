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
"""nGraph helper functions."""

from ngraph.impl import Function
from openvino.inference_engine import IENetwork


def get_function(cnn_network: IENetwork) -> Function:
    """Get nGraph function from CNN network."""
    capsule = cnn_network._get_function_capsule()
    ng_function = Function.from_capsule(capsule)
    return ng_function

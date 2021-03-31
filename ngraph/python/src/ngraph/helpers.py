# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""nGraph helper functions."""

from ngraph.impl import Function
from openvino.inference_engine import IENetwork


def function_from_cnn(cnn_network: IENetwork) -> Function:
    """Get nGraph function from Inference Engine CNN network."""
    ng_function = cnn_network.get_function()
    return ng_function


def function_to_cnn(ng_function: Function) -> Function:
    """Get Inference Engine CNN network from nGraph function."""
    capsule = Function.to_capsule(ng_function)
    return IENetwork(capsule)

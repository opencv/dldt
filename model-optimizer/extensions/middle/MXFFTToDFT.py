# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.dft import DFT, IDFT
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.ops.pad import Pad
from mo.ops.unsqueeze import Unsqueeze


class MXFFTToDFT(FrontReplacementSubgraph):
    """
    This transformation converts the operation MXFFT into OpenVINO DFT (if the attribute 'is_inverse' is False),
    or into OpenVINO IDFT (otherwise).
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for mx_fft in graph.get_op_nodes(op='MXFFT'):
            mx_fft_name = mx_fft.soft_get('name', mx_fft.id)

            mx_fft_input_shape = mx_fft.in_port(0).data.get_shape()
            assert mx_fft_input_shape is not None, 'Input shape of MXFFT node {} must not be None'.format(mx_fft_name)
            input_rank = len(mx_fft_input_shape)

            pads_begin = np.zeros(input_rank + 1, dtype=np.int64)
            pads_end = np.zeros(input_rank + 1, dtype=np.int64)
            pads_end[-1] = 1

            unsqueeze_node = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array([-1])},
                                                         {'name': mx_fft_name + '/Unsqueeze'})
            pad_node = create_op_with_const_inputs(graph, Pad, {1: pads_begin, 2: pads_end},
                                                   {'name': mx_fft_name + '/Pad', 'mode': 'constant'},
                                                   unsqueeze_node)

            op = IDFT if mx_fft.soft_get('is_inverse', False) else DFT
            dft_node = create_op_with_const_inputs(graph, op, {1: int64_array([-1])}, {'in_ports_count': 2}, pad_node)

            mx_fft.in_port(0).get_connection().set_destination(unsqueeze_node.in_port(0))
            mx_fft.out_port(0).get_connection().set_source(dft_node.in_port(0))
            rename_nodes([(mx_fft, mx_fft_name + '/to_be_removed'), (dft_node, mx_fft_name)])

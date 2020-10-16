"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class LookupTableInsert(Op):
    '''
    In some models this operation has only output control flow edges and no output data edges.
    And for these cases implementation of the shape inference is needed since the shape inference is executed
    before control flow edges resolving. This operation has non-tensor output so the output shape is empty.
    '''
    enabled = False
    op = 'LookupTableInsert'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': self.infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) == 3, \
            "Incorrect number of inputs for {} node".format(node_name)

        # check shapes of input tensors
        keys_shape = node.in_port(1).data.get_shape()
        values_shape = node.in_port(2).data.get_shape()
        assert np.array_equal(keys_shape, values_shape), \
            'Shapes of tensors with keys and values must be equal for {} node'.format(node_name)

        # set output shape that must be empty
        # since output is not a tensor
        node.out_port(0).data.set_shape(int64_array([]))


class LookupTableInsertV2(LookupTableInsert):
    enabled = False
    op = 'LookupTableInsertV2'

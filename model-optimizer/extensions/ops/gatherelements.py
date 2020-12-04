"""
 Copyright (C) 2017-2020 Intel Corporation

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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class GatherElements(Op):
    op = 'GatherElements'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset6',
            'infer': self.infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'axis': 0,
        }, attrs)

    def backend_attrs(self):
        return ['axis']

    @staticmethod
    def infer(node: Node):
        indices_shape = node.in_port(1).data.get_shape()
        node.out_port(0).data.set_shape(indices_shape)

        # todo: add value_inference

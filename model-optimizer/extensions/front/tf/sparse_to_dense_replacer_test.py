"""
 Copyright (C) 2018-2021 Intel Corporation

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

import unittest

from extensions.front.tf.sparse_to_dense_replacer import SparseToDenseReplacer
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph
from mo.utils.unittest.graph import build_graph, const


class SparseToDenseFrontReplacersTest(unittest.TestCase):
    def test1(self):
        nodes_attributes = {
            'input_indices': {'shape': int64_array([5, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_values' : {'shape': int64_array([5]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},

            'sparse_to_dense' : {'kind': 'op', 'op': 'SparseToDense'},
            'broadcast' : {'kind': 'op', 'op': 'Broadcast'},
            'scatternd' : {'kind': 'op', 'op': 'ScatterNDUpdate'},
            'cast_default_value': {'kind': 'op', 'op': 'Cast'},

            'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},

            **const('input_dense_shape', int64_array([50, 40])),
            **const('input_default_value', int64_array(0))}

        graph = build_graph(nodes_attributes,
                            [('input_indices', 'sparse_to_dense', {'out': 0, 'in': 0}),
                             ('input_dense_shape', 'sparse_to_dense', {'out': 0, 'in': 1}),
                             ('input_values', 'sparse_to_dense', {'out': 0, 'in': 2}),
                             ('input_default_value', 'sparse_to_dense', {'out': 0, 'in': 3}),
                             ('sparse_to_dense', 'last', {'out': 0, 'in': 0})],
                             nodes_with_edges_only=True)
        graph.stage = 'front'
        SparseToDenseReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('input_default_value', 'cast_default_value', {'in': 0}),
                                 ('cast_default_value', 'broadcast', {'in': 0}),
                                 ('input_dense_shape', 'broadcast', {'in': 1}),
                                 ('broadcast', 'scatternd', {'in': 0}),
                                 ('input_indices', 'scatternd', {'in': 1}),
                                 ('input_values', 'scatternd', {'in': 2}),
                                 ('scatternd', 'last', {'in': 0})],
                                 nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

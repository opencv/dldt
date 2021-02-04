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

import numpy as np
from generator import generator, generate

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.strided_slice import StridedSlice
from mo.front.common.partial_infer.concat import concat_infer
from extensions.ops.split import VariadicSplit
from extensions.middle.StridedSliceNormalizer import StridedSliceNormalizer
from mo.middle.passes.infer import partial_infer
from mo.utils.error import Error
from mo.utils.unittest.graph import build_graph, valued_const_with_data, valued_data, regular_op_with_empty_data, \
    connect, shaped_data, shaped_const_with_data, result, build_graph_with_attrs, regular_op, empty_data
from mo.utils.ir_engine.compare_graphs import compare_graphs

# extended with existing concat

edges = (
    *connect('input', '0:strided_slice'),
    *connect('begin', '1:strided_slice'),
    *connect('end', '2:strided_slice'),
    *connect('strides', '3:strided_slice'),
    *connect('strided_slice', 'res')
)


class TestStridedSliceNormalizer(unittest.TestCase):

    def test_strided_slice_extend_inputs(self):
        input_shape = (16, 100, 100, 3)
        nodes = {
            **valued_const_with_data('input', np.arange(np.product(input_shape)).reshape(*input_shape)),
            **regular_op_with_empty_data('strided_slice', {'op': 'StridedSlice', 'begin_mask': [1, 1, 1], 'end_mask': [1, 1, 1],
                                                           'shrink_axis_mask': [0, 0, 0],
                                                           'new_axis_mask': [0, 0, 0],
                                                           'ellipsis_mask': [0, 0, 0],
                                                           'infer': StridedSlice.infer}),

            **regular_op_with_empty_data('strided_slice_ref', {'op': 'StridedSlice', 'begin_mask': [1, 1, 1, 0],
                                                               'end_mask': [1, 1, 1, 0], 'ellipsis_mask': [0, 0, 0, 0],
                                                               'new_axis_mask': [0, 0, 0, 0],
                                                               'shrink_axis_mask': [0, 0, 0, 0],
                                                               'infer': StridedSlice.infer}),
            **valued_const_with_data('begin', int64_array([0, 0, 0])),
            **valued_const_with_data('begin_placeholder', int64_array([0])),
            **regular_op_with_empty_data('begin_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),
            **valued_const_with_data('end', int64_array([4, 25, 50])),
            **valued_const_with_data('end_placeholder', int64_array([0])),
            **regular_op_with_empty_data('end_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),
            **valued_const_with_data('strides', int64_array([1, 1, 1])),
            **valued_const_with_data('strides_placeholder', int64_array([1])),
            **regular_op_with_empty_data('strides_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),
            **regular_op('res', {'kind': 'op', 'type': 'Result', 'op': 'Result', 'infer': lambda x: None})
        }

        edges_ref_extended_inputs = (
            *connect('input', '0:strided_slice_ref'),

            *connect('begin', '0:begin_concat'),
            *connect('begin_placeholder', '1:begin_concat'),
            *connect('begin_concat', '1:strided_slice_ref'),

            *connect('end', '0:end_concat'),
            *connect('end_placeholder', '1:end_concat'),
            *connect('end_concat', '2:strided_slice_ref'),

            *connect('strides', '0:strides_concat'),
            *connect('strides_placeholder', '1:strides_concat'),
            *connect('strides_concat', '3:strided_slice_ref'),

            *connect('strided_slice_ref', 'res')
        )

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph_ref = build_graph(nodes, edges_ref_extended_inputs, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph_ref.stage = 'middle'

        StridedSliceNormalizer().find_and_replace_pattern(graph)
        graph = partial_infer(graph)
        graph_ref = partial_infer(graph_ref)

        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=False)
        self.assertTrue(flag, 'Graphs after StridedSliceNormalizer do not match to reference: {}'.format(resp))

    def test_strided_slice_unrooll_ellipsis(self):
        input_shape = (10, 10, 10, 10)
        # out = inp[1:4, ..., 0:5] -> inp[1:4, :, :, 0:5] => out_shape = (3, 10, 10, 5)
        ellipsis_start = 1

        nodes = {
            **valued_const_with_data('input', np.arange(np.product(input_shape)).reshape(*input_shape)),
            **regular_op_with_empty_data('strided_slice', {'op': 'StridedSlice', 'begin_mask': [1, 1, 1], 'end_mask': [1, 1, 1],
                                                           'shrink_axis_mask': [0, 0, 0],
                                                           'new_axis_mask': [0, 0, 0],
                                                           'ellipsis_mask': [0, 1, 0],
                                                           'infer': StridedSlice.infer}),

            **regular_op_with_empty_data('strided_slice_ref', {'op': 'StridedSlice', 'begin_mask': [1, 0, 0, 1],
                                                               'end_mask': [1, 0, 0, 1], 'ellipsis_mask': [0, 0, 0, 0],
                                                               'new_axis_mask': [0, 0, 0, 0],
                                                               'shrink_axis_mask': [0, 0, 0, 0],
                                                               'infer': StridedSlice.infer}),

            **valued_const_with_data('begin', int64_array([1, 0, 0])),
            **valued_const_with_data('split_axis_begin', int64_array(0)),
            **valued_const_with_data('splits_lengths_begin', int64_array([ellipsis_start, -1])),
            **regular_op_with_empty_data('split_for_begin', {'op': 'VariadicSplit', 'infer': VariadicSplit.infer}),
            **empty_data('split_for_begin_data_1'),
            **valued_const_with_data('begin_placeholder', int64_array([0])),
            **regular_op_with_empty_data('begin_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),


            **valued_const_with_data('end', int64_array([4, 0, 5])),
            **valued_const_with_data('split_axis_end', int64_array(0)),
            **valued_const_with_data('splits_lengths_end', int64_array([ellipsis_start, -1])),
            **regular_op_with_empty_data('split_for_end', {'op': 'VariadicSplit', 'infer': VariadicSplit.infer}),
            **empty_data('split_for_end_data_1'),
            **valued_const_with_data('end_placeholder', int64_array([0])),
            **regular_op_with_empty_data('end_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),

            **valued_const_with_data('strides', int64_array([1, 1, 1])),
            **valued_const_with_data('split_axis_strides', int64_array(0)),
            **valued_const_with_data('splits_lengths_strides', int64_array([ellipsis_start, -1])),
            **regular_op_with_empty_data('split_for_strides', {'op': 'VariadicSplit', 'infer': VariadicSplit.infer}),
            **empty_data('split_for_strides_data_1'),
            **valued_const_with_data('strides_placeholder', int64_array([1])),
            **regular_op_with_empty_data('strides_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),

            **regular_op('res', {'kind': 'op', 'type': 'Result', 'op': 'Result', 'infer': lambda x: None})
        }

        edges_ref_ellipsis_unrolled = (
            *connect('input', '0:strided_slice_ref'),

            *connect('begin', '0:split_for_begin'),
            *connect('split_axis_begin', '1:split_for_begin'),
            *connect('splits_lengths_begin', '2:split_for_begin'),
            *connect('split_for_begin:0', '0:begin_concat'),
            *connect('begin_placeholder', '1:begin_concat'),
            ('split_for_begin', 'split_for_begin_data_1', {'out': 1, 'in': 2}),
            ('split_for_begin_data_1', 'begin_concat', {'out': 1, 'in': 2}),
            *connect('begin_concat', '1:strided_slice_ref'),

            *connect('end', '0:split_for_end'),
            *connect('split_axis_end', '1:split_for_end'),
            *connect('splits_lengths_end', '2:split_for_end'),
            *connect('split_for_end:0', '0:end_concat'),
            *connect('end_placeholder', '1:end_concat'),
            ('split_for_end', 'split_for_end_data_1', {'out': 1, 'in': 2}),
            ('split_for_end_data_1', 'end_concat', {'out': 1, 'in': 2}),
            *connect('end_concat', '2:strided_slice_ref'),

            *connect('strides', '0:split_for_strides'),
            *connect('split_axis_strides', '1:split_for_strides'),
            *connect('splits_lengths_strides', '2:split_for_strides'),
            *connect('split_for_strides:0', '0:strides_concat'),
            *connect('strides_placeholder', '1:strides_concat'),
            ('split_for_strides', 'split_for_strides_data_1', {'out': 1, 'in': 2}),
            ('split_for_strides_data_1', 'strides_concat', {'out': 1, 'in': 2}),
            *connect('strides_concat', '3:strided_slice_ref'),

            *connect('strided_slice_ref', 'res')
        )

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph_ref = build_graph(nodes, edges_ref_ellipsis_unrolled, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph_ref.stage = 'middle'
        graph = partial_infer(graph)
        StridedSliceNormalizer().find_and_replace_pattern(graph)
        graph = partial_infer(graph)
        graph_ref = partial_infer(graph_ref)

        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=False)
        self.assertTrue(flag, 'Graphs after StridedSliceNormalizer do not match to reference: {}'.format(resp))

    def run_infer_test(self, inp, ref_res, begin, end, strides, begin_mask, end_mask,
                       shrink_axis_mask, new_axis_mask, ellipsis_mask):
        nodes = {
            **valued_const_with_data('input', np.arange(np.product(inp)).reshape(*inp)),
            **valued_const_with_data('begin', int64_array(begin)),
            **valued_const_with_data('end', int64_array(end)),
            **valued_const_with_data('strides', int64_array(strides)),
            **regular_op_with_empty_data('strided_slice', {'begin_mask': begin_mask, 'end_mask': end_mask,
                                                           'shrink_axis_mask': shrink_axis_mask,
                                                           'new_axis_mask': new_axis_mask,
                                                           'ellipsis_mask': ellipsis_mask,
                                                           'infer': StridedSlice.infer}),
            **regular_op('res', {'kind': 'op', 'type': 'Result', 'op': 'Result', 'infer': lambda x: None})
        }

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        StridedSliceNormalizer().find_and_replace_pattern(graph)
        graph = partial_infer(graph)

        node = Node(graph, 'strided_slice')
        res = node.out_port(0).data.get_shape()
        self.assertTrue(np.array_equal(res, ref_res))

    def test_strided_slice_infer_after_normalizer_1(
            self,  # inp[0, :34, 20, :2]
            inp=(1, 35, 35, 3), ref_res=(34, 2),
            begin=(0, 0, 0, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1),
            begin_mask=(0,), end_mask=(0,),
            shrink_axis_mask=(1, 0, 1, 0), new_axis_mask=(0,),
            ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

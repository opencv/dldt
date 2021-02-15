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
import numpy as np

from extensions.ops.split import VariadicSplit
from mo.front.common.partial_infer.utils import int64_array
from mo.ops.strided_slice import StridedSlice
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node
from mo.graph.perm_inputs import PermuteInputs
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.utils.error import Error
from mo.ops.const import Const
from mo.ops.op import PermuteAttrs


class StridedSliceNormalizer(MiddleReplacementPattern):
    """
    StridedSlice is not normal if it cannot be permuted by ApplyPermutations. This normalizer
    inserts blank colons ':' in slice expression so that it can be correctly permuted
    from NHWC to NCHW layout. It changes masks and inserts blank begin, end and strides values.
    In order to successfully run in ShapeOf subgraphs insertations  must be done by inserting nodes
    not just by rewritting constants.

    StridedSlice is not normal in 2 cases:
        1. rank of a slice expression is less than rank of input tensor
        2. there is an ellipsis

    1st case example
    BEFORE:
                  |
                begin
           value=[0, 0]
                  |

    AFTER:
                  |
                begin          Const
           value=[0, 0]     value=[0, 0]
                  \             /
                   \           /
                      Concat
                 value=[0, 0, 0, 0]
                        |

    Input of a shape [16, 100, 100, 3] in NHWC layout, output = input[:, 0:50] will be extended to input[:, 0:50, :, :]
    after permutation to NCHW output = input[:, :, 0:50, :]. Above is show only for begin input, for end and strides
    changes are analogously.

    2nd case example
    BEFORE:
                  |
                begin
           value=[1, 50]
                  |

    AFTER:
                  |
                begin
           value=[1, 1, 1]
                  |
             VariadicSplit
           /              \
          /                \
         /       Const      \
         \     val=[0, 0]   /
          \       |        /
           \      |       /
               Concat
           value=[1, 0, 0, 1, 1]
                  |

    Input of a shape [16, 10, 100, 100, 3] in NDHWC layout output = input[1:4, ..., 1:51, 1:3],
    output_shape = [3, 10, 100, 50, 2]. In order to do correctly layout permutation in slice expression
    input[1:4, ..., 1:51, 1:3] ellipsis should be exended => input[1:4, :, :, 1:51, 1:3]. After
    layour permutation input[1:4, 1:3, :, : 1:5].

    In the places of colons blank zero begin, end and strides values
    should be inserted. In order to do that we split begin, and concatenate with the blank zeros in the middle. Above
    is show only for begin input, for end and strides changes are analogously.
    """
    enabled = True

    def run_before(self):
        from extensions.middle.LayoutChangeForConstantShapePaths import LayoutChangeForConstantShapePaths
        return [LayoutChangeForConstantShapePaths]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='StridedSlice'):
            StridedSliceNormalizer.normalize_strided_slice(graph, node)
            PermuteAttrs.create_permute_attrs(node, attrs=[('begin_mask', 'input:0'),  # but indeed depends from slice_rank
                                                           ('end_mask', 'input:0'),
                                                           ('new_axis_mask', 'input:0'),
                                                           ('shrink_axis_mask', 'input:0'),
                                                           ('ellipsis_mask', 'input:0')])

            # exceptional case, need to add permute edge ettributes here not during shape_infer
            # if we specify during shape_infer we will get wrong ApplyPermutation because
            # input data node will be changed
            PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:1', 'slice', 'dim_size')
            PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:2', 'slice', 'dim_size')
            PermuteInputs().set_input_permutation(node.in_node(3), node, 'input:3', 'slice', 'dim_size')

    @staticmethod
    def normalize_strided_slice(graph: Graph, node: Node):
        input_shape = node.in_port(0).data.get_shape()
        input_rank = len(input_shape)
        slice_rank = node.in_port(1).data.get_shape()[0]

        StridedSlice.align_mask_with_slice_rank(node, slice_rank)  # if StridedSlice is created after partial_infer
        StridedSliceNormalizer.normalize_slices_attr(node)

        num_insertions = input_rank - slice_rank + np.count_nonzero(node.new_axis_mask)
        if np.any(node.ellipsis_mask):
            assert np.count_nonzero(node.ellipsis_mask) == 1, 'only one ellipsis_mask nonzero value is allowed'
            ellipsis_start = np.nonzero(node.ellipsis_mask)[0][0]
            # since we don't expect values in begin and end: take the whole range along ellipsis_start
            node.begin_mask[ellipsis_start] = 0
            node.end_mask[ellipsis_start] = 0
            node.ellipsis_mask[ellipsis_start] = 0
            insertation_start_idx = ellipsis_start + 1

            StridedSliceNormalizer.unroll_ellipsis_for_inputs(graph, node, ellipsis_start, num_insertions)
        elif num_insertions > 0:
            insertation_start_idx = slice_rank  # insert blank values to mask ends
            StridedSliceNormalizer.extend_inputs(node, num_insertions)

        if num_insertions > 0:
            # insert blank values for ellipsis unrolling and extending
            for mask_name in StridedSlice.get_mask_names():
                node[mask_name] = np.insert(node[mask_name], insertation_start_idx, [0] * num_insertions).astype(int)

    @staticmethod
    def unroll_ellipsis_for_inputs(graph: Graph, node: Node, ellipsis_start: int, num_ellipsis_ext: int):
        node_name = node.soft_get('name', node.id)
        for i, slice_name in enumerate(('begin', 'end', 'strides')):
            i += 1
            placeholder_arr = np.zeros(num_ellipsis_ext) if i != 3 else np.ones(num_ellipsis_ext)
            placeholder_node = Const(graph, {'name': node_name + '/const_to_unroll_{}_ellipsis'.format(slice_name),
                                             'value': int64_array(placeholder_arr)}).create_node()

            concat_in_ports_count = 3 if ellipsis_start != 0 else 2
            concat = Concat(graph, {'axis': 0, 'name': node_name + '/concat_{}'.format(slice_name),
                                    'in_ports_count': concat_in_ports_count}).create_node()

            if ellipsis_start != 0:
                split = create_op_with_const_inputs(graph, VariadicSplit, {1: int64_array(0),
                                                                           2: int64_array([ellipsis_start, -1])},
                                                    {'name': node_name + '/split_for_{}_ellipsis'.format(slice_name),
                                                     'out_ports_count': 2})
                node.in_port(i).get_connection().set_destination(split.in_port(0))

                concat.in_port(0).connect(split.out_port(0))
                concat.in_port(1).connect(placeholder_node.out_port(0))
                concat.in_port(2).connect(split.out_port(1))
            else:
                concat.in_port(0).connect(placeholder_node.out_port(0))
                node.in_port(i).get_connection().set_destination(concat.in_port(1))

            concat.out_port(0).get_connection().set_destination(node.in_port(i))

    @staticmethod
    def extend_inputs(node: Node, num_insertations: int):
        graph = node.graph
        node_name = node.soft_get('name', node.id)

        for i, slice_name in enumerate(('begin', 'end', 'strides')):
            i += 1
            if node.in_port(3).disconnected():
                break  # if strides are not specified no need for extending

            # for strides blank ones should be inserted
            blank_values_arr = np.zeros(num_insertations) if i != 3 else np.ones(num_insertations)
            blank_values_node = Const(graph, {'name': node_name + '/extend_{}_const'.format(slice_name),
                                             'value': int64_array(blank_values_arr)}).create_node()

            if node.in_port(i).get_source().node.soft_get('type') == 'Concat':
                # concat already exists
                concat = node.in_port(i).get_source().node
                last_in_port = max(concat.in_ports().keys())
                assert not concat.in_port(last_in_port).disconnected(), 'The last in_port of Concat node {}' \
                                                                            'should be connected'.\
                    format(concat.soft_get('name', node.id))

                concat.add_input_port(last_in_port + 1)
                concat.in_port(last_in_port + 1).connect(blank_values_node.out_port(0))
            else:
                # have to create concat
                concat = Concat(graph, {'axis': 0, 'name': node_name + '/concat_{}'.format(slice_name),
                                        'in_ports_count': 2}).create_node()
                node.in_port(i).get_connection().set_destination(concat.in_port(0))
                concat.in_port(1).connect(blank_values_node.out_port(0))
                concat.out_port(0).get_connection().set_destination(node.in_port(i))

    @staticmethod
    def normalize_slices_attr(node: Node):
        # removes negative starts, ends and magic numbers from 'slice' attr which is used by ConvertGroupedStridedSlice
        slice_rank = len(node['slices'])
        data_shape = node.in_port(0).data.get_shape()

        node_name = node.soft_get('name', node.id)
        if node.is_in_port_connected(3):
            strides = node.in_port(3).data.get_value()
            if strides is None:
                raise Error('StridedSlice operation for node {} supports only constant strides input'.format(node_name))
        else:
            strides = np.ones(slice_rank)

        num_ellipsis_inserts = len(data_shape) - slice_rank + np.count_nonzero(node.new_axis_mask) + 1
        res_slices = []

        in_idx = 0
        for i, s in enumerate(node['slices']):
            if node.new_axis_mask[i]:
                res_slices.append(slice(0, 1, 1))
            elif node.shrink_axis_mask[i]:
                res_slices.append(slice(s, s + 1, strides[i]))  # need strides if shrink index is negative
            elif node.ellipsis_mask[i]:
                for idx in range(num_ellipsis_inserts):
                    res_slices.append(slice(0, data_shape[in_idx], 1))
                    in_idx += 1
            else:
                res_slices.append(s)

            if not (node.new_axis_mask[i] or node.ellipsis_mask[i]):
                res_slices[-1] = slice(*res_slices[-1].indices(data_shape[in_idx]))  # convert negative begins/ends
                in_idx += 1
        node['slices'] = np.array(res_slices)

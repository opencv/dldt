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

import logging as log

import numpy as np

from extensions.ops.normalize_l2 import NormalizeL2Op
from mo.front.common.layout import get_features_dim
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_node
from mo.middle.replacement import MiddleReplacementPattern


class L2NormToNorm(MiddleReplacementPattern):
    """
        Transformation fuses sub-graph performing l2 normalization into the NormalizeL2 operation. The transformation is
        allowed only for cases when normalization is performed over just C dimension or "C + all spatial dimensions".
    """
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict(kind='data')),
                ('l2_normalize', dict(kind='op', op='Mul')),
                ('l2_normalize_data', dict(kind='data')),
                ('maximum', dict(kind='op', op='Maximum')),
                ('maximum_data', dict(kind='data')),
                ('maximum_y_data', dict(kind='data')),
                ('rsqrt_pow', dict(kind='data', value=lambda x: np.all(x == -0.5) if x is not None else False)),
                ('rsqrt', dict(kind='op', op='Pow')),
                ('rsqrt_data', dict(kind='data')),
                ('square_pow', dict(kind='data', value=lambda x: np.all(x == 2) if x is not None else False)),
                ('square', dict(kind='op', op='Pow')),
                ('square_data', dict(kind='data')),
                ('sum', dict(kind='op', op='ReduceSum')),
                ('sum_data', dict(kind='data')),
            ],
            edges=[
                ('input', 'square', {'in': 0}),
                ('square_pow', 'square', {'in': 1}),
                ('square', 'square_data'),
                ('square_data', 'sum'),
                ('sum', 'sum_data'),
                ('maximum_y_data', 'maximum'),
                ('sum_data', 'maximum'),
                ('maximum', 'maximum_data'),
                ('maximum_data', 'rsqrt', {'in': 0}),
                ('rsqrt_pow', 'rsqrt', {'in': 1}),
                ('rsqrt', 'rsqrt_data'),
                ('rsqrt_data', 'l2_normalize'),
                ('input', 'l2_normalize'),
                ('l2_normalize', 'l2_normalize_data'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        y = match['maximum'].in_port(0).data.get_value()
        if y is None:
            y = match['maximum'].in_port(1).data.get_value()

        if y is None or y.shape != ():
            log.debug('The value of the "maximum_y_data" is not defined or is not constant')
            return

        # We need to check axes which performed reduction because IE supports only 2D, 3D, 4D inputs and
        # reduction only along spatial and channel dimensions.
        input_rank = len(match['sum'].in_port(0).data.get_shape())
        if input_rank not in [2, 3, 4]:
            log.debug('IE supports L2 normalization only for 2D, 3D and 4D tensors, skip fusing transformation.')
            return

        axes = match['sum'].in_port(1).data.get_value()
        axes = int64_array(axes)
        if axes.shape == ():
            axes = int64_array([axes])
        axes = int64_array([axis if axis >= 0 else axis + input_rank for axis in axes])
        axes.sort()

        transformation_applicable = False
        # check for case C + all spatial dims. Works for 2D (NC), 3D (NCH) and 4D (NCHW and NHWC)
        if len(axes) + 1 == input_rank and np.array_equal(axes, int64_array(np.arange(start=1, stop=input_rank))):
            transformation_applicable = True

        # check for pure C channel normalization
        if len(axes) == 1 and ((input_rank == 4 and get_features_dim(graph.graph['layout'], input_rank) == axes[0]) or
                               (input_rank != 4 and axes[0] == 1)):
            transformation_applicable = True

        if not transformation_applicable:
            log.debug('IE doesn\'t support l2 normalization with reduction along axes {}'.format(axes))
            return

        # rename l2_normalize node since it will be no longer output after the transformation
        output_name = match['l2_normalize'].soft_get('name', match['l2_normalize'].id)
        normalizel2_name = output_name + '/normalizel2'
        rename_node(match['l2_normalize'], normalizel2_name)

        normalize_node = create_op_node_with_second_input(graph, NormalizeL2Op, axes, {'name': output_name,
                                                                                       'eps_mode': 'max', 'eps': y})
        rename_node(normalize_node, output_name)

        match['square'].in_port(0).get_source().connect(normalize_node.in_port(0))

        match['square'].in_port(0).disconnect()
        if match['l2_normalize'].in_port(0).get_source().node.id == match['rsqrt'].id:
            match['l2_normalize'].in_port(1).disconnect()
        else:
            match['l2_normalize'].in_port(0).disconnect()

        match['l2_normalize'].out_port(0).get_connection().set_source(normalize_node.out_port(0))

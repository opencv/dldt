# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from extensions.ops.dft import DFT, IDFT
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes


class TFFFTToDFT(FrontReplacementSubgraph):
    """
    This transformation converts the operation TFFFT into MO DFT (if the attribute 'is_inverse' is False),
    or into MO IDFT (otherwise).
    """
    enabled = True

    def run_after(self):
        from extensions.front.tf.RollRealImagPackReplacement import RollRealImagPackReplacement
        return [RollRealImagPackReplacement]

    def pattern(self):
        return dict(
            nodes=[('fft', dict(op='TFFFT'))],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        tf_fft = match['fft']
        tf_fft_name = tf_fft.soft_get('name', tf_fft.id)

        dft_node = create_dft_from_tffft(graph, tf_fft, tf_fft.in_port(0).get_source().node)
        tf_fft.out_port(0).get_connection().set_source(dft_node.out_port(0))

        rename_nodes([(tf_fft, tf_fft_name + '/to_be_removed'), (dft_node, tf_fft_name)])

        if not graph.graph['cmd_params'].disable_nhwc_to_nchw or graph.graph['layout'] == 'NHWC':
            dft_node['need_insert_transposes_for_dft'] = True


def create_dft_from_tffft(graph: Graph, tffft: Node, input_node=None) -> Node:
    num_of_dims = tffft.soft_get('num_of_dimensions', 1)
    axes = int64_array(range(-num_of_dims, 0))
    op = IDFT if tffft.soft_get('is_inverse', False) else DFT
    return create_op_with_const_inputs(graph, op, {1: axes}, {'in_ports_count': 2}, input_node)

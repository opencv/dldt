# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from typing import Callable, List, Dict

import numpy as np

from extensions.middle.MarkSubgraphsWithCorrectLayout import MarkSubGraphsWithCorrectLayout
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node


class MarkNodesWithShapeValues(BackReplacementPattern):
    """
    This transformation marks op nodes in ShapeOf subgraphs with 'returns_shape_value' bool attribute and
    data nodes of float32 constants with 'correct_data_type' attribute.
    So that float Consts and Cast float will be kept in FP32 even if argument --data_type=FP16 is specified.

    This is needed to enable conversion to FP16 even if values in ShapeOf subgraphs exceed max(float16)
    or because of FP16 lower precession shape inference is incorrect on some nodes (e.g. if Interpolate in scales mode
    accepts values from ShapeOf subgraph).

    This transformation should be executed after shape inference and after all transformations which insert/modify
    Cast nodes in ShapeOf subgraphs therefore it's placed at the end of the back phase.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].data_type == 'FP16']

    def run_after(self):
        from extensions.back.pass_separator import BackFinish
        return [BackFinish]

    def run_before(self):
        return []

    @staticmethod
    def get_shape_accepting_ops() -> Dict[str, List]:
        return {
            'Interpolate': [1, 2],  # sizes, scales inputs
            'Reshape': [1],  # shape
            'Broadcast': [1],  # target_shape
            'ConvBackPropData ': [2],  # output_shape
            'GroupConvolutionBackpropData ': [2],  # output_shape
            'BatchToSpace': [1, 2, 3],  # block_shape, crops_begin, crops_end
            'SpaceToBatch': [1, 2, 3],  # block_shape, pads_begin, pads_end
            'StridedSlice': [1, 2, 3],  # begin, end, strides
            'VariadicSplit': [2],  # split_lengths
            'Tile': [1],  # repeats input
            'TopK': [1],  # K input
            'Pad': [1, 2],  # pads_begin, pads_end
            'Range': [0, 1, 2],  # start, stop, step inputs
            'OneHot': [1],  # depth input
        }

    @staticmethod
    def get_nodes_with_shape_inputs(graph: Graph) -> List[Node]:
        shape_accepting_ops = MarkNodesWithShapeValues.get_shape_accepting_ops()
        shape_accepting_nodes = []
        for node in graph.get_op_nodes():
            if node.soft_get('type') in shape_accepting_ops:
                shape_accepting_nodes.append(node)
        return shape_accepting_nodes

    @staticmethod
    def get_sources_for_nodes(nodes_with_shape_inputs: List[Node], filter_condition: Callable) -> List[Node]:
        shape_accepting_ops = MarkNodesWithShapeValues.get_shape_accepting_ops()
        sources = []
        for node in nodes_with_shape_inputs:
            for port_idx in shape_accepting_ops[node.soft_get('type')]:
                if not node.is_in_port_connected(port_idx):
                    continue

                source_node = node.in_port(port_idx).get_source().node
                if not filter_condition(source_node):
                    continue

                sources.append(source_node)
        return sources

    @staticmethod
    def mark_nodes(shape_returning_nodes: List[Node]):
        for node in shape_returning_nodes:
            node['returns_shape_value'] = True
            if node.soft_get('type') == 'Const':
                if node.value.dtype == np.float32:
                    node.out_node(0)['correct_data_type'] = True
                elif node.value.dtype in [np.float16, np.float64]:
                    log.error("Const node '{}' returns shape values of '{}' type but it must be integer or float32. "
                              'During Elementwise type inference will attempt to cast to float32'.
                              format(node.soft_get('name', node.id), node.value.dtype), extra={'is_warning': True})

    def find_and_replace_pattern(self, graph: Graph):
        shape_accepting_nodes = self.get_nodes_with_shape_inputs(graph)

        condition = lambda node: node.soft_get('type') != 'ShapeOf'
        sources_of_shape_accepting_nodes = self.get_sources_for_nodes(shape_accepting_nodes, condition)
        shape_returning_nodes = MarkSubGraphsWithCorrectLayout.bfs(sources_of_shape_accepting_nodes, set(),
                                                                   condition, forward=False)
        self.mark_nodes(shape_returning_nodes)

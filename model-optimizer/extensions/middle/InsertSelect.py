# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import numpy as np

from extensions.front.kaldi.memory_offset_adjustment import find_max_frame_time
from extensions.middle.MakeKaldiConstReshapable import create_const_with_batch_from_input
from extensions.ops.elementwise import Equal
from extensions.ops.select import Select
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.middle.pattern_match import find_pattern_matches, inverse_dict
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.assign import Assign
from mo.ops.concat import Concat
from mo.ops.crop import Crop
from mo.ops.read_value import ReadValue
from mo.ops.result import Result
from mo.utils.graph import bfs_search
from mo.utils.error import Error


class AddSelectBeforeMemoryNodePattern(MiddleReplacementPattern):
    """
    Add Select before saving state with Memory to avoid garbage saving
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'kaldi']

    def run_after(self):
        from extensions.middle.ReplaceMemoryOffsetWithSplice import ReplaceMemoryOffsetWithMemoryNodePattern
        from extensions.middle.RemoveDuplicationMemory import MergeNeighborSplicePattern
        return [ReplaceMemoryOffsetWithMemoryNodePattern,
                MergeNeighborSplicePattern]

    def run_before(self):
        from extensions.middle.ReplaceSpliceNodePattern import ReplaceSpliceNodePattern
        return [ReplaceSpliceNodePattern]

    @staticmethod
    def calculate_frame_time(graph: Graph):
        # in Kaldi can be 1 or 2 inputs. Only main input can change delay in network.
        # Usually ivector input have name 'ivector'.
        inputs = graph.get_op_nodes(op='Parameter')
        if len(inputs) == 1:
            inp_name = inputs[0].name
        elif len(inputs) == 2:
            if inputs[0].name == 'ivector':
                inp_name = inputs[1].name
            elif inputs[1].name == 'ivector':
                inp_name = inputs[0].name
            else:
                raise Error("There are 2 inputs for Kaldi model but we can't find out which one is ivector. " +
                            "Use name \'ivector\' for according input")
        else:
            raise Error("There are {} inputs for Kaldi model but we expect only 1 or 2".format(len(inputs)))

        # sort nodes to calculate delays
        nodes = list(bfs_search(graph, [inp_name]))
        nx.set_node_attributes(G=graph, name='frame_time', values=-1)

        for n in nodes:
            node = Node(graph, n)

            # just ignore data nodes
            if node.kind != 'op':
                continue

            # calculate frame_time (delay) that was not calculated
            if node.frame_time < 0:
                # Splice increases frame delay
                if node.kind == 'op' and node.op == "Splice":
                    node.frame_time = node.in_port(0).get_source().node.frame_time + len(node.context) - 1
                # crop often used to get concrete time frame, set frame_time correctly for this case
                elif node.kind == 'op' and node.op == 'Crop':
                    if node.in_port(0).get_connection().get_source().node.op == 'Splice':
                        splice_node = node.in_port(0).get_source().node
                        assert len(node.offset) == 1
                        assert len(node.dim) == 1
                        new_delay = splice_node.context[node.offset[0]//node.dim[0]] - splice_node.context[0]
                        node.frame_time = splice_node.in_port(0).get_source().node.frame_time + new_delay
                    else:
                        node.frame_time = node.in_port(0).get_source().node.frame_time
                # for node with several inputs frame_time = maximum of delays from branches
                elif len(node.in_edges()) > 1:
                    # find out maximum of delay and check that we have at least one branch with another delay
                    node.frame_time, _ = find_max_frame_time(node)
                elif len(node.in_edges()) == 1:
                    for p in node.in_ports():
                        if not node.in_port(p).disconnected():
                            node.frame_time = node.in_port(p).get_source().node.frame_time
                            break
                else:
                    # for all input nodes (without inputs) frame_time is 0
                    node.frame_time = 0

    @staticmethod
    def insert_select(graph: Graph, node: Node):
        context_len = node.frame_time + 1

        if context_len == 1:
            return

        in_node_port = node.in_port(0).get_source()
        in_node_shape = node.in_port(0).data.get_shape()
        node.in_port(0).disconnect()

        # add Select before saving state to avoid saving garbage
        select_node = Select(graph, {'name': 'select_' + node.name}).create_node()
        zero_else = create_const_with_batch_from_input(in_node_port, in_node_shape[1])
        select_node.in_port(1).connect(in_node_port)
        select_node.in_port(2).connect(zero_else.out_port(0))

        # check if we have already appropriate iteration counter
        existing_counters = find_pattern_matches(graph, nodes=[('mem_in', dict(op='ReadValue')),
                                                               ('mem_in_data', dict(shape=int64_array([context_len]))),
                                                               ('crop_mem_in', dict(op='Crop', axis=int64_array([1]),
                                                                                    offset=int64_array([1]),
                                                                                    dim=int64_array(
                                                                                        [context_len - 1]))),
                                                               ('crop_mem_in_data', dict()),
                                                               ('concat', dict(op='Concat', axis=1)),
                                                               ('concat_data', dict()),
                                                               ('const_1', dict(op='Const')),
                                                               ('const_1_data', dict()),
                                                               ('mem_out', dict(op='Assign')),
                                                               ('crop_out', dict(op='Crop', axis=int64_array([1]),
                                                                                 offset=int64_array([0]),
                                                                                 dim=int64_array([1]))),
                                                               ('crop_out_data', dict()),
                                                               ('select', dict(op='Select'))
                                                               ],
                                                 edges=[('mem_in', 'mem_in_data'), ('mem_in_data', 'crop_mem_in'),
                                                        ('crop_mem_in', 'crop_mem_in_data'),
                                                        ('crop_mem_in_data', 'concat', {'in': 0}),
                                                        ('const_1', 'const_1_data'),
                                                        ('const_1_data', 'concat', {'in': 1}),
                                                        ('concat', 'concat_data'), ('concat_data', 'mem_out'),
                                                        ('concat_data', 'crop_out'), ('crop_out', 'crop_out_data'),
                                                        ('crop_out_data', 'select')])
        counter_match = next(existing_counters, None)
        if counter_match is not None:
            ones = Node(graph, inverse_dict(counter_match)['const_1'])
            input_port = Node(graph, inverse_dict(counter_match)['crop_out']).out_port(0)
        else:
            init_value_mem_out = create_const_with_batch_from_input(in_node_port, context_len, precision=np.int32)
            mem_out = ReadValue(graph, {'name': 'iteration_number',
                                        'variable_id': 'iteration_' + node.name}).create_node()
            mem_out.in_port(0).connect(init_value_mem_out.out_port(0))
            cut_first = Crop(graph, {'name': 'cut_first', 'axis': int64_array([1]),
                                     'offset': int64_array([1]), 'dim': int64_array([context_len - 1])}).create_node()
            cut_first.in_port(0).connect(mem_out.out_port(0))
            ones = create_const_with_batch_from_input(in_node_port, 1, 1, np.int32)
            concat = Concat(graph, {'name': 'concat_ones', 'in_ports_count': 2, 'axis': 1}).create_node()
            concat.in_port(0).connect(cut_first.out_port(0))
            concat.in_port(1).connect(ones.out_port(0))
            mem_in = Assign(graph, {'name': 'iteration_number_out',
                                    'variable_id': 'iteration_' + node.name}).create_node()
            mem_in.in_port(0).connect(concat.out_port(0))
            res = Result(graph, {}).create_node()
            mem_in.out_port(0).connect(res.in_port(0))
            cut_last = Crop(graph, {'name': 'cut_last', 'axis': int64_array([1]),
                                    'offset': int64_array([0]), 'dim': int64_array([1])}).create_node()
            cut_last.in_port(0).connect(concat.out_port(0))
            input_port = cut_last.out_port(0)

        # Check if data from memory is 1
        # if it is True, we have correct data and should proceed with saving it to memory
        # else we have not gathered context and have garbage here, shouldn't change initial state of memory
        cast_in = Equal(graph, {'name': input_port.node.name + '/cast_to_bool'}).create_node()
        cast_in.in_port(0).connect(ones.out_port(0))
        cast_in.in_port(1).connect(input_port)
        select_node.in_port(0).connect(cast_in.out_port(0))
        select_node.out_port(0).connect(node.in_port(0))
        select_node.out_port(0).data.set_shape(in_node_shape)

    def find_and_replace_pattern(self, graph: Graph):
        should_continue = False
        for n in graph:
            if Node(graph, n).kind == 'op' and Node(graph, n).op == 'Assign' and \
                    Node(graph, n).name != 'iteration_number_out':
                should_continue = True
                break

        if not should_continue:
            return

        self.calculate_frame_time(graph)

        for node in graph.get_op_nodes(op='Assign'):
            if node.name == 'iteration_number_out':
                continue
            self.insert_select(graph, node)

        for n in graph:
            node = Node(graph, n)
            if 'frame_time' in node:
                del node['frame_time']

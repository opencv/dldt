"""
 Copyright (c) 2019 Intel Corporation

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

from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.lin_op import Add, Mul
from mo.ops.op import Op
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class AddMeanScaleValues(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        return []

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    @staticmethod
    def apply_scale(graph: Graph, input_node: Node, node_mean_scale_values: dict):
        if 'scale' in node_mean_scale_values and node_mean_scale_values['scale'] is not None:
            if all([x == 1 for x in node_mean_scale_values['scale']]):
                return
            out_node = input_node.out_node()
            if not input_node.has_valid('shape'):
                raise Error("Node {} has not valid shape attribute".format(input_node.id))
            input_shape = input_node.shape

            # Create Mul node
            value = 1 / np.array(node_mean_scale_values['scale'])
            graph.remove_edge(input_node.id, out_node.id)

            mul_node = Mul(graph, dict(name="Mul_"))
            mul_data = Op.create_input_data_node(graph, "data_mul_", np.array(value))
            Op.expand_node_shape(mul_data, (len(input_shape) - 2 if graph.graph['layout'] == 'NCHW' else 0))
            mul_input = Op.create_data_node(graph, input_node, {'shape': out_node.shape})

            mul_node.create_node_with_data(inputs=[mul_input, mul_data], data_nodes=out_node)

    @staticmethod
    def apply_mean_value(graph: Graph, input_node: Node, node_mean_scale_values: dict):
        if 'mean' in node_mean_scale_values and node_mean_scale_values['mean'] is not None:
            if all([x == 0 for x in node_mean_scale_values['mean']]):
                return
            out_node = input_node.out_node()
            if not input_node.has_valid('shape'):
                raise Error("Node {} has not valid shape attribute".format(input_node.id))
            input_shape = input_node.shape
            # Create Add node
            graph.remove_edge(input_node.id, out_node.id)

            value = np.array(node_mean_scale_values['mean']) * (-1)

            add_node = Add(graph, dict(name="Add_"))
            add_data = Op.create_input_data_node(graph, "data_add_", np.array(value))
            Op.expand_node_shape(add_data, (len(input_shape) - 2 if graph.graph['layout'] == 'NCHW' else 0))
            add_input = Op.create_data_node(graph, input_node, {'shape': out_node.shape})

            add_node.create_node_with_data(inputs=[add_input, add_data], data_nodes=out_node)

    def find_and_replace_pattern(self, graph: Graph):
        input_nodes = {}
        values = graph.graph['cmd_params'].mean_scale_values
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == 'Placeholder':
                input_nodes.update({node.id: node})

        if not isinstance(values, dict):
            if len(values) != len(input_nodes):
                raise Error('Numbers of inputs and mean/scale values do not match. ' +
                            refer_to_faq_msg(61))

            data = np.copy(values)
            values = {}
            for idx, key in enumerate(input_nodes.keys()):
                values.update(
                    {
                        input_nodes[key]['name']: {
                            'mean': data[idx][0],
                            'scale': data[idx][1]
                        }
                    }
                )

        for node_name in values:
            node_id = graph.get_node_id_by_name(node_name)
            node_mean_scale_values = values[node_name]
            if node_id not in input_nodes:
                # if the user cutted-off input of the network then input node name specified in the --scale_values
                # or --mean_values doesn't correspond to a real input node generated by Model Optimizer. But
                # the information about initial input node name is stored in Placeholder's attribute 'initial_node_name'
                new_node_id = None
                for placeholder in input_nodes.values():
                    if placeholder.has('initial_node_name') and placeholder.initial_node_name == node_name:
                        new_node_id = placeholder.id
                        break
                if new_node_id is None:
                    raise Error('Input with name {} wasn\'t found!'.format(node_name) +
                                refer_to_faq_msg(83))
                node_id = new_node_id

            input_node = Node(graph, node_id)
            AddMeanScaleValues.apply_scale(graph, input_node, node_mean_scale_values)
            AddMeanScaleValues.apply_mean_value(graph, input_node, node_mean_scale_values)

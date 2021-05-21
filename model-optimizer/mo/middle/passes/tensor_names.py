# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Objects from xml.etree.ElementTree are imported to modify XML and aren't used to parse.
from xml.etree.ElementTree import Element, SubElement, tostring  # nosec

from defusedxml import defuse_stdlib
from defusedxml.minidom import parseString

from mo.graph.graph import Node, Graph

# To eliminate a risk of xml.etree.ElementTree objects to be used to parse XML in future development
# defusedxml.defuse_stdlib() is called to patch xml library with safe methods.
defuse_stdlib()


def propagate_op_name_to_tensor(graph: Graph):
    for node in graph.nodes():
        node = Node(graph, node)
        if node.kind == 'op' and node.has_valid('name'):
            for out_node, edge in node.out_nodes_edges().values():
                assert out_node.kind == 'data'
                out_node['ie_tensor_name'] = node.name
                out_node['ie_tensor_port'] = edge['out']
                out_node['ie_tensor_id'] = node.node


def output_tensor_names_map(graph: Graph, xml_file_name: str):
    mapping = Element('mapping')
    for node in graph:
        node = Node(graph, node)
        if node.has_valid('fw_tensor_debug_info') and node.has_valid('ie_tensor_name'):
            for fw_tensor_debug_info in node.fw_tensor_debug_info:
                # Check that debug info has valid fw attrs
                if not all(attr is not None for attr in fw_tensor_debug_info):
                    continue
                map = SubElement(mapping, 'map')
                fw = SubElement(map, 'framework')
                ie = SubElement(map, 'IR')

                fw.set('name', fw_tensor_debug_info[0])
                fw.set('out_port_id', str(fw_tensor_debug_info[1]))

                if node.has_valid('ie_tensor_name'):
                    ie.set('name', node.ie_tensor_name)
                if node.has_valid('ie_tensor_port'):
                    ie.set('out_port_id', str(node.ie_tensor_port))
                if node.has_valid('ie_tensor_id'):
                    ie.set('id', str(node.ie_tensor_id))
    with open(xml_file_name, 'w') as file:
        file.write(parseString(tostring(mapping)).toprettyxml())

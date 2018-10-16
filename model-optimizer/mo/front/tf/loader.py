"""
 Copyright (c) 2018 Intel Corporation

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

from mo.utils.error import Error
import re
import os
from mo.utils.utils import refer_to_faq_msg

try:
    import tensorflow as tf
except ImportError:
    raise Error('Module tensorflow was not found. Please install tensorflow 1.2 or higher. ' +
                refer_to_faq_msg(42))

from google.protobuf import text_format
from mo.graph.graph import create_graph_with_nodes

from mo.utils.summarize_graph import summarize_graph


def freeze_checkpoint(graph_def, checkpoint, output_node_names):
    """
    Replaces all the variables in a graph with constants of the same values.
    :param graph_def: GraphDef object holding the network.
    :param checkpoint: path to checkpoint file with values of variables.
    :param output_node_names: list of output node names
    :return: GraphDef containing a simplified version of the original.
    """
    tf.import_graph_def(graph_def, name="")
    with tf.Session() as sess:
        var_list = {}
        var_to_shape_map = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint).get_variable_to_shape_map()
        for key in var_to_shape_map:
            try:
                tensor = sess.graph.get_operation_by_name(key).outputs[0]
            except KeyError:
                continue
            var_list[key] = tensor
        tf.train.Saver(var_list=var_list).restore(sess, checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, output_node_names)
    return output_graph_def


def read_file_to_graph_def(graph_def: [tf.GraphDef, tf.MetaGraphDef], graph_file_name: str = "", is_binary: bool = True):
    """
    Reads file to protobuf
    :param graph_def: GraphDef orr MetaGraphDef object to store the network
    :param graph_file_name: path to file with graph
    :param is_binary: flag to switch between binary and test protobuf format of graph file
    :return: GraphDef or MetaGaphDef containing the network with cleared device info.
    """
    try:
        if is_binary:
            with open(graph_file_name, "rb") as f:
                graph_def.ParseFromString(f.read())
        else:
            with open(graph_file_name, "r") as f:
                text_format.Merge(f.read(), graph_def)
        nodes_to_clear_device = graph_def.node if isinstance(graph_def, tf.GraphDef) else graph_def.graph_def.node
        for node in nodes_to_clear_device:
            node.device = ""
    except Exception as e:
        raise Error(
            'Cannot read the model file: "{}" is incorrect TensorFlow model file or missing. ' \
            'The file should contain frozen TensorFlow graph in text or binary format. ' \
            'Make sure that --input_model_is_text is provided for a model in text format. ' \
            'By the default, a model is interpreted in binary format. Details: {}. ' +
            refer_to_faq_msg(43),
            graph_file_name,
            str(e)
        ) from e
    return graph_def


def get_output_node_names_list(graph_def, user_defined_output_node_names_list: list):
    return summarize_graph(graph_def)['outputs'] if user_defined_output_node_names_list is None \
            or len(user_defined_output_node_names_list) == 0 else user_defined_output_node_names_list


def deducing_metagraph_path(meta_graph_file: str):
    match = re.search('^(.*)\.(data-\d*-of-\d*|index|meta)$', meta_graph_file)
    if match is not None:
        deduced_meta_graph_file = match.group(1) + '.meta'
        if not os.path.isfile(deduced_meta_graph_file):
            raise Error('\n\nMetaGraph freezing mechanism was enabled. '
                        '\n{} file does not represent MetaGraph. '
                        '\n{} path to MetaGraph was deduced, but it does not exist'
                        '\n\nModel with MetaGraph consists of 3-4 files:'
                        '\n1. model_name.meta'
                        '\n2. model_name.index'
                        '\n3. model_name.data-00000-of-00001 (digit part may vary)'
                        '\n4. checkpoint (optional)'.format(meta_graph_file, deduced_meta_graph_file))
        else:
            meta_graph_file = deduced_meta_graph_file
    else:
        raise Error('\n\nMetaGraph freezing mechanism was enabled. '
                    '\n{} file does not represent MetaGraph. '
                    '\n\nModel with MetaGraph consists of 3-4 files:'
                    '\n1. model_name.meta'
                    '\n2. model_name.index'
                    '\n3. model_name.data-00000-of-00001 (digit part may vary)'
                    '\n4. checkpoint (optional)'
                    '\n\nTo load this model, simply run:'
                    '\npython3 mo_tf.py --input_meta_graph model_name.meta'
                    ''.format(meta_graph_file))
    return meta_graph_file


def load_tf_graph_def(graph_file_name: str = "", is_binary: bool = True, checkpoint: str = "",
                      model_dir: str = "", saved_model_tags: list = [], meta_graph_file: str = "",
                      user_defined_output_node_names_list: list = []):
    # As a provisional solution, use a native TF methods to load a model protobuf
    graph_def = tf.GraphDef()
    try:
        if graph_file_name and not meta_graph_file and not checkpoint:
            # frozen graph
            return read_file_to_graph_def(graph_def, graph_file_name, is_binary)
        if graph_file_name and not meta_graph_file and checkpoint:
            # inference graph and checkpoint
            graph_def = read_file_to_graph_def(graph_def, graph_file_name, is_binary)
            outputs = get_output_node_names_list(graph_def, user_defined_output_node_names_list)
            return freeze_checkpoint(graph_def=graph_def, checkpoint=checkpoint, output_node_names=outputs)
        if not graph_file_name and meta_graph_file:
            meta_graph_file = deducing_metagraph_path(meta_graph_file)
            input_meta_graph_def = read_file_to_graph_def(tf.MetaGraphDef(), meta_graph_file, is_binary)
            with tf.Session() as sess:
                restorer = tf.train.import_meta_graph(input_meta_graph_def)
                restorer.restore(sess, meta_graph_file.strip('.meta'))
                outputs = get_output_node_names_list(input_meta_graph_def.graph_def, user_defined_output_node_names_list)
                return tf.graph_util.convert_variables_to_constants(sess, input_meta_graph_def.graph_def, outputs)
        if model_dir:
            # saved model directory
            tags = saved_model_tags if saved_model_tags is not None else [tf.saved_model.tag_constants.SERVING]
            with tf.Session() as sess:
                meta_graph_def = tf.saved_model.loader.load(sess, tags, model_dir)
                outputs = get_output_node_names_list(meta_graph_def.graph_def, user_defined_output_node_names_list)
                return tf.graph_util.convert_variables_to_constants(sess, meta_graph_def.graph_def, outputs)
    except Exception as e:
        raise Error('Cannot load input model: {}', e) from e
    raise Error("Unknown configuration of input model parameters")


def protobuf_attrs(pb: tf.NodeDef):
    return {'pb': pb}


def protobuf2nx(pb: tf.GraphDef):
    graph = create_graph_with_nodes(pb.node, get_id=lambda pb: pb.name, get_attrs=protobuf_attrs)
    # initial order of nodes in the GraphDef. It is used to specify order in
    # which merged nodes are added to the generated sub-graph GraphDef for the TensorFlow offload feature.
    graph.graph['initial_nodes_order'] = [node.name for node in pb.node]

    # Remove data dependency edges. This is needed for the TF offload case
    for _, attrs in list(graph.nodes(data=True)):
        pb = attrs['pb']
        if '_class' in pb.attr:
            index = 0
            while index < len(pb.attr['_class'].list.s):
                if re.match('^loc:@.*', pb.attr['_class'].list.s[index].decode('utf-8')):
                    del pb.attr['_class'].list.s[index]
                else:
                    index = index + 1

    return graph

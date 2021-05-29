import os

from common.layer_test_class import CommonLayerTest

from common.utils.tf_utils import summarize_graph


def save_to_pb(tf_model, path_to_saved_tf_model):
    import tensorflow as tf
    tf.io.write_graph(tf_model, path_to_saved_tf_model, 'model.pb', False)
    assert os.path.isfile(os.path.join(path_to_saved_tf_model, 'model.pb')), "model.pb haven't been saved " \
                                                                             "here: {}".format(path_to_saved_tf_model)
    return os.path.join(path_to_saved_tf_model, 'model.pb')


class CommonTFLayerTest(CommonLayerTest):
    def produce_model_path(self, framework_model, save_path):
        return save_to_pb(framework_model, save_path)

    def get_framework_results(self, inputs_dict, model_path):
        # Evaluate model via Tensorflow and IE
        # Load the Tensorflow model
        import tensorflow as tf
        from tensorflow.python.platform import gfile

        graph_summary = summarize_graph(model_path=model_path)
        outputs_list = graph_summary["outputs"]

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            with gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.compat.v1.import_graph_def(graph_def, name='')

                input = dict()
                for key in inputs_dict.keys():
                    data = inputs_dict.get(key)
                    if len(data.shape) == 4:        # reshaping for 4D tensors
                        input[key+':0'] = data.transpose(0, 2, 3, 1)
                    elif len(data.shape) == 5:      # reshaping for 5D tensors
                        input[key+':0'] = data.transpose(0, 2, 3, 4, 1)
                    else:
                        input[key+':0'] = data
                tf_res = sess.run([out + ":0" for out in outputs_list], input)

                result = dict()
                for i, output in enumerate(outputs_list):
                    _tf_res = tf_res[i]
                    if len(_tf_res.shape) == 4:      # reshaping for 4D tensors
                        result[output] = _tf_res.transpose(0, 3, 1, 2) # 2, 0, 1
                    elif len(_tf_res.shape) == 5:    # reshaping for 5D tensors
                        result[output] = _tf_res.transpose(0, 4, 1, 2, 3)  # 3, 0, 1, 2
                    else:
                        result[output] = _tf_res
                return result

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.random_uniform import RandomUniform
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_dtype_extractor


class RandomUniformIntExtractor(FrontExtractorOp):
    op = 'RandomUniformInt'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'output_type': tf_dtype_extractor(node.pb.attr["Tout"].type),
            'initial_type': tf_dtype_extractor(node.pb.attr["Tout"].type),
            'seed': node.pb.attr['seed'].i,
            'seed2': node.pb.attr['seed2'].i
        }
        RandomUniform.update_node_stat(node, attrs)
        return cls.enabled

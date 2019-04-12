"""
 Copyright (c) 2018-2019 Intel Corporation

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

import unittest

import numpy as np
import onnx

from extensions.front.onnx.crop_ext import CropFrontExtractor
from mo.utils.unittest.graph import build_graph
from mo.graph.graph import Node


class CropONNXExtractorTest(unittest.TestCase):
    @staticmethod
    def _create_node(attrs: dict):
        pb = onnx.helper.make_node("Crop", ["X"], ["Y"], **attrs)
        graph = build_graph({'node_0': {'pb': pb}}, [])
        return Node(graph, 'node_0')

    @staticmethod
    def _base_attrs():
        # Commonly used attributes in the tests
        # Each test takes these ones and then adds/modifies/deletes particular fields
        return (
            # test input ONNX attributes
            dict(
                border=[5, 10, 15, 20],
            ),
            # reference output Node attributes
            dict(
                op='Crop',
                crop_begin=np.array([10, 5]),
                crop_end=np.array([20, 15]),
                axis=np.array([2, 3])
            )
        )

    @staticmethod
    def _extract(inp):
        node = __class__._create_node(inp)
        CropFrontExtractor.extract(node)
        return node.graph.node[node.id]

    def _match(self, out, ref):
        for key in ref.keys():
            status = out[key] == ref[key]
            if type(status) in [list, np.ndarray]:
                status = np.all(status)
            self.assertTrue(status, 'Mismatch for field {}, observed: {}, expected: {}'.format(key, out[key], ref[key]))

    def test_default(self):
        inp, ref = self._base_attrs()
        out = self._extract(inp)
        self._match(out, ref)

    def test_with_scale(self):
        inp, ref = self._base_attrs()
        inp['scale'] = np.array([34, 50])

        del ref['crop_begin']
        del ref['crop_end']
        ref['dim'] = np.array([34, 50])
        ref['offset'] = np.array([10, 5])

        out = self._extract(inp)
        self._match(out, ref)

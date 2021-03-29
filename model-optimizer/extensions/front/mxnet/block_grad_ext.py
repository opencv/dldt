# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.identity import Identity
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node


class BlockGradExt(FrontExtractorOp):
    op = 'BlockGrad'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Identity.update_node_stat(node, {})
        return cls.enabled

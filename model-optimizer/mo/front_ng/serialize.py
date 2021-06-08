# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from mo.pipeline.common import get_ir_version


def ngraph_emit_ir(nGraphFunction, argv: argparse.Namespace):
    output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()

    from ngraph import function_to_cnn  # pylint: disable=no-name-in-module,import-error
    network = function_to_cnn(nGraphFunction)

    orig_model_name = os.path.normpath(os.path.join(output_dir, argv.model_name))
    network.serialize(orig_model_name + ".xml", orig_model_name + ".bin")
    print('[ SUCCESS ] Converted with nGraph Serializer')
    print('[ SUCCESS ] Generated IR version {} model.'.format(get_ir_version(argv)))
    print('[ SUCCESS ] XML file: {}.xml'.format(orig_model_name))
    print('[ SUCCESS ] BIN file: {}.bin'.format(orig_model_name))
    return 0

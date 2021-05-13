# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.pyopenvino import IECore
from openvino.pyopenvino import IENetwork
from openvino.pyopenvino import ExecutableNetwork
from openvino.pyopenvino import Version
from openvino.pyopenvino import Parameter
from openvino.pyopenvino import InputInfoPtr
from openvino.pyopenvino import InputInfoCPtr
from openvino.pyopenvino import DataPtr
from openvino.pyopenvino import TensorDesc
from openvino.pyopenvino import get_version
from openvino.pyopenvino import StatusCode
from openvino.pyopenvino import InferQueue
from openvino.pyopenvino import InferRequest # TODO: move to ie_api?
from openvino.pyopenvino import Blob
from openvino.pyopenvino import PreProcessInfo
from openvino.pyopenvino import MeanVariant
from openvino.pyopenvino import ResizeAlgorithm
from openvino.pyopenvino import ColorFormat
from openvino.pyopenvino import PreProcessChannel

from openvino.inference_engine.ie_api import BlobWrapper
from openvino.inference_engine.ie_api import infer
from openvino.inference_engine.ie_api import async_infer
from openvino.inference_engine.ie_api import get_result

# Patching for Blob class
Blob = BlobWrapper
# Patching ExecutableNetwork
ExecutableNetwork.infer = infer
# Patching InferRequest
InferRequest.infer = infer
InferRequest.async_infer = async_infer
InferRequest.get_result = get_result
# Patching InferQueue
InferQueue.async_infer = async_infer

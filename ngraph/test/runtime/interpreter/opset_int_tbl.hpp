//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#ifndef  NGRAPH_OP
#warning "NGRAPH_OP not defined"
#define NGRAPH_OP(x, y)
#endif

NGRAPH_OP(BatchNormInference, op::v0)
NGRAPH_OP(Ceiling, op::v0)
NGRAPH_OP(Convert, op::v0)
NGRAPH_OP(CumSum, ngraph::op::v0)
NGRAPH_OP(DetectionOutput, op::v0)
NGRAPH_OP(Elu, op::v0)
NGRAPH_OP(Gelu, op::v0)
NGRAPH_OP(HardSigmoid, op::v0)
NGRAPH_OP(LRN, ngraph::op::v0)
NGRAPH_OP(MVN, ngraph::op::v0)
NGRAPH_OP(ReverseSequence, op::v0)
NGRAPH_OP(RNNCell, op::v0)
NGRAPH_OP(Selu, op::v0)

NGRAPH_OP(AvgPool, op::v1)
NGRAPH_OP(Convolution, ngraph::op::v1)
NGRAPH_OP(ConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(GroupConvolution, ngraph::op::v1)
NGRAPH_OP(GroupConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(LessEqual, op::v1)
NGRAPH_OP(LogicalAnd, op::v1)
NGRAPH_OP(LogicalOr, op::v1)
NGRAPH_OP(LogicalXor, op::v1)
NGRAPH_OP(LogicalNot, op::v1)
NGRAPH_OP(MaxPool, op::v1)
NGRAPH_OP(OneHot, op::v1)
NGRAPH_OP(Pad, op::v1)
NGRAPH_OP(Select, op::v1)

NGRAPH_OP(EmbeddingBagOffsetsSum, ngraph::op::v3)
NGRAPH_OP(EmbeddingBagPackedSum, ngraph::op::v3)
NGRAPH_OP(ExtractImagePatches, op::v3)
NGRAPH_OP(EmbeddingSegmentsSum, ngraph::op::v3)
NGRAPH_OP(GRUCell, ngraph::op::v3)
NGRAPH_OP(NonZero, op::v3)
NGRAPH_OP(ScatterNDUpdate, op::v3)
NGRAPH_OP(ShapeOf, op::v3)

NGRAPH_OP(CTCLoss, op::v4)
NGRAPH_OP(LSTMCell, op::v4)

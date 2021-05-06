// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NGRAPH_OP
#warning "NGRAPH_OP not defined"
#define NGRAPH_OP(x, y)
#endif

NGRAPH_OP(Abs, ngraph::op::v0)
NGRAPH_OP(Acos, ngraph::op::v0)
NGRAPH_OP(Add, ngraph::op::v1)
NGRAPH_OP(Asin, ngraph::op::v0)
NGRAPH_OP(Atan, ngraph::op::v0)
NGRAPH_OP(AvgPool, ngraph::op::v1)
NGRAPH_OP(BatchNormInference, ngraph::op::v5)
NGRAPH_OP(BinaryConvolution, ngraph::op::v1)
NGRAPH_OP(Broadcast, ngraph::op::v3)
NGRAPH_OP(Bucketize, ngraph::op::v3)
NGRAPH_OP(CTCGreedyDecoder, ngraph::op::v0)
NGRAPH_OP(Ceiling, ngraph::op::v0)
NGRAPH_OP(Clamp, ngraph::op::v0)
NGRAPH_OP(Concat, ngraph::op::v0)
NGRAPH_OP(Constant, ngraph::op)
NGRAPH_OP(Convert, ngraph::op::v0)
NGRAPH_OP(ConvertLike, ngraph::op::v1)
NGRAPH_OP(Convolution, ngraph::op::v1)
NGRAPH_OP(ConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(Cos, ngraph::op::v0)
NGRAPH_OP(Cosh, ngraph::op::v0)
NGRAPH_OP(CumSum, ngraph::op::v0)
NGRAPH_OP(DeformableConvolution, ngraph::op::v1)
NGRAPH_OP(DeformablePSROIPooling, ngraph::op::v1)
NGRAPH_OP(DepthToSpace, ngraph::op::v0)
NGRAPH_OP(DetectionOutput, ngraph::op::v0)
NGRAPH_OP(Divide, ngraph::op::v1)
NGRAPH_OP(Elu, ngraph::op::v0)
NGRAPH_OP(Erf, ngraph::op::v0)
NGRAPH_OP(Equal, ngraph::op::v1)
NGRAPH_OP(Exp, ngraph::op::v0)
NGRAPH_OP(ExtractImagePatches, ngraph::op::v3)
NGRAPH_OP(FakeQuantize, ngraph::op::v0)
NGRAPH_OP(Floor, ngraph::op::v0)
NGRAPH_OP(FloorMod, ngraph::op::v1)
NGRAPH_OP(Gather, ngraph::op::v7)
NGRAPH_OP(GatherTree, ngraph::op::v1)
NGRAPH_OP(Greater, ngraph::op::v1)
NGRAPH_OP(GreaterEqual, ngraph::op::v1)
NGRAPH_OP(GroupConvolution, ngraph::op::v1)
NGRAPH_OP(GroupConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(GRN, ngraph::op::v0)
NGRAPH_OP(HardSigmoid, ngraph::op::v0)
NGRAPH_OP(Less, ngraph::op::v1)
NGRAPH_OP(LessEqual, ngraph::op::v1)
NGRAPH_OP(Log, ngraph::op::v0)
NGRAPH_OP(LogicalAnd, ngraph::op::v1)
NGRAPH_OP(LogicalNot, ngraph::op::v1)
NGRAPH_OP(LogicalOr, ngraph::op::v1)
NGRAPH_OP(LogicalXor, ngraph::op::v1)
NGRAPH_OP(LRN, ngraph::op::v0)
NGRAPH_OP(LSTMCell, ngraph::op::v4)
NGRAPH_OP(MatMul, ngraph::op::v0)
NGRAPH_OP(MaxPool, ngraph::op::v1)
NGRAPH_OP(Maximum, ngraph::op::v1)
NGRAPH_OP(Minimum, ngraph::op::v1)
NGRAPH_OP(Mod, ngraph::op::v1)
NGRAPH_OP(Multiply, ngraph::op::v1)
NGRAPH_OP(Negative, ngraph::op::v0)
NGRAPH_OP(NormalizeL2, ngraph::op::v0)
NGRAPH_OP(NotEqual, ngraph::op::v1)
NGRAPH_OP(OneHot, ngraph::op::v1)
NGRAPH_OP(PRelu, ngraph::op::v0)
NGRAPH_OP(PSROIPooling, ngraph::op::v0)
NGRAPH_OP(Pad, ngraph::op::v1)
NGRAPH_OP(Parameter, ngraph::op::v0)
NGRAPH_OP(Power, ngraph::op::v1)
NGRAPH_OP(PriorBox, ngraph::op::v0)
NGRAPH_OP(PriorBoxClustered, ngraph::op::v0)
NGRAPH_OP(Proposal, ngraph::op::v4)
NGRAPH_OP(Range, ngraph::op::v4)
NGRAPH_OP(Relu, ngraph::op::v0)
NGRAPH_OP(ReduceMax, ngraph::op::v1)
NGRAPH_OP(ReduceLogicalAnd, ngraph::op::v1)
NGRAPH_OP(ReduceLogicalOr, ngraph::op::v1)
NGRAPH_OP(ReduceMean, ngraph::op::v1)
NGRAPH_OP(ReduceMin, ngraph::op::v1)
NGRAPH_OP(ReduceProd, ngraph::op::v1)
NGRAPH_OP(ReduceSum, ngraph::op::v1)
NGRAPH_OP(RegionYolo, ngraph::op::v0)
NGRAPH_OP(ReorgYolo, ngraph::op::v0)
NGRAPH_OP(Reshape, ngraph::op::v1)
NGRAPH_OP(Result, ngraph::op::v0)
NGRAPH_OP(Reverse, ngraph::op::v1)
NGRAPH_OP(ReverseSequence, ngraph::op::v0)
NGRAPH_OP(ROIPooling, ngraph::op::v0)
NGRAPH_OP(ScatterNDUpdate, ngraph::op::v3)
NGRAPH_OP(Select, ngraph::op::v1)
NGRAPH_OP(Selu, ngraph::op::v0)
NGRAPH_OP(Sign, ngraph::op::v0)
NGRAPH_OP(Sigmoid, ngraph::op::v0)
NGRAPH_OP(Sin, ngraph::op::v0)
NGRAPH_OP(Sinh, ngraph::op::v0)
NGRAPH_OP(Softmax, ngraph::op::v1)
NGRAPH_OP(Sqrt, ngraph::op::v0)
NGRAPH_OP(SpaceToDepth, ngraph::op::v0)
NGRAPH_OP(Split, ngraph::op::v1)
NGRAPH_OP(SquaredDifference, ngraph::op::v0)
NGRAPH_OP(Squeeze, ngraph::op::v0)
NGRAPH_OP(StridedSlice, ngraph::op::v1)
NGRAPH_OP(Subtract, ngraph::op::v1)
NGRAPH_OP(Tan, ngraph::op::v0)
NGRAPH_OP(Tanh, ngraph::op::v0)
NGRAPH_OP(TensorIterator, ngraph::op::v0)
NGRAPH_OP(Tile, ngraph::op::v0)
NGRAPH_OP(Transpose, ngraph::op::v1)
NGRAPH_OP(Unsqueeze, ngraph::op::v0)
NGRAPH_OP(VariadicSplit, ngraph::op::v1)

// New operations added in opset2
NGRAPH_OP(BatchToSpace, ngraph::op::v1)
NGRAPH_OP(SpaceToBatch, ngraph::op::v1)

// New operations added in opset3
NGRAPH_OP(EmbeddingBagPackedSum, ngraph::op::v3)
NGRAPH_OP(EmbeddingSegmentsSum, ngraph::op::v3)
NGRAPH_OP(EmbeddingBagOffsetsSum, ngraph::op::v3)
NGRAPH_OP(GRUCell, ngraph::op::v3)
NGRAPH_OP(NonZero, ngraph::op::v3)
NGRAPH_OP(RNNCell, ngraph::op::v0)
NGRAPH_OP(ROIAlign, ngraph::op::v3)
NGRAPH_OP(ScatterElementsUpdate, ngraph::op::v3)
NGRAPH_OP(ScatterUpdate, ngraph::op::v3)
NGRAPH_OP(ShuffleChannels, ngraph::op::v0)
NGRAPH_OP(ShapeOf, ngraph::op::v3)
NGRAPH_OP(TopK, ngraph::op::v3)

// New operations added in opset4
NGRAPH_OP(Acosh, ngraph::op::v3)
NGRAPH_OP(Asinh, ngraph::op::v3)
NGRAPH_OP(Atanh, ngraph::op::v3)
NGRAPH_OP(CTCLoss, ngraph::op::v4)
NGRAPH_OP(HSwish, ngraph::op::v4)
NGRAPH_OP(Interpolate, ngraph::op::v4)
NGRAPH_OP(Mish, ngraph::op::v4)
NGRAPH_OP(ReduceL1, ngraph::op::v4)
NGRAPH_OP(ReduceL2, ngraph::op::v4)
NGRAPH_OP(SoftPlus, ngraph::op::v4)
NGRAPH_OP(Swish, ngraph::op::v4)

// New operations added in opset5
NGRAPH_OP(GatherND, ngraph::op::v5)
NGRAPH_OP(GRUSequence, ngraph::op::v5)
NGRAPH_OP(HSigmoid, ngraph::op::v5)
NGRAPH_OP(LogSoftmax, ngraph::op::v5)
NGRAPH_OP(Loop, ngraph::op::v5)
NGRAPH_OP(LSTMSequence, ngraph::op::v5)
NGRAPH_OP(NonMaxSuppression, ngraph::op::v5)
NGRAPH_OP(RNNSequence, ngraph::op::v5)
NGRAPH_OP(Round, ngraph::op::v5)

// New operations added in opset6
NGRAPH_OP(CTCGreedyDecoderSeqLen, ngraph::op::v6)
NGRAPH_OP(ExperimentalDetectronDetectionOutput, ngraph::op::v6)
NGRAPH_OP(ExperimentalDetectronGenerateProposalsSingleImage, ngraph::op::v6)
NGRAPH_OP(ExperimentalDetectronPriorGridGenerator, ngraph::op::v6)
NGRAPH_OP(ExperimentalDetectronROIFeatureExtractor, ngraph::op::v6)
NGRAPH_OP(ExperimentalDetectronTopKROIs, ngraph::op::v6)
NGRAPH_OP(GatherElements, ngraph::op::v6)
NGRAPH_OP(MVN, ngraph::op::v6)
NGRAPH_OP(Assign, ngraph::op::v6)    // new version
NGRAPH_OP(ReadValue, ngraph::op::v6) // new version

// New operations added in opset7
NGRAPH_OP(DFT, ngraph::op::v7)
NGRAPH_OP(Einsum, ngraph::op::v7)
NGRAPH_OP(Gelu, ngraph::op::v7)
NGRAPH_OP(IDFT, ngraph::op::v7)
NGRAPH_OP(Roll, ngraph::op::v7)

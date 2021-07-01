// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_ada_pooling.h"
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <math.h>
#include <mkldnn_extension_utils.h>
#include <utils/general_utils.h>
#include <mkldnn_types.h>
#include <utils/bfloat16.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include "ie_parallel.hpp"
#include <mkldnn_selective_build.h>
#include <ngraph/opsets/opset8.hpp>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::cpu::x64;


bool MKLDNNAdaPoolingNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (one_of(op->get_type_info(), ngraph::op::v8::AdaptiveAvgPool::type_info)) {
            auto adaPool = std::dynamic_pointer_cast<ngraph::opset8::AdaptiveAvgPool>(op);
            if (!adaPool) {
                errorMessage = "Only opset8 AdaptiveAvgPooling operation is supported";
                return false;
            }
        } else if (one_of(op->get_type_info(), ngraph::op::v8::AdaptiveMaxPool::type_info)) {
            auto adaPool = std::dynamic_pointer_cast<ngraph::opset8::AdaptiveMaxPool>(op);
            if (!adaPool) {
                errorMessage = "Only opset8 AdaptiveMaxPooling operation is supported";
                return false;
            }
        } else {
            errorMessage = "Unsupported Adaptive pooling mode";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNAdaPoolingNode::MKLDNNAdaPoolingNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                       MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {

//    if (one_of(op->get_type_info(), ngraph::op::v8::AdaptiveAvgPool::type_info)) {
//        mode = AdaPoolingMode::AVG;
//    } else if (one_of(op->get_type_info(), ngraph::op::v8::AdaptiveMaxPool::type_info)) {
//        mode = AdaPoolingMode::MAX;
//    }
    auto maxPoolOp = ngraph::as_type_ptr<ngraph::op::v8::AdaptiveMaxPool>(op);
    auto avgPoolOp = ngraph::as_type_ptr<ngraph::op::v8::AdaptiveAvgPool>(op);
    std::string errorMessage;
    if (maxPoolOp) {
        mode = AdaPoolingMode::MAX;
        algorithm = Algorithm::AdaptivePoolingMax;
    } else if (avgPoolOp) {
        mode = AdaPoolingMode::AVG;
        algorithm = Algorithm::AdaptivePoolingAvg;
    }
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "AdaPooling layer with name '" + getName() + "' ";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNAdaPoolingNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 2)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();

    inputPrecision = getOriginalInputPrecisionAtPort(0);
    outputPrecision = getOriginalOutputPrecisionAtPort(0);
    outputPrecision = inputPrecision = Precision::FP32;  // TODO: remove

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    auto parentDims = getParentEdgeAt(0)->getDims();
    auto childDims = getChildEdgeAt(0)->getDims();

    const int spatialDimsCount = parentDims.ndims() - 2;
    if (spatialDimsCount != 1 && spatialDimsCount != 2 && spatialDimsCount != 3) {
        IE_THROW() << errorPrefix << "doesn't support 0th input with rank: " << getParentEdgeAt(0)->getDims().ndims();
    }

    if (getParentEdgeAt(1)->getDims().ndims() != 1) {
        IE_THROW() << errorPrefix << "doesn't support 1st input with rank: " << getParentEdgeAt(1)->getDims().ndims();
    }

    if (getChildEdgeAt(0)->getDims().ndims() != getParentEdgeAt(0)->getDims().ndims()) {
        IE_THROW() << errorPrefix << "must keep data rank";
    }

    MKLDNNMemoryDesc in_candidate, out_candidate;
    switch (spatialDimsCount) {
        case 1:
            in_candidate = MKLDNNMemoryDesc{parentDims, inputDataType, memory::format_tag::abc};
            out_candidate = MKLDNNMemoryDesc{childDims, outputDataType, memory::format_tag::abc};
            break;
        case 2:
            in_candidate = MKLDNNMemoryDesc{parentDims, inputDataType, memory::format_tag::nchw};
            out_candidate = MKLDNNMemoryDesc{childDims, outputDataType, memory::format_tag::nchw};
            break;
        case 3:
        default:
            in_candidate = MKLDNNMemoryDesc{parentDims, inputDataType, memory::format_tag::ncdhw};
            out_candidate = MKLDNNMemoryDesc{childDims, outputDataType, memory::format_tag::ncdhw};
    }
    createDescriptor({in_candidate}, {out_candidate});

}

void MKLDNNAdaPoolingNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;


//    if (!mayiuse(avx512_core)) {
//        if (outputPrec == Precision::BF16 || inputPrec0 == Precision::BF16)
//            outputPrec = inputPrec0 = Precision::FP32;
//    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(2);
    config.outConfs.resize(1);

    std::vector<std::pair<memory::format_tag, memory::format_tag>> supportedFormats {
            {memory::format_tag::nchw, memory::format_tag::nchw},  // TODO: as pooling
//            {memory::format_tag::nhwc, memory::format_tag::nhwc},
//            {memory::format_tag::nChw16c, memory::format_tag::nChw16c},
//            {memory::format_tag::nChw8c, memory::format_tag::nChw8c}
    };

    for (auto fmts : supportedFormats) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, fmts.first);
        config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::data_type::s32, memory::format_tag::x);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmts.second);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, fmts.second});
    }
}

std::vector<memory::format_tag> MKLDNNAdaPoolingNode::getAvailableFormatsForDims(const MKLDNNDims &dims) const {
//    if (dims.ndims() == 0)
//        return {memory::format_tag::x};
//    else if (dims.ndims() == 1)
//        return {memory::format_tag::x};
//    else if (dims.ndims() == 2)
//        return {memory::format_tag::nc};
//    else if (dims.ndims() == 3)
//        return {memory::format_tag::tnc, memory::format_tag::ntc};
//    else if (dims.ndims() == 4)
//        return {memory::format_tag::nChw8c, memory::format_tag::nChw16c, memory::format_tag::nhwc, memory::format_tag::nchw};
//    else if (dims.ndims() == 5)
//        return {memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
//    return {memory::format_tag::any};
// TODO: why ? ^

// TODO: only for not plane
    if (dims.ndims() == 3)
        return {memory::format_tag::tnc, memory::format_tag::ntc};
    else if (dims.ndims() == 4)
        return {memory::format_tag::nChw8c, memory::format_tag::nChw16c, memory::format_tag::nhwc, memory::format_tag::nchw};
    else if (dims.ndims() == 5)
        return {memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
    return {memory::format_tag::any};
}

namespace {
    struct AdaPoolingContext {
        MKLDNNAdaPoolingNode &node;
    };
}

template<typename T>
struct MKLDNNAdaPoolingNode::AdaPoolingExecute {
    using srcT = typename std::tuple_element<0, T>::type;
    using dstT = typename std::tuple_element<1, T>::type;

    void operator()(AdaPoolingContext & ctx) {
        ctx.node.executeSpecified<srcT, dstT>();
    }
};
void MKLDNNAdaPoolingNode::execute(mkldnn::stream strm) {
    auto inputPrec = getParentEdgeAt(0)->getMemory().GetDescriptor().data.data_type;
    auto outputPrec = getChildEdgeAt(0)->getMemory().GetDescriptor().data.data_type;
    if (
//            !((inputPrec == mkldnn_bf16 && outputPrec == mkldnn_bf16) ||
          (inputPrec == mkldnn_f32 && outputPrec == mkldnn_f32)
//          )
        )
        IE_THROW() <<"AdaPooling doesn't support demanded precisions";

    AdaPoolingContext ctx = {
            *this
    };

    OV_SWITCH(MKLDNNPlugin, AdaPoolingExecute, ctx, std::tie(inputPrec, outputPrec),
              OV_CASE2(mkldnn_f32, mkldnn_f32, float, float)
//              ,
//              OV_CASE2(mkldnn_bf16, mkldnn_bf16, bfloat16_t, bfloat16_t)
              )
}

template <typename inputType, typename outputType>
void MKLDNNAdaPoolingNode::executeSpecified() {
    auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto &dstMemory = getChildEdgeAt(0)->getMemory();

    auto srcBlockDesc = srcMemory0.GetDescriptor().data.format_desc.blocking;
    auto dstBlockDesc = dstMemory.GetDescriptor().data.format_desc.blocking;

    int blockSize = srcBlockDesc.inner_nblks > 0 ? srcBlockDesc.inner_blks[0] : 1;
    auto isPlainFmt = srcMemory0.GetDesc().isPlainFormat();
    auto isNhwcFmt = srcMemory0.GetDesc().isTailCFormat();

    const auto *srcData = reinterpret_cast<const inputType *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    const auto *srcPooledSpatialShapes = reinterpret_cast<const int *>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
    auto *dst = reinterpret_cast<outputType *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

//    auto nominalRoiCount = static_cast<int>(srcMemory1.GetDims()[0]);
//    int realRois = 0;
    auto inputDimVector = srcMemory0.GetDims();
    const int C = static_cast<int>(inputDimVector[1]);
    const int H = static_cast<int>(inputDimVector[2]);
    const int W = static_cast<int>(inputDimVector[3]);

    const int binCount = pooledH * pooledW;

    const int hInputStride = srcBlockDesc.strides[2];
    const int wInputStride = srcBlockDesc.strides[3];
    const int hOutputStride = dstBlockDesc.strides[2];
    const int wOutputStride = dstBlockDesc.strides[3];
    const int chPadding = srcMemory0.GetDescriptor().data.padded_dims[1];
    const int blockCount = chPadding / blockSize;

//    for (; realRois < nominalRoiCount; realRois++) {
//        auto roiBatchInd = srcRoiIdx[realRois];
//        if (roiBatchInd == -1) {
//            break;
//        }
//    }
//
//    for (int n = 0; n < realRois; ++n) {
//        int roiOff = n * 4;
//        const float* srcRoiPtr = &srcRoi[roiOff];
//        int roiBatchInd = srcRoiIdx[n];
//        if (roiBatchInd < -1) {  // -1 means switched off region
//            IE_THROW() << "Batch index cannot be less, than -1";
//        } else if (roiBatchInd >= inputDimVector[0]) {
//            IE_THROW() << "Demanded batch (id = " << roiBatchInd << ") doesn't exist";
//        }
//
//        float x1 = srcRoiPtr[0] * spatialScale;
//        float y1 = srcRoiPtr[1] * spatialScale;
//        float x2 = srcRoiPtr[2] * spatialScale;
//        float y2 = srcRoiPtr[3] * spatialScale;
//
//        float roiHeight = std::max(y2 - y1, 1.0f);
//        float roiWidth = std::max(x2 - x1, 1.0f);
//        float binHeight = roiHeight / pooledH;
//        float binWidth = roiWidth / pooledW;
//
//        auto samplingRatioX = samplingRatio == 0 ? static_cast<int>(ceil(binWidth)) : samplingRatio;
//        auto samplingRatioY = samplingRatio == 0 ? static_cast<int>(ceil(binHeight)) : samplingRatio;
//
//        uint64_t numSamplesInBin = samplingRatioX * samplingRatioY;
//
//        float sampleDistanceX = binWidth / samplingRatioX;
//        float sampleDistanceY = binHeight / samplingRatioY;
//        // prepare arrays for sampling points and weights
//        std::vector<std::pair<int, int>> pointVector;
//        std::vector<float> weightVector;
//        pointVector.reserve(4 * numSamplesInBin * binCount);
//        weightVector.reserve(4 * numSamplesInBin * binCount);
//
//        for (int yBinInd = 0; yBinInd < pooledH; ++yBinInd) {
//            for (int xBinInd = 0; xBinInd < pooledW; ++xBinInd) {
//                // run into bin
//                for (int ySampleInd = 0; ySampleInd < samplingRatioY; ySampleInd++) {
//                    float sampleY = y1 + yBinInd * binHeight + sampleDistanceY * (0.5f + ySampleInd);
//                    for (int xSampleInd = 0; xSampleInd < samplingRatioX; xSampleInd++) {
//                        float sampleX = x1 + xBinInd * binWidth + sampleDistanceX * (0.5f + xSampleInd);
//                        if (sampleX < -1.0 || sampleX > W ||
//                            sampleY < -1.0 || sampleY > H) {
//                            // For this sample we save 4x point (0,0) with weight 0
//                            pointVector.insert(pointVector.end(), 4, {0, 0});
//                            weightVector.insert(weightVector.end(), 4, float{0});
//                            continue;
//                        }
//                        sampleX = std::max(sampleX, float{0});
//                        sampleY = std::max(sampleY, float{0});
//
//                        auto sampleYLow = static_cast<unsigned int>(sampleY);
//                        auto sampleXLow = static_cast<unsigned int>(sampleX);
//                        unsigned int sampleYHigh;
//                        unsigned int sampleXHigh;
//                        if (sampleYLow >= H - 1) {
//                            sampleYHigh = sampleYLow = H - 1;
//                            sampleY = static_cast<float>(sampleYLow);
//                        } else {
//                            sampleYHigh = sampleYLow + 1;
//                        }
//                        if (sampleXLow >= W - 1) {
//                            sampleXHigh = sampleXLow = W - 1;
//                            sampleX = static_cast<float>(sampleXLow);
//                        } else {
//                            sampleXHigh = sampleXLow + 1;
//                        }
//                        pointVector.push_back({sampleYLow, sampleXLow});
//                        pointVector.push_back({sampleYLow, sampleXHigh});
//                        pointVector.push_back({sampleYHigh, sampleXLow});
//                        pointVector.push_back({sampleYHigh, sampleXHigh});
//
//                        // weight calculation for bilinear interpolation
//                        auto ly = sampleY - sampleYLow;
//                        auto lx = sampleX - sampleXLow;
//                        auto hy = 1.0f - ly;
//                        auto hx = 1.0f - lx;
//
//                        weightVector.push_back(hy * hx);
//                        weightVector.push_back(hy * lx);
//                        weightVector.push_back(ly * hx);
//                        weightVector.push_back(ly * lx);
//                    }
//                }
//            }
//        }
//        auto pool = [&] (int xBinInd_, int yBinInd_, int binOffsetInput_, int binOffsetOutput_, int blockResidual_) {
//            float pooledValue = 0;
//            unsigned int sampleIndex = 4 * (yBinInd_ * pooledW + xBinInd_) * numSamplesInBin;
//            for (unsigned int binSampleInd = 0; binSampleInd < numSamplesInBin; binSampleInd++) {
//                size_t part1Index = binOffsetInput_ + pointVector[sampleIndex].first * hInputStride +
//                                    pointVector[sampleIndex].second * wInputStride + blockResidual_;
//                float part1 = srcData[part1Index];
//                size_t part2Index = binOffsetInput_ + pointVector[sampleIndex + 1].first * hInputStride +
//                                    pointVector[sampleIndex + 1].second * wInputStride + blockResidual_;
//                float part2 = srcData[part2Index];
//                size_t part3Index = binOffsetInput_ + pointVector[sampleIndex + 2].first * hInputStride +
//                                    pointVector[sampleIndex + 2].second * wInputStride + blockResidual_;
//                float part3 = srcData[part3Index];
//                size_t part4Index = binOffsetInput_ + pointVector[sampleIndex + 3].first * hInputStride +
//                                    pointVector[sampleIndex + 3].second * wInputStride + blockResidual_;
//                float part4 = srcData[part4Index];
//
//                switch (getAlgorithm()) {
//                    case Algorithm::ROIAlignMax:
//                    {
//                        float sampleValue = std::max(
//                                {weightVector[sampleIndex] * part1,
//                                 weightVector[sampleIndex + 1] * part2,
//                                 weightVector[sampleIndex + 2] * part3,
//                                 weightVector[sampleIndex + 3] * part4});
//                        pooledValue = sampleValue > pooledValue ? sampleValue : pooledValue;
//                        break;
//                    }
//                    case Algorithm::ROIAlignAvg:
//                    default:
//                    {
//                        float sampleValue =
//                                weightVector[sampleIndex] * part1 +
//                                weightVector[sampleIndex + 1] * part2 +
//                                weightVector[sampleIndex + 2] * part3 +
//                                weightVector[sampleIndex + 3] * part4;
//                        pooledValue += sampleValue / numSamplesInBin;
//                    }
//                }
//                sampleIndex += 4;
//            }
//            size_t dstIndex = binOffsetOutput_ + yBinInd_ * hOutputStride +
//                              xBinInd_ * wOutputStride + blockResidual_;
//            dst[dstIndex] = pooledValue;
//        };
//        if (isNhwcFmt) {
//            parallel_for2d(pooledH, pooledW, [&](int yBinInd, int xBinInd) {
//                for (int c = 0; c < C; c++) {
//                    size_t binOffsetInput = roiBatchInd * C * H * W + c;
//                    size_t binOffsetOutput = n * C * binCount + c;
//                    pool(xBinInd, yBinInd, binOffsetInput, binOffsetOutput, 0);
//                }
//            });
//        } else {  // nchw, nChw16c, nChw8c
//            parallel_for3d(blockCount, pooledH, pooledW, [&](int blkIdx, int yBinInd, int xBinInd) {
//                int cStart = blkIdx * blockSize;
//                int cEnd = (blkIdx == blockCount - 1 ? C : cStart + blockSize);
//                for (int c = cStart; c < cEnd; c++) {
//                    const int blockResidual = (isPlainFmt ? 0 : c % blockSize);
//                    const int blockIdx = (c / blockSize) * blockSize;
//                    size_t binOffsetInput = (roiBatchInd * chPadding + blockIdx) * H * W;
//                    size_t binOffsetOutput = (n * chPadding + blockIdx) * binCount;
//                    pool(xBinInd, yBinInd, binOffsetInput, binOffsetOutput, blockResidual);
//                }
//            });
//        }
//    }
}

bool MKLDNNAdaPoolingNode::created() const {
    return getType() == (mode == AdaPoolingMode::AVG ? AdaptiveAvgPooling : AdaptiveMaxPooling);
}

void MKLDNNAdaPoolingNode::createPrimitive() {}

REG_MKLDNN_PRIM_FOR(MKLDNNAdaPoolingNode, AdaptiveAvgPooling)
REG_MKLDNN_PRIM_FOR(MKLDNNAdaPoolingNode, AdaptiveMaxPooling)

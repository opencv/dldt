// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_depth_to_space_node.h"

#include <legacy/ie_layers.h>
#include <cpu/x64/jit_generator.hpp>
#include <mkldnn_extension_utils.h>
#include "common/tensor_desc_creator.h"

#include <string>
#include <cmath>

#define THROW_ERROR IE_THROW() << "DepthToSpace layer with name '" << getName() << "' "

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;

MKLDNNDepthToSpaceNode::MKLDNNDepthToSpaceNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNDepthToSpaceNode::getSupportedDescriptors() {
    auto* depthToSpaceLayer = dynamic_cast<DepthToSpaceLayer*>(getCnnLayer().get());
    if (depthToSpaceLayer == nullptr)
        THROW_ERROR << "cannot convert from CNN layer";

    if (depthToSpaceLayer->insData[0].lock() == nullptr)
        THROW_ERROR << "has nullable input data";

    SizeVector srcDims = depthToSpaceLayer->insData[0].lock()->getTensorDesc().getDims();
    if (srcDims.size() < 3)
        THROW_ERROR << "has incorrect number of input dimensions";
    if (srcDims.size() > 5)
        THROW_ERROR << "doesn't support dimensions with rank greater than 5";

    if (depthToSpaceLayer->outData[0] == nullptr)
        THROW_ERROR << "has nullable output data";

    SizeVector dstDims = depthToSpaceLayer->outData[0]->getTensorDesc().getDims();
    if (srcDims.size() != dstDims.size())
        THROW_ERROR << "has incorrect number of input/output dimensions";

    std::string modeString = depthToSpaceLayer->GetParamAsString("mode");
    if (modeString == "blocks_first") {
        mode = Mode::BLOCKS_FIRST;
    } else if (modeString == "depth_first") {
        mode = Mode::DEPTH_FIRST;
    } else {
        THROW_ERROR << "doesn't support mode: " << modeString;
    }

    blockSize = depthToSpaceLayer->GetParamAsUInt("block_size", 1);
    if (blockSize == 0)
        THROW_ERROR << "has incorrect block_size parameter is zero!";

    size_t nSpatialDims = srcDims.size() - 2;
    blockStep = static_cast<size_t>(std::pow(blockSize, nSpatialDims));
    if (srcDims[1] % blockStep)
        THROW_ERROR << "has block_size parameter which is incompatible with input tensor channels dimension size";

    if (srcDims[1] / blockStep != dstDims[1])
        THROW_ERROR << "has incompatible input/output channels";

    for (size_t i = 0; i < nSpatialDims; ++i) {
        if (srcDims[i + 2] * blockSize != dstDims[i + 2])
            THROW_ERROR << "has incompatible spatial dims";
    }

    if (getParentEdges().size() != 1)
        THROW_ERROR << "has incorrect number of input edges";
    if (getChildEdges().empty())
        THROW_ERROR << "has incorrect number of output edges";
}

void MKLDNNDepthToSpaceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    auto srcDims = getParentEdgeAt(0)->getDims();
    const size_t nDims = srcDims.ndims();

    impl_desc_type impl_type;
    if (mayiuse(impl::cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;

    std::vector<TensorDescCreatorTypes> supportedTypes;
    if (nDims > 2 && impl_type != impl_desc_type::ref) {
        auto canUseBlocked = [=](const size_t block) {
            return srcDims[1] % block == 0 && (mode == Mode::BLOCKS_FIRST ? (srcDims[1] / block) % blockStep == 0 : block % blockStep == 0);
        };

        supportedTypes.push_back(TensorDescCreatorTypes::nspc);
        if (canUseBlocked(8lu))
            supportedTypes.push_back(TensorDescCreatorTypes::nCsp8c);
        if (canUseBlocked(16lu))
            supportedTypes.push_back(TensorDescCreatorTypes::nCsp16c);
    }
    supportedTypes.push_back(TensorDescCreatorTypes::ncsp);
    auto creators = TensorDescCreator::getCommonCreators();
    auto range = TensorDescCreator::makeFilteredRange(creators, nDims, supportedTypes);

    for (auto itr = range.first; itr != range.second; ++itr) {
        config.inConfs[0].desc = itr->second->createDesc(precision, getParentEdgeAt(0)->getDims().ToSizeVector());
        config.outConfs[0].desc = itr->second->createDesc(precision, getChildEdgeAt(0)->getDims().ToSizeVector());
        supportedPrimitiveDescriptors.emplace_back(config, impl_type, MKLDNNMemoryDesc(config.outConfs.front().desc).getFormat());
    }
}

void MKLDNNDepthToSpaceNode::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated destination memory";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "has unidentified preferable primitive descriptor";

    SizeVector srcDims = getParentEdgeAt(0)->getBlob()->getTensorDesc().getDims();
    SizeVector dstDims = getChildEdgeAt(0)->getBlob()->getTensorDesc().getDims();

    size_t nDims = srcDims.size();
    const size_t nSpatialDims = nDims - 2;
    const bool isBlocked = getParentEdgeAt(0)->getMemory().GetDesc().isBlockedCFormat();
    const size_t reshapedRank = nDims + nSpatialDims + static_cast<int>(isBlocked) + static_cast<int>(isBlocked && mode == Mode::DEPTH_FIRST);
    const size_t lastIdx = reshapedRank - 1;
    size_t firstSpatialOrder = 2;

    PermuteUtils::PermuteParams params;
    params.data_size = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision().size();
    params.order.resize(reshapedRank, 0);
    params.src_block_dims.resize(reshapedRank);
    params.src_block_dims[0] = srcDims[0];

    // reshaping of src dimensions and creating the permutation order for each layout:
    // new shape: mode = blocks_first [N, block_size, block_size, ..., block_size, C / (block_size ^ K), D1, D2, ..., DK]
    //            mode = depth_first  [N, C / (block_size ^ K), block_size, block_size, ..., block_size, D1, D2, ..., DK]
    // order    : mode = blocks_first : [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ..., K + (K + 1), K]
    //            mode = depth_first  : [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1]
    // where `k` is number of spatial dimensions

    auto reshapeAndSetPermOrder = [&](const size_t idx1, const size_t idx2, const size_t shift, const SizeVector& dims) {
        for (size_t i = 0; i < nSpatialDims; i++) {
            params.order[i * 2 + shift] = i + idx1;
            params.order[i * 2 + shift + 1] = i + idx2;

            params.src_block_dims[params.order[i * 2 + shift]] = dims[i + shift];
            params.src_block_dims[params.order[i * 2 + shift + 1]] = blockSize;
        }
    };

    if (isBlocked) {
        SizeVector srcBlockedDims = getParentEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();
        SizeVector dstBlockedDims = getChildEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();

        size_t orderShiftForBlocks, orderShiftForDims;
        if (mode == Mode::BLOCKS_FIRST) {
            orderShiftForBlocks = 1;
            orderShiftForDims = nSpatialDims + 2;

            params.src_block_dims[nSpatialDims + 1] = srcBlockedDims[1] / blockStep;
            params.src_block_dims[lastIdx] = srcBlockedDims.back();

            params.order[1] = nSpatialDims + 1;
            params.order[lastIdx] = lastIdx;
        } else {
            orderShiftForBlocks = nSpatialDims + 4;
            orderShiftForDims = 3;

            size_t newBlockSize = srcBlockedDims.back() / blockStep;
            size_t newBlocksCount = srcBlockedDims[1] * newBlockSize / srcBlockedDims.back();
            params.src_block_dims[1] = newBlocksCount;
            params.src_block_dims[2] = srcBlockedDims[1] / newBlocksCount;
            params.src_block_dims[lastIdx - nSpatialDims] = newBlockSize;

            params.order[1] = 1;
            params.order[2] = 3;
            params.order[lastIdx - 1] = 2;
            params.order[lastIdx] = lastIdx - nSpatialDims;
        }

        reshapeAndSetPermOrder(orderShiftForDims, orderShiftForBlocks, firstSpatialOrder, srcBlockedDims);
    } else if (getParentEdgeAt(0)->getMemory().GetDesc().isTailCFormat()) {
        srcDims.push_back(srcDims[1]);
        dstDims.push_back(dstDims[1]);
        srcDims.erase(srcDims.begin() + 1);
        dstDims.erase(dstDims.begin() + 1);
        firstSpatialOrder = 1;

        size_t shift = static_cast<size_t>(mode == DEPTH_FIRST) + nSpatialDims + 1;
        params.order[lastIdx] = mode == Mode::DEPTH_FIRST ? nSpatialDims + 1 : lastIdx;
        params.src_block_dims[params.order[lastIdx]] = srcDims.back() / blockStep;

        reshapeAndSetPermOrder(firstSpatialOrder, shift, firstSpatialOrder, srcDims);
    } else {
        size_t shift = static_cast<size_t>(mode == DEPTH_FIRST) + 1;
        params.order[1] = mode == DEPTH_FIRST ? 1 : nSpatialDims + 1;
        params.src_block_dims[params.order[1]] = srcDims[1] / blockStep;

        reshapeAndSetPermOrder(nSpatialDims + firstSpatialOrder, shift, firstSpatialOrder, srcDims);
    }

    permuteKernel = std::unique_ptr<PermuteKernel>(new PermuteKernel(params, false));
}

void MKLDNNDepthToSpaceNode::execute(mkldnn::stream strm) {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    int MB = batchToProcess();
    if (permuteKernel->isOptimized()) {
        permuteKernel->optimizedExecute(srcData, dstData, MB);
        return;
    }

    permuteKernel->execute(srcData, dstData, MB);
}

bool MKLDNNDepthToSpaceNode::created() const {
    return getType() == DepthToSpace;
}
REG_MKLDNN_PRIM_FOR(MKLDNNDepthToSpaceNode, DepthToSpace);

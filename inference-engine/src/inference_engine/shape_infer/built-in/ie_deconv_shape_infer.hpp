// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <ie_format_parser.h>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Deconvolution layer
 */
class DeconvShapeProp : public BuiltInShapeInferImpl {
public:
    explicit DeconvShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        DeconvolutionLayer deconvLayer(lp);
        deconvLayer.params = params;
        deconvLayer.type = _type;
        validate(&deconvLayer, inShapes, params, blobs);

        auto dims = inShapes[0];
        size_t inputN = dims[0];
        size_t IH = dims[2];
        size_t IW = dims[3];
        int PR = -1, PB = -1;
        float OHTemp, OWTemp, KH, KW;
        if (deconvLayer._dilation[Y_AXIS])
            KH = (deconvLayer._kernel[Y_AXIS] - 1) * deconvLayer._dilation[Y_AXIS] + 1;
        else
            KH = deconvLayer._kernel[Y_AXIS];
        if (deconvLayer._dilation[X_AXIS])
            KW = (deconvLayer._kernel[X_AXIS] - 1) * deconvLayer._dilation[X_AXIS] + 1;
        else
            KW = deconvLayer._kernel[X_AXIS];
        size_t SH = deconvLayer._stride[Y_AXIS];
        size_t SW = deconvLayer._stride[X_AXIS];
        size_t PH = deconvLayer._padding[Y_AXIS];
        size_t PW = deconvLayer._padding[X_AXIS];
        size_t OC = deconvLayer._out_depth;
        std::string padType = deconvLayer._auto_pad;
        if (padType == "valid") {
            OHTemp = IH * SH + KH - 1;
            OWTemp = IW * SW + KW - 1;
        } else if ((padType == "same_upper") || (padType == "same_lower")) {
            OHTemp = IH * SH;
            OWTemp = IW * SW;
        } else {
            PR = deconvLayer._pads_end[X_AXIS];
            PB = deconvLayer._pads_end[Y_AXIS];
            OHTemp = SH * (IH - 1) + KH - PH - PB;
            OWTemp = SW * (IW - 1) + KW - PW - PR;
        }
        if (OHTemp < 0 || OWTemp < 0)
            THROW_IE_EXCEPTION << "New shapes " << details::dumpVec(dims) << " make output shape negative";
        size_t OH = static_cast<size_t>(OHTemp);
        size_t OW = static_cast<size_t>(OWTemp);
        outShapes.emplace_back(std::initializer_list<size_t>{inputN, OC, OH, OW});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

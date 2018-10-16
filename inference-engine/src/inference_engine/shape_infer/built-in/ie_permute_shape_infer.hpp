// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include <ie_layer_validators.hpp>
#include "impl_register.hpp"
#include <ie_layers.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Permute layer
 */
class PermuteShapeProp : public BuiltInShapeInferImpl {
public:
    explicit PermuteShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CNNLayer permuteLayer(lp);
        permuteLayer.params = params;
        permuteLayer.type = _type;
        validate(&permuteLayer, inShapes, params, blobs);

        std::vector<size_t> order;
        std::vector<int> layerOrder = permuteLayer.GetParamAsInts("order");
        for (auto ord : layerOrder)
            order.push_back(static_cast<size_t>(ord));

        SizeVector outShape;
        for (size_t i = 0; i < inShapes[0].size(); i++) {
            outShape.push_back(inShapes[0][order[i]]);
        }
        outShapes.emplace_back(outShape);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

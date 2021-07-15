// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>

#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

namespace {

class TanHStage final : public PostOpStage {
public:
    using PostOpStage::PostOpStage;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<TanHStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

void FrontEnd::parseTanH(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    model->addNewStage<TanHStage>(node->get_friendly_name(), StageType::Tanh, node, inputs, outputs);
}

}  // namespace vpu

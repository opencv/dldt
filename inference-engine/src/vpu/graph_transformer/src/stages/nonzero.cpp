// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/ngraph/operations/static_shape_nonzero.hpp>
#include <precision_utils.h>
#include <memory>
#include <set>

namespace vpu {

namespace {

class NonZero : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<NonZero>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        auto inputStrides = input(0)->requiredStrides();
        auto outIndicesStrides = output(0)->requiredStrides();
        auto outDimsStrides = output(1)->requiredStrides();

        stridesInfo.setInput(inputEdge(0), inputStrides.add(0, DimStride::Compact));
        stridesInfo.setOutput(outputEdge(0), outIndicesStrides.add(0, DimStride::Compact));
        stridesInfo.setOutput(outputEdge(1), outDimsStrides.add(0, DimStride::Compact));
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
                                 {{DataType::FP16, DataType::U8, DataType::S32}},
                                 {{DataType::S32}, {DataType::S32}});
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        VPU_INTERNAL_CHECK(numInputs() == 1,
                           "Nonzero stage with name %s must have only 1 input, "
                           "actually provided %d", name(), numInputs());
        VPU_INTERNAL_CHECK(numOutputs() == 2,
                           "Nonzero stage with name %s must have only 2 outputs, "
                           "actually provided %d", name(), numOutputs());

        input(0)->serializeBuffer(serializer);
        output(0)->serializeBuffer(serializer);
        output(1)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseNonZero(
        const Model& model,
        const NodePtr& node,
        const DataVector& inputs,
        const DataVector& outputs) const {
    auto nonZero = ngraph::as_type_ptr<ngraph::vpu::op::StaticShapeNonZero>(node);
    VPU_THROW_UNLESS(nonZero != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", nonZero->get_friendly_name(), nonZero->get_type_name());
    VPU_THROW_UNLESS(inputs.size() == 1,
                     "Nonzero layer with name %s must have only 1 input, actually provided %d",
                     nonZero->get_friendly_name(), inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 2,
                     "Nonzero layer with name %s must have only 2 outputs, actually provided %d",
                     nonZero->get_friendly_name(), outputs.size());

    const auto input = inputs[0];
    const auto inputNumDims = input->desc().numDims();
    const auto totalIndicesDimSize = input->desc().totalDimSize();

    const auto outIndicesDesc = outputs[0]->desc();
    const auto outIndicesPerm = outIndicesDesc.dimsOrder().toPermutation();
    const auto minorIndicesDim = outIndicesDesc.dim(outIndicesPerm.at(0));
    const auto majorIndicesDim = outIndicesDesc.dim(outIndicesPerm.at(1));
    VPU_THROW_UNLESS(outIndicesDesc.numDims() == 2,
                     "NonZero layer with name %s must have 2D output Indices tensor, "
                     "actually provided %dD tensor",
                     nonZero->get_friendly_name(), outIndicesDesc.numDims());
    VPU_THROW_UNLESS(minorIndicesDim >= totalIndicesDimSize,
                     "NonZero layer with name %s must have output Indices tensor with minor dim "
                     "size >= total amount of elements of input tensor, actually provided %d >= %d",
                     nonZero->get_friendly_name(), minorIndicesDim, totalIndicesDimSize);
    VPU_THROW_UNLESS(majorIndicesDim == inputNumDims,
                     "NonZero layer with name %s must have output Indices tensor with major dim "
                     "size == number of dimensions of input tensor, actually provided %d == %d",
                     nonZero->get_friendly_name(), majorIndicesDim, inputNumDims);

    const auto outDimsDesc = outputs[1]->desc();
    const auto outDimsPerm = outDimsDesc.dimsOrder().toPermutation();
    const auto minorDimsDim = outDimsDesc.dim(outDimsPerm.at(0));
    VPU_THROW_UNLESS(outDimsDesc.numDims() == 1,
                     "NonZero layer with name %s must have 1D output Dims tensor, "
                     "actually provided %dD tensor",
                     nonZero->get_friendly_name(), outDimsDesc.numDims());
    VPU_THROW_UNLESS(minorDimsDim >= 2,
                     "NonZero layer with name %s must have output Dims tensor with minor dim "
                     "size >= 2, actually provided %d",
                     nonZero->get_friendly_name(), minorDimsDim);

    model->addNewStage<NonZero>(
            nonZero->get_friendly_name(),
            StageType::NonZero,
            nonZero,
            inputs,
            outputs);
}

}  // namespace vpu

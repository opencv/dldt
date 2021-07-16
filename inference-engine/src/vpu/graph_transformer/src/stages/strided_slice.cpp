// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <debug.h>
#include <memory>
#include <vector>

namespace vpu {

namespace {

std::uint32_t maskStrToInt(std::string mask) {
    std::uint32_t result = 0;
    int idx = 0;

    for (const auto& character : mask) {
        switch (character) {
            case ',':
                continue;
            case '1':
                result |= (0x1 << idx++);
                break;
            case '0':
                idx++;
                break;
            default:
                VPU_THROW_FORMAT("Unsupported mask value: only 0 or 1 are supported, but got {} instead", character);
        }
    }

    return result;
}

std::string stridedSliceMaskStr (const std::vector<int64_t>& mask) {
    std::string value;
    for (const auto &val : mask) {
        if (!value.empty())
            value += ",";
        value += std::to_string(val);
    }
    return value;
};

std::string stridedSliceInvertMaskStr (const std::vector<int64_t>& mask) {
    std::string value;
    for (const auto &val : mask) {
        if (!value.empty())
            value += ",";
        value += std::to_string(1 - val);
    }
    return value;
};

class StridedSliceStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<StridedSliceStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& /*batchInfo*/) override {
    }

    void initialCheckImpl() const override {
        VPU_THROW_UNLESS(numInputs() == 3 || numInputs() == 4,
            "Validating layer {} with type {} failed: number of input should be 3 or 4, but {} were provided",
            name(), type(), numInputs());
        VPU_THROW_UNLESS(numOutputs() == 1,
            "Validating layer {} with type {} failed: number of outputs should be 1, but {} were provided",
            name(), type(), numOutputs());

        const auto& input0DataType = input(0)->desc().type();

        std::vector<EnumSet<DataType>> expectedInputs3Types =
            { {input0DataType}, {DataType::S32}, {DataType::S32} };
        std::vector<EnumSet<DataType>> expectedInputs4Types =
            { {input0DataType}, {DataType::S32}, {DataType::S32}, {DataType::S32} };


        assertInputsOutputsTypes(
            this,
            numInputs() == 3 ? expectedInputs3Types : expectedInputs4Types,
            {{input0DataType}});
    }
    // need to rework
    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto stridedSlice = ngraph::as_type_ptr<ngraph::opset4::StridedSlice>(origNode());
        VPU_THROW_UNLESS(stridedSlice != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", origNode()->get_friendly_name(), origNode()->get_type_name());
        stridedSlice->get_begin_mask();
        stridedSlice->get_end_mask();
        stridedSlice->get_new_axis_mask();
        stridedSlice->get_shrink_axis_mask();
        stridedSlice->get_ellipsis_mask();
        const auto& beginMask = stridedSliceInvertMaskStr(stridedSlice->get_begin_mask());
        const auto& endMask = stridedSliceInvertMaskStr(stridedSlice->get_end_mask());
        serializer.append(maskStrToInt(beginMask));
        serializer.append(maskStrToInt(endMask));

        const auto& newAxisMask =    stridedSliceMaskStr(stridedSlice->get_new_axis_mask());
        const auto& shrinkAxisMask = stridedSliceMaskStr(stridedSlice->get_shrink_axis_mask());
        const auto& ellipsisMask =   stridedSliceMaskStr(stridedSlice->get_ellipsis_mask());
        serializer.append(maskStrToInt(newAxisMask));
        serializer.append(maskStrToInt(shrinkAxisMask));
        serializer.append(maskStrToInt(ellipsisMask));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        input(0)->serializeBuffer(serializer);
        input(1)->serializeBuffer(serializer);
        input(2)->serializeBuffer(serializer);
        input(3)->serializeBuffer(serializer);
        output(0)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseStridedSlice(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 3 || inputs.size() == 4,
        "Parsing layer {} with type {} failed: number of input should be 3 or 4, but {} were provided",
        node->get_friendly_name(), node->get_type_name(), inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
        "Parsing layer {} with type {} failed: number of outputs should be 1, but {} were provided",
        node->get_friendly_name(), node->get_type_name(), outputs.size());

    DataVector extendedInputs{inputs.begin(), inputs.end()};
    if (inputs.size() == 3) {
        extendedInputs.push_back(model->addFakeData());
    }

    model->addNewStage<StridedSliceStage>(node->get_friendly_name(), StageType::StridedSlice, node, extendedInputs, outputs);
}

}  // namespace vpu

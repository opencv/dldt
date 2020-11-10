// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/convert_extract_image_patches_to_reorg_yolo_vpu.hpp"

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>

#include <memory>
#include <vector>

namespace vpu {

ConvertExtractImagePatchesToReorgYoloVPU::ConvertExtractImagePatchesToReorgYoloVPU() {
    const auto image = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
    const auto eip = std::make_shared<ngraph::opset5::ExtractImagePatches>(
            image, ngraph::Shape{1, 1}, ngraph::Strides{1, 1}, ngraph::Shape{1, 1}, ngraph::op::PadType::VALID);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto extractImagePatches =  std::dynamic_pointer_cast<ngraph::opset5::ExtractImagePatches>(m.get_match_root());

        /*
         * In this transformation we replace ExtractImagePatches operation to ReorgYolo operation
         * if ExtractImagePatches operation attributes obey the following conditions:
         *
         * EIP.sizes = EIP.strides
         * EIP.rates = {1, 1}
         * Spatial dimensions of input tensor must be divisible by EIP.strides
         *
         */

        if (!extractImagePatches || m_transformation_callback(extractImagePatches)) {
            return false;
        }


        if (extractImagePatches->get_strides() != extractImagePatches->get_sizes()) {
            return false;
        }

        const auto& inputPartialShape = extractImagePatches->get_input_partial_shape(0);
        const auto& sizes = extractImagePatches->get_sizes();
        const auto& strides = extractImagePatches->get_strides();
        const auto& rates = extractImagePatches->get_rates();

        // Check that ExtractImagePatches input have static shape and rank == 4
        if (!inputPartialShape.rank().is_static() || inputPartialShape.rank().get_length() != 4) {
            return false;
        }

        // Check that ExtractImagePatches input spatial dimensions are not dynamic
        if (inputPartialShape[2].is_dynamic() || inputPartialShape[3].is_dynamic()) {
            return false;
        }

        // Check that ExtractImagePatches input spatial dimensions are divisible by EIP.strides
        if (inputPartialShape[2].get_length() % strides[0] != 0 || inputPartialShape[3].get_length() % strides[1] != 0) {
            return false;
        }

        // Check that EIP.sizes = EIP.strides
        if (sizes[0] != strides[0] || sizes[1] != strides[1]) {
            return false;
        }

        // Check that EIP.rates = {1, 1}
        if (rates[0] != 1 || rates[1] != 1) {
            return false;
        }

        const auto reorgYolo = std::make_shared<ngraph::opset5::ReorgYolo>(
                extractImagePatches->input(0).get_source_output(),
                ngraph::Strides{extractImagePatches->get_strides()});

        reorgYolo->set_friendly_name(extractImagePatches->get_friendly_name());
        ngraph::copy_runtime_info(extractImagePatches, reorgYolo);
        ngraph::replace_node(extractImagePatches, reorgYolo);
        return true;
    };

    const auto matcher = std::make_shared<ngraph::pattern::Matcher>(eip, "ConvertExtractImagePatchesToReorgYolo");
    register_matcher(matcher, callback);
}

}  // namespace vpu

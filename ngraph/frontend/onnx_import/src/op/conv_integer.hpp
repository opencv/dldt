// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#pragma once

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                /// \brief Performs ONNX ConvInteger operation.
                ///
                /// \param node   The ONNX node object representing this operation.
                ///
                /// \return The vector containing Ngraph nodes producing output of quantized ONNX
                ///         convolution operation.
                OutputVector conv_integer(const Node& node);

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph

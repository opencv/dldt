// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
                ///
                /// \brief      Reshape the input tensor similar to numpy.reshape.
                ///
                /// \param[in]  node  The ONNX node representing this operation.
                ///
                /// \return     Ngraph node representing this operation.
                ///
                OutputVector reshape(const Node& node);

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph

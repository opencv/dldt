//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cstdint> // std::int64_t
#include <memory>  // std::make_shared
#include <utility> // std::move

#include "ngraph/axis_set.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "onnx_import/core/node.hpp"
#include "onnx_import/utils/common.hpp"
#include "onnx_import/utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reduction
        {
            // An alias for a function that produces reduction operator (typically make_shared<...>)
            using ReductionOpProvider = std::function<std::shared_ptr<ngraph::Node>(
                const Output<ngraph::Node>&, const std::shared_ptr<ngraph::Node>&, bool)>;

            ///
            /// \brief      Create an nGraph version of an ONNX reduction operation.
            ///
            /// \param[in]  node                   The node representing incoming ONNX operation.
            /// \param[in]  ng_input               The input (nGraph) Tensor.
            /// \param[in]  reduction_op_provider  A function producing a reduction operation node
            ///
            /// \return     nGraph node equivalent of the ONNX operation.
            ///
            std::shared_ptr<ngraph::Node>
                make_ng_reduction_op(const Node& node,
                                     const Output<ngraph::Node>& ng_input,
                                     ReductionOpProvider reduction_op_provider);

        } // namespace  reduction
    }     // namespace onnx_import
} // namespace ngraph

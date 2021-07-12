// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <numeric>

#include "ngraph/runtime/reference/utils/nms_common.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace nms_common
            {
                void nms_common_postprocessing(void* prois,
                                               void* pscores,
                                               void* pselected_num,
                                               const ngraph::element::Type& output_type,
                                               const std::vector<float>& selected_outputs,
                                               const std::vector<int64_t>& selected_indices,
                                               const Shape& selected_indices_shape,
                                               const std::vector<int64_t>& valid_outputs)
                {
                    // for static shape
                    int64_t total_num = static_cast<int64_t>(selected_indices_shape[0]);

                    float* ptr = static_cast<float*>(prois);
                    memcpy(ptr, selected_outputs.data(), total_num * sizeof(float) * 6);

                    if (pscores)
                    {
                        if (output_type == ngraph::element::i64)
                        {
                            int64_t* indices_ptr = static_cast<int64_t*>(pscores);
                            memcpy(
                                indices_ptr, selected_indices.data(), total_num * sizeof(int64_t));
                        }
                        else
                        {
                            int32_t* indices_ptr = static_cast<int32_t*>(pscores);
                            for (int64_t i = 0; i < total_num; ++i)
                            {
                                indices_ptr[i] = static_cast<int32_t>(selected_indices[i]);
                            }
                        }
                    }

                    if (pselected_num)
                    {
                        if (output_type == ngraph::element::i64)
                        {
                            int64_t* valid_outputs_ptr = static_cast<int64_t*>(pselected_num);
                            std::copy(
                                valid_outputs.begin(), valid_outputs.end(), valid_outputs_ptr);
                        }
                        else
                        {
                            int32_t* valid_outputs_ptr = static_cast<int32_t*>(pselected_num);
                            for (size_t i = 0; i < valid_outputs.size(); ++i)
                            {
                                valid_outputs_ptr[i] = static_cast<int32_t>(valid_outputs[i]);
                            }
                        }
                    }
                }
            } // namespace nms_common
        }     // namespace reference
    }         // namespace runtime
} // namespace ngraph

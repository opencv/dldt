//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <cassert>
#include <cfenv>
#include <cmath>
#include <functional>
#include <numeric>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/helpers.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/split.hpp"
#include "ngraph/util.hpp"

// can't be removed currently due to arm-plugin dependency
#include "ngraph/runtime/reference/convolution_backprop_data.hpp"
namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace
            {
                constexpr size_t in_batch_axis = 0;
                constexpr size_t in_channel_axis = 1;
                constexpr size_t filter_out_ch_axis = 0;
                constexpr size_t filter_in_ch_axis = 1;
                constexpr size_t out_batch_axis = 0;
                constexpr size_t out_channel_axis = 1;
                constexpr size_t spatial_axis = 2;

                template <typename T>
                struct Tensor
                {
                    T buffer;
                    const Shape shape;
                    const size_t size;

                    Tensor& operator++()
                    {
                        buffer += size;
                        return *this;
                    }
                };

                template <typename Buf>
                Tensor<Buf> tensor(Buf buffer, const Shape& s)
                {
                    static_assert(std::is_pointer<Buf>::value, "expecting ptr");
                    return Tensor<Buf>{buffer, s, shape_size(s)};
                }

                template <typename Tens>
                Tens tensor_slice(Tens t)
                {
                    Shape new_shape(++t.shape.begin(), t.shape.end());
                    return tensor(t.buffer, new_shape);
                }

                struct ConvolutionParams
                {
                    std::vector<int> strides;
                    std::vector<int> dilation;
                    std::vector<int> pads_begin;
                    std::vector<int> pads_end;

                    ConvolutionParams(const Strides& strides_,
                                      const Strides& dilation_,
                                      const CoordinateDiff& pads_begin_,
                                      const CoordinateDiff& pads_end_)
                        : strides{strides_.begin(), strides_.end()}
                        , dilation{dilation_.begin(), dilation_.end()}
                        , pads_begin{pads_begin_.begin(), pads_begin_.end()}
                        , pads_end{pads_end_.begin(), pads_end_.end()} {};
                };

                template <typename Int>
                constexpr inline bool in_range(Int val, std::pair<Int, Int> range) noexcept
                {
                    return val >= range.first && val < range.second;
                }

                template <typename INPUT, typename FILTER, typename OUTPUT, typename ACCU>
                void convolve_3D_channels(const ConvolutionParams& p,
                                          const Tensor<const INPUT*>& batch,
                                          const Tensor<const FILTER*>& filter,
                                          Tensor<OUTPUT*>& out)
                {
                    const int input_size_z = batch.shape[1];
                    const int input_size_y = batch.shape[2];
                    const int input_size_x = batch.shape[3];
                    const int filter_size_z = filter.shape[1];
                    const int filter_size_y = filter.shape[2];
                    const int filter_size_x = filter.shape[3];
                    const int dilated_filter_size_z =
                        filter_size_z + (filter_size_z - 1) * (p.dilation[0] - 1);
                    const int dilated_filter_size_y =
                        filter_size_y + (filter_size_y - 1) * (p.dilation[1] - 1);
                    const int dilated_filter_size_x =
                        filter_size_x + (filter_size_x - 1) * (p.dilation[2] - 1);

                    for (int i_z = -p.pads_begin[0];
                         i_z <= (p.pads_end[0] + input_size_z - dilated_filter_size_z);
                         i_z += p.strides[0])
                    {
                        for (int i_y = -p.pads_begin[1];
                             i_y <= (p.pads_end[1] + input_size_y - dilated_filter_size_y);
                             i_y += p.strides[1])
                        {
                            for (int i_x = -p.pads_begin[2];
                                 i_x <= (p.pads_end[2] + input_size_x - dilated_filter_size_x);
                                 i_x += p.strides[2])
                            {
                                ACCU sum = 0;
                                auto input_channel = tensor_slice(batch);
                                auto filter_channel = tensor_slice(filter);
                                size_t filter_channels_count = filter.shape[0];
                                while (filter_channels_count--)
                                {
                                    for (int f_z = 0; f_z < filter_size_z; ++f_z)
                                    {
                                        for (int f_y = 0; f_y < filter_size_y; ++f_y)
                                        {
                                            for (int f_x = 0; f_x < filter_size_x; ++f_x)
                                            {
                                                int rel_i_z = i_z + (f_z * p.dilation[0]);
                                                int rel_i_y = i_y + (f_y * p.dilation[1]);
                                                int rel_i_x = i_x + (f_x * p.dilation[2]);

                                                bool padding =
                                                    !(in_range(rel_i_x, {0, input_size_x}) &&
                                                      in_range(rel_i_y, {0, input_size_y}) &&
                                                      in_range(rel_i_z, {0, input_size_z}));
                                                if (padding)
                                                    continue;

                                                int f_buf_idx =
                                                    (f_z * filter_size_y * filter_size_x) +
                                                    (f_y * filter_size_x) + f_x;
                                                int i_buf_idx =
                                                    (rel_i_z * input_size_y * input_size_x) +
                                                    (rel_i_y * input_size_x) + rel_i_x;
                                                sum += static_cast<ACCU>(
                                                           input_channel.buffer[i_buf_idx]) *
                                                       static_cast<ACCU>(
                                                           filter_channel.buffer[f_buf_idx]);
                                            }
                                        }
                                    }
                                    ++input_channel;
                                    ++filter_channel;
                                }
                                *out.buffer = sum;
                                ++out.buffer;
                            }
                        }
                    }
                }

                void extend_to_3D(ConvolutionParams& p, Shape& in_shape, Shape& filter_shape)
                {
                    int spatial_rank = in_shape.size() - 2;
                    if (spatial_rank < 3)
                    {
                        int missing_dims = 3 - spatial_rank;
                        p.dilation.insert(
                            std::next(p.dilation.end(), -spatial_rank), missing_dims, 1);
                        p.strides.insert(
                            std::next(p.strides.end(), -spatial_rank), missing_dims, 1);
                        p.pads_begin.insert(
                            std::next(p.pads_begin.end(), -spatial_rank), missing_dims, 0);
                        p.pads_end.insert(
                            std::next(p.pads_end.end(), -spatial_rank), missing_dims, 0);
                        in_shape.insert(std::next(in_shape.end(), -spatial_rank), missing_dims, 1);
                        filter_shape.insert(
                            std::next(filter_shape.end(), -spatial_rank), missing_dims, 1);
                    }
                }
            }

            template <typename INPUT,
                      typename FILTER,
                      typename OUTPUT,
                      typename ACCU = typename widen<OUTPUT>::type>
            void convolution(const INPUT* in,
                             const FILTER* f,
                             OUTPUT* out,
                             const Shape& in_shape,
                             const Shape& filter_shape,
                             const Shape& out_shape,
                             const Strides& strides,
                             const Strides& dilation,
                             const CoordinateDiff& pads_begin,
                             const CoordinateDiff& pads_end,
                             const Strides&)

            {
                // this implementation supports 1D, 2D & 3D convolutions
                NGRAPH_CHECK(in_shape.size() >= 3 && in_shape.size() <= 5,
                             "Unsupported input rank: ",
                             in_shape);

                NGRAPH_CHECK(filter_shape.size() >= 3 && filter_shape.size() <= 5,
                             "Unsupported kernel rank: ",
                             filter_shape);

                auto old_mode = std::fegetround();
                std::fesetround(FE_TONEAREST);

                // here we are converting all param types to int's to avoid arithmetic issues
                // (e.g signed + unsigned) in indexes calculation later
                ConvolutionParams params{strides, dilation, pads_begin, pads_end};

                // here we are extending spatial dimensions to 3D, because we are going to use 3D
                // convolution implementation to convolve also in 1D & 2D case
                Shape input_shape{in_shape};
                Shape filters_shape{filter_shape};
                if (in_shape.size() < 5)
                {
                    extend_to_3D(params, input_shape, filters_shape);
                }

                const auto input = tensor(in, input_shape);
                const auto filters = tensor(f, filters_shape);
                auto output = tensor(out, out_shape);

                const size_t batch_count = input.shape[in_batch_axis];
                const size_t filters_count = filters.shape[filter_out_ch_axis];

                auto batch = tensor_slice(input);
                for (size_t batch_idx = 0; batch_idx < batch_count; ++batch_idx, ++batch)
                {
                    auto filter = tensor_slice(filters);
                    for (size_t f_idx = 0; f_idx < filters_count; ++f_idx, ++filter)
                    {
                        convolve_3D_channels<INPUT, FILTER, OUTPUT, ACCU>(
                            params, batch, filter, output);
                    }
                }

                std::fesetround(old_mode);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph

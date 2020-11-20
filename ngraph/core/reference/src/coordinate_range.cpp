//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include "ngraph/coordinate_range.hpp"

#include <stdexcept>

#include "ngraph/coordinate_index.hpp"

namespace ngraph
{
    namespace coordinates
    {
        SliceRange::SliceRange(const Shape& source_shape,
                               const Coordinate& source_start_corner,
                               const Coordinate& source_end_corner,
                               const Strides& source_strides)
            : m_source_shape{source_shape}
            , m_bounds{source_start_corner, source_end_corner}
            , m_source_strides{source_strides}
            , m_coordinate{source_start_corner}
        {
            const auto axes = m_source_shape.size();

            if (axes != m_bounds.lower.size())
            {
                throw std::domain_error(
                    "Source start corner does not have the same number of axes as the source space "
                    "shape");
            }
            if (axes != m_bounds.upper.size())
            {
                throw std::domain_error(
                    "Source end corner does not have the same number of axes as the source space "
                    "shape");
            }
            if (axes != m_source_strides.size())
            {
                throw std::domain_error(
                    "Source strides do not have the same number of axes as the source space shape");
            }
        }

        size_t SliceRange::index() const { return coordinate_index(m_coordinate, m_source_shape); }
        bool SliceRange::increment()
        {
            for (auto axis = m_coordinate.size(); axis-- > 0;)
            {
                m_coordinate[axis] += m_source_strides[axis];
                if (m_coordinate[axis] < m_bounds.upper[axis])
                {
                    return true;
                }
                m_coordinate[axis] = m_bounds.lower[axis];
            }

            return false;
        }

        ReverseRange::ReverseRange(const Shape& source_shape, const AxisSet& reversed_axes)
            : m_source_shape{source_shape}
            , m_reversed_axes{reversed_axes}
            , m_coordinate(source_shape.size(), 0)
            , m_reversed_coordinate(source_shape.size(), 0)
        {
            const auto max_reversed_axes = [&] {
                return *std::max_element(reversed_axes.begin(), reversed_axes.end());
            };
            if (!m_reversed_axes.empty() && !(max_reversed_axes() < m_source_shape.size()))
            {
                throw std::domain_error("Reversed axes have axes above the source space shape");
            }
        }

        size_t ReverseRange::index() const
        {
            do_reverse();
            return coordinate_index(m_reversed_coordinate, m_source_shape);
        }

        bool ReverseRange::increment()
        {
            for (auto axis = m_coordinate.size(); axis-- > 0;)
            {
                ++m_coordinate[axis];
                if (m_coordinate[axis] < m_source_shape[axis])
                {
                    return true;
                }
                m_coordinate[axis] = 0;
            }
            return false;
        }

        void ReverseRange::do_reverse() const
        {
            m_reversed_coordinate = m_coordinate;

            for (const auto& i : m_reversed_axes)
            {
                m_reversed_coordinate[i] = m_source_shape[i] - m_reversed_coordinate[i] - 1;
            }
        }

    } // namespace coordinates
} // namespace ngraph

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

#pragma once

#include <iterator>
#include <limits>
#include <type_traits>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace details
            {
                template <bool check>
                using Required = typename std::enable_if<check, bool>::type;

                template <typename It>
                struct IsRandomAccessIt
                {
                    static constexpr bool value =
                        std::is_same<typename It::iterator_category,
                                     std::random_access_iterator_tag>::value;
                };

                template <typename... Args>
                using void_t = void;

                template <typename, typename = size_t>
                struct is_complete : std::false_type
                {
                };

                template <typename T>
                struct is_complete<T, decltype(sizeof(T))> : std::true_type
                {
                };

                template <typename It>
                struct from_iterator
                {
                    using stored_value = typename std::remove_pointer<
                        typename std::iterator_traits<It>::pointer>::type;
                };

            } // namespace details

            /// @brief Span should mimic std::span
            template <typename Element>
            class Span
            {
            public:
                static_assert(std::is_object<Element>::value,
                              "Element must be an object type (not a reference type or void)");
                static_assert(details::is_complete<Element>::value,
                              "Element must be a complete type (not a forward declaration)");
                static_assert(!std::is_abstract<Element>::value,
                              "Element cannot be an abstract class type");

                constexpr Span() = default;

                constexpr Span(Element* data, std::size_t size)
                    : m_data{data}
                    , m_size{size}
                {
                }

                using value_type = Element;
                using size_type = std::size_t;

                constexpr Element* begin() const noexcept { return m_data; }
                constexpr Element* end() const noexcept { return m_data + m_size; }
                friend constexpr Element* begin(const Span& s) noexcept { return s.begin(); }
                friend constexpr Element* end(const Span& s) noexcept { return s.end(); }
                constexpr std::size_t size() const noexcept { return m_size; }
                constexpr bool empty() const noexcept { return !m_size; }
                constexpr Element& front() const noexcept { return *m_data; }
                constexpr Element& back() const noexcept { return *(m_data + (m_size - 1)); }
                constexpr Element& operator[](std::size_t idx) const { return *(m_data + idx); }
                Element& at(std::size_t idx) const { return *(m_data + idx); }
                Span subspan(std::size_t offset,
                             std::size_t size = std::numeric_limits<std::size_t>::max())
                {
                    if (offset > m_size)
                    {
                        return {};
                    }
                    return {m_data + offset, std::min(size, m_size - offset)};
                }

            private:
                Element* m_data{nullptr};
                std::size_t m_size{0};
            };

            template <typename Iterator,
                      typename value = typename details::from_iterator<Iterator>::stored_value,
                      details::Required<details::IsRandomAccessIt<Iterator>::value> = true>
            constexpr auto span(Iterator first, Iterator second) -> Span<value>
            {
                return Span<value>{
                    std::addressof(*first),
                    static_cast<typename Span<value>::size_type>(std::distance(first, second))};
            }

            template <typename Container,
                      // check if Container has contiguous range memory
                      typename = details::void_t<decltype(std::declval<Container>().data()),
                                                 decltype(std::declval<Container>().size())>>
            constexpr auto span(const Container& c) -> Span<const typename Container::value_type>
            {
                return {c.data(), c.size()};
            }

            template <typename Container,
                      // check if Container has contiguous range memory
                      typename = details::void_t<decltype(std::declval<Container>().data()),
                                                 decltype(std::declval<Container>().size())>>
            constexpr auto span(Container& c) -> Span<typename Container::value_type>
            {
                return {c.data(), c.size()};
            }

            template <typename Element>
            constexpr auto span(const Element* data, std::size_t size) -> Span<const Element>
            {
                return {data, size};
            }

            template <typename Element>
            constexpr auto span(Element* data, std::size_t size) -> Span<Element>
            {
                return {data, size};
            }

        } // namespace reference
    }     // namespace runtime
} // namespace ngraph

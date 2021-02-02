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

#include <functional>

#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            class Label;
        }

        class Matcher;

        using RPatternValueMap = std::map<std::shared_ptr<Node>, OutputVector>;
        using PatternValueMap = std::map<std::shared_ptr<Node>, Output<Node>>;
        using PatternValueMaps = std::vector<PatternValueMap>;

        using PatternMap = std::map<std::shared_ptr<Node>, std::shared_ptr<Node>>;

        PatternMap as_pattern_map(const PatternValueMap& pattern_value_map);
        PatternValueMap as_pattern_value_map(const PatternMap& pattern_map);

        namespace op
        {
            using NodePredicate = std::function<bool(std::shared_ptr<Node>)>;
            using ValuePredicate = std::function<bool(const Output<Node>& value)>;

            NGRAPH_API
            ValuePredicate as_value_predicate(NodePredicate pred);

            class NGRAPH_API Pattern : public Node
            {
            public:
                /// \brief \p a base class for \sa Skip and \sa Label
                ///
                Pattern(const OutputVector& patterns, ValuePredicate pred)
                    : Node(patterns)
                    , m_predicate(pred)
                {
                    if (!m_predicate)
                    {
                        m_predicate = [](const Output<Node>&) { return true; };
                    }
                }

                Pattern(const OutputVector& patterns)
                    : Pattern(patterns, nullptr)
                {
                }

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& /* new_args */) const override
                {
                    throw ngraph_error("Uncopyable");
                }

                ValuePredicate get_predicate() const;

                // May be we can skip adding Labels inputs by default
                std::shared_ptr<Node> skip_value_map()
                {
                    m_skip_value_map = true;
                    return this->shared_from_this();
                }

            protected:
                ValuePredicate m_predicate;
                bool m_skip_value_map = false;
            };
        }

        template <typename T>
        std::function<bool(std::shared_ptr<Node>)> has_class()
        {
            auto pred = [](std::shared_ptr<Node> node) -> bool { return is_type<T>(node); };

            return pred;
        }

        NGRAPH_API
        op::ValuePredicate consumers_count(size_t n);

        NGRAPH_API
        op::ValuePredicate has_static_dim(size_t pos);

        NGRAPH_API
        op::ValuePredicate has_static_dims(const std::vector<size_t>& dims);

        NGRAPH_API
        op::ValuePredicate has_static_shape();

        NGRAPH_API
        op::ValuePredicate rank_is_equal_to(size_t len);

        NGRAPH_API
        op::ValuePredicate dim_is_equal_to(size_t pos, size_t len);

        NGRAPH_API
        op::ValuePredicate type_matches(const element::Type& type);

        NGRAPH_API
        op::ValuePredicate type_matches_any(const std::vector<element::Type>& types);

        // TODO: implement
        template <typename T>
        op::ValuePredicate value_is_equal_to(const std::vector<T>& ref_value)
        {
            return [=](Output<Node> output) -> bool { return true; };
        }

        template <typename... Args>
        op::ValuePredicate all_of(Args... args)
        {
            std::array<op::ValuePredicate, sizeof...(args)> predicates{args...};
            return [=](Output<Node> output) -> bool {
                for (const auto& pred : predicates)
                {
                    if (!pred(output))
                        return false;
                }
                return true;
            };
        }

        template <typename... Args>
        op::ValuePredicate any_of(Args... args)
        {
            std::array<op::ValuePredicate, sizeof...(args)> predicates{args...};
            return [=](Output<Node> output) -> bool {
                for (const auto& pred : predicates)
                {
                    if (pred(output))
                        return true;
                }
                return false;
            };
        }
    }
}

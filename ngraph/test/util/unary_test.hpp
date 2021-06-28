// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

namespace ngraph
{
    namespace test
    {
        template <typename V>
        struct ArgumentTolerance
        {
            ArgumentTolerance() = default;
            ArgumentTolerance(V v)
                : in_use{true}
                , value{v}
            {
            }
            bool in_use{false};
            V value{};
        };

        using BitTolerance = ArgumentTolerance<int>;
        using FloatTolerance = ArgumentTolerance<float>;

        struct Tolerance
        {
            BitTolerance bit{};
            FloatTolerance fp{};

            Tolerance() = default;
            Tolerance(const float tolerance)
                : fp(tolerance)
            {
            }
            Tolerance(const int tolerance)
                : bit(tolerance)
            {
            }
        };

        template <typename TestEngine, ngraph::element::Type_t et>
        class unary_test;

        template <ngraph::element::Type_t eleType>
        class Data
        {
        public:
            using T = ngraph::fundamental_type_for<eleType>;

            static struct rand_normal_t
            {
            } rand_normal;
            static struct rand_uniform_int_t
            {
            } rand_uniform_int;
            static struct rand_uniform_real_t
            {
            } rand_uniform_real;

            // random (normal-distrubition) generate data
            Data(rand_normal_t, double mean, double stddev, const ngraph::Shape& s)
                : value(ngraph::shape_size(s))
                , shape(s)
            {
                std::mt19937 gen{0};
                std::normal_distribution<> d{mean, stddev};
                for (size_t i = 0; i < value.size(); ++i)
                    value[i] = d(gen);
            }

            // random (uniform-int-distrubition) generate data
            Data(rand_uniform_int_t, int64_t low, int64_t high, const ngraph::Shape& s)
                : value(ngraph::shape_size(s))
                , shape(s)
            {
                std::mt19937 gen{0};
                std::uniform_int_distribution<int64_t> d{low, high};
                for (size_t i = 0; i < value.size(); ++i)
                    value[i] = d(gen);
            }

            // random (uniform-real-distrubition) generate data
            Data(rand_uniform_real_t, double low, double high, const ngraph::Shape& s)
                : value(ngraph::shape_size(s))
                , shape(s)
            {
                std::mt19937 gen{0};
                std::uniform_real_distribution<> d{low, high};
                for (size_t i = 0; i < value.size(); ++i)
                    value[i] = d(gen);
            }

            Data() = default;

            // only vector, no shape specified
            Data(const std::vector<T>& v)
                : value(v)
                , shape{}
                , no_shape(true)
            {
            }

            // tensor with shape
            Data(const std::vector<T>& v, const ngraph::Shape& s)
                : value(ngraph::shape_size(s))
                , shape{s}
            {
                for (size_t i = 0; i < value.size(); ++i)
                    value[i] = v[i % v.size()];
            }

            // caller have to pass two level of braces when aggregate-initialize input argument of
            // type std::vector<T>. by adding an overload of type std::initializer_list<T>, we can
            // save one level of braces
            Data(const std::initializer_list<T>& v)
                : value(v.size())
                , shape{}
                , no_shape(true)
            {
                std::copy(v.begin(), v.end(), value.begin());
            }

            Data(const std::initializer_list<T>& v, const ngraph::Shape& s)
                : value(ngraph::shape_size(s))
                , shape{s}
            {
                auto it = v.begin();
                for (size_t i = 0; i < value.size(); ++i, ++it)
                {
                    if (it == v.end())
                        it = v.begin();
                    value[i] = *it;
                }
            }

        private:
            std::vector<T> value{};
            ngraph::Shape shape{};
            bool no_shape = false;

            template <typename TestEngine, ngraph::element::Type_t et>
            friend class unary_test;
        };

        template <typename TestEngine, ngraph::element::Type_t et>
        class unary_test
        {
        public:
            unary_test(std::shared_ptr<ngraph::Function> function)
                : m_function(function)
            {
            }

            template <ngraph::element::Type_t eleType = et>
            void test(const Data<eleType>& input, const Data<eleType>& expeted, Tolerance tol = {})
            {
                if (m_function->is_dynamic())
                {
                    auto test_case =
                        ngraph::test::TestCase<TestEngine, ngraph::test::TestCaseType::DYNAMIC>(
                            m_function);
                    do_test<eleType>(test_case, {input}, {expeted}, tol);
                }
                else
                {
                    auto test_case =
                        ngraph::test::TestCase<TestEngine, ngraph::test::TestCaseType::STATIC>(
                            m_function);
                    do_test<eleType>(test_case, {input}, {expeted}, tol);
                }
            }

            // this overload supports passing a predictor with overloaded i/o types
            template <ngraph::element::Type_t eleType = et,
                      typename T = ngraph::fundamental_type_for<eleType>>
            void test(const Data<eleType>& input, T (*elewise_predictor)(T), Tolerance tol = {})
            {
                Data<eleType> expeted = input;

                for (size_t i = 0; i < input.value.size(); i++)
                    expeted.value[i] = elewise_predictor(input.value[i]);

                test<eleType>({input}, {expeted}, tol);
            }

            // this overload supports passing a lambda as predictor
            template <ngraph::element::Type_t eleType = et, typename Predictor>
            void test(const Data<eleType>& input, Predictor elewise_predictor, Tolerance tol = {})
            {
                Data<eleType> expeted = input;

                for (size_t i = 0; i < input.value.size(); i++)
                    expeted.value[i] = elewise_predictor(input.value[i]);

                test<eleType>({input}, {expeted}, tol);
            }

        private:
            std::shared_ptr<ngraph::Function> m_function;

            template <ngraph::element::Type_t eleType,
                      typename TC,
                      typename T = ngraph::fundamental_type_for<eleType>>
            void do_test(TC& test_case,
                         const std::vector<Data<eleType>>& input,
                         const std::vector<Data<eleType>>& expeted,
                         Tolerance tol)
            {
                for (size_t i = 0; i < input.size(); i++)
                {
                    const auto& item = input[i];
                    if (item.no_shape)
                        test_case.template add_input<T>(item.value);
                    else
                        test_case.template add_input<T>(item.shape, item.value);
                }

                for (size_t i = 0; i < expeted.size(); i++)
                {
                    const auto& item = expeted[i];

                    if (item.no_shape)
                        test_case.template add_expected_output<T>(item.value);
                    else
                        test_case.template add_expected_output<T>(item.shape, item.value);
                }

                if (tol.bit.in_use)
                    test_case.run(tol.bit.value);
                else if (tol.fp.in_use)
                    test_case.run_with_tolerance_as_fp(tol.fp.value);
                else
                    test_case.run();
            }
        };

        template <typename TestEngine,
                  typename OpType,
                  ngraph::element::Type_t et,
                  typename... Args>
        unary_test<TestEngine, et>
            make_unary_test(const ngraph::PartialShape& pshape = ngraph::PartialShape::dynamic(),
                            Args&&... args)
        {
            auto param = std::make_shared<ngraph::op::Parameter>(et, pshape);
            auto op = std::make_shared<OpType>(param, std::forward<Args>(args)...);
            auto function = std::make_shared<ngraph::Function>(op, ngraph::ParameterVector{param});
            return unary_test<TestEngine, et>(function);
        }
    }
}
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

#include "evaluates_map.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/runtime/reference/cum_sum.hpp"
#include "ngraph/runtime/reference/embedding_segments_sum.hpp"
#include "ngraph/runtime/reference/embedding_bag_offsets_sum.hpp"
#include "ngraph/runtime/reference/embedding_bag_packed_sum.hpp"
#include "ngraph/runtime/reference/mvn.hpp"
#include "ngraph/runtime/reference/shuffle_channels.hpp"
#include "ngraph/runtime/reference/lrn.hpp"

using namespace ngraph;
using namespace std;

namespace {
    template<element::Type_t ET>
    bool evaluate(shared_ptr<Node> op, const HostTensorVector &outputs, const HostTensorVector &inputs) {
        return false;
    }

    template<element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Convolution> &op, const HostTensorVector &outputs,
                  const HostTensorVector &inputs) {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto &out_shape = outputs[0]->get_shape();
        const auto &in_shape = inputs[0]->get_shape();
        const auto &filter_shape = inputs[1]->get_shape();
        Strides in_dilation(std::vector<size_t>(in_shape.size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        runtime::reference::convolution<typename element_type_traits<ET>::value_type>(in_data_ptr, filter_data,
                                                                                      out_data_ptr,
                                                                                      in_shape,
                                                                                      filter_shape,
                                                                                      out_shape,
                                                                                      op->get_strides(),
                                                                                      op->get_dilations(),
                                                                                      op->get_pads_begin(),
                                                                                      op->get_pads_end(),
                                                                                      in_dilation);
        return true;
    }


    template<element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::ConvolutionBackpropData> &op, const HostTensorVector &outputs,
                  const HostTensorVector &inputs) {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto &out_shape = outputs[0]->get_shape();
        const auto &in_shape = inputs[0]->get_shape();
        const auto &filter_shape = inputs[1]->get_shape();
        Strides in_dilation(std::vector<size_t>(in_shape.size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        runtime::reference::convolution_backprop_in<typename element_type_traits<ET>::value_type>(in_data_ptr,
                                                                                                  filter_data,
                                                                                                  out_data_ptr,
                                                                                                  in_shape,
                                                                                                  filter_shape,
                                                                                                  out_shape,
                                                                                                  in_dilation,
                                                                                                  op->get_dilations(),
                                                                                                  op->get_pads_begin(),
                                                                                                  op->get_pads_end(),
                                                                                                  op->get_strides());
        return true;
    }

    template<element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::GroupConvolution> &op, const HostTensorVector &outputs,
                  const HostTensorVector &inputs) {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto &out_shape = outputs[0]->get_shape();
        const auto &in_shape = inputs[0]->get_shape();
        const auto &filter_shape = inputs[1]->get_shape();
        Strides in_dilation(std::vector<size_t>(in_shape.size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        runtime::reference::convolution<typename element_type_traits<ET>::value_type>(in_data_ptr, filter_data,
                                                                                      out_data_ptr,
                                                                                      in_shape,
                                                                                      filter_shape,
                                                                                      out_shape,
                                                                                      op->get_strides(),
                                                                                      op->get_dilations(),
                                                                                      op->get_pads_begin(),
                                                                                      op->get_pads_end(),
                                                                                      in_dilation,
                                                                                      filter_shape.at(0));
        return true;
    }

    template<element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::GroupConvolutionBackpropData> &op, const HostTensorVector &outputs,
                  const HostTensorVector &inputs) {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto &out_shape = outputs[0]->get_shape();
        const auto &in_shape = inputs[0]->get_shape();
        const auto &filter_shape = inputs[1]->get_shape();
        Strides in_dilation(std::vector<size_t>(in_shape.size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        runtime::reference::convolution_backprop_in<typename element_type_traits<ET>::value_type>(in_data_ptr,
                                                                                                  filter_data,
                                                                                                  out_data_ptr,
                                                                                                  in_shape,
                                                                                                  filter_shape,
                                                                                                  out_shape,
                                                                                                  in_dilation,
                                                                                                  op->get_dilations(),
                                                                                                  op->get_pads_begin(),
                                                                                                  op->get_pads_end(),
                                                                                                  op->get_strides(),
                                                                                                  filter_shape.at(0));
        return true;
    }

    template<element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::CumSum> &op, const HostTensorVector &outputs,
                  const HostTensorVector &inputs) {
        using T = typename element_type_traits<ET>::value_type;
        // TODO: For validation purposes only i64 axis_tensor is used. Types coverage have to be extended if needed
        using P = typename element_type_traits<ngraph::element::Type_t::i64>::value_type;
        runtime::reference::cumsum<T, P>(inputs[0]->get_data_ptr<ET>(),
                                         inputs[1]->get_data_ptr<ngraph::element::Type_t::i64>(),
                                         outputs[0]->get_data_ptr<ET>(), inputs[0]->get_shape(),
                                         op->is_exclusive(), op->is_reverse());
        return true;
    }

    template<element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingSegmentsSum> &op, const HostTensorVector &outputs,
                  const HostTensorVector &inputs) {
        using T = typename element_type_traits<ET>::value_type;
#define REF_CALL(elType) \
        runtime::reference::embeddingSegmentsSum<T, typename element_type_traits<elType>::value_type>( \
                                                       inputs[0]->get_data_ptr<ET>(), \
                                                       inputs[1]->get_data_ptr<elType>(), \
                                                       inputs[2]->get_data_ptr<elType>(), \
                                                       inputs.size() > 4 ? inputs[4]->get_data_ptr<elType>() : nullptr, \
                                                       inputs.size() > 5 ? inputs[5]->get_data_ptr<ET>() : nullptr, \
                                                       outputs[0]->get_data_ptr<ET>(), \
                                                       inputs[0]->get_shape(), \
                                                       inputs[1]->get_shape(), \
                                                       outputs[0]->get_shape()); \
        break;

        switch (inputs[1]->get_element_type()) {
            case element::Type_t::i32:
            REF_CALL(element::Type_t::i32);
            case element::Type_t::i64:
            REF_CALL(element::Type_t::i64);
            default:
                return false;
        }
#undef REF_CALL
        return true;
    }

    template<element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingBagOffsetsSum> &op, const HostTensorVector &outputs,
                  const HostTensorVector &inputs) {
        using T = typename element_type_traits<ET>::value_type;
#define REF_CALL(elType) \
        runtime::reference::embeddingBagOffsetsSum<T, typename element_type_traits<elType>::value_type>( \
                                                       inputs[0]->get_data_ptr<ET>(), \
                                                       inputs[1]->get_data_ptr<elType>(), \
                                                       inputs[2]->get_data_ptr<elType>(), \
                                                       inputs.size() > 3 ? inputs[3]->get_data_ptr<elType>() : nullptr, \
                                                       inputs.size() > 4 ? inputs[4]->get_data_ptr<ET>() : nullptr, \
                                                       outputs[0]->get_data_ptr<ET>(), \
                                                       shape_size(inputs[1]->get_shape()), \
                                                       outputs[0]->get_shape()); \
        break;

        switch (inputs[1]->get_element_type()) {
            case element::Type_t::i32:
            REF_CALL(element::Type_t::i32);
            case element::Type_t::i64:
            REF_CALL(element::Type_t::i64);
            default:
                return false;
        }
#undef REF_CALL
        return true;
    }

    template<element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingBagPackedSum> &op, const HostTensorVector &outputs,
                  const HostTensorVector &inputs) {
        using T = typename element_type_traits<ET>::value_type;
#define REF_CALL(elType) \
        runtime::reference::embeddingBagPackedSum<T, typename element_type_traits<elType>::value_type>( \
                                                       inputs[0]->get_data_ptr<ET>(), \
                                                       inputs[1]->get_data_ptr<elType>(), \
                                                       inputs.size() > 2 ? inputs[2]->get_data_ptr<ET>() : nullptr, \
                                                       outputs[0]->get_data_ptr<ET>(), \
                                                       inputs[1]->get_shape(), \
                                                       outputs[0]->get_shape()); \
        break;

        switch (inputs[1]->get_element_type()) {
            case element::Type_t::i32:
            REF_CALL(element::Type_t::i32);
            case element::Type_t::i64:
            REF_CALL(element::Type_t::i64);
            default:
                return false;
        }
#undef REF_CALL
        return true;
    }

    template<element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::MVN> &op, const HostTensorVector &outputs,
                  const HostTensorVector &inputs) {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::mvn<T>(inputs[0]->get_data_ptr<ET>(),
                                   outputs[0]->get_data_ptr<ET>(),
                                   inputs[0]->get_shape(),
                                   op->get_normalize_variance(),
                                   op->get_reduction_axes(),
                                   op->get_eps());
        return true;
    }

    template<element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::ShuffleChannels> &op, const HostTensorVector &outputs,
                  const HostTensorVector &inputs) {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::shuffle_channels(inputs[0]->get_data_ptr<T>(),
                                             outputs[0]->get_data_ptr<T>(),
                                             inputs[0]->get_shape(), op->get_axis(),
                                             op->get_group());
        return true;
    }

    template<element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::LRN> &op, const HostTensorVector &outputs,
                  const HostTensorVector &inputs) {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::lrn<T>(inputs[0]->get_data_ptr<ET>(), op->get_reduction_axes(),
                                   outputs[0]->get_data_ptr<ET>(), inputs[0]->get_shape(),
                                   op->get_alpha(), op->get_beta(), op->get_bias(), op->get_nsize());
        return true;
    }

    template<typename T>
    bool evaluate_node(std::shared_ptr<Node> node, const HostTensorVector &outputs, const HostTensorVector &inputs) {
        switch (node->get_element_type()) {
            case element::Type_t::boolean:
                return evaluate<element::Type_t::boolean>(as_type_ptr<T>(node), outputs, inputs);;
//            case element::Type_t::bf16:
//                break;
            case element::Type_t::f16:
                return evaluate<element::Type_t::f16>(as_type_ptr<T>(node), outputs, inputs);
            case element::Type_t::f32:
                return evaluate<element::Type_t::f32>(as_type_ptr<T>(node), outputs, inputs);
            case element::Type_t::i8:
                return evaluate<element::Type_t::i8>(as_type_ptr<T>(node), outputs, inputs);
            case element::Type_t::i16:
                return evaluate<element::Type_t::i16>(as_type_ptr<T>(node), outputs, inputs);
            case element::Type_t::i32:
                return evaluate<element::Type_t::i32>(as_type_ptr<T>(node), outputs, inputs);
            case element::Type_t::i64:
                return evaluate<element::Type_t::i64>(as_type_ptr<T>(node), outputs, inputs);
            case element::Type_t::u8:
                return evaluate<element::Type_t::u8>(as_type_ptr<T>(node), outputs, inputs);
            case element::Type_t::u16:
                return evaluate<element::Type_t::u16>(as_type_ptr<T>(node), outputs, inputs);
            case element::Type_t::u32:
                return evaluate<element::Type_t::u32>(as_type_ptr<T>(node), outputs, inputs);
            default:
                throw ngraph_error(std::string("Unhandled data type ")
                                   + node->get_element_type().get_type_name() + std::string("i n evaluate_node()"));
        }

    }
}  // namespace

runtime::interpreter::EvaluatorsMap &runtime::interpreter::get_evaluators_map() {
    static runtime::interpreter::EvaluatorsMap evaluatorsMap{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, evaluate_node<NAMESPACE::NAME>},

#include "opset_int_tbl.hpp"

#undef NGRAPH_OP
    };
    return evaluatorsMap;
}
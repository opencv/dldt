// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <memory>
#include <vector>
#include <unordered_set>
#include <regex>

#include <cnn_network_ngraph_impl.hpp>
#include "ngraph_ops/convolution_ie.hpp"
#include "ngraph_ops/deconvolution_ie.hpp"
#include "legacy/ngraph_ops/eltwise.hpp"
#include "legacy/ngraph_ops/fully_connected.hpp"
#include "legacy/ngraph_ops/gather_ie.hpp"
#include "legacy/ngraph_ops/gather_tree_ie.hpp"
#include "legacy/ngraph_ops/gru_cell_ie.hpp"
#include "legacy/ngraph_ops/interp.hpp"
#include "legacy/ngraph_ops/lrn_ie.hpp"
#include "legacy/ngraph_ops/lstm_cell_ie.hpp"
#include "legacy/ngraph_ops/normalize_ie.hpp"
#include "legacy/ngraph_ops/pad_ie.hpp"
#include "legacy/ngraph_ops/onehot_ie.hpp"
#include "legacy/ngraph_ops/power.hpp"
#include "legacy/ngraph_ops/prior_box_clustered_ie.hpp"
#include "legacy/ngraph_ops/prior_box_ie.hpp"
#include "legacy/ngraph_ops/proposal_ie.hpp"
#include "legacy/ngraph_ops/relu_ie.hpp"
#include "legacy/ngraph_ops/scaleshift.hpp"
#include "legacy/ngraph_ops/tile_ie.hpp"
#include "legacy/ngraph_ops/hard_sigmoid_ie.hpp"
#include "legacy/ngraph_ops/nms_ie.hpp"
#include "legacy/ngraph_ops/crop_ie.hpp"
#include "legacy/ngraph_ops/selu_ie.hpp"
#include "legacy/ngraph_ops/rnn_cell_ie.hpp"
#include "legacy/ngraph_ops/topk_ie.hpp"
#include "legacy/ngraph_ops/rnn_sequence_ie.hpp"
#include "legacy/ngraph_ops/lstm_sequence_ie.hpp"
#include "legacy/ngraph_ops/gru_sequence_ie.hpp"
#include "generic_ie.hpp"
#include "exec_graph_info.hpp"

#include "caseless.hpp"
#include <debug.h>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include "transformations/utils/utils.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

#include "legacy/convert_function_to_cnn_network.hpp"
#include "ie_legacy_itt.hpp"
#include "ie_cnn_layer_builder_ngraph.h"

namespace InferenceEngine {
namespace details {

// helper for adding creators with a specific exception
#define REQUIRED_IE_CONVERSION_CREATOR(type_name, ie_type_name)\
    addSpecificCreator({type_name}, [](const std::shared_ptr<::ngraph::Node>& node,\
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {\
        THROW_IE_EXCEPTION << type_name  << " operation has a form that is not supported. " << node->get_friendly_name()\
        << " should be converted to " << ie_type_name << " operation.";\
        return nullptr;\
    });\

/**
 * @brief Creator for CNNLayer from nGraph op
 */
class CNNLayerCreator : public ::ngraph::AttributeVisitor {
public:
    using CreatorFor = std::function<CNNLayerPtr(const std::shared_ptr<::ngraph::Node>& node,
                                                 const std::map<std::string, std::string>& param)>;
    explicit CNNLayerCreator(const std::shared_ptr<::ngraph::Node>& node);

    CNNLayerPtr create();

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<bool> &value) override {
        params[name] = value.get() ? "true" : "false";
    }

    void addSpecificCreator(const std::vector<std::string>& forTypes, const CreatorFor& creator) {
        for (const auto type : forTypes) {
            creators[type] = creator;
        }
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<std::string>& adapter) override {
        std::string data = adapter.get();
        std::transform(data.begin(), data.end(), data.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        params[name] = data;
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override {
        auto shape = adapter.get();
        params[name] = joinVec(shape);
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        auto shape = adapter.get();
        params[name] = joinVec(shape);
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<double>& adapter) override {
        params[name] = std::to_string(adapter.get());
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<int64_t>& adapter) override {
        params[name] = std::to_string(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        std::vector<std::string> data = adapter.get();
        for (auto& str : data) {
            std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
                return std::tolower(c);
            });
        }

        std::stringstream ss;
        std::copy(data.begin(), data.end(), std::ostream_iterator<std::string>(ss, ","));
        params[name] = ss.str();
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        auto data = adapter.get();
        params[name] = joinVec(data);
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<void>& adapter) override;

private:
    std::shared_ptr<::ngraph::Node> node;
    std::map<std::string, std::string> params;
    std::map<std::string, CreatorFor> creators;
};

void InferenceEngine::details::CNNLayerCreator::on_adapter(const std::string& name,
                                                           ::ngraph::ValueAccessor<void>& adapter) {
    if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::element::Type>>(&adapter)) {
        auto type = static_cast<::ngraph::element::Type&>(*a);
        params[name] = details::convertPrecision(type).name();
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::PartialShape>>(&adapter)) {
        std::string dims;
        auto shape = static_cast<::ngraph::PartialShape&>(*a);
        for (int64_t i = 0; i < shape.rank().get_length(); i++) {
            if (!dims.empty()) dims += ",";
            dims += std::to_string(shape[i].get_length());
        }
        params[name] = dims;
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::Shape>>(&adapter)) {
        auto shape = static_cast<::ngraph::Shape&>(*a);
        params[name] = joinVec(shape);
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::Strides>>(&adapter)) {
        auto shape = static_cast<::ngraph::Strides&>(*a);
        params[name] = joinVec(shape);
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<std::vector<size_t>>>(& adapter)) {
        auto data = a->get();
        params[name] = joinVec(data);
    } else {
        THROW_IE_EXCEPTION << "Error converting ngraph to CNN network. "
                              "Attribute adapter can not be found for " << name << " parameter";
    }
}

InferenceEngine::details::CNNLayerCreator::CNNLayerCreator(const std::shared_ptr<::ngraph::Node>& node): node(node) {
    addSpecificCreator({"Parameter"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                         const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Input",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        return res;
    });
    // TODO - Remove "GreaterEq" once ngraph transitions to GreaterEqual
    addSpecificCreator({"Eltwise", "Subtract", "Power", "Maximum", "Divide", "Greater", "GreaterEqual", "FloorMod", "LogicalOr", "LogicalAnd", "LogicalXor",
        "GreaterEq", "Less", "LessEqual", "Equal", "NotEqual", "Multiply", "Add"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                                 const std::map<std::string, std::string>& params) -> CNNLayerPtr {
            LayerParams attrs = {node->get_friendly_name(), "Eltwise",
                details::convertPrecision(node->get_output_element_type(0))};
            auto res = std::make_shared<EltwiseLayer>(attrs);
            res->params = params;
            if (node->description() == "Maximum") {
                res->params["operation"] = "max";
            } else if (node->description() == "Power") {
                res->params["operation"] = "pow";
            } else if (node->description() == "Subtract") {
                res->params["operation"] = "sub";
            } else if (node->description() == "Divide") {
                res->params["operation"] = "div";
            } else if (node->description() == "LessEqual") {
                res->params["operation"] = "less_equal";
            } else if (node->description() == "Less") {
                res->params["operation"] = "less";
            } else if (node->description() == "Equal") {
                res->params["operation"] = "equal";
            } else if (node->description() == "NotEqual") {
                res->params["operation"] = "not_equal";
            } else if (node->description() == "FloorMod") {
                res->params["operation"] = "floor_mod";
            } else if (node->description() == "Multiply") {
                res->params["operation"] = "prod";
            } else if (node->description() == "Add") {
                res->params["operation"] = "sum";
            } else if (node->description() == "Greater") {
                res->params["operation"] = "greater";
            } else if (node->description() == "GreaterEq") {
                res->params["operation"] = "greater_equal";
            } else if (node->description() == "GreaterEqual") {
                res->params["operation"] = "greater_equal";
            } else if (node->description() == "LogicalOr") {
                res->params["operation"] = "logical_or";
            } else if (node->description() == "LogicalAnd") {
                res->params["operation"] = "logical_and";
            } else if (node->description() == "LogicalXor") {
                res->params["operation"] = "logical_xor";
            } else if (node->description() == "Eltwise") {
                auto castedLayer = std::dynamic_pointer_cast<::ngraph::op::Eltwise>(node);
                if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << attrs.type << " layer " << attrs.name;
                std::string type;
                switch (castedLayer->eltwise_type) {
                case ELTWISE_TYPE::Sum:
                    type = "sum";
                    break;
                case ELTWISE_TYPE::Prod:
                    type = "prod";
                    break;
                default:
                    THROW_IE_EXCEPTION << "Not supported eltwise type!";
                }

                res->params["operation"] = type;
            }
            return res;
        });
    addSpecificCreator({"Concat"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ConcatLayer>(attrs);
        res->params = params;
        auto axis = std::stoi(res->params["axis"]);
        res->params["axis"] = Builder::asString(axis < 0 ? axis + node->get_input_shape(0).size() : axis);
        return res;
    });
    addSpecificCreator({"AvgPool", "MaxPool"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                  const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Pooling",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<PoolingLayer>(attrs);
        res->params = params;
        if (res->params.find("auto_pad") != res->params.end() &&
            details::CaselessEq<std::string>()(res->params["auto_pad"], "EXPLICIT"))
            res->params.erase("auto_pad");

        if (res->params.find("exclude_pad") != res->params.end()) {
            res->params["exclude-pad"] = res->params["exclude_pad"];
            res->params.erase("exclude_pad");
        }

        if (node->description() == "MaxPool") {
            res->params["pool-method"] = "max";
        } else if (node->description() == "AvgPool") {
            res->params["pool-method"] = "avg";
        }
        return res;
    });
    addSpecificCreator({"Select"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SelectLayer>(attrs);
        res->params = params;
        return res;
    });
    addSpecificCreator({"BinaryConvolution"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<BinaryConvolutionLayer>(attrs);

        // todo: investigate difference between ngraph parameters for BinConvolution and the implementation above
        // this leads to accuracy issue for Precollected_ONNX_ResNet50_88percentinto1bit e2e test
        // res->params = params;

        auto castedLayer = ::ngraph::as_type_ptr<::ngraph::op::v1::BinaryConvolution>(node);
        IE_ASSERT(castedLayer) << " Operation " << node->description() << " with name "
            << node->get_friendly_name() << " cannot be casted to ngraph::op::v1::BinaryConvolution";

        std::string value;
        for (const auto& val : castedLayer->get_pads_begin()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["pads_begin"] = value;

        value.clear();
        for (const auto& val : castedLayer->get_pads_end()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["pads_end"] = value;

        switch (castedLayer->get_auto_pad()) {
            case ::ngraph::op::PadType::SAME_UPPER:
                res->params["auto_pad"] = "same_upper";
                break;
            case ::ngraph::op::PadType::SAME_LOWER:
                res->params["auto_pad"] = "same_lower";
                break;
            case ::ngraph::op::PadType::VALID:
                res->params["auto_pad"] = "valid";
                break;
            default:
                break;
        }

        value.clear();
        for (const auto& val : castedLayer->get_strides()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["strides"] = value;

        value.clear();
        for (const auto& val : castedLayer->get_dilations()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["dilations"] = value;

        // Restore kernel size and output
        const auto& shape = castedLayer->get_input_shape(1);
        res->params["output"] = Builder::asString(shape[0]);

        value.clear();
        for (size_t i = 2; i < shape.size(); i++) {
            if (!value.empty()) value += ",";
            value += Builder::asString(shape[i]);
        }
        res->params["kernel"] = value;

        switch (castedLayer->get_mode()) {
            case ::ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT:
                res->params["mode"] = "xnor-popcount";
        }

        IE_ASSERT(castedLayer->input(1).get_partial_shape().is_static()) << " Weights for binary convolution "
            << castedLayer->get_friendly_name() << " should have static shapes!";
        auto weights_shape = castedLayer->input(1).get_source_output().get_shape();
        res->params["input"] = Builder::asString(weights_shape[1]);
        res->params["pad_value"] = Builder::asString(castedLayer->get_pad_value());

        Builder::NodeConverter<::ngraph::op::Constant> converter;

        const auto weightsNode = castedLayer->input(1).get_source_output().get_node_shared_ptr();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }
        return res;
    });

    addSpecificCreator({"SpaceToBatch"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SpaceToBatchLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"BatchToSpace"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<BatchToSpaceLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Assign"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Memory",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params["id"] = params.at("variable_id");
        res->params["index"] = "0";
        res->params["size"] = "2";
        return res;
    });

    addSpecificCreator({"ReadValue"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Memory",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params["id"] = params.at("variable_id");
        res->params["index"] = "1";
        res->params["size"] = "2";
        return res;
    });

    addSpecificCreator({"DepthToSpace"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<DepthToSpaceLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"SpaceToDepth"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SpaceToDepthLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"DeconvolutionIE"},
                       [](const std::shared_ptr<::ngraph::Node> &node,
                          const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Deconvolution",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<DeconvolutionLayer>(attrs);

        res->params = params;
        const auto& shape = node->get_input_shape(1);
        res->params["output"] = Builder::asString(shape[1]);
        std::string kernel_value;
        for (size_t i = 2; i < shape.size(); i++) {
            if (!kernel_value.empty()) kernel_value += ",";
            kernel_value += Builder::asString(shape[i]);
        }
        res->params["kernel"] = kernel_value;

        Builder::NodeConverter<ngraph::op::Constant> converter;
        const auto weightsNode = node->input_value(1).get_node_shared_ptr();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];

            if (node->inputs().size() == 3) {
                const auto biasNode = node->input_value(2).get_node_shared_ptr();
                if (converter.canCreate(biasNode)) {
                    const auto& bias = converter.createLayer(biasNode);
                    res->blobs["biases"] = bias->blobs["custom"];
                    res->_biases = bias->blobs["custom"];
                }
            }
        }
        return res;
    });

    addSpecificCreator({"DetectionOutput"},
                       [](const std::shared_ptr<::ngraph::Node> &node,
                          const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "DetectionOutput",
                            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        auto parseBoolStrToIntStr = [](const std::string &param) -> const std::string {
            if (param == "true") {
                return "1";
            }
            else if (param == "false") {
                return "0";
            }
            return param;
        };
        if (res->params["code_type"] == "caffe.priorboxparameter.center_size"){
            res->params["code_type"] = "caffe.PriorBoxParameter.CENTER_SIZE";
        }
        else{
            res->params["code_type"] =  "caffe.PriorBoxParameter.CORNER";
        }
        res->params["variance_encoded_in_target"] = parseBoolStrToIntStr(res->params["variance_encoded_in_target"]);
        res->params["share_location"] = parseBoolStrToIntStr(res->params["share_location"]);
        res->params["clip_after_nms"] = parseBoolStrToIntStr(res->params["clip_after_nms"]);
        res->params["clip_before_nms"] = parseBoolStrToIntStr(res->params["clip_before_nms"]);
        res->params["decrease_label_id"] = parseBoolStrToIntStr(res->params["decrease_label_id"]);
        res->params["normalized"] = parseBoolStrToIntStr(res->params["normalized"]);
        return res;
    });

    addSpecificCreator({"LogicalNot"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Activation",
                              details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params["type"] = "not";
        return res;                                
    });

    addSpecificCreator({"LSTMCellIE"},
                        [](const std::shared_ptr<::ngraph::Node>& node,
                           const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "LSTMCell",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<LSTMCell>(attrs);
        res->params = params;
        Builder::NodeConverter<ngraph::op::Constant> converter;
        const auto weightsNode = node->input_value(3).get_node_shared_ptr();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }

        const auto biasNode = node->input_value(4).get_node_shared_ptr();
        if (converter.canCreate(biasNode)) {
            const auto& bias = converter.createLayer(biasNode);
            res->blobs["biases"] = bias->blobs["custom"];
            res->_biases = bias->blobs["custom"];
        }
        return res;
    });

    addSpecificCreator({"RNNCellIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "RNNCell",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<RNNCell>(attrs);
        res->params = params;

        Builder::NodeConverter<ngraph::op::Constant> converter;
        const auto weightsNode = node->input_value(2).get_node_shared_ptr();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }

        const auto biasNode = node->input_value(3).get_node_shared_ptr();
        if (converter.canCreate(biasNode)) {
            const auto& bias = converter.createLayer(biasNode);
            res->blobs["biases"] = bias->blobs["custom"];
            res->_biases = bias->blobs["custom"];
        }
        return res;
    });

    addSpecificCreator({"GRUCellIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "GRUCell",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<GRUCell>(attrs);
        res->params = params;

        Builder::NodeConverter<ngraph::op::Constant> converter;
        const auto weightsNode = node->input_value(2).get_node_shared_ptr();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }

        const auto biasNode = node->input_value(3).get_node_shared_ptr();
        if (converter.canCreate(biasNode)) {
            const auto& bias = converter.createLayer(biasNode);
            res->blobs["biases"] = bias->blobs["custom"];
            res->_biases = bias->blobs["custom"];
        }
        return res;
    });

    addSpecificCreator({"PRelu"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "PReLU",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<PReLULayer>(attrs);
        res->params = params;

        Builder::NodeConverter<ngraph::op::Constant> converter;
        const auto weightsNode = node->input_value(1).get_node_shared_ptr();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }
        return res;
    });

    addSpecificCreator({"TileIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Tile",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<TileLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"ProposalIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Proposal",
            details::convertPrecision(node->get_output_element_type(0))};
        auto parseBoolStrToIntStr = [](const std::string &param) -> const std::string {
            if (param == "true") {
                return "1";
            }
            else if (param == "false") {
                return "0";
            }
            return param;
        };
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params = params;
        res->params["clip_before_nms"] = parseBoolStrToIntStr(res->params["clip_before_nms"]);
        res->params["clip_after_nms"] = parseBoolStrToIntStr(res->params["clip_after_nms"]);
        res->params["normalize"] = parseBoolStrToIntStr(res->params["normalize"]);
        return res;
    });

    addSpecificCreator({"Relu"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "ReLU",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ReLULayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Reshape"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Reshape",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ReshapeLayer>(attrs);
        return res;
    });

    addSpecificCreator({"ReverseSequence"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "ReverseSequence",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ReverseSequenceLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"SeluIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Selu",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Softmax"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "SoftMax",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SoftMaxLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Split"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Split",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SplitLayer>(attrs);

        auto axis_node = node->input_value(1).get_node_shared_ptr();
        const auto axis_node_const = std::dynamic_pointer_cast<ngraph::op::Constant>(axis_node);
        if (!axis_node_const) {
            THROW_IE_EXCEPTION << "Split " << node->get_friendly_name() << " has no axes as Constant";
        }
        auto axis = axis_node_const->cast_vector<int64_t>()[0];
        if (axis < 0) {
            axis += node->get_input_shape(0).size();
        }
        res->params["axis"] = Builder::asString(axis);

        return res;
    });

    addSpecificCreator({"Tanh"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "TanH",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"ScatterElementsUpdate"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ScatterElementsUpdateLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"ScatterUpdate"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ScatterUpdateLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"StaticShapeTopK"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "TopK",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<TopKLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"StridedSlice"}, [](const std::shared_ptr<::ngraph::Node> &node,
        const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "StridedSlice",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::StridedSliceLayer>(attrs);
        auto stridedSliceInvertMaskStr = [](const std::string& str) -> std::string {
            std::string value;
            auto found_numbers = details::split(str,",");
            for (const auto &val : found_numbers)
            {
                if (!value.empty())
                    value += ",";
                value += Builder::asString((1 - std::stoi(val)));
            }
            return value;
        };

        res->params = params;
        // plugins require reversed value of begin_mask and end_mask
        res->params["begin_mask"] = stridedSliceInvertMaskStr(res->params["begin_mask"]);
        res->params["end_mask"] = stridedSliceInvertMaskStr(res->params["end_mask"]);

        return res;
    });

    addSpecificCreator({"TopK","TopKIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "TopK",
                          details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::TopKLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Transpose"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Permute",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        if (auto transpose_const = std::dynamic_pointer_cast<ngraph::op::Constant>(node->input_value(1).get_node_shared_ptr())) {
            res->params["order"] = Builder::asString(transpose_const->cast_vector<int64_t>());
        }
        return res;

    });

    addSpecificCreator({"SwishIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Swish",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        return res;

    });

    addSpecificCreator({"NonMaxSuppressionIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                 const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "NonMaxSuppression", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::NonMaxSuppressionLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"GRUSequenceIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                    const std::map<std::string, std::string>& params) -> CNNLayerPtr {

        LayerParams attrs = {node->get_friendly_name(), "GRUSequence",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<RNNSequenceLayer>(attrs);
        res->params = params;

        if (res->params["direction"] == "reverse")
            res->params["direction"] = "Backward";
        else if (res->params["direction"] == "forward")
            res->params["direction"] = "Forward";
        else
            res->params["direction"] = "Bidirectional";

        res->cellType = RNNSequenceLayer::CellType::GRU;
        if (res->params["linear_before_reset"] == "true") {
            res->cellType = RNNSequenceLayer::CellType::GRU_LBR;
        }

        Builder::NodeConverter<ngraph::op::Constant> converter;
        const auto weightsNode = node->input_value(3).get_node_shared_ptr();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }

        const auto biasNode = node->input_value(4).get_node_shared_ptr();
        if (converter.canCreate(biasNode)) {
            const auto& bias = converter.createLayer(biasNode);
            res->blobs["biases"] = bias->blobs["custom"];
            res->_biases = bias->blobs["custom"];
        }
        return res;
    });

    addSpecificCreator({"RNNSequenceIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                    const std::map<std::string, std::string>& params) -> CNNLayerPtr {

        LayerParams attrs = {node->get_friendly_name(), "RNNSequence",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<RNNSequenceLayer>(attrs);
        res->params = params;

        res->cellType = RNNSequenceLayer::CellType::RNN;

        if (res->params["direction"] == "reverse")
            res->params["direction"] = "Backward";
        else if (res->params["direction"] == "forward")
            res->params["direction"] = "Forward";
        else
            res->params["direction"] = "Bidirectional";

        Builder::NodeConverter<ngraph::op::Constant> converter;
        const auto weightsNode = node->input_value(3).get_node_shared_ptr();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }

        const auto biasNode = node->input_value(4).get_node_shared_ptr();
        if (converter.canCreate(biasNode)) {
            const auto& bias = converter.createLayer(biasNode);
            res->blobs["biases"] = bias->blobs["custom"];
            res->_biases = bias->blobs["custom"];
        }
        return res;
    });

    addSpecificCreator({"LSTMSequenceIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                    const std::map<std::string, std::string>& params) -> CNNLayerPtr {

        LayerParams attrs = {node->get_friendly_name(), "LSTMSequence",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<RNNSequenceLayer>(attrs);
        res->params = params;

        res->cellType = RNNSequenceLayer::CellType::LSTM;

        if (res->params["direction"] == "reverse")
            res->params["direction"] = "Backward";
        else if (res->params["direction"] == "forward")
            res->params["direction"] = "Forward";
        else
            res->params["direction"] = "Bidirectional";

        Builder::NodeConverter<ngraph::op::Constant> converter;
        const auto weightsNode = node->input_value(4).get_node_shared_ptr();
        if (converter.canCreate(weightsNode)) {
            const auto &weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }

        const auto biasNode = node->input_value(5).get_node_shared_ptr();
        if (converter.canCreate(biasNode)) {
            const auto &bias = converter.createLayer(biasNode);
            res->blobs["biases"] = bias->blobs["custom"];
            res->_biases = bias->blobs["custom"];
        }
        return res;
    });

    REQUIRED_IE_CONVERSION_CREATOR("Broadcast", "Tile");
    REQUIRED_IE_CONVERSION_CREATOR("Interpolate", "Interp");
    REQUIRED_IE_CONVERSION_CREATOR("NormalizeL2", "NormalizeIE");
    REQUIRED_IE_CONVERSION_CREATOR("GroupConvolution", "ConvolutionIE");
    REQUIRED_IE_CONVERSION_CREATOR("GroupConvolutionBackpropData", "DeconvolutionIE");

    addSpecificCreator({ "Convolution", "Gather", "GatherTree", "GRUCell", "GRUSequence", "HardSigmoid",
                      "LRN", "LSTMCell", "LSTMSequence", "NonMaxSuppression", "RNNCell", "RNNSequence", "OneHot",
                      "Pad", "PriorBoxClustered", "PriorBox", "Proposal", "Selu", "Swish", "Tile"},
            [](const std::shared_ptr<::ngraph::Node>& node, const std::map<std::string, std::string>& params)
            -> CNNLayerPtr {
        const std::string& type_name = node->get_type_name();
        THROW_IE_EXCEPTION << type_name << " operation has a form that is not supported. " << node->get_friendly_name()
                           << " should be converted to " << type_name + "IE operation.";
        return nullptr;
    });

    addSpecificCreator({"ReduceMin", "ReduceMax", "ReduceMean", "ReduceProd", "ReduceSum", "ReduceL1", "ReduceL2"},
                       [](const std::shared_ptr<::ngraph::Node>& node, const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(), details::convertPrecision(node->get_output_element_type(0))};
        auto reduce_node = std::dynamic_pointer_cast<ngraph::op::util::ArithmeticReductionKeepDims>(node);
        if (reduce_node == nullptr)
            THROW_IE_EXCEPTION << "Node '" << node->get_name() << "' is not an instance of ArithmeticReductionKeepDims.";
        auto res = std::make_shared<InferenceEngine::ReduceLayer>(attrs);
        res->params = params;
        res->params["keep_dims"] = reduce_node->get_keep_dims() ? "True" : "False";
        return res;
    });

    addSpecificCreator({"ReduceLogicalAnd"}, [](const std::shared_ptr<::ngraph::Node>& node, const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "ReduceAnd", details::convertPrecision(node->get_output_element_type(0))};
        auto reduce_node = std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(node);
        if (reduce_node == nullptr)
            THROW_IE_EXCEPTION << "Node '" << node->get_name() << "' is not an instance of LogicalReductionKeepDims.";
        auto res = std::make_shared<InferenceEngine::ReduceLayer>(attrs);
        res->params = params;
        res->params["keep_dims"] = reduce_node->get_keep_dims() ? "True" : "False";
        return res;
    });

    addSpecificCreator({"ReduceLogicalOr"}, [](const std::shared_ptr<::ngraph::Node>& node, const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "ReduceOr", details::convertPrecision(node->get_output_element_type(0))};
        auto reduce_node = std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(node);
        if (reduce_node == nullptr)
            THROW_IE_EXCEPTION << "Node '" << node->get_name() << "' is not an instance of LogicalReductionKeepDims.";
        auto res = std::make_shared<InferenceEngine::ReduceLayer>(attrs);
        res->params = params;
        res->params["keep_dims"] = reduce_node->get_keep_dims() ? "True" : "False";
        return res;
    });
}

CNNLayerPtr InferenceEngine::details::CNNLayerCreator::create() {
    LayerParams attrs = {node->get_friendly_name(), node->description(),
                         details::convertPrecision(node->get_output_element_type(0))};
    if (creators.find(node->description()) != creators.end())
        return creators[node->description()](node, params);

    auto res = std::make_shared<CNNLayer>(attrs);
    res->params = params;
    return res;
}

void convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function> &graph,
                                 const ICNNNetwork &network,
                                 CNNNetworkImpl* cnnNetworkImpl,
                                 bool keep_constant_inputs) {
    OV_ITT_SCOPED_TASK(itt::domains::IELegacy, "details::convertFunctionToICNNNetwork");

    const auto createCNNLayer = [](const std::shared_ptr<::ngraph::Node> &node) -> CNNLayerPtr {
        class NGraphCNNLayer: public CNNLayer {
        public:
            void setNode(const std::shared_ptr<::ngraph::Node>& node) {
                this->node = node;
            }
        };
        const static std::vector<std::shared_ptr<Builder::INodeConverter>> convertors = {
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::AvgPool>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Clamp>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Constant>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ConvolutionIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::CropIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Convert>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::CTCGreedyDecoder>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::DeformableConvolution>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::DeformablePSROIPooling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Eltwise>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Elu>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::FakeQuantize>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Ceiling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GatherIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GatherTreeIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Interp>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v0::Interpolate>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::LRN_IE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::MVN>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::FullyConnected>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::MatMul>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GenericIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GRN>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::MaxPool>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Minimum>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::NormalizeIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::OneHotIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PadIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PowerIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PriorBoxClusteredIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PriorBoxIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ReLUIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ResampleV2>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::RegionYolo>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ReorgYolo>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ROIPooling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PSROIPooling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ScaleShiftIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::SquaredDifference>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::VariadicSplit>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Subtract>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::TensorIterator>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::opset5::Loop>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::HardSigmoid_IE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ShuffleChannels>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v4::Interpolate>>(),
                std::make_shared<Builder::NodeConverter<::ExecGraphInfoSerialization::ExecutionNode>>(),
        };
        CNNLayerPtr result;

        for (auto &convertor : convertors) {
            if (!convertor->canCreate(node)) continue;
            result = convertor->createLayer(node);
            break;
        }

        if (!result) {
            CNNLayerCreator visitor(node);
            if (node->visit_attributes(visitor)) result = visitor.create();
        }

        if (!result)
            THROW_IE_EXCEPTION << "Cannot cast ngraph node " << node->get_friendly_name() << " to CNNLayer!";
        NGraphCNNLayer * layer = reinterpret_cast<NGraphCNNLayer*>(result.get());
        layer->setNode(node);
        return result;
    };

    const auto isInternalConstLayer = [](const std::shared_ptr<::ngraph::op::Constant> &constLayer,
                                         const std::shared_ptr<::ngraph::Node> &consumerLayer,
                                         bool keep_constants) -> bool {
        if (((::ngraph::as_type_ptr<::ngraph::op::ConvolutionIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::FullyConnected>(consumerLayer)) && !keep_constants) ||
            ::ngraph::as_type_ptr<::ngraph::op::v1::BinaryConvolution>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::DeconvolutionIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::v1::DeformableConvolution>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::Elu>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::NormalizeIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::PRelu>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::v1::Split>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::VariadicSplit>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::ScaleShiftIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::Transpose>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::LSTMSequenceIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::RNNSequenceIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::GRUSequenceIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::RNNCellIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::GRUCellIE>(consumerLayer)) {
            // Check that all input nodes except zero input are Constants for all ops except DeformableConvolutions
            // for which the input with index 1 is also dynamic
            size_t inputID = 1;
            if (::ngraph::as_type_ptr<::ngraph::op::v1::DeformableConvolution>(consumerLayer) ||
                             ::ngraph::as_type_ptr<::ngraph::op::GRUCellIE>(consumerLayer) ||
                             ::ngraph::as_type_ptr<::ngraph::op::RNNCellIE>(consumerLayer) ||
                    ::ngraph::as_type_ptr<::ngraph::op::GRUSequenceIE>(consumerLayer) ||
                    ::ngraph::as_type_ptr<::ngraph::op::RNNSequenceIE>(consumerLayer)) {
                inputID = 2;
            } else if (::ngraph::as_type_ptr<::ngraph::op::LSTMSequenceIE>(consumerLayer)) {
                inputID = 3;
            }

            for (; inputID < consumerLayer->inputs().size(); ++inputID) {
                auto inputLayer = consumerLayer->input(inputID).get_source_output().get_node_shared_ptr();
                if (inputLayer == constLayer) {
                    return true;
                }
            }
        } else if (::ngraph::as_type_ptr<::ngraph::op::LSTMCellIE>(consumerLayer)) {
            for (size_t inputID = 3; inputID < consumerLayer->inputs().size(); ++inputID) {
                auto inputLayer = consumerLayer->input(inputID).get_source_output().get_node_shared_ptr();
                if (inputLayer == constLayer) {
                    return true;
                }
            }
        }
        return false;
    };

    // Checks that node is internal layer for all layers from specific function
    const auto isInternalLayer = [=](const std::shared_ptr<::ngraph::Node> &node,
                                     bool keep_constant) -> bool {
        if (auto constantNode = ::ngraph::as_type_ptr<::ngraph::op::Constant>(node)) {
            for (const auto &consumerInputPort : constantNode->output(0).get_target_inputs()) {
                const auto &consumerLayer = consumerInputPort.get_node()->shared_from_this();
                if (!isInternalConstLayer(constantNode, consumerLayer, keep_constant))
                    return false;
            }
            return true;
        }

        return ::ngraph::as_type_ptr<::ngraph::op::Result>(node) != nullptr;
    };

    const auto keep_input_info = [](CNNNetworkImpl *network, const DataPtr &inData) {
        InputInfo::Ptr info(new InputInfo());
        info->setInputData(inData);
        network->setInputInfo(info);
    };

    // Check if some of function nodes has dynamic input or output shape
    // we collect this nodes and then throw an exception with the list
    // of dynamic nodes.
    std::stringstream err_log;
    for (const auto & node : graph->get_ordered_ops()) {
        bool is_dynamic = false;
        for (const auto & input : node->inputs()) {
            if (input.get_partial_shape().is_dynamic()) {
                is_dynamic = true;
                break;
            }
        }
        for (const auto & output : node->outputs()) {
            if (output.get_partial_shape().is_dynamic()) {
                is_dynamic = true;
                break;
            }
        }
        if (is_dynamic) err_log << node << std::endl;
    }
    if (!err_log.str().empty()) {
        THROW_IE_EXCEPTION << "\nUnsupported dynamic ops: \n" << err_log.str();
    }

    const CNNNetworkNGraphImpl* nGraphImpl = dynamic_cast<const CNNNetworkNGraphImpl*>(&network);

    InputsDataMap thisInputDataMap;
    network.getInputsInfo(thisInputDataMap);

    // Construct network
    cnnNetworkImpl->setName(graph->get_friendly_name());

    const ngraph::NodeVector& nodes = graph->get_ops();
    bool keep_constants = keep_constant_inputs || ::ngraph::op::util::has_op_with_type<::ngraph::op::FakeQuantize>(graph);

    std::unordered_map<std::string, std::shared_ptr<ngraph::Node>> unique_names;
    auto can_change_name = [](const std::shared_ptr<ngraph::Node> & node) -> bool {
        if (ngraph::as_type_ptr<ngraph::op::Parameter>(node) ||
            ngraph::as_type_ptr<ngraph::op::Result>(node)) {
            return false;
        }
        for (const auto & output : node->outputs()) {
            for (const auto & consumer : output.get_target_inputs()) {
                if (ngraph::is_type<ngraph::op::Result>(consumer.get_node())) {
                    return false;
                }
            }
        }
        return true;
    };

    auto generate_unique_name = [&unique_names](std::string name) -> std::string {
        size_t suffix = 1;
        while(unique_names.count(name + "/" + std::to_string(suffix))) {
            ++suffix;
        }
        return name + "/" + std::to_string(suffix);
    };

    // normalize nodes names to be unique
    for (auto & node : nodes) {
        // skip Result operations as they have the same friendly name as their parent
        if (ngraph::is_type<ngraph::op::Result>(node.get())) {
            continue;
        }

        auto & duplicate = unique_names[node->get_friendly_name()];
        if (!duplicate) {
            duplicate = node;
            continue;
        }

        if (!can_change_name(duplicate) && !can_change_name(node)) {
            THROW_IE_EXCEPTION << "Detected two output operations with the same name: " << duplicate << " and " << node;
        }

        auto & renamed = can_change_name(duplicate) ? duplicate : node;
        renamed->set_friendly_name(generate_unique_name(renamed->get_friendly_name()));

        unique_names[duplicate->get_friendly_name()] = duplicate;
        unique_names[node->get_friendly_name()] = node;
    }

    // Create layers and output data
    for (const auto &layer : nodes) {
        if (isInternalLayer(layer, keep_constants)) continue;

        // TODO: remove this rt info when all blobs will be inputs
        auto &rt_info = layer->get_rt_info();
        rt_info["keep_constants"] = std::make_shared<::ngraph::VariantWrapper<int64_t>> (keep_constants);

        CNNLayerPtr cnnLayer = createCNNLayer(layer);

        // Set originalLayersNames from FusedNames
        std::string originalNames = ::ngraph::getFusedNames(layer);
        if (!originalNames.empty()) {
            cnnLayer->params[ExecGraphInfoSerialization::ORIGINAL_NAMES] = originalNames;
        }

        std::string primitivesPriority = ::ngraph::getPrimitivesPriority(layer);
        if (!primitivesPriority.empty()) {
            cnnLayer->params["PrimitivesPriority"] = primitivesPriority;
        }

        // Copy runtime info attributes from Nodes to CNNLayers if they have VariantWrapper<std::string> type
        using VariantString = ::ngraph::VariantWrapper<std::string>;
        for (const auto &rt : rt_info) {
            if (auto str_attr = std::dynamic_pointer_cast<VariantString>(rt.second)) {
                if (details::CaselessEq<std::string>()(rt.first, "affinity")) {
                    cnnLayer->affinity = str_attr->get();
                } else {
                    cnnLayer->params[rt.first] = str_attr->get();
                }
            }
        }

        size_t inputCount(0);
        for (size_t i = 0; i < layer->get_input_size(); i++) {
            const auto &constant = ngraph::as_type_ptr<ngraph::op::Constant>(layer->input(i).get_source_output().get_node_shared_ptr());
            if (constant && isInternalConstLayer(constant, layer, keep_constants)) {
                continue;
            }
            inputCount++;
        }

        if (cnnLayer->type == "Memory" && cnnLayer->params["index"] == "1") {
            inputCount = 0;
        }

        cnnLayer->insData.resize(inputCount);

        for (size_t i = 0; i < layer->get_output_size(); i++) {
            // Memory node with index = 1 has no inputs according to the specification.
            // For proper conversion, we must cut off all the layers and data nodes above ReadValue,
            // if they are connected only with this layer.
            // Now MO generates only constants or constant sub-graphs as input to ReadValue op.
            if (std::dynamic_pointer_cast<::ngraph::op::Constant>(layer)) {
                bool all_to_read_value = !layer->output(i).get_target_inputs().empty();
                for (const auto &output_input : layer->output(i).get_target_inputs()) {
                    all_to_read_value
                            &= dynamic_cast<ngraph::op::ReadValue *>(output_input.get_node()) != nullptr;
                }
                if (all_to_read_value)
                    continue;
            }

            if (cnnLayer->type == "Memory" && cnnLayer->params["index"] == "0") {
                cnnLayer->outData.clear();
                continue;
            }
            auto outName = layer->output(i).get_tensor().get_name();
            if (outName.empty()) {
                outName = ngraph::op::util::create_ie_output_name(layer->output(i));
            }

            DataPtr &ptr = cnnNetworkImpl->getData(outName.c_str());
            IE_ASSERT(layer->get_output_partial_shape(i).is_static()) << " nGraph "
                << layer->description() << " operation with name: "
                << layer->get_friendly_name() << " cannot be converted to " << cnnLayer->type
                << " layer with name: " << cnnLayer->name << " because output with index "
                << i << " contains dynamic shapes: " << layer->get_output_partial_shape(i)
                << ". Try to use CNNNetwork::reshape() method in order to specialize shapes "
                << "before the conversion.";
            SizeVector dims = layer->get_output_shape(i);
            for (const auto &dim : dims) {
                if (!dim)
                    THROW_IE_EXCEPTION << cnnLayer->type << " layer " << cnnLayer->name
                        << " has incorrect dimensions in the output data " << i;
            }
            if (!ptr && nGraphImpl && nGraphImpl->_data.find(outName) != nGraphImpl->_data.end()) {
                ptr = nGraphImpl->_data.at(outName);
                {
                    const auto layout =
                        dims.size() == ptr->getTensorDesc().getDims().size() ?
                        ptr->getTensorDesc().getLayout() :
                        TensorDesc::getLayoutByDims(dims);

                    ptr->reshape(dims, layout);
                }
                cnnNetworkImpl->addData(outName.c_str(), ptr);
            }

            if (!ptr) {
                ptr.reset(new Data(outName,
                                   {details::convertPrecision(layer->get_output_element_type(i)), dims,
                                    TensorDesc::getLayoutByDims(dims)}));
            }

            getCreatorLayer(ptr) = cnnLayer;
            cnnLayer->outData.push_back(ptr);
            if (std::dynamic_pointer_cast<::ngraph::op::Parameter>(layer)) {
                keep_input_info(cnnNetworkImpl, ptr);
            }
        }
        cnnNetworkImpl->addLayer(cnnLayer);
    }

    // Set input data
    for (const auto &layer : graph->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<::ngraph::op::ReadValue>(layer))
            continue;
        if (std::dynamic_pointer_cast<::ngraph::op::Result>(layer)) {
            IE_ASSERT(layer->get_input_size() == 1);
            const auto &input = layer->input_value(0);
            auto name = input.get_tensor().get_name();
            if (!name.empty())
                cnnNetworkImpl->addOutput(name);
            else
                cnnNetworkImpl->addOutput(ngraph::op::util::create_ie_output_name(input));
            continue;
        }

        uint64_t count_of_skipped = 0;
        for (size_t i = 0; i < layer->get_input_size(); i++) {
            const auto &output_port = layer->input_value(i);
            const auto &input = output_port.get_node_shared_ptr();

            if (auto const_node = std::dynamic_pointer_cast<::ngraph::op::Constant>(input)) {
                if (isInternalConstLayer(const_node, layer, keep_constants)) {
                    count_of_skipped++;
                    continue;
                }
            }

            CNNLayerPtr prevCnnLayer;
            StatusCode ret = cnnNetworkImpl->getLayerByName(input->get_friendly_name().c_str(), prevCnnLayer, nullptr);
            if (ret != OK)
                THROW_IE_EXCEPTION << "Cannot find layer with name: " << input->get_friendly_name();

            CNNLayerPtr cnnLayer;
            ret = cnnNetworkImpl->getLayerByName(layer->get_friendly_name().c_str(), cnnLayer, nullptr);
            if (ret != OK)
                THROW_IE_EXCEPTION << "Cannot find layer with name: " << layer->get_friendly_name();

            auto inIndex = layer->input(i).get_index();
            if (cnnLayer->insData.size() <= (inIndex - count_of_skipped) ||
                prevCnnLayer->outData.size() <= output_port.get_index() || count_of_skipped > inIndex)
                THROW_IE_EXCEPTION << "Cannot create ICNNNetwork. Network structure is incorrect! "
                                   << "Input port " << inIndex << " (max " << cnnLayer->insData.size() << ") of "
                                   << cnnLayer->type << " layer " << cnnLayer->name
                                   << " cannot be connected with output port " << output_port.get_index()
                                   << " (max " << prevCnnLayer->outData.size() << ") of " << prevCnnLayer->type
                                   << " layer " << prevCnnLayer->name;
            cnnLayer->insData[inIndex - count_of_skipped] = prevCnnLayer->outData[output_port.get_index()];
            getInputTo(prevCnnLayer->outData[output_port.get_index()])[cnnLayer->name] = cnnLayer;
        }
    }

    // check all input ports are occupied
    for (const auto &kvp : cnnNetworkImpl->allLayers()) {
        const CNNLayer::Ptr &layer = kvp.second;
        size_t inSize = layer->insData.size();

        for (unsigned i = 0; i < inSize; i++) {
            if (!layer->insData[i].lock()) {
                THROW_IE_EXCEPTION << "Layer " << layer->name.c_str() << " input port " << i
                                   << " is not connected to any data";
            }
        }

        // execution ngraph is fake graph and should not be validated
        if (layer->params.count(ExecGraphInfoSerialization::PERF_COUNTER) == 0) {
            layer->parseParams();
        }
    }

    if (!cnnNetworkImpl) THROW_IE_EXCEPTION << "Cannot convert nGraph function to CNNNetworkImpl!";

    // update input preprocessing info
    InputsDataMap resultInputDataMap;
    cnnNetworkImpl->getInputsInfo(resultInputDataMap);
    IE_ASSERT(resultInputDataMap.size() == thisInputDataMap.size());
    for (auto i : resultInputDataMap) {
        auto &thisInputData = *thisInputDataMap[i.first];
        i.second->setPrecision(thisInputData.getPrecision());
        i.second->setLayout(thisInputData.getLayout());
        i.second->getPreProcess() = thisInputData.getPreProcess();
    }
}

std::shared_ptr<CNNNetworkImpl> convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function> &graph,
                                                             const ICNNNetwork &network,
                                                             bool keep_constant_inputs) {
    auto cnnNetworkImpl = std::make_shared<details::CNNNetworkImpl>();
    convertFunctionToICNNNetwork(graph, network, cnnNetworkImpl.get(), keep_constant_inputs);
    return cnnNetworkImpl;
}

}  // namespace details
}  // namespace InferenceEngine

// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <paddlepaddle_frontend/model.hpp>
#include <paddlepaddle_frontend/place.hpp>
#include <paddlepaddle_frontend/utility.hpp>

#include <fstream>
#include "framework.pb.h"
#include "decoder.hpp"
#include "node_context.hpp"
#include <ngraph/opsets/opset7.hpp>

namespace ngraph {
namespace frontend {

using namespace paddle::framework::proto;

class InputModelPDPD::InputModelPDPDImpl {
public:

    InputModelPDPDImpl (const std::string& _path, const IInputModel& input_model);
    std::vector<IPlace::Ptr> getInputs () const;
    std::vector<IPlace::Ptr> getOutputs () const;
    IPlace::Ptr getPlaceByTensorName (const std::string& tensorName) const;
    void overrideAllOutputs (const std::vector<IPlace::Ptr>& outputs);
    void overrideAllInputs (const std::vector<IPlace::Ptr>& inputs);
    void extractSubgraph (const std::vector<IPlace::Ptr>& inputs, const std::vector<IPlace::Ptr>& outputs);
    void setDefaultShape (IPlace::Ptr place, const ngraph::Shape&);
    void setPartialShape (IPlace::Ptr place, const ngraph::PartialShape&);
    void setElementType (IPlace::Ptr place, const ngraph::element::Type&);
    void setTensorValue (IPlace::Ptr place, const void* value);

    std::vector<uint8_t> readWeight(const std::string& name, int64_t len);
    std::vector<std::shared_ptr<OpPlacePDPD>> getOpPlaces(int i) const { return m_op_places_blocks[i]; }
    std::map<std::string, std::shared_ptr<TensorPlacePDPD>> getVarPlaces(int i) const { return m_var_places_blocks[i]; }
    size_t getBlockNumber() const { return m_op_places_blocks.size(); }
    std::map<pdpd::TensorName, Output<Node>> getTensorValues() const { return m_tensor_values; };

private:
    std::string m_path;
    std::vector<std::vector<std::shared_ptr<OpPlacePDPD>>> m_op_places_blocks;
    std::vector<std::map<std::string, std::shared_ptr<TensorPlacePDPD>>> m_var_places_blocks;
    std::shared_ptr<ProgramDesc> m_fw_ptr;
    // GCC 4.8 limitation: can't create 'm_weights_stream = std::ifstream(...)'
    std::unique_ptr<std::ifstream> m_weights_stream;
    bool m_weights_composed = false;
    const IInputModel& m_input_model;
    std::vector<IPlace::Ptr> m_inputs;
    std::vector<IPlace::Ptr> m_outputs;
    std::map<pdpd::TensorName, Output<Node>> m_tensor_values;
};

InputModelPDPD::InputModelPDPDImpl::InputModelPDPDImpl(const std::string& _path, const IInputModel& input_model)
    : m_path(_path),
      m_fw_ptr{std::make_shared<ProgramDesc>()},
      m_input_model(input_model) {
    std::string ext = ".pdmodel";            
    std::string model_file(m_path);
    if (m_path.length() >= ext.length() && (0 == m_path.compare(m_path.length() - ext.length(), ext.length(), ext)))
    {
        m_weights_composed = true;
        auto weights_file = m_path.replace(m_path.size() - ext.size(), ext.size(), ".pdiparams");
        m_weights_stream = std::unique_ptr<std::ifstream>(new std::ifstream(weights_file, std::ios::binary));
        if (!m_weights_stream || !m_weights_stream->is_open()) {
            std::cerr << "Cannot open file containing weights: " << weights_file << std::endl;
        }
    } else {
        m_weights_composed = false;
        model_file += "/__model__";
    }

    std::ifstream pb_stream(model_file, std::ios::binary);
    std::cout << "Model Parsed: " << m_fw_ptr->ParseFromIstream(&pb_stream) << std::endl;

    std::cout << "Blocks number: " << m_fw_ptr->blocks().size() << std::endl;

    const int cnt_of_blocks = m_fw_ptr->blocks_size();
    const auto& blocks = m_fw_ptr->blocks();
    m_var_places_blocks.resize(cnt_of_blocks);
    m_op_places_blocks.resize(cnt_of_blocks);

    for (int block_idx = 0; block_idx < cnt_of_blocks; block_idx++) {
        const auto& block = blocks[block_idx];
        auto& var_place_block = m_var_places_blocks[block_idx];
        auto& op_place_block = m_op_places_blocks[block_idx];

        for (const auto& var : block.vars()) {
            var_place_block[var.name()] = std::make_shared<TensorPlacePDPD>(m_input_model, std::make_shared<VarDesc>(var));
        }

        for (const auto& op : block.ops()) {
            auto op_place = std::make_shared<OpPlacePDPD>(m_input_model, std::make_shared<OpDesc>(op));
            op_place_block.push_back(op_place);

            for (const auto &output : op.outputs()) {
                for (const auto &var_name : output.arguments()) {
                    auto out_port = std::make_shared<OutPortPlacePDPD>(m_input_model);

                    // connect out_port and tensor
                    const auto& tensor = var_place_block.at(var_name);
                    tensor->addProducingPort(out_port);
                    out_port->setTargetTensor(tensor);

                    // connect out_port and op
                    op_place->addOutPort(out_port, output.parameter());
                    out_port->setOp(op_place);
                }
            }

            for (const auto &input : op.inputs()) {
                for (const auto &var_name : input.arguments()) {
                    auto in_port = std::make_shared<InPortPlacePDPD>(m_input_model);

                    // connect in_port and tensor
                    const auto& tensor = var_place_block.at(var_name);
                    tensor->addConsumingPort(in_port);
                    in_port->setSourceTensor(tensor);

                    // connect in_port and op
                    op_place->addInPort(in_port, input.parameter());
                    in_port->setOp(op_place);
                }
            }

            // Determine outputs and inputs
            if (op.type() == "feed") {
                const auto& place = op_place->getOutputPortPDPD("Out", 0);
                const auto& var_place = std::dynamic_pointer_cast<TensorPlacePDPD>(place->getTargetTensorPDPD());
                const auto& tensor_desc = var_place->getDesc()->type().lod_tensor().tensor();
                const auto& dims = tensor_desc.dims();

                var_place->setElementType(TYPE_MAP[tensor_desc.data_type()]);
                var_place->setPartialShape(PartialShape(std::vector<Dimension>(dims.begin(), dims.end())));
                m_inputs.push_back(var_place);
            } else if (op.type() == "fetch") {
                auto place = op_place->getInputPortPDPD("X", 0);
                m_outputs.push_back(place->getSourceTensorPDPD());
            }
        }
    }
}

std::vector<uint8_t> InputModelPDPD::InputModelPDPDImpl::readWeight(const std::string& name, int64_t len) {
    std::vector<uint8_t> tensor_data(len);

    std::unique_ptr<std::ifstream> is;
    std::ifstream* stream_ptr;
    if (m_weights_composed) {
        stream_ptr = m_weights_stream.get();
    } else {
        is = std::unique_ptr<std::ifstream>(new std::ifstream(m_path + "/" + name, std::ios::in | std::ifstream::binary));
        PDPD_ASSERT(is && is->is_open(), "Cannot open file for constant value.");
        stream_ptr = is.get();
    }
    std::vector<char> header(16);
    stream_ptr->read(&header[0], 16);
    uint32_t dims_len = 0;
    stream_ptr->read(reinterpret_cast<char*>(&dims_len), 4);
    std::vector<char> dims_struct(dims_len);
    stream_ptr->read(&dims_struct[0], dims_len);
    stream_ptr->read(reinterpret_cast<char*>(&tensor_data[0]), len);
    return tensor_data;
}

std::vector<IPlace::Ptr> InputModelPDPD::InputModelPDPDImpl::getInputs () const {
    return m_inputs;
}

std::vector<IPlace::Ptr> InputModelPDPD::InputModelPDPDImpl::getOutputs () const {
    return m_outputs;
}

IPlace::Ptr InputModelPDPD::InputModelPDPDImpl::getPlaceByTensorName (const std::string& tensorName) const {
    for (const auto& var_places_in_block : m_var_places_blocks) {
        if (var_places_in_block.count(tensorName))
            return var_places_in_block.at(tensorName);
    }
    return nullptr;
}

namespace pdpd {

std::shared_ptr<TensorPlacePDPD> castToTensorPlace(const IPlace::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<TensorPlacePDPD>(place)) {
        return var_place;
    } else if (auto in_port_place = std::dynamic_pointer_cast<InPortPlacePDPD>(place)) {
        return in_port_place->getSourceTensorPDPD();
    } else if (auto out_port_place = std::dynamic_pointer_cast<OutPortPlacePDPD>(place)) {
        return out_port_place->getTargetTensorPDPD();
    }
    PDPD_THROW("Cannot cast this Place to TensorPlacePDPD.");
}

} // namespace pdpd

void InputModelPDPD::InputModelPDPDImpl::overrideAllInputs (const std::vector<IPlace::Ptr>& inputs) {
    m_inputs.clear();
    for (const auto& inp : inputs) {
        m_inputs.push_back(pdpd::castToTensorPlace(inp));
    }
}

void InputModelPDPD::InputModelPDPDImpl::overrideAllOutputs (const std::vector<IPlace::Ptr>& outputs) {
    m_outputs.clear();
    for (const auto& outp : outputs) {
        m_outputs.push_back(pdpd::castToTensorPlace(outp));
    }
}

void InputModelPDPD::InputModelPDPDImpl::extractSubgraph (const std::vector<IPlace::Ptr>& inputs, const std::vector<IPlace::Ptr>& outputs) {
    overrideAllInputs(inputs);
    overrideAllOutputs(outputs);
}

void InputModelPDPD::InputModelPDPDImpl::setDefaultShape (IPlace::Ptr place, const ngraph::Shape& shape) {
    NOT_IMPLEMENTED("setDefaultShape");
}

void InputModelPDPD::InputModelPDPDImpl::setPartialShape (IPlace::Ptr place, const ngraph::PartialShape& p_shape) {
    pdpd::castToTensorPlace(place)->setPartialShape(p_shape);
}

void InputModelPDPD::InputModelPDPDImpl::setElementType (IPlace::Ptr place, const ngraph::element::Type& type) {
    pdpd::castToTensorPlace(place)->setElementType(type);
}

void InputModelPDPD::InputModelPDPDImpl::setTensorValue (IPlace::Ptr place, const void* value) {
    auto tensor_place = pdpd::castToTensorPlace(place);
    auto p_shape = tensor_place->getPartialShape();
    auto type = tensor_place->getElementType();
    auto constant = opset7::Constant::create(type, p_shape.to_shape(), value);
    auto name = tensor_place->getNames()[0];
    constant->set_friendly_name(name);
    m_tensor_values[name] = constant;
}

InputModelPDPD::InputModelPDPD (const std::string& _path) : _impl{std::make_shared<InputModelPDPDImpl>(_path, *this)} {}

std::vector<uint8_t> InputModelPDPD::readWeight(const std::string& name, int64_t len) {
    return _impl->readWeight(name, len);
}

std::vector<std::shared_ptr<OpPlacePDPD>> InputModelPDPD::getOpPlaces(int i) const {
    return _impl->getOpPlaces(i);
}

std::map<std::string, std::shared_ptr<TensorPlacePDPD>> InputModelPDPD::getVarPlaces(int i) const {
    return _impl->getVarPlaces(i);
}

size_t InputModelPDPD::getBlockNumber() const {
    return _impl->getBlockNumber();
}

std::map<pdpd::TensorName, Output<Node>> InputModelPDPD::getTensorValues() const {
    return _impl->getTensorValues();
}

std::vector<IPlace::Ptr> InputModelPDPD::getInputs () const {
    return _impl->getInputs();
}

std::vector<IPlace::Ptr> InputModelPDPD::getOutputs () const {
    return _impl->getOutputs();
}

IPlace::Ptr InputModelPDPD::getPlaceByTensorName (const std::string& tensorName) const {
    return _impl->getPlaceByTensorName(tensorName);
}

void InputModelPDPD::overrideAllOutputs (const std::vector<IPlace::Ptr>& outputs) {
    return _impl->overrideAllOutputs(outputs);
}

void InputModelPDPD::overrideAllInputs (const std::vector<IPlace::Ptr>& inputs) {
    return _impl->overrideAllInputs(inputs);
}

void InputModelPDPD::extractSubgraph (const std::vector<IPlace::Ptr>& inputs, const std::vector<IPlace::Ptr>& outputs) {
    return _impl->extractSubgraph(inputs, outputs);
}

void InputModelPDPD::setDefaultShape (IPlace::Ptr place, const ngraph::Shape& shape) {
    return _impl->setDefaultShape(place, shape);
}

void InputModelPDPD::setPartialShape (IPlace::Ptr place, const ngraph::PartialShape& p_shape) {
    return _impl->setPartialShape(place, p_shape);
}

void InputModelPDPD::setElementType (IPlace::Ptr place, const ngraph::element::Type& type) {
    return _impl->setElementType(place, type);
}

void InputModelPDPD::setTensorValue (IPlace::Ptr place, const void* value) {
    return _impl->setTensorValue(place, value);
}

} // namespace frontend
} // namespace ngraph

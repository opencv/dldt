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

#include <frontend_manager/frontend_manager.hpp>

namespace ngraph {
namespace frontend {

class OpPlacePDPD;
class TensorPlacePDPD;

class NGRAPH_API InputModelPDPD : public InputModel
{
    friend class FrontEndPDPD;
    class InputModelPDPDImpl;
    std::shared_ptr<InputModelPDPDImpl> _impl;

    std::vector<std::shared_ptr<OpPlacePDPD>> getOpPlaces() const;
    std::map<std::string, std::shared_ptr<TensorPlacePDPD>> getVarPlaces() const;
    std::map<std::string, Output<Node>> getTensorValues() const;

public:
    explicit InputModelPDPD (const std::string& path);
    explicit InputModelPDPD (const std::vector<std::istream*>& streams);
    std::vector<Place::Ptr> getInputs () const override;
    std::vector<Place::Ptr> getOutputs () const override;
    Place::Ptr getPlaceByTensorName (const std::string& tensorName) const override;
    void overrideAllOutputs (const std::vector<Place::Ptr>& outputs) override;
    void overrideAllInputs (const std::vector<Place::Ptr>& inputs) override;
    void extractSubgraph (const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override;
    void setDefaultShape (Place::Ptr place, const ngraph::Shape&) override;
    void setPartialShape (Place::Ptr place, const ngraph::PartialShape&) override;
    ngraph::PartialShape getPartialShape (Place::Ptr place) const override;
    void setElementType (Place::Ptr place, const ngraph::element::Type&) override;
    void setTensorValue (Place::Ptr place, const void* value) override;

};

} // namespace frontend
} // namespace ngraph

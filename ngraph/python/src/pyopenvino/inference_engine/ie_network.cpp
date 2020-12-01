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

#include <cpp/ie_cnn_network.h>

#include "pyopenvino/inference_engine/ie_network.hpp"
//#include "../../../pybind11/include/pybind11/pybind11.h"


namespace py = pybind11;

void regclass_IENetwork(py::module m)
{
    py::class_<InferenceEngine::CNNNetwork, std::shared_ptr<InferenceEngine::CNNNetwork>> cls(m, "IENetwork");
    cls.def(py::init([](py::object* capsule) {
            // get the underlying PyObject* which is a PyCapsule pointer
            auto* pybind_capsule_ptr = capsule->ptr();

            // extract the pointer stored in the PyCapsule under the name "ngraph_function"
            auto* capsule_ptr = PyCapsule_GetPointer(pybind_capsule_ptr, "ngraph_function");

            auto* function_sp = static_cast<std::shared_ptr<ngraph::Function>*>(capsule_ptr);
            if (function_sp == nullptr)
                THROW_IE_EXCEPTION << "Cannot create CNNNetwork from capsule! Capsule doesn't contain nGraph function!";

            InferenceEngine::CNNNetwork cnnNetwork(*function_sp);
            return std::make_shared<InferenceEngine::CNNNetwork>(cnnNetwork);
        }));
}

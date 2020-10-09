// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <utility>
#include <algorithm>
#include <memory>
#include <string>
#include <map>

#include <ie_blob.h>
#include <description_buffer.hpp>
#include <debug.h>
#include <ie_layouts.h>
#include <threading/ie_executor_manager.hpp>
#include <blob_transform.hpp>
#include <ie_parallel.hpp>
#include <ie_memcpy.h>
#include <precision_utils.h>

#include "template/template_config.hpp"
#include "template_infer_request.hpp"
#include "template_executable_network.hpp"
#include "template_plugin.hpp"
#include "template_itt.hpp"

using namespace TemplatePlugin;
using namespace InferenceEngine;

using Time = std::chrono::high_resolution_clock;

// ! [infer_request:ctor]
TemplateInferRequest::TemplateInferRequest(const InferenceEngine::InputsDataMap&                     networkInputs,
                                           const InferenceEngine::OutputsDataMap&                    networkOutputs,
                                           const std::shared_ptr<TemplatePlugin::ExecutableNetwork>& executableNetwork) :
    InferRequestInternal(networkInputs, networkOutputs),
    _executableNetwork(executableNetwork) {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

    auto requestID = std::to_string(_executableNetwork->_requestId.fetch_add(1));

    std::string name = _executableNetwork->_function->get_friendly_name() + "_Req" + requestID;
    _profilingTask = {
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_Preprocess"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_Postprocess"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_StartPipline"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_WaitPipline"),
    };

    _executable = _executableNetwork->_plugin->_backend->compile(_executableNetwork->_function);
    _parameters = _executableNetwork->_function->get_parameters();
    _results = _executableNetwork->_function->get_results();

    allocateDeviceBuffers();
    allocateBlobs();
}
// ! [infer_request:ctor]

// ! [infer_request:dtor]
TemplateInferRequest::~TemplateInferRequest() {
    _executableNetwork->_requestId--;
}
// ! [infer_request:dtor]

void TemplateInferRequest::allocateDeviceBuffers() {
    // Allocate plugin backend specific memory handles
    _inputTensors.resize(_networkInputs.size());
    _outputTensors.resize(_networkOutputs.size());
}

template<typename BlobData, typename GetNetworkPrecisionF>
static void AllocateImplSingle(
                         BlobMap& blobMap,
                         BlobMap& networkBlobMap,
                         BlobData& blobData,
                         GetNetworkPrecisionF&& GetNetworkPrecision,
                         const SizeVector& dims) {
    auto& precision = blobData.second->getTensorDesc().getPrecision();
    auto layout = blobData.second->getTensorDesc().getLayout();
    Blob::Ptr blob;
    switch (precision) {
        case Precision::U8: {
            blob = InferenceEngine::make_shared_blob<std::uint8_t>({precision, dims, layout});
        } break;
        case Precision::FP32 : {
            blob = InferenceEngine::make_shared_blob<float>({precision, dims, layout});
        } break;
        case Precision::I32 : {
            blob = InferenceEngine::make_shared_blob<std::int32_t>({precision, dims, layout});
        } break;
        default: THROW_IE_EXCEPTION << "Template Plugin: Unsupported Input/Output Presision";
    }
    blob->allocate();
    blobMap[blobData.first] = blob;

    auto networkPresion = GetNetworkPrecision(blobData.first);
    Blob::Ptr networkBlob;
    switch (networkPresion) {
        case ngraph::element::Type_t::f32 : {
            if (precision == Precision::FP32) {
                networkBlob = blob;
            } else {
                networkBlob = InferenceEngine::make_shared_blob<float>({Precision::FP32, dims, layout});
            }
        } break;
        case ngraph::element::Type_t::i32 : {
            if (precision == Precision::I32) {
                networkBlob = blob;
            } else {
                networkBlob = InferenceEngine::make_shared_blob<std::int32_t>({Precision::I32, dims, layout});
            }
        } break;
        default: THROW_IE_EXCEPTION << "Template Plugin: Unsupported network Input/Output Presision";
    }
    if (blob != networkBlob) {
        networkBlob->allocate();
    }
    networkBlobMap[blobData.first] = networkBlob;
}

template<typename BlobDataMap, typename GetNetworkPrecisionF>
static void AllocateImpl(const BlobDataMap& blobDataMap,
                         BlobMap& blobMap,
                         BlobMap& networkBlobMap,
                         GetNetworkPrecisionF&& GetNetworkPrecision) {
    for (auto&& blobData : blobDataMap) {
        if (blobData.second->getTensorDesc().getPartialShape().is_dynamic()) {
            // Cannot pre-allocate blob for a tensor with unknown dimensions
            continue;
        }
        auto& dims = blobData.second->getTensorDesc().getDims();
        AllocateImplSingle(blobMap, networkBlobMap, blobData, GetNetworkPrecision, dims);
    }
}

void TemplateInferRequest::allocateBlobs() {
    auto&& parameters = _executableNetwork->_function->get_parameters();
    AllocateImpl(_networkInputs, _inputs, _networkInputBlobs, [&] (const std::string& blobName) {
        return parameters.at(_executableNetwork->_inputIndex.at(blobName))->get_element_type();
    });
    auto&& results = _executableNetwork->_function->get_results();
    AllocateImpl(_networkOutputs, _outputs, _networkOutputBlobs, [&] (const std::string& blobName) {
        return results.at(_executableNetwork->_outputIndex.at(blobName))->get_element_type();
    });
}

// ! [infer_request:infer_impl]
void TemplateInferRequest::InferImpl() {
    // TODO: fill with actual list of pipeline stages, which are executed synchronously for sync infer requests
    inferPreprocess();
    startPipeline();
    waitPipeline();  // does nothing in current implementation
    inferPostprocess();
}
// ! [infer_request:infer_impl]

template<typename SrcT, typename DstT>
static void blobCopy(const Blob::Ptr& src, const Blob::Ptr& dst) {
    std::copy_n(InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const SrcT*>(),
                src->size(),
                InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<DstT*>());
}

static void blobCopy(const Blob::Ptr& src, const Blob::Ptr& dst) {
    switch (src->getTensorDesc().getPrecision()) {
        case Precision::U8 : {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::U8 : break;
                case Precision::FP32 : {
                    blobCopy<std::uint8_t, float>(src, dst);
                } break;
                default : {
                    THROW_IE_EXCEPTION << "Unsupported precision conversion from "
                        << src->getTensorDesc().getPrecision() <<" to " << dst->getTensorDesc().getPrecision();
                }
            }
        } break;
        case Precision::FP32 : {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::FP32 : break;
                case Precision::U8 : {
                    blobCopy<float, std::uint8_t>(src, dst);
                } break;
                default : {
                    THROW_IE_EXCEPTION << "Unsupported precision conversion from "
                        << src->getTensorDesc().getPrecision() <<" to " << dst->getTensorDesc().getPrecision();
                }
            }
        } break;
        default : {
            THROW_IE_EXCEPTION << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision();
        }
    }
}

// ! [infer_request:infer_preprocess]
void TemplateInferRequest::inferPreprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[Preprocess]);
    auto start = Time::now();
    // NOTE: After InferRequestInternal::execDataPreprocessing call
    //       input can points to other memory region than it was allocated in constructor.
    InferRequestInternal::execDataPreprocessing(_inputs);
    for (auto&& input : _inputs) {
        auto inputBlob = input.second;
        auto networkInput = _networkInputBlobs[input.first];
        if (inputBlob->getTensorDesc().getPrecision() == networkInput->getTensorDesc().getPrecision()) {
            networkInput = inputBlob;
        } else {
            blobCopy(inputBlob, networkInput);
        }
        auto index = _executableNetwork->_inputIndex[input.first];
        const auto& parameter = _parameters[index];
        auto parameterShape = m_realShapes.find(input.first) != m_realShapes.end() ? ngraph::Shape(m_realShapes.at(input.first)) : parameter->get_shape();
        const auto& parameterType = parameter->get_element_type();
        _inputTensors[index] = _executableNetwork->_plugin->_backend->create_tensor(parameterType, parameterShape,
            InferenceEngine::as<InferenceEngine::MemoryBlob>(networkInput)->rmap().as<void*>());
    }
    // Go over all outputs in the model, not over all allocated blobs because for a part of the outputs
    // blobs may not be yet allocated due to unknown dimensions
    // TODO: should we really go over all results in the network, or it is better to go over _networkOutputs?
    for (size_t index = 0; index < _results.size(); ++index) {
        _outputTensors[index] = _executableNetwork->_plugin->_backend->create_tensor();
    }
    _durations[Preprocess] = Time::now() - start;
}
// ! [infer_request:infer_preprocess]

// ! [infer_request:start_pipeline]
void TemplateInferRequest::startPipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[StartPipeline])
    auto start = Time::now();
    _executable->call(_outputTensors, _inputTensors);
    _durations[StartPipeline] = Time::now() - start;
}
// ! [infer_request:start_pipeline]

void TemplateInferRequest::waitPipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[WaitPipeline])
    auto start = Time::now();
    // TODO: Wait pipeline using driver API or other synchronizations methods
    // NOTE: not used in current implementation since `startPipeline` executes pipiline synchronously
    _durations[WaitPipeline] = Time::now() - start;
}

// ! [infer_request:infer_postprocess]
void TemplateInferRequest::inferPostprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[Postprocess]);
    auto start = Time::now();
    for (auto&& networkOutput : _networkOutputs) {
        {
            // Touch blob to allocate it
            Blob::Ptr blob;
            GetBlob(networkOutput.first.c_str(), blob);
        }
        auto outputBlob = _outputs.at(networkOutput.first);
        auto networkOutputBlob = _networkOutputBlobs[networkOutput.first];
        // perform precision conversion of network output's precision and computational
        // graph output's precision are different
        if (outputBlob->getTensorDesc().getPrecision() != networkOutputBlob->getTensorDesc().getPrecision()) {
            blobCopy(networkOutputBlob, outputBlob);
        }
    }
    _durations[Postprocess] = Time::now() - start;
}
// ! [infer_request:infer_postprocess]

void TemplateInferRequest::GetBlob(const char* name, Blob::Ptr& data) {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "GetBlob");
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    const SizeVector oneVector = { 1 };
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        // ROI blob is returned only if it was set previously. Otherwise default blob is returned.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
            data = it->second->getRoiBlob();
        } else {
            data = _inputs[name];
            const auto& dims = m_realShapes.find(name) != m_realShapes.end() ? m_realShapes[name] : foundInput->getTensorDesc().getDims();
            if (!data) {
                auto&& parameters = _executableNetwork->_function->get_parameters();
                AllocateImplSingle(_inputs, _networkInputBlobs, *_networkInputs.find(name), [&] (const std::string& blobName) {
                    return parameters.at(_executableNetwork->_inputIndex.at(blobName))->get_element_type();
                }, dims);
                data = _inputs[name];
            }
            checkBlob(data, name, true,
                      foundInput->getTensorDesc().getLayout() != SCALAR
                      ? dims
                      : oneVector);
        }
    } else {
        data = _outputs[name];
        SizeVector dims;
        if (foundOutput->getTensorDesc().getPartialShape().is_static()) {
            dims = foundOutput->getTensorDesc().getDims();
        } else if (_outputTensors[_executableNetwork->_outputIndex.at(name)]->get_partial_shape().is_static()) {
            dims = _outputTensors[_executableNetwork->_outputIndex.at(name)]->get_shape();
        } else {
            THROW_IE_EXCEPTION << "Output blob dimensions are not all known for output name " <<
                name << " with partial shape: " << foundOutput->getTensorDesc().getPartialShape();
        }

        if (!data) {
            auto &&results = _executableNetwork->_function->get_results();
            AllocateImplSingle(_outputs, _networkOutputBlobs, *_networkOutputs.find(name),
                               [&](const std::string &blobName) {
                                   return results.at(_executableNetwork->_outputIndex.at(blobName))->get_element_type();
                               }, dims);
            data = _outputs[name];
        }
        checkBlob(data, name, false,
                  foundOutput->getTensorDesc().getLayout() != SCALAR
                  ? dims
                  : oneVector);
    }
}

// ! [infer_request:get_performance_counts]
void TemplateInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const {
    InferenceEngineProfileInfo info;
    info.execution_index = 0;
    info.status = InferenceEngineProfileInfo::EXECUTED;

    info.cpu_uSec = info.realTime_uSec = _durations[Preprocess].count();
    perfMap["1. input preprocessing"] = info;
    info.cpu_uSec = info.realTime_uSec = 0;
    perfMap["2. input transfer to a device"] = info;
    info.cpu_uSec = info.realTime_uSec = _durations[StartPipeline].count();
    perfMap["3. execution time"] = info;
    info.cpu_uSec = info.realTime_uSec = 0;
    perfMap["4. output transfer from a device"] = info;
    info.cpu_uSec = info.realTime_uSec = _durations[Postprocess].count();
    perfMap["5. output postprocessing"] = info;
}
// ! [infer_request:get_performance_counts]

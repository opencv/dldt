// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "auto_infer_request.hpp"
#include <ie_input_info.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>

namespace AutoPlugin {
    using namespace InferenceEngine;

AutoInferRequest::AutoInferRequest(const InputsDataMap&              networkInputs,
                                   const OutputsDataMap&             networkOutputs,
                                   const SoIInferRequestInternal&    inferRequest,
                                   AutoPlugin::NetworkSharedFuture f, std::atomic<bool>& anyRequestHasHotSwapped)
    : IInferRequestInternal(networkInputs, networkOutputs)
    , _inferRequest(inferRequest), _sharedFutureForActualNetwork(f), _anyRequestHasHotSwapped(anyRequestHasHotSwapped) {
    for (const auto &it : _networkInputs)
        _inputs[it.first] = _inferRequest->GetBlob(it.first);
    for (const auto &it : _networkOutputs)
        _outputs[it.first] = _inferRequest->GetBlob(it.first);
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoInferRequest::GetPerformanceCounts() const {
    return _inferRequest->GetPerformanceCounts();
}

void AutoInferRequest::InferImpl() {
    HotSwapRequests(); //safe to call here (before actual inference started)
    SetBlobsToDeviceRequest();
    _inferRequest->Infer();
}

void AutoInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data) {
    IInferRequestInternal::SetBlob(name, data);
}

Blob::Ptr AutoInferRequest::GetBlob(const std::string& name) {
    return IInferRequestInternal::GetBlob(name);
}

void AutoInferRequest::Cancel() {
    _inferRequest->Cancel();
}

void AutoInferRequest::StartAsync() {
    HotSwapRequests(); //safe to call here (before actual inference started)
    SetBlobsToDeviceRequest();
    _inferRequest->StartAsync();
}

InferenceEngine::StatusCode AutoInferRequest::Wait(int64_t millis_timeout) {
    return _inferRequest->Wait(millis_timeout);
}

void AutoInferRequest::SetCallback(Callback callback) {
    _callback = callback;
    _inferRequest->SetCallback(callback);
}

void AutoInferRequest::HotSwapRequests() {
    if (_sharedFutureForActualNetwork.valid() && _sharedFutureForActualNetwork.wait_for(std::chrono::nanoseconds(0)) == std::future_status::ready) {
        std::lock_guard<std::mutex>{_hotswapMutex};
        if (!_hotswapDone) {
            _hotswapDone = true;
            auto actualNetwork = _sharedFutureForActualNetwork.get();
            if (actualNetwork) { // if this was CPU only and no accel, the network would be NULL
                std::cout << "!!! DEBUG: HotSwapRequests !!!" << std::endl;
                _inferRequest = {actualNetwork, actualNetwork->CreateInferRequest()};
                _inferRequest->SetCallback(_callback);
                _anyRequestHasHotSwapped = true;
            }
        }
    }
}

void AutoInferRequest::SetBlobsToDeviceRequest() {
        for (const auto &it : _networkInputs) {
            const auto &name = it.first;
            // this assumes the request is already in BUSY state
            auto blob = GetBlob(name);
            if (_inferRequest->GetBlob(name) != blob)
                _inferRequest->SetBlob(name, blob);
        }
        for (const auto &it : _networkOutputs) {
            const auto &name = it.first;
            // this assumes the request is already in BUSY state
            auto blob = GetBlob(name);
            if (_inferRequest->GetBlob(name) != blob)
                _inferRequest->SetBlob(name, blob);
        }
    }


}  // namespace AutoPlugin

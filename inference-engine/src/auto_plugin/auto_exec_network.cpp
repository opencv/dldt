// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <memory>
#include <map>

#include "ie_metric_helpers.hpp"
#include "auto_exec_network.hpp"
#include "auto_infer_request.hpp"
#define MAGIC_NUM 3
namespace AutoPlugin {
using namespace InferenceEngine;

AutoExecutableNetwork::AutoExecutableNetwork(NetworkFuture cpuFuture,
                                             NetworkFuture acceleratorFuture,
                                             bool          enablePerfCount)
                                             : _cpuFuture(std::move(cpuFuture))
                                             , _acceleratorFuture(std::move(acceleratorFuture))
                                             , _enablePerfCount(enablePerfCount) {
    // both are valid, like AUTO:CPU,GPU
    if (_cpuFuture.valid() && _acceleratorFuture.valid()) {
        try {
            _networkFirstReady = _cpuFuture.get();
            _alreadyActualNetwork = false;
        } catch (const std::exception& e) {
            printf("Warning: load network to CPU failed: %s\n", e.what());
            _networkActualNeeded = _acceleratorFuture.get();
            _alreadyActualNetwork = true;
        }
    } else if (_acceleratorFuture.valid()) {  // only accelerator is valid, like AUTO:GPU
        _networkActualNeeded = _acceleratorFuture.get();
        _alreadyActualNetwork = true;
    } else if (_cpuFuture.valid()) {  // only CPU is valid, like AUTO:CPU
        _networkActualNeeded = _cpuFuture.get();
        _alreadyActualNetwork = true;
    } else {
        IE_THROW() << "No device task available";
    }

    // set request queue to flow control
    SetRequestQueue();
}

AutoExecutableNetwork::~AutoExecutableNetwork() = default;

InferenceEngine::IInferRequestInternal::Ptr AutoExecutableNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                                          OutputsDataMap networkOutputs) {
    InferenceEngine::SoExecutableNetworkInternal network;
    SoIInferRequestInternal inferRequest;
    if (TryGetActualNetwork(network)) {
        inferRequest = {_networkActualNeeded, _networkActualNeeded->CreateInferRequest()};
    } else {
        inferRequest = {_networkFirstReady, _networkFirstReady->CreateInferRequest()};
    }
    return std::make_shared<AutoInferRequest>(_networkInputs, _networkOutputs, inferRequest,
                                              shared_from_this(), _alreadyActualNetwork,
                                              _enablePerfCount);
}

bool AutoExecutableNetwork::TryGetActualNetwork(InferenceEngine::SoExecutableNetworkInternal& soExecNetwork) {
    // try to get actual network
    if (_acceleratorFuture.valid() && _acceleratorFuture.wait_for(std::chrono::nanoseconds(0)) == std::future_status::ready) {
        soExecNetwork = _acceleratorFuture.get();
        _alreadyActualNetwork = true;
        _networkActualNeeded = soExecNetwork;
        // reset the capacity of request queue for actual network
        ResetRequestQueue();
        // reapply config to actual network
        // fixme: GPU doesn't support SetConfig and throw exception
        try {
            _networkActualNeeded->SetConfig(_cacheConfig);
        } catch (...) {
        }
        return true;
    }
    // if already get actual network
    if (_alreadyActualNetwork) {
        soExecNetwork = _networkActualNeeded;
        return true;
    }
    return false;
}

void AutoExecutableNetwork::WaitForActualDevice() const {
    if (_alreadyActualNetwork) {
        return;
    }

    if (_acceleratorFuture.valid()) {
        _networkActualNeeded = _acceleratorFuture.get();
        _alreadyActualNetwork = true;
        // reset the capacity of request queue
        ResetRequestQueue();
    } else {
        IE_THROW() << "Export failed due to no valid executable network";
    }
}

void AutoExecutableNetwork::Export(std::ostream& networkModel) {
    //fixme: the Export  should work with actual device, so we have to wait!!!
    WaitForActualDevice();
    _networkActualNeeded->Export(networkModel);
}

RemoteContext::Ptr AutoExecutableNetwork::GetContext() const {
    // fixme: the GetContext  should work with actual device, so we have to wait!!!
    WaitForActualDevice();
    return _networkActualNeeded->GetContext();
}

InferenceEngine::CNNNetwork AutoExecutableNetwork::GetExecGraphInfo() {
    WaitForActualDevice();
    return _networkActualNeeded->GetExecGraphInfo();
}

Parameter AutoExecutableNetwork::GetMetric(const std::string &name) const {
    // fixme: should we wait actual device? meanwhile it will block inference, how to fix?
//    WaitForActualDevice();
    if (_alreadyActualNetwork) {
        return _networkActualNeeded->GetMetric(name);
    } else {
        return _networkFirstReady->GetMetric(name);
    }
}

void AutoExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config) {
    //fixme: have to store the config and reapply when the networks swapped
    _cacheConfig = config;
    if (_alreadyActualNetwork) {
        _networkActualNeeded->SetConfig(config);
    } else {
        _networkFirstReady->SetConfig(config);
    }
}

Parameter AutoExecutableNetwork::GetConfig(const std::string& name) const {
    //fixme: carefuly select between FirstLoaded and ActuallyNeeded
    return _cacheConfig;
}
// Block until an item becomes available, and then dequeue it.
void AutoExecutableNetwork::PopRequest(int32_t& id) {
    _requests.pop(id);
    // printf("!!! DEBUG: pop request with id(%d), _request.size=%ld !!!\n", id, _requests.size());
}

void AutoExecutableNetwork::PushRequest() const {
    // actually we don't care what it is in the queue, we only need these tokens
    _requests.try_push(MAGIC_NUM);
    // printf("!!! DEBUG: push request with id(%d), _request.size=%ld !!!\n", id, _requests.size());
}

void AutoExecutableNetwork::SetRequestQueue() {
    int optimalNum { 1 };
    if (_networkActualNeeded) {
        optimalNum = _networkActualNeeded->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        _requests.set_capacity(optimalNum);
    } else if (_networkFirstReady) {
        optimalNum = _networkFirstReady->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        _requests.set_capacity(optimalNum);
    } else {
        IE_THROW() << "No available executable network";
    }
    for (int i = 0; i < optimalNum; ++i) {
        // this MAGIC_NUM is used to indicate element in queue
        _requests.try_push(MAGIC_NUM);
    }
}

inline void AutoExecutableNetwork::ResetRequestQueue() const {
    unsigned int optimalNum = _networkActualNeeded->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
    _requests.set_capacity(1);
    _requests.clear();
    for (unsigned int i = 0; i < optimalNum; ++i) {
        PushRequest();
    }
}

}  // namespace AutoPlugin

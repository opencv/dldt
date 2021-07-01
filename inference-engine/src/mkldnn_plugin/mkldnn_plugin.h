// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include "mkldnn_exec_network.h"

#include <string>
#include <map>
#include <unordered_map>
#include <memory>
#include <functional>
#include <vector>
#include <cfloat>

namespace MKLDNNPlugin {

class Engine : public InferenceEngine::IInferencePlugin {
public:
    Engine();
    ~Engine();

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>
    LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;

    void AddExtension(const InferenceEngine::IExtensionPtr& extension) override;

    void SetConfig(const std::map<std::string, std::string> &config) override;

    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::Parameter GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;

private:
    Config engConfig;
    NumaNodesWeights weightsSharing;
    MKLDNNExtensionManager::Ptr extensionManager = std::make_shared<MKLDNNExtensionManager>();
    bool streamsSet = false;

    struct NetworkPerfStats {
        float maxMemTolerance = -1;
        float ratio_compute_convs = 0;
        float ratio_mem_limited_convs = 0;
        float ratio_compute_deconvs = 0;

        static constexpr float memThresholdNotLimited = 1.0f;
        static constexpr float memThresholdAssumeLimited = 0.5f;
        static constexpr float memThresholdAssumeLimitedAVX512 = memThresholdAssumeLimited/2;
        static constexpr float memThresholdAssumeLimitedMuch = memThresholdAssumeLimited/4;
        static constexpr float memThresholdUnknown = FLT_MAX;

        static constexpr float memLimitedRatioThresholdAVX512 = 0.10;
        static constexpr float ALL = 1.0f;
    };
    static NetworkPerfStats NetworkMemBandwidthTolerance(const InferenceEngine::CNNNetwork &network);
};

}  // namespace MKLDNNPlugin

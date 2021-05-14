// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_set>

#include <ie_metric_helpers.hpp>
#include <threading/ie_executor_manager.hpp>
#include <ie_algorithm.hpp>

#include "auto_plugin.hpp"

namespace AutoPlugin {
namespace {
    ConfigType mergeConfigs(ConfigType config, const ConfigType& local) {
        for (auto && kvp : local) {
            config[kvp.first] = kvp.second;
        }
        return config;
    }

    std::string GetNetworkPrecision(const InferenceEngine::CNNNetwork &network) {
        for (auto&& layer : network.getInputsInfo()) {
            auto precision = layer.second->getPrecision();
            auto name = std::string(precision.name());
            if (name == "I8") {
                name = "INT8";
            }
            return name;
        }
        return {};
    }

    DeviceInformation SelectDevice(const InferenceEngine::CNNNetwork &network,
                                   const std::vector<DeviceInformation>& metaDevices,
                                   const std::vector<std::string>& optCap) {
        if (metaDevices.empty()) {
            IE_THROW(NotFound) << "No available device to select in AUTO plugin";
        }
        if (metaDevices.size() == 1) {
            return metaDevices.at(0);
        }

        std::vector<DeviceInformation> CPU;
        std::vector<DeviceInformation> GPU;

        for (auto& item : metaDevices) {
            if (item.deviceName.find("CPU") == 0) {
                CPU.push_back(item);
                continue;
            }
            if (item.deviceName.find("GPU") == 0) {
                GPU.push_back(item);
                continue;
            }
        }

        auto networkPrecision = GetNetworkPrecision(network);
        auto getCap = [&](std::string&& substr){
        auto capability = std::find_if(optCap.begin(), optCap.end(),
                                    [&](const std::string& c)->bool{ return (c.find(substr) != std::string::npos);});
            return capability;
        };

        if (CPU.empty() && GPU.empty()) {
            IE_THROW(NotFound) << "No available device found";
        }

        std::sort(GPU.begin(), GPU.end(), [](const DeviceInformation& a, const DeviceInformation& b)->bool{return b.deviceName < a.deviceName;});

        if (!GPU.empty()) {
            auto capability = getCap("GPU");
            if (capability != optCap.end() && capability->find(networkPrecision) != std::string::npos) {
                return GPU[0];
            }
        }
        if (CPU.empty()) {
            IE_THROW() << "Cannot select any device";
        }
        return CPU[0];
    }
}  // namespace

AutoInferencePlugin::AutoInferencePlugin() {
    _pluginName = "AUTO";
}

IE::IExecutableNetworkInternal::Ptr AutoInferencePlugin::LoadNetwork(const std::string& fileName,
                                                                     const ConfigType&  config) {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
    }

    auto fullConfig = mergeConfigs(_config, config);
    auto metaDevices = GetDeviceChoice(fullConfig);

    // FIXME: always select CPU device now
    DeviceInformation selectedDevice = SelectDevice(metaDevices);
    IE::SoExecutableNetworkInternal executableNetwork;
    try {
        executableNetwork = GetCore()->LoadNetwork(fileName, selectedDevice.deviceName, selectedDevice.config);
    } catch(const IE::Exception &iie) {
        IE_THROW() << "Failed to load network to device named " << selectedDevice.deviceName
                   << " with exception " << iie.what();
    }

    bool enablePerfCounters = false;
    try {
        enablePerfCounters =
            executableNetwork->GetConfig(IE::PluginConfigParams::KEY_PERF_COUNT).as<std::string>() ==
                IE::PluginConfigParams::YES;
    } catch (...) {
    }

    return std::make_shared<AutoExecutableNetwork>(executableNetwork,
                                                   selectedDevice,
                                                   enablePerfCounters);
}

IE::ExecutableNetworkInternal::Ptr AutoInferencePlugin::LoadExeNetworkImpl(const IE::CNNNetwork& network,
                                                                           const ConfigType&     config) {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
    }

    if (network.getFunction() == nullptr) {
        IE_THROW() << "AUTO device supports just ngraph network representation";
    }

    auto fullConfig = mergeConfigs(_config, config);
    auto metaDevices = GetDeviceChoice(fullConfig);
    auto optCap = GetOptimizationCapabilities();

    DeviceInformation selectedDevice = SelectDevice(network, metaDevices, optCap);
    IE::SoExecutableNetworkInternal executableNetwork;
    try {
        executableNetwork = GetCore()->LoadNetwork(network, selectedDevice.deviceName, selectedDevice.config);
    } catch(const IE::Exception &iie) {
        IE_THROW() << "Failed to load network to device named " << selectedDevice.deviceName
                   << " with exception " << iie.what();
    }

    bool enablePerfCounters = false;
    try {
        enablePerfCounters =
            executableNetwork->GetConfig(IE::PluginConfigParams::KEY_PERF_COUNT).as<std::string>() ==
                IE::PluginConfigParams::YES;
    } catch (...) {
    }

    return std::make_shared<AutoExecutableNetwork>(executableNetwork,
                                                   selectedDevice,
                                                   enablePerfCounters);
}

IE::QueryNetworkResult AutoInferencePlugin::QueryNetwork(const IE::CNNNetwork& network, const ConfigType& config) const {
    IE::QueryNetworkResult queryResult = {};
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
    }

    if (network.getFunction() == nullptr) {
        IE_THROW() << "AUTO device supports just ngraph network representation";
    }

    auto fullConfig = mergeConfigs(_config, config);
    auto metaDevices = GetDeviceChoice(fullConfig);
    std::unordered_set<std::string> supportedLayers;
    for (auto&& value : metaDevices) {
        try {
            auto deviceQr = GetCore()->QueryNetwork(network, value.deviceName, value.config);
            std::unordered_set<std::string> deviceSupportedLayers;
            for (auto &&layerQr : deviceQr.supportedLayersMap) {
                deviceSupportedLayers.emplace(layerQr.first);
            }
            supportedLayers = supportedLayers.empty()
                            ? deviceSupportedLayers : (deviceSupportedLayers.empty()
                            ? supportedLayers : IE::details::Intersection(
                                 supportedLayers, deviceSupportedLayers));
            break;
        } catch (...) {
        }
    }

    for (auto&& supportedLayer : supportedLayers) {
        queryResult.supportedLayersMap[supportedLayer] = GetName();
    }
    return queryResult;
}

IE::Parameter AutoInferencePlugin::GetConfig(const std::string& name,
                                             const std::map<std::string, IE::Parameter> & options) const {
    auto it = _config.find(name);
    if (it == _config.end()) {
        IE_THROW() << "Unsupported config key: " << name;
    } else {
        return { it->second };
    }
}

void AutoInferencePlugin::SetConfig(const ConfigType& config) {
    for (auto && kvp : config) {
        _config[kvp.first] = kvp.second;
    }
}

IE::Parameter AutoInferencePlugin::GetMetric(const std::string& name,
                                             const std::map<std::string, IE::Parameter> & options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.emplace_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.emplace_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.emplace_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.emplace_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        std::string device_name = {"Inference Engine AUTO device"};
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, device_name);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities = { "" };
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else {
        IE_THROW() << "Unsupported metric key " << name;
    }
}

std::vector<AutoPlugin::DeviceInformation> AutoInferencePlugin::GetDeviceChoice(const ConfigType&  config) const {
    std::vector<DeviceInformation> metaDevices;
    std::vector<std::string> availableDevices = GetCore()->GetAvailableDevices();

    auto getDeviceConfig = [&] (const DeviceName & deviceWithID) {
        IE::DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        ConfigType tconfig = mergeConfigs(_config, config);

        // set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[IE::PluginConfigParams::KEY_DEVICE_ID] = deviceIDLocal;
        }

        return GetSupportedConfig(tconfig, deviceName);
    };

    for (auto && d : availableDevices) {
        if (d != _pluginName) {
            metaDevices.push_back({ d, getDeviceConfig(d)});
        }
    }

    if (metaDevices.empty()) {
        IE_THROW() << "Please, check environment due to no supported devices can be used";
    }

    return metaDevices;
}

std::vector<std::string> AutoInferencePlugin::GetOptimizationCapabilities() const {
    // FIXME: workaround to get devicelist.
    std::unordered_set<std::string> capabilities;
    std::vector<std::string> queryDeviceLists{"CPU", "GPU"};
    for (auto &item : queryDeviceLists) {
        try {
            std::vector<std::string> device_cap =
                GetCore()->GetMetric(item, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
            for (auto &dc : device_cap) {
                capabilities.insert(dc);
            }
        } catch (...) {
        }
    }
    return {capabilities.begin(), capabilities.end()};
}

//////////////////////////////////// private & protected functions ///////////////////
ConfigType AutoInferencePlugin::GetSupportedConfig(const ConfigType&  config,
                                                   const std::string& deviceName) const {
    std::vector<std::string> supportedConfigKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    ConfigType supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = config.find(key);
        if (config.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    return supportedConfig;
}

// define CreatePluginEngine to create plugin instance
static const IE::Version version = {{2, 1}, CI_BUILD_NUMBER, "AutoPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(AutoInferencePlugin, version)
}  // namespace AutoPlugin

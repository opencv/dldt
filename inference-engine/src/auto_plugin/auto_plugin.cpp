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
#include <transformations/utils/utils.hpp>

#include <auto_plugin/auto_config.hpp>
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
        auto nGraphFunc = network.getFunction();
        bool isINTModel = ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(nGraphFunc);
        if (isINTModel) {
            return "INT8";
        }
        for (auto & node : nGraphFunc->get_ordered_ops()) {
            auto nodeSize = node->inputs().size();
            auto nodeName = node->get_name();
            if (nodeSize != 0) {
                if (nodeName.find("Convolution") != std::string::npos) {
                    auto layerType = node->input(0).get_element_type().get_type_name();
                    if (layerType == "f32")
                        return "FP32";
                    if (layerType == "f16")
                        return "FP16";
                }
            }
        }
        return "FP32";
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

    DeviceInformation selectedDevice;
    IE::SoExecutableNetworkInternal executableNetwork;
    while (!metaDevices.empty()) {
        selectedDevice = SelectDevice(metaDevices);
        try {
            executableNetwork = GetCore()->LoadNetwork(fileName, selectedDevice.deviceName, config);
            break;
        } catch (...) {
            auto eraseDevice = std::find_if(metaDevices.begin(), metaDevices.end(),
                                            [=](const DeviceInformation &d) -> bool { return d.deviceName == selectedDevice.deviceName; });
            if (eraseDevice == metaDevices.end()) {
                IE_THROW() << "Didn't find the selected device name";
            }
            metaDevices.erase(eraseDevice);
            executableNetwork = {};
        }
    }
    if (!executableNetwork) {
        IE_THROW() << "Failed to load network by AUTO plugin";
    }

    bool enablePerfCounters = false;
    try {
        enablePerfCounters =
            executableNetwork->GetConfig(IE::PluginConfigParams::KEY_PERF_COUNT).as<std::string>() ==
                IE::PluginConfigParams::YES;
    }
    catch (...) {
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
    DeviceInformation selectedDevice;
    IE::SoExecutableNetworkInternal executableNetwork;
    while (!metaDevices.empty()) {
        selectedDevice = SelectDevice(network, metaDevices);
        try {
            executableNetwork = GetCore()->LoadNetwork(
                network, selectedDevice.deviceName, selectedDevice.config);
            break;
        } catch (...) {
            auto eraseDevice = std::find_if(metaDevices.begin(), metaDevices.end(),
                                            [=](const DeviceInformation& d)->bool{return d.deviceName == selectedDevice.deviceName;});
            if (eraseDevice == metaDevices.end()) {
                IE_THROW() << "Didn't find the selected device name";
            }
            metaDevices.erase(eraseDevice);
            executableNetwork = {};
        }
    }
    if (!executableNetwork) {
        IE_THROW() << "Failed to load network by AUTO plugin";
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
        std::vector<std::string> capabilities = GetOptimizationCapabilities();
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else {
        IE_THROW() << "Unsupported metric key " << name;
    }
}

//////////////////////////////////// private & protected functions ///////////////////
std::vector<AutoPlugin::DeviceInformation> AutoInferencePlugin::GetDeviceChoice(const ConfigType&  config) const {
    std::vector<DeviceInformation> metaDevices;
    std::vector<std::string> availableDevices;

    auto deviceListConfig = config.find(IE::AutoConfigParams::KEY_AUTO_DEVICE_LIST);
    if (deviceListConfig == config.end()) {
        availableDevices = GetCore()->GetAvailableDevices();
    } else {
        availableDevices = IE::DeviceIDParser::getHeteroDevices(deviceListConfig->second);
    }

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
            for (auto &cap : device_cap) {
                // For CPU test SetBlobOfKindTest::CompareWithRefs which checks BATCHED_BLOB capability,
                // and AUTO select CPU but not GPU (GPU has this capability).
                if (cap == METRIC_VALUE(BATCHED_BLOB)) {
                    continue;
                }
                capabilities.insert(cap);
            }
        } catch (...) {
        }
    }
    return {capabilities.begin(), capabilities.end()};
}

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

DeviceInformation AutoInferencePlugin::SelectDevice(const std::vector<DeviceInformation>& metaDevices) {
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

    if (CPU.empty() && GPU.empty()) {
        IE_THROW(NotFound) << "No available device found";
    }

    // Sort GPU by name: GPU.2 > GPU.1 > GPU.0 > GPU, so we always choose the GPU[0] as best device
    std::sort(GPU.begin(), GPU.end(), [](const DeviceInformation& a, const DeviceInformation& b)->bool{return b.deviceName < a.deviceName;});

    if (!GPU.empty()) {
        return GPU[0];
    }
    if (CPU.empty()) {
        IE_THROW() << "Cannot select any device";
    }
    return CPU[0];
}

DeviceInformation AutoInferencePlugin::SelectDevice(const InferenceEngine::CNNNetwork&    network,
                                                    const std::vector<DeviceInformation>& metaDevices) {
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
    if (CPU.empty() && GPU.empty()) {
        IE_THROW(NotFound) << "No available device found";
    }

    // Sort GPU by name: GPU.2 > GPU.1 > GPU.0 > GPU, so we always choose the GPU[0] as best device
    std::sort(GPU.begin(), GPU.end(), [](const DeviceInformation& a, const DeviceInformation& b)->bool{return b.deviceName < a.deviceName;});

    for (auto&& item : GPU) {
        std::vector<std::string> capability = GetCore()->GetMetric(item.deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        auto res = std::find(capability.begin(), capability.end(), networkPrecision);
        if (res != capability.end()) {
            return item;
        }
    }

    if (CPU.empty()) {
        IE_THROW() << "Cannot select any device";
    }
    return CPU[0];
}

// define CreatePluginEngine to create plugin instance
static const IE::Version version = {{2, 1}, CI_BUILD_NUMBER, "AutoPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(AutoInferencePlugin, version)
}  // namespace AutoPlugin

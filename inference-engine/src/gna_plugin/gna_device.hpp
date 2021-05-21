// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <thread>

#include <ie_common.h>

#if GNA_LIB_VER == 2
#include "gna2-common-api.h"
#include "gna2-inference-api.h"
#include "gna2-instrumentation-api.h"

#include "gna2-memory-api.h"
#include "gna2-model-api.h"
#include "gna2-model-suecreek-header.h"
#else
#include <gna-api.h>
#include "gna-api-dumper.h"
#include "gna-api-instrumentation.h"
#endif

enum GnaWaitStatus : int {
    GNA_REQUEST_COMPLETED = 0,  // and removed from GNA library queue
    GNA_REQUEST_ABORTED = 1,    // for QoS purposes
    GNA_REQUEST_PENDING = 2     // for device busy purposes
};


struct SingleGnaPerf
{
    std::string graphPartName;
    uint32_t perfConfigId;
    std::array<uint64_t, 2> perfData;
};
using GNAPluginNS::Gna2ModelWithMeta;
/**
 * holds gna - style handle in RAII way
 */
class GNADeviceHelper {
    static std::mutex acrossPluginsSync;
    static std::string decoratedGnaLibVersion() {
        static std::string gnaLibraryVersion{ ", GNA library version: " + GNADeviceHelper::GetGnaLibraryVersion() };
        return gnaLibraryVersion;
    }
#if GNA_LIB_VER == 1
    intel_gna_status_t nGNAStatus = GNA_NOERROR;
    intel_gna_handle_t nGNAHandle = 0;
    intel_gna_perf_t nGNAPerfResults;
    intel_gna_perf_t nGNAPerfResultsTotal;
#else
    uint32_t nGnaDeviceIndex = 0;
    bool swExactMode = false;
    Gna2DeviceVersion detectedGnaDevVersion = Gna2DeviceVersionSoftwareEmulation;
    std::string executionTarget;
    std::string compileTarget;
    bool isGnaLibVersion2_1 = false;

    static const uint32_t TotalGna2InstrumentationPoints = 2;
    Gna2InstrumentationPoint gna2InstrumentationPoints[TotalGna2InstrumentationPoints] = {
        Gna2InstrumentationPointHwTotalCycles,
        Gna2InstrumentationPointHwStallCycles };

    std::set<uint32_t> unwaitedRequestIds;
#define MAX_TIMEOUT 500000
#endif
    bool isPerformanceMeasuring = false;
    bool perLayerPerformanceMeasuring = false;
    bool deviceOpened = false;
public:
#if GNA_LIB_VER == 1
    explicit GNADeviceHelper(uint8_t lib_async_n_threads = 1,
                            bool use_openmp = false,
                            bool isPerformanceMeasuring = false) :
                                    isPerformanceMeasuring(isPerformanceMeasuring) {
#else
    explicit GNADeviceHelper(std::string executionTargetIn = "",
         std::string compileTargetIn = "",
         bool swExactModeIn = false,
         uint8_t lib_async_n_threads = 1,
         bool use_openmp = false,
         bool isPerformanceMeasuring = false) :
         perLayerPerformanceMeasuring(isPerformanceMeasuring),
         swExactMode(swExactModeIn),
         executionTarget(executionTargetIn),
         compileTarget(compileTargetIn),
         isPerformanceMeasuring(isPerformanceMeasuring),
         nGnaDeviceIndex{selectGnaDevice()} {
#endif
        open(lib_async_n_threads);
        initGnaPerfCounters();

        // check GNA Library version
        const auto gnaLibVersion = GetGnaLibraryVersion();
#if GNA_LIB_VER == 2
        if (gnaLibVersion.rfind("2.1", 0) == 0) {
            isGnaLibVersion2_1 = true;
        }
#endif

        if (use_openmp) {
            uint8_t num_cores = std::thread::hardware_concurrency();
            setOMPThreads((num_cores != 0) ? num_cores : 1);
        }
    }

    GNADeviceHelper(const GNADeviceHelper&) = delete;
    GNADeviceHelper& operator= (const GNADeviceHelper&) = delete;
    ~GNADeviceHelper() {
        if (deviceOpened) {
            close();
        }
    }

    uint8_t *alloc(uint32_t size_requested, uint32_t *size_granted);

#if GNA_LIB_VER == 1
    void propagateSync(const intel_nnet_type_t *pNeuralNetwork,
                       const uint32_t *pActiveIndices,
                       uint32_t nActiveIndices,
                       intel_gna_proc_t nGNAProcType);

    uint32_t propagate(const intel_nnet_type_t *pNeuralNetwork,
                       const uint32_t *pActiveIndices,
                       uint32_t nActiveIndices,
                       intel_gna_proc_t nGNAProcType);
#else
    void setUpActiveList(unsigned req_config_id, uint32_t layerIndex, uint32_t* ptr_active_indices, uint32_t num_active_indices);
    void propagateSync(const uint32_t requestConfigId, Gna2AccelerationMode gna2AccelerationMode);
    uint32_t propagate(const uint32_t requestConfigId, Gna2AccelerationMode gna2AccelerationMode);
    std::vector<std::pair<uint32_t, std::string> > createModels(Gna2ModelWithMeta& gnaModel) const;
    uint32_t createModel(Gna2ModelWithMeta& gnaModel);
    void releaseModel(const uint32_t model_id);
    std::vector<uint32_t> createRequestConfigs(const std::vector<std::pair<uint32_t, std::string>>& modelsIdAndName);
    uint32_t createRequestConfig(const uint32_t modelId);
    static uint32_t getNumberOfGnaDevices();
    static uint32_t selectGnaDevice();
    static bool isGnaHw(const Gna2DeviceVersion dev) {
        return Gna2DeviceVersionSoftwareEmulation != dev;
    }
    bool hasGnaHw() const {
        return isGnaHw(detectedGnaDevVersion);
    }
    static bool isUpTo20HwGnaDevice(const Gna2DeviceVersion dev) {
        return dev <= Gna2DeviceVersion2_0 && isGnaHw(dev);
    }
    bool enforceLegacyCnnNeeded() const;
    static void checkGna2Status(Gna2Status status, const std::string& from);
    static void checkGna2Status(Gna2Status status, const Gna2Model& gnaModel, uint32_t operationOffset = 0);
    std::vector<SingleGnaPerf> partialPerformanceResults;
    std::map< uint32_t, std::vector< uint32_t> > allGnaReqestIdsMap;
    std::map< uint32_t, std::vector< uint32_t> > allGnaReqConfigIdsMap;
    std::map< uint32_t, std::vector<std::pair<uint32_t, std::string> > > allGnaModelsIdsMap;
#endif
    GnaWaitStatus wait(uint32_t id, int64_t millisTimeout = MAX_TIMEOUT);

    struct DumpResult {
#if GNA_LIB_VER == 2
        Gna2ModelSueCreekHeader header;
#else
        intel_gna_model_header header;
#endif
        std::shared_ptr<void> model;
    };

    const void * dumpXNNROPtr = nullptr;
    uint32_t dumpXNNROSize = 0;

#if GNA_LIB_VER == 1
    DumpResult dumpXnn(const intel_nnet_type_t *pNeuralNetwork,
                 const uint32_t *pActiveIndices,
                 uint32_t nActiveIndices);
    intel_gna_status_t getGNAStatus() const noexcept {
        return nGNAStatus;
    }
    void updateGnaPerfCounters();
#else

    DumpResult dumpXnn(const uint32_t modelId);
    void dumpXnnForDeviceVersion(const uint32_t modelId,
        std::ostream & outStream,
        Gna2DeviceVersion targetDeviceVersion);
#endif
    void free(void * ptr);

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getGnaPerfCounters();
    static std::string GetGnaLibraryVersion();
 private:
    void open(uint8_t const n_threads);

    void close();
    static std::string getGnaLibraryVersionPrivate();
#if GNA_LIB_VER == 1
    void checkStatus() const;
#else
    static const std::map <Gna2ItemType, const std::string> errorTypes;
    static const std::map <Gna2ErrorType, const std::string> errorReasons;
    static const std::map <Gna2OperationType, const std::string> operationTypes;
    static const std::map <const std::pair<Gna2OperationType, int32_t>, const std::string > operandTypes;

    static void enforceLegacyCnns(Gna2Model& gnaModel);
    Gna2DeviceVersion parseDeclaredTarget(std::string target, const bool execTarget) const;
    Gna2DeviceVersion getDefaultTarget() const;
    Gna2DeviceVersion getTargetDevice(bool execTarget) const;
#endif
    void setOMPThreads(uint8_t const n_threads);

    void initGnaPerfCounters() {
#if GNA_LIB_VER == 1
        nGNAPerfResults = {{0, 0, 0, 0, 0, 0, 0}, {0, 0}, {0, 0, 0}, {0, 0}};
        nGNAPerfResultsTotal = {{0, 0, 0, 0, 0, 0, 0}, {0, 0}, {0, 0, 0}, {0, 0}};
#endif
    }

#if GNA_LIB_VER == 2
    void initAndAssignGnaPerfCounters(const std::string& layerName, const uint32_t requestConfigId, SingleGnaPerf &);
#endif

};  // NOLINT

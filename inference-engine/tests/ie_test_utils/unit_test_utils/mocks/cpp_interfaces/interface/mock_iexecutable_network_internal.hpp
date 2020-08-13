// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <gmock/gmock.h>

#include "ie_input_info.hpp"
#include "ie_icnn_network.hpp"
#include "ie_iexecutable_network.hpp"
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>

#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_infer_request_internal.hpp"

using namespace InferenceEngine;

IE_SUPPRESS_DEPRECATED_START
class MockIExecutableNetworkInternal : public IExecutableNetworkInternal {
public:
    MOCK_CONST_METHOD0(GetOutputsInfo, ConstOutputsDataMap());
    MOCK_CONST_METHOD0(GetInputsInfo, ConstInputsDataMap());
    MOCK_METHOD1(CreateInferRequest, void(IInferRequest::Ptr &));
    MOCK_METHOD1(Export, void(const std::string &));
    void Export(std::ostream &) override {};
    MOCK_METHOD0(QueryState, std::vector<IMemoryStateInternal::Ptr>());
    MOCK_METHOD1(GetExecGraphInfo, void(ICNNNetwork::Ptr &));

    MOCK_METHOD1(SetConfig, void(const std::map<std::string, Parameter> &config));
    MOCK_CONST_METHOD2(GetConfig, void(const std::string &name, Parameter &result));
    MOCK_CONST_METHOD2(GetMetric, void(const std::string &name, Parameter &result));
    MOCK_CONST_METHOD1(GetContext, void(RemoteContext::Ptr &pContext));
};
IE_SUPPRESS_DEPRECATED_END
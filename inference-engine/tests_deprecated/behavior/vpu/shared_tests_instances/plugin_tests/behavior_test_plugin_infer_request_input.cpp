// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_infer_request_input.hpp"
#include "vpu_test_data.hpp"

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginTestInferRequestInput, ValuesIn(allInputSupportedValues),
                        getTestCaseName);

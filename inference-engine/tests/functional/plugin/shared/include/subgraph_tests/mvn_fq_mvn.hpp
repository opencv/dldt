// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/mvn_fq_mvn.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MvnFqMvnSubgraphTest, CompareWithRefs){
    Run();
};

}  // namespace SubgraphTestsDefinitions

// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "ie_core.hpp"
#include "pugixml.hpp"

namespace ExecutionGraphTests {

#ifndef IR_SERIALIZATION_MODELS_PATH  // should be already defined by cmake
#define IR_SERIALIZATION_MODELS_PATH ""
#endif

using ExecGraphSerializationParam = std::tuple<std::pair<std::string, std::string>, std::string>;

class ExecGraphSerializationTest : public ::testing::Test, public testing::WithParamInterface<ExecGraphSerializationParam> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ExecGraphSerializationParam> obj);
    void SetUp() override;
    void TearDown() override;

private:
    // walker traverse (DFS) xml document and store layer & data nodes in
    // vector which is later used for comparison
    struct exec_graph_walker : pugi::xml_tree_walker {
        std::vector<pugi::xml_node> nodes;
        virtual bool for_each(pugi::xml_node &node);
    };

    // compare_docs() helper
    std::pair<bool, std::string> compare_nodes(const pugi::xml_node &node1,
                                               const pugi::xml_node &node2);

protected:
    // checks if two exec graph xml's are equivalent:
    // - the same count of <layer> and <data> nodes
    // - the same count of attributes of each node
    // - the same name of each attribute (value is not checked, since it can differ
    // beetween different devices)
    std::pair<bool, std::string> compare_docs(const pugi::xml_document &doc1,
                                              const pugi::xml_document &doc2);

    std::string deviceName;
    std::string m_out_xml_path, m_out_bin_path;
    std::string source_model, expected_model;
};
} // namespace ExecutionGraphTests

﻿// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <fstream>

#include <ie_core.hpp>

namespace {
    std::string model_path(const char* model) {
        std::string path = ONNX_TEST_MODELS;
        path += "support_test/";
        path += model;
        return path;
    }
}

TEST(ONNXReader_ModelSupported, basic_model) {
    // this model is a basic ONNX model taken from ngraph's unit test (add_abc.onnx)
    // it contains the minimum number of fields required to accept this file as a valid model
    EXPECT_NO_THROW(InferenceEngine::Core{}.ReadNetwork(model_path("supported/basic.onnx")));
}

TEST(ONNXReader_ModelSupported, basic_reverse_fields_order) {
    // this model contains the same fields as basic.onnx but serialized in reverse order
    EXPECT_NO_THROW(InferenceEngine::Core{}.ReadNetwork(model_path("supported/basic_reverse_fields_order.onnx")));
}

TEST(ONNXReader_ModelSupported, more_fields) {
    // this model contains some optional fields (producer_name and doc_string) but 5 fields in total
    EXPECT_NO_THROW(InferenceEngine::Core{}.ReadNetwork(model_path("supported/more_fields.onnx")));
}

TEST(ONNXReader_ModelUnsupported, no_graph_field) {
    // this model contains only 2 fields (it doesn't contain a graph in particular)
    EXPECT_THROW(InferenceEngine::Core{}.ReadNetwork(model_path("unsupported/no_graph_field.onnx")),
                 InferenceEngine::details::InferenceEngineException);
}

TEST(ONNXReader_ModelUnsupported, incorrect_onnx_field) {
    // in this model the second field's key is F8 (field number 31) which is doesn't exist in ONNX
    // this  test will have to be changed if the number of fields in onnx.proto
    // (ModelProto message definition) ever reaches 31 or more
    EXPECT_THROW(InferenceEngine::Core{}.ReadNetwork(model_path("unsupported/incorrect_onnx_field.onnx")),
                 InferenceEngine::details::InferenceEngineException);
}

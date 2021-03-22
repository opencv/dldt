// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include "../../../ngraph/core/include/ngraph/partial_shape.hpp"

namespace benchmark_app {
    struct InputInfo {
        InferenceEngine::Precision precision;
        ngraph::PartialShape partialShape;
        InferenceEngine::SizeVector blobShape;
        std::string layout;
        bool isImage() const;
        bool isImageInfo() const;
        size_t getDimentionByLayout(char character) const;
        size_t width() const;
        size_t height() const;
        size_t channels() const;
        size_t batch() const;
        size_t depth() const;
    };
    using InputsInfo = std::map<std::string, InputInfo>;
}

std::vector<std::string> parseDevices(const std::string& device_string);
uint32_t deviceDefaultDeviceDurationInSeconds(const std::string& device);
std::map<std::string, std::string> parseNStreamsValuePerDevice(const std::vector<std::string>& devices,
                                                               const std::string& values_string);
#if 0
InferenceEngine::ICNNNetwork::InputPartialShapes parseShapes(const std::string& shapes_string);
bool updateShapes(InferenceEngine::ICNNNetwork::InputPartialShapes& shapes,
                  const std::string shapes_string, const InferenceEngine::InputsDataMap& input_info);
bool adjustShapesBatch(InferenceEngine::ICNNNetwork::InputPartialShapes& shapes,
                       const size_t batch_size, const InferenceEngine::InputsDataMap& input_info);
std::string getShapesString(const InferenceEngine::ICNNNetwork::InputPartialShapes& shapes);
#endif
std::string getShapesString(const InferenceEngine::ICNNNetwork::InputPartialShapes& shapes);
size_t getBatchSize(const benchmark_app::InputsInfo& inputs_info);
std::vector<std::string> split(const std::string &s, char delim);

template <typename T>
std::map<std::string, std::string> parseInputParameters(const std::string parameter_string,
                                                        const std::map<std::string, T>& input_info) {
    // Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all inputs)
    std::map<std::string, std::string> return_value;
    std::string search_string = parameter_string;
    auto start_pos = search_string.find_first_of('[');
    while (start_pos != std::string::npos) {
        auto end_pos = search_string.find_first_of(']');
        if (end_pos == std::string::npos)
            break;
        auto input_name = search_string.substr(0, start_pos);
        auto input_value = search_string.substr(start_pos + 1, end_pos - start_pos - 1);
        if (!input_name.empty()) {
            return_value[input_name] = input_value;
        } else {
            for (auto& item : input_info) {
                return_value[item.first] = input_value;
            }
        }
        search_string = search_string.substr(end_pos + 1);
        if (search_string.empty() || search_string.front() != ',')
            break;
        search_string = search_string.substr(1);
        start_pos = search_string.find_first_of('[');
    }
    if (!search_string.empty())
        throw std::logic_error("Can't parse input parameter string: " + parameter_string);
    return return_value;
}

template <typename T>
benchmark_app::InputsInfo getInputsInfo(const std::string& shape_string,
                                        const std::string& layout_string,
                                        const size_t batch_size,
                                        const std::string& blobs_shape_string,
                                        const std::map<std::string, T>& input_info,
                                        bool& reshape_required) {
    std::map<std::string, std::string> shape_map = parseInputParameters(shape_string, input_info);
    std::map<std::string, std::string> blobs_shape_map = parseInputParameters(blobs_shape_string, input_info);
    std::map<std::string, std::string> layout_map = parseInputParameters(layout_string, input_info);
    reshape_required = false;
    benchmark_app::InputsInfo info_map;
    for (auto& item : input_info) {
        benchmark_app::InputInfo info;
        auto name = item.first;
        auto descriptor = item.second->getTensorDesc();
        // Precision
        info.precision = descriptor.getPrecision();
        // Partial Shape
        if (shape_map.count(name)) {
            std::vector<ngraph::Dimension> parsed_shape;
            for (auto& dim : split(shape_map[name], ',')) {
                if (dim == "?" || dim == "-1") {
                    parsed_shape.push_back(ngraph::Dimension());
                } else {
                    const std::string range_divider = "..";
                    size_t range_index = dim.find(range_divider);
                    if (range_index != std::string::npos) {
                        std::string min = dim.substr(0, range_index);
                        std::string max = dim.substr(range_index + range_divider.length());
                        parsed_shape.push_back(ngraph::Dimension(
                                min.empty() ? 0 : std::stoi(min),
                                max.empty() ? ngraph::Interval::s_max : std::stoi(max)));
                    } else {
                        parsed_shape.push_back(std::stoi(dim));
                    }
                }
            }
            info.partialShape = parsed_shape;
            reshape_required = true;
        } else {
            info.partialShape = descriptor.getPartialShape();
        }
        // Blob Shape
        if (blobs_shape_map.count(name)) {
            std::vector<size_t> parsed_shape;
            for (auto& dim : split(blobs_shape_map.at(name), ',')) {
                parsed_shape.push_back(std::stoi(dim));
            }
            info.blobShape = parsed_shape;
        } else {
            info.blobShape = descriptor.getDims();
        }
        // Layout
        if (layout_map.count(name)) {
            info.layout = layout_map.at(name);
            std::transform(info.layout.begin(), info.layout.end(), info.layout.begin(), ::toupper);
        } else {
            std::stringstream ss;
            ss << descriptor.getLayout();
            info.layout = ss.str();
        }
        // Update shape with batch if needed
        // Update blob shape only not affecting network shape to trigger dynamic batch size case
        if (batch_size != 0) {
            std::size_t batch_index = info.layout.find("N");
            if ((batch_index != std::string::npos) && (info.blobShape.at(batch_index) != batch_size)) {
                info.blobShape[batch_index] = batch_size;
                reshape_required = true;
            }
        }
        info_map[name] = info;
    }
    return info_map;
}

template <typename T>
benchmark_app::InputsInfo getInputsInfo(const std::string& shape_string,
                                        const std::string& layout_string,
                                        const size_t batch_size,
                                        const std::string& blobs_shape_string,
                                        const std::map<std::string, T>& input_info) {
    bool reshape_required = false;
    return getInputsInfo<T>(shape_string, layout_string, batch_size, blobs_shape_string, input_info, reshape_required);
}

#ifdef USE_OPENCV
void dump_config(const std::string& filename,
                 const std::map<std::string, std::map<std::string, std::string>>& config);
void load_config(const std::string& filename,
                 std::map<std::string, std::map<std::string, std::string>>& config);
#endif
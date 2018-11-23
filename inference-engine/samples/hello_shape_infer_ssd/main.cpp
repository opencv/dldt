// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <samples/common.hpp>
#include <ext_list.hpp>

#include "shape_infer_extension.hpp"

using namespace InferenceEngine;

int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (argc != 5) {
            std::cout << "Usage : ./hello_shape_infer_ssd <path_to_model> <path_to_image> <device> <batch>"
                      << std::endl;
            return EXIT_FAILURE;
        }
        const std::string input_model{argv[1]};
        const std::string input_image_path{argv[2]};
        const std::string device_name{argv[3]};
        const size_t batch_size{std::stoul(argv[4])};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(device_name);
        IExtensionPtr cpuExtension, inPlaceExtension;
        if (device_name == "CPU") {
            cpuExtension = std::make_shared<Extensions::Cpu::CpuExtensions>();
            inPlaceExtension = std::make_shared<InPlaceExtension>();
            plugin.AddExtension(cpuExtension);
            // register sample's custom kernel (CustomReLU)
            plugin.AddExtension(inPlaceExtension);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        CNNNetReader network_reader;
        network_reader.ReadNetwork(input_model);
        network_reader.ReadWeights(input_model.substr(0, input_model.size() - 4) + ".bin");
        CNNNetwork network = network_reader.getNetwork();

        OutputsDataMap outputs_info(network.getOutputsInfo());
        InputsDataMap inputs_info(network.getInputsInfo());
        if (inputs_info.size() != 1 && outputs_info.size() != 1)
            throw std::logic_error("Sample supports clean SSD network with one input and one output");

        // --------------------------- Resize network to match image sizes and given batch----------------------
        if (device_name == "CPU") {
            // register shape inference functions (SpatialTransformer) from CPU Extension
            network.AddExtension(cpuExtension);
            // register sample's custom shape inference (CustomReLU)
            network.AddExtension(inPlaceExtension);
        }
        auto input_shapes = network.getInputShapes();
        std::string input_name;
        SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        cv::Mat image = cv::imread(input_image_path);
        input_shape[0] = batch_size;
        input_shape[2] = image.rows;
        input_shape[3] = image.cols;
        input_shapes[input_name] = input_shape;
        std::cout << "Resizing network to the image size = [" << image.rows << "x" << image.cols << "] "
                  << "with batch = " << batch_size << std::endl;
        network.reshape(input_shapes);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        InputInfo::Ptr input_info;
        std::tie(input_name, input_info) = *inputs_info.begin();
        input_info->setLayout(Layout::NCHW);
        input_info->setPrecision(Precision::U8);
        // --------------------------- Prepare output blobs ----------------------------------------------------
        DataPtr output_info;
        std::string output_name;
        std::tie(output_name, output_info) = *outputs_info.begin();
        if (output_info->creatorLayer.lock()->type != "DetectionOutput")
            throw std::logic_error("Can't find a DetectionOutput layer in the topology");
        const SizeVector output_shape = output_info->getTensorDesc().getDims();
        const int max_proposal_count = output_shape[2];
        const int object_size = output_shape[3];
        if (object_size != 7) {
            throw std::logic_error("Output item should have 7 as a last dimension");
        }
        if (output_shape.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD model");
        }
        if (output_info == nullptr) {
            THROW_IE_EXCEPTION << "[SAMPLES] shared_ptr ouput_info == nullptr";
        }

        output_info->setPrecision(Precision::FP32);

        auto dumpVec = [](const SizeVector& vec) -> std::string {
            if (vec.empty()) return "[]";
            std::stringstream oss;
            oss << "[" << vec[0];
            for (size_t i = 1; i < vec.size(); i++) oss << "," << vec[i];
            oss << "]";
            return oss.str();
        };
        std::cout << "Resulting input shape = " << dumpVec(input_shape) << std::endl;
        std::cout << "Resulting output shape = " << dumpVec(output_shape) << std::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        Blob::Ptr input = infer_request.GetBlob(input_name);
        for (int b = 0; b < batch_size; b++) {
            matU8ToBlob<uint8_t>(image, input, b);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference --------------------------------------------------------
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output ------------------------------------------------------
        Blob::Ptr output = infer_request.GetBlob(output_name);
        const float* detection = output->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

        /* Each detection has image_id that denotes processed image */
        for (int cur_proposal = 0; cur_proposal < max_proposal_count; cur_proposal++) {
            float image_id = detection[cur_proposal * object_size + 0];
            float label = detection[cur_proposal * object_size + 1];
            float confidence = detection[cur_proposal * object_size + 2];
            /* CPU and GPU plugins have difference in DetectionOutput layer, so we need both checks */
            if (image_id < 0 || confidence == 0) {
                continue;
            }

            float xmin = detection[cur_proposal * object_size + 3] * image.cols;
            float ymin = detection[cur_proposal * object_size + 4] * image.rows;
            float xmax = detection[cur_proposal * object_size + 5] * image.cols;
            float ymax = detection[cur_proposal * object_size + 6] * image.rows;

            if (confidence > 0.5) {
                /** Drawing only objects with >50% probability **/
                std::ostringstream conf;
                conf << ":" << std::fixed << std::setprecision(3) << confidence;
                cv::rectangle(image, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), cv::Scalar(0, 0, 255));
                std::cout << "[" << cur_proposal << "," << label << "] element, prob = " << confidence <<
                          ", bbox = (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << ", batch id = "
                          << image_id << std::endl;
            }
        }
        cv::imwrite("hello_shape_infer_ssd_output.jpg", image);
        std::cout << "The resulting image was saved in the file: hello_shape_infer_ssd_output.jpg" << std::endl;

        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

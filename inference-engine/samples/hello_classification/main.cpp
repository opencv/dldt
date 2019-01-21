// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iomanip>
#include <vector>
#include <memory>
#include <string>
#include <cstdlib>

#ifdef UNICODE
#include <tchar.h>
#endif

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;

#ifndef UNICODE
#define tcout std::cout
#define _T(STR) STR
#else
#define tcout std::wcout
#endif

#ifndef UNICODE
int main(int argc, char *argv[]) {
#else
int wmain(int argc, wchar_t *argv[]) {
#endif
    try {
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (argc != 3) {
            tcout << _T("Usage : ./hello_classification <path_to_model> <path_to_image>") << std::endl;
            return EXIT_FAILURE;
        }

        const file_name_t input_model{argv[1]};
        const file_name_t input_image_path{argv[2]};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        PluginDispatcher dispatcher({_T("../../../lib/intel64"), _T("")});
        InferencePlugin plugin(dispatcher.getSuitablePlugin(TargetDevice::eCPU));
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        CNNNetReader network_reader;
        network_reader.ReadNetwork(fileNameToString(input_model));
        network_reader.ReadWeights(fileNameToString(input_model).substr(0, input_model.size() - 4) + ".bin");
        network_reader.getNetwork().setBatchSize(1);
        CNNNetwork network = network_reader.getNetwork();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
        std::string input_name = network.getInputsInfo().begin()->first;

        input_info->setLayout(Layout::NCHW);
        input_info->setPrecision(Precision::U8);

        // --------------------------- Prepare output blobs ----------------------------------------------------
        DataPtr output_info = network.getOutputsInfo().begin()->second;
        std::string output_name = network.getOutputsInfo().begin()->first;

        output_info->setPrecision(Precision::FP32);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------

        cv::Mat image = cv::imread(fileNameToString(input_image_path));

        /* Resize manually and copy data from the image to the input blob */
        Blob::Ptr input = infer_request.GetBlob(input_name);
        auto input_data = input->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();

        cv::resize(image, image, cv::Size(input_info->getTensorDesc().getDims()[3], input_info->getTensorDesc().getDims()[2]));

        size_t channels_number = input->getTensorDesc().getDims()[1];
        size_t image_size = input->getTensorDesc().getDims()[3] * input->getTensorDesc().getDims()[2];

        for (size_t pid = 0; pid < image_size; ++pid) {
            for (size_t ch = 0; ch < channels_number; ++ch) {
                input_data[ch * image_size + pid] = image.at<cv::Vec3b>(pid)[ch];
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference --------------------------------------------------------
        /* Running the request synchronously */
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output ------------------------------------------------------
        Blob::Ptr output = infer_request.GetBlob(output_name);
        auto output_data = output->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

        std::vector<unsigned> results;
        /*  This is to sort output probabilities and put it to results vector */
        TopResults(10, *output, results);

        std::cout << std::endl << "Top 10 results:" << std::endl << std::endl;
        for (size_t id = 0; id < 10; ++id) {
            std::cout.precision(7);
            auto result = output_data[results[id]];
            std::cout << std::left << std::fixed << result << " label #" << results[id] << std::endl;
        }
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

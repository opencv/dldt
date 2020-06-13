// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_network_reader.hpp"

#include <details/ie_so_pointer.hpp>
#include <details/ie_mmap_allocator.hpp>
#include <file_utils.h>
#include <ie_blob_stream.hpp>
#include <ie_profiling.hpp>
#include <ie_reader.hpp>

#include <fstream>
#include <istream>
#include <mutex>
#include <map>

namespace InferenceEngine {

namespace details {

/**
 * @brief This class defines the name of the fabric for creating an IReader object in DLL
 */
template <>
class SOCreatorTrait<IReader> {
public:
    /**
     * @brief A name of the fabric for creating IReader object in DLL
     */
    static constexpr auto name = "CreateReader";
};

}  // namespace details

/**
 * @brief This class is a wrapper for reader interfaces
 */
class Reader: public IReader {
private:
    InferenceEngine::details::SOPointer<IReader> ptr;
    std::once_flag readFlag;
    std::string name;
    std::string location;

    InferenceEngine::details::SOPointer<IReader> getReaderPtr() {
        std::call_once(readFlag, [&] () {
            FileUtils::FilePath libraryName = FileUtils::toFilePath(location);
            FileUtils::FilePath readersLibraryPath = FileUtils::makeSharedLibraryName(getInferenceEngineLibraryPath(), libraryName);

            if (!FileUtils::fileExist(readersLibraryPath)) {
                THROW_IE_EXCEPTION << "Please, make sure that Inference Engine ONNX reader library "
                    << FileUtils::fromFilePath(::FileUtils::makeSharedLibraryName({}, libraryName)) << " is in "
                    << getIELibraryPath();
            }
            ptr = InferenceEngine::details::SOPointer<IReader>(readersLibraryPath);
        });

        return ptr;
    }

    InferenceEngine::details::SOPointer<IReader> getReaderPtr() const {
        return const_cast<Reader*>(this)->getReaderPtr();
    }

    void Release() noexcept override {
        delete this;
    }

public:
    using Ptr = std::shared_ptr<Reader>;
    Reader(const std::string& name, const std::string location): name(name), location(location) {}
    bool supportModel(std::istream& model) const override {
        auto reader = getReaderPtr();
        return reader->supportModel(model);
    }
    CNNNetwork read(std::istream& model, const std::vector<IExtensionPtr>& exts) const override {
        auto reader = getReaderPtr();
        return reader->read(model, exts);
    }
    CNNNetwork read(std::istream& model, const Blob::CPtr& weights, const std::vector<IExtensionPtr>& exts) const override {
        auto reader = getReaderPtr();
        return reader->read(model, weights, exts);
    }
    std::vector<std::string> getDataFileExtensions() const override {
        auto reader = getReaderPtr();
        return reader->getDataFileExtensions();
    }
    std::string getName() const {
        return name;
    }
};

namespace {

// Extension to plugins creator
std::multimap<std::string, Reader::Ptr> readers;

void registerReaders() {
    IE_PROFILING_AUTO_SCOPE(details::registerReaders)
    static bool initialized = false;
    static std::mutex readerMutex;
    std::lock_guard<std::mutex> lock(readerMutex);
    if (initialized) return;
    // TODO: Read readers info from XML
#ifdef ONNX_IMPORT_ENABLE
    auto onnxReader = std::make_shared<Reader>("ONNX", std::string("inference_engine_onnx_reader") + std::string(IE_BUILD_POSTFIX));
    readers.emplace("onnx", onnxReader);
    readers.emplace("prototxt", onnxReader);
#endif
    auto irReader = std::make_shared<Reader>("IR", std::string("inference_engine_ir_reader") + std::string(IE_BUILD_POSTFIX));
    readers.emplace("xml", irReader);
    initialized = true;
}

}  // namespace

CNNNetwork details::ReadNetwork(const std::string& modelPath, const std::string& binPath, const std::vector<IExtensionPtr>& exts) {
    IE_PROFILING_AUTO_SCOPE(details::ReadNetwork)
    // Register readers if it is needed
    registerReaders();

    // Fix unicode name
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring model_path = InferenceEngine::details::multiByteCharToWString(modelPath.c_str());
#else
    std::string model_path = modelPath;
#endif
    // Try to open model file
    std::ifstream modelStream(model_path, std::ios::binary);
    if (!modelStream.is_open())
        THROW_IE_EXCEPTION << "Model file " << modelPath << " cannot be opened!";

    // Find reader for model extension
    auto fileExt = modelPath.substr(modelPath.find_last_of(".") + 1);
    for (auto it = readers.lower_bound(fileExt); it != readers.upper_bound(fileExt); it++) {
        auto reader = it->second;
        // Check that reader supports the model
        if (reader->supportModel(modelStream)) {
            // Find weights
            std::string bPath = binPath;
            if (bPath.empty()) {
                auto pathWoExt = modelPath;
                auto pos = modelPath.rfind('.');
                if (pos != std::string::npos) pathWoExt = modelPath.substr(0, pos);
                for (const auto& ext : reader->getDataFileExtensions()) {
                    bPath = pathWoExt + "." + ext;
                    if (!FileUtils::fileExist(bPath)) {
                        bPath.clear();
                    } else {
                        break;
                    }
                }
            }
            if (!bPath.empty()) {
                // Open weights file
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
                std::wstring weights_path = InferenceEngine::details::multiByteCharToWString(bPath.c_str());
#else
                std::string weights_path = bPath;
#endif
                std::shared_ptr<IAllocator> allocator = details::make_mmap_allocator(weights_path.c_str());
                void* data = allocator->alloc(0);

                if (data == nullptr)
                    THROW_IE_EXCEPTION << "Weights file " << bPath << " cannot be opened!";

                details::MmapAllocator* mmap = reinterpret_cast<details::MmapAllocator*>(allocator.get());
                const Blob::CPtr weights = make_shared_blob<uint8_t>({Precision::U8, { mmap->size() }, C }, reinterpret_cast<unsigned char*>(data));

                // read model with weights
                CNNNetwork network = reader->read(modelStream, weights, exts);
                network.setWeightsAllocator(allocator);

                return network;
            }
            // read model without weights
            return reader->read(modelStream, exts);
        }
    }
    THROW_IE_EXCEPTION << "Unknown model format! Cannot read the model: " << modelPath;
}

CNNNetwork details::ReadNetwork(const std::string& model, const Blob::CPtr& weights, const std::vector<IExtensionPtr>& exts) {
    IE_PROFILING_AUTO_SCOPE(details::ReadNetwork)
    // Register readers if it is needed
    registerReaders();
    std::istringstream modelStream(model);

    for (auto it = readers.begin(); it != readers.end(); it++) {
        auto reader = it->second;
        if (reader->supportModel(modelStream)) {
            if (weights)
                return reader->read(modelStream, weights, exts);
            return reader->read(modelStream, exts);
        }
    }
    THROW_IE_EXCEPTION << "Unknown model format! Cannot read the model from string!";
}

}  // namespace InferenceEngine

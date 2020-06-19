// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that defines a wrapper class for handling extension instantiation and releasing resources
 *
 * @file ie_extension.h
 */
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "details/ie_so_pointer.hpp"
#include "ie_iextension.h"

namespace InferenceEngine {
namespace details {

/**
 * @brief The SOCreatorTrait class specialization for IExtension case, defines the name of the fabric method for
 * creating IExtension object in DLL
 */
template <>
class SOCreatorTrait<IExtension> {
public:
    /**
     * @brief A name of the fabric method for creating an IExtension object in DLL
     */
    static constexpr auto name = "CreateExtension";
};

}  // namespace details

/**
 * @brief This class is a C++ helper to work with objects created using extensions.
 */
IE_SUPPRESS_DEPRECATED_START_WIN
class INFERENCE_ENGINE_API_CLASS(Extension) : public IExtension {
public:
    /**
     * @brief Loads extension from a shared library
     *
     * @param name Full or relative path to extension library
     */
    explicit Extension(const file_name_t& name): actual(name) {}

    /**
     * @brief Gets the extension version information
     *
     * @param versionInfo A pointer to version info, set by the plugin
     */
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {
        actual->GetVersion(versionInfo);
    }

    /**
     * @brief Cleans the resources up
     */
    void Unload() noexcept override {
        actual->Unload();
    }

    /**
     * @brief Does nothing since destruction is done via the regular mechanism
     */
    void Release() noexcept override {}

    /**
     * @deprecated Use IExtension::getImplTypes to get implementation types for a particular node.
     * The method will removed in 2021.1 release.
     * @brief Gets the array with types of layers which are included in the extension
     *
     * @param types Types array
     * @param size Size of the types array
     * @param resp Response descriptor
     * @return Status code
     */
    INFERENCE_ENGINE_DEPRECATED("Use IExtension::getImplTypes to get implementation types for a particular node")
    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        IE_SUPPRESS_DEPRECATED_START
        return actual->getPrimitiveTypes(types, size, resp);
        IE_SUPPRESS_DEPRECATED_END
    }

    /**
     * @deprecated Use IExtension::getImplementation to get a concrete implementation.
     * The method will be removed in 2021.1 release.
     * @brief Gets the factory with implementations for a given layer
     *
     * @param factory Factory with implementations
     * @param cnnLayer A layer to get the factory for
     * @param resp Response descriptor
     * @return Status code
     */
    IE_SUPPRESS_DEPRECATED_START
    INFERENCE_ENGINE_DEPRECATED("Use IExtension::getImplementation to get a concrete implementation")
    StatusCode getFactoryFor(ILayerImplFactory*& factory, const CNNLayer* cnnLayer,
                             ResponseDesc* resp) noexcept override {
        return actual->getFactoryFor(factory, cnnLayer, resp);
    }
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Returns operation sets
     * This method throws an exception if it was not implemented
     * @return map of opset name to opset
     */
    std::map<std::string, ngraph::OpSet> getOpSets() override;

    /**
     * @brief Returns vector of implementation types
     * @param node shared pointer to nGraph op
     * @return vector of strings
     */
    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override {
        if (node == nullptr) THROW_IE_EXCEPTION << "Provided ngraph::Node pointer is nullptr.";
        return actual->getImplTypes(node);
    }

    /**
     * @brief Returns implementation for specific nGraph op
     * @param node shared pointer to nGraph op
     * @param implType implementation type
     * @return shared pointer to implementation
     */
    ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override {
        if (node == nullptr) THROW_IE_EXCEPTION << "Provided ngraph::Node pointer is nullptr.";
        return actual->getImplementation(node, implType);
    }

protected:
    /**
     * @brief A SOPointer instance to the loaded templated object
     */
    InferenceEngine::details::SOPointer<IExtension> actual;
};

/**
 * @brief Creates a special shared_pointer wrapper for the given type from a specific shared module
 *
 * @param name Name of the shared library file
 * @return shared_pointer A wrapper for the given type from a specific shared module
 */
template <>
inline std::shared_ptr<IExtension> make_so_pointer(const file_name_t& name) {
    return std::make_shared<Extension>(name);
}

}  // namespace InferenceEngine

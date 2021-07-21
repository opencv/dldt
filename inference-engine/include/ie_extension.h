// Copyright (C) 2018-2021 Intel Corporation
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

#include <ngraph/ngraph.hpp>
#include "ie_iextension.h"
#include "details/ie_so_pointer.hpp"

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
class INFERENCE_ENGINE_API_CLASS(Extension) final : public IExtension {
public:
    /**
     * @brief Loads extension from a shared library
     *
     * @param name Full or relative path to extension library
     */
    template <typename C,
              typename = details::enableIfSupportedChar<C>>
    explicit Extension(const std::basic_string<C>& name): actual(name) {}

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
        if (node == nullptr) IE_THROW() << "Provided ngraph::Node pointer is nullptr.";
        return actual->getImplTypes(node);
    }

    /**
     * @brief Returns implementation for specific nGraph op
     * @param node shared pointer to nGraph op
     * @param implType implementation type
     * @return shared pointer to implementation
     */
    ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override {
        if (node == nullptr) IE_THROW() << "Provided ngraph::Node pointer is nullptr.";
        return actual->getImplementation(node, implType);
    }

protected:
    /**
     * @brief A SOPointer instance to the loaded templated object
     */
    details::SOPointer<IExtension> actual;
};

/**
 * @brief Creates extension using deprecated API
 * @tparam T extension type
 * @param name extension library name
 * @return shared pointer to extension
 */
template<typename T = IExtension>
INFERENCE_ENGINE_DEPRECATED("Use std::make_shared<Extension>")
inline std::shared_ptr<T> make_so_pointer(const std::string& name) {
    return std::make_shared<Extension>(name);
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief Creates extension using deprecated API
 * @param name extension library name
 * @return shared pointer to extension
 */
template<typename T = IExtension>
INFERENCE_ENGINE_DEPRECATED("Use std::make_shared<Extension>")
inline std::shared_ptr<IExtension> make_so_pointer(const std::wstring& name) {
    return std::make_shared<Extension>(name);
}

#endif

// ===========================================================================================================================================
// New extensions experiment
// ===========================================================================================================================================
class INFERENCE_ENGINE_API_CLASS(NewExtension) {
public:
    using Ptr = std::shared_ptr<NewExtension>;

    using type_info_t = ngraph::Node::type_info_t;

    virtual ~NewExtension() = default;

    virtual const type_info_t& get_type_info() const = 0;
};

class INFERENCE_ENGINE_API_CLASS(IRExtension): public NewExtension {
    std::string m_opset;
public:
    NGRAPH_RTTI_DECLARATION;
    IRExtension(const std::string& opsetVersion): m_opset(opsetVersion) {}

    const std::string& get_opset() {
        return m_opset;
    }

    virtual const ngraph::Node::type_info_t get_type() = 0;
    virtual ngraph::OutputVector create(const ngraph::OutputVector& inputs, ngraph::AttributeVisitor& visitor) = 0;
};

template<class T>
class DefaultIRExtension: public IRExtension {
public:
    DefaultIRExtension(const std::string opsetVersion): IRExtension(opsetVersion) {}
    ngraph::OutputVector create(const ngraph::OutputVector& inputs, ngraph::AttributeVisitor& visitor) override {
        std::shared_ptr<ngraph::Node> node = std::make_shared<T>();

        node->set_arguments(inputs);
        if (node->visit_attributes(visitor)) {
            node->constructor_validate_and_infer_types();
        }
        return node->outputs();
    }
    const ngraph::Node::type_info_t get_type() override {
        return T::type_info;
    }
};

class INFERENCE_ENGINE_API_CLASS(SOExtension) final : public NewExtension {
public:
    NGRAPH_RTTI_DECLARATION;
    SOExtension(const details::SharedObjectLoader& actual, const NewExtension::Ptr& ext) : so(actual), extension(ext) {
        IE_ASSERT(extension);
    }
    const NewExtension::Ptr& getExtension() {
        return extension;
    }
private:
    NewExtension::Ptr extension;
    details::SharedObjectLoader so;
};

/**
 * @brief Creates the default instance of the extension
 *
 * @param ext Extension interface
 */
INFERENCE_EXTENSION_API(void) CreateExtensions(std::vector<InferenceEngine::NewExtension::Ptr>&);

/**
 * @def IE_DEFINE_EXTENSION_CREATE_FUNCTION
 * @brief Generates extension creation function
 */
#define IE_CREATE_EXTENSIONS(extensions)                                                      \
    INFERENCE_EXTENSION_API(void)                                                             \
    InferenceEngine::CreateExtensions(std::vector<InferenceEngine::NewExtension::Ptr>& ext) { \
        ext = extensions;                                                                     \
    }

template <typename C, typename = details::enableIfSupportedChar<C>>
std::vector<NewExtension::Ptr> load_extensions(const std::basic_string<C>& name) {
    details::SharedObjectLoader so(name.c_str());
    std::vector<NewExtension::Ptr> extensions;
    try {
        using CreateF = void(std::vector<InferenceEngine::NewExtension::Ptr>&);
        std::vector<InferenceEngine::NewExtension::Ptr> ext;
        reinterpret_cast<CreateF*>(so.get_symbol("CreateExtensions"))(ext);
        for (const auto& ex : ext) {
            extensions.emplace_back(std::make_shared<SOExtension>(so, ex));
        }
    } catch (...) {
        details::Rethrow();
    }
    return extensions;
}
}  // namespace InferenceEngine

namespace ngraph {
    /// \brief Tests if value is a pointer/shared_ptr that can be statically cast to a
    /// Type*/shared_ptr<Type>
    template <typename Type>
    typename std::enable_if<
        std::is_convertible<
            decltype(std::declval<std::shared_ptr<InferenceEngine::NewExtension>>()->get_type_info().is_castable(Type::type_info)),
            bool>::value,
        bool>::type
        is_type(const std::shared_ptr<InferenceEngine::NewExtension>& value) {
            bool result = value->get_type_info().is_castable(Type::type_info);
            if (value->get_type_info().is_castable(InferenceEngine::SOExtension::type_info)) {
                auto realExt = std::static_pointer_cast<InferenceEngine::SOExtension>(value)->getExtension();
                result |= realExt->get_type_info().is_castable(Type::type_info);
            }
            return result;
    }
    /// Casts a Value* to a Type* if it is of type Type, nullptr otherwise
    template <typename Type>
    typename std::enable_if<
        std::is_convertible<decltype(static_cast<Type*>(std::declval<InferenceEngine::NewExtension*>())), Type*>::value,
        Type*>::type
        as_type(InferenceEngine::NewExtension* value) {
        if (is_type<InferenceEngine::SOExtension>(value)) {
            auto* realExt = static_cast<InferenceEngine::SOExtension*>(value)->getExtension().get();
            if (is_type<Type>(realExt)) return static_cast<Type*>(realExt);
        }
        return is_type<Type>(value) ? static_cast<Type*>(value) : nullptr;
    }

    /// Casts a std::shared_ptr<Value> to a std::shared_ptr<Type> if it is of type
    /// Type, nullptr otherwise
    template <typename Type>
    typename std::enable_if<
        std::is_convertible<decltype(std::static_pointer_cast<Type>(std::declval<std::shared_ptr<InferenceEngine::NewExtension>>())),
                            std::shared_ptr<Type>>::value,
        std::shared_ptr<Type>>::type
        as_type_ptr(std::shared_ptr<InferenceEngine::NewExtension> value) {
        if (is_type<InferenceEngine::SOExtension>(value)) {
            auto realExt = std::static_pointer_cast<InferenceEngine::SOExtension>(value)->getExtension();

            if (is_type<Type>(realExt)) return std::static_pointer_cast<Type>(realExt);
        }
        return is_type<Type>(value) ? std::static_pointer_cast<Type>(value) : std::shared_ptr<Type>();
    }
} // namespace ngraph

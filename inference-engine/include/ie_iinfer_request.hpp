// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for IInferRequest interface
 *
 * @file ie_iinfer_request.hpp
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include "ie_blob.h"
#include "ie_common.h"
#include "ie_preprocess.hpp"
#include "ie_imemory_state.hpp"

namespace InferenceEngine {

/**
 * @brief This is an interface of asynchronous infer request
 *
 */
class IInferRequest : public std::enable_shared_from_this<IInferRequest> {
public:
    /**
     * @enum WaitMode
     * @brief Enumeration to hold wait mode for IInferRequest
     */
    enum WaitMode : int64_t {
        /** Wait until inference result becomes available */
        RESULT_READY = -1,
        /** IInferRequest doesn't block or interrupt current thread and immediately returns inference status */
        STATUS_ONLY = 0,
    };
    /**
     * @brief A shared pointer to the IInferRequest object
     */
    using Ptr = std::shared_ptr<IInferRequest>;
    /**
     * @brief A smart pointer to the IInferRequest object
     */
    using WeakPtr = std::weak_ptr<IInferRequest>;

    /**
     * @brief Sets input/output data to infer
     *
     * @note Memory allocation does not happen
     * @param name Name of input or output blob.
     * @param data Reference to input or output blob. The type of a blob must match the network input precision and
     * size.
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode SetBlob(const char* name, const Blob::Ptr& data, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Gets input/output data for inference
     *
     * @note Memory allocation does not happen
     * @param name Name of input or output blob.
     * @param data Reference to input or output blob. The type of Blob must match the network input precision and size.
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode GetBlob(const char* name, Blob::Ptr& data, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Sets pre-process for input data
     * @param name Name of input blob.
     * @param data Reference to input or output blob. The type of Blob must match the network input precision and size.
     * @param info Preprocess info for blob.
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: OK (0) for success
     */
    virtual StatusCode SetBlob(const char *name, const Blob::Ptr &data, const PreProcessInfo& info, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Gets pre-process for input data
     * @param name Name of input blob.
     * @param info pointer to a pointer to PreProcessInfo structure
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: OK (0) for success
     */
    virtual StatusCode GetPreProcess(const char* name, const PreProcessInfo** info, ResponseDesc *resp) const noexcept = 0;
    /**
     * @brief Infers specified input(s) in synchronous mode
     *
     * @note blocks all methods of IInferRequest while request is ongoing (running or waiting in queue)
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode Infer(ResponseDesc* resp) noexcept = 0;
    /**
     * @brief Cancels current async inference request
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode Cancel(ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer
     *
     * @note not all plugins provide meaningful data
     * @param perfMap Map of layer names to profiling information for that layer
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>& perfMap,
                                            ResponseDesc* resp) const noexcept = 0;

    /**
     * @brief Waits for the result to become available. Blocks until specified millis_timeout has elapsed or the result
     * becomes available, whichever comes first.
     *
     * @param millis_timeout Maximum duration in milliseconds to block for
     * @note There are special cases when millis_timeout is equal some value of the WaitMode enum:
     * * STATUS_ONLY - immediately returns inference status (IInferRequest::RequestStatus). It does not block or
     * interrupt current thread
     * * RESULT_READY - waits until inference result becomes available
     * @param resp Optional: a pointer to an already allocated object to contain extra information of a failure (if
     * occurred)
     * @return Enumeration of the resulted action: InferenceEngine::OK (0) for success
     */
    virtual InferenceEngine::StatusCode Wait(int64_t millis_timeout, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Starts inference of specified input(s) in asynchronous mode
     *
     * @note It returns immediately. Inference starts also immediately
     * @param resp Optional: a pointer to an already allocated object to contain extra information of a failure (if
     * occurred)
     * @return Enumeration of the resulted action: InferenceEngine::OK (0) for success
     */
    virtual StatusCode StartAsync(ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Completion callback definition as pointer to a function
     *
     * @param context Pointer to request for providing context inside callback
     * @param code Completion result status: InferenceEngine::OK (0) for success
     */
    typedef void (*CompletionCallback)(InferenceEngine::IInferRequest::Ptr context, InferenceEngine::StatusCode code);

    /**
     * @brief Sets a callback function that will be called on success or failure of asynchronous request
     *
     * @param callback A function to be called
     * @return Enumeration of the resulted action: InferenceEngine::OK (0) for success
     */
    virtual StatusCode SetCompletionCallback(CompletionCallback callback) noexcept = 0;

    /**
     * @brief Gets arbitrary data for the request and stores a pointer to a pointer to the obtained data
     *
     * @param data Pointer to a pointer to the gotten arbitrary data
     * @param resp Optional: a pointer to an already allocated object to contain extra information of a failure (if
     * occurred)
     * @return Enumeration of the resulted action: InferenceEngine::OK (0) for success
     */
    virtual StatusCode GetUserData(void** data, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Sets arbitrary data for the request
     *
     * @param data Pointer to a pointer to arbitrary data to set
     * @param resp Optional: a pointer to an already allocated object to contain extra information of a failure (if
     * occurred)
     * @return Enumeration of the resulted action: InferenceEngine::OK (0) for success
     */
    virtual StatusCode SetUserData(void* data, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Sets new batch size when dynamic batching is enabled in executable network that created this request.
     *
     * @param batch_size new batch size to be used by all the following inference calls for this request.
     * @param resp Optional: a pointer to an already allocated object to contain extra information of a failure (if
     * occurred)
     * @return Enumeration of the resulted action: InferenceEngine::OK (0) for success
     */
    virtual InferenceEngine::StatusCode SetBatch(int batch_size, ResponseDesc* resp) noexcept = 0;

    virtual InferenceEngine::StatusCode SetShape(const char* /*name*/, const SizeVector& /*dims*/, ResponseDesc* /*resp*/) noexcept {
        return NOT_IMPLEMENTED;
    }

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @brief Gets state control interface for given infer request.
     *
     * State control essential for recurrent networks
     *
     * @param pState reference to a pointer that receives internal states
     * @param idx requested index for receiving memory state
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success, OUT_OF_BOUNDS (-6) no memory state for
     * given index
     */
    virtual StatusCode QueryState(IVariableState::Ptr& pState, size_t idx, ResponseDesc* resp) noexcept = 0;

protected:
    ~IInferRequest() = default;
};

}  // namespace InferenceEngine

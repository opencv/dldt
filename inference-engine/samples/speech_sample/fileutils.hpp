// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cnpy.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>

class BaseFile {
public:
    virtual void LoadFile(const char* fileName, uint32_t arrayIndex, std::string& ptrName, std::vector<uint8_t>& memory, uint32_t* ptrNumRows,
                          uint32_t* ptrNumColumns, uint32_t* ptrNumBytesPerElement) = 0;

    virtual void SaveFile(const char* fileName, bool shouldAppend, std::string name, void* ptrMemory, uint32_t numRows, uint32_t numColumns) = 0;

    virtual void GetFileInfo(const char* fileName, uint32_t numArrayToFindSize, uint32_t* ptrNumArrays, uint32_t* ptrNumMemoryBytes) = 0;
};

class ArkFile : public BaseFile {
public:
    /**
     * @brief Get info from Kaldi ARK speech feature vector file
     * @param fileName .ark file name
     * @param numArrayToFindSize number speech feature vectors in the file
     * @param ptrNumArrays pointer to specific number array
     * @param ptrNumMemoryBytes pointer to specific number of memory bytes
     * @return none.
     */
    virtual void GetFileInfo(const char* fileName, uint32_t numArrayToFindSize, uint32_t* ptrNumArrays, uint32_t* ptrNumMemoryBytes);

    /**
     * @brief Load Kaldi ARK speech feature vector file
     * @param fileName .ark file name
     * @param arrayIndex number speech feature vector in the file
     * @param ptrName reference to variable length name
     * @param memory reference to speech feature vector to save
     * @param ptrNumRows pointer to number of rows to read
     * @param ptrNumColumns pointer to number of columns to read
     * @param ptrNumBytesPerElement pointer to number bytes per element (size of float by default)
     * @return none.
     */
    virtual void LoadFile(const char* fileName, uint32_t arrayIndex, std::string& ptrName, std::vector<uint8_t>& memory, uint32_t* ptrNumRows,
                          uint32_t* ptrNumColumns, uint32_t* ptrNumBytesPerElement);

    /**
     * @brief Save Kaldi ARK speech feature vector file
     * @param fileName .ark file name
     * @param shouldAppend bool flag to rewrite or add to the end of file
     * @param name reference to variable length name
     * @param ptrMemory pointer to speech feature vector to save
     * @param numRows number of rows
     * @param numColumns number of columns
     * @return none.
     */
    virtual void SaveFile(const char* fileName, bool shouldAppend, std::string name, void* ptrMemory, uint32_t numRows, uint32_t numColumns);
};

class NumpyFile : public BaseFile {
public:
    virtual void GetFileInfo(const char* fileName, uint32_t numArrayToFindSize, uint32_t* ptrNumArrays, uint32_t* ptrNumMemoryBytes);

    virtual void LoadFile(const char* fileName, uint32_t arrayIndex, std::string& ptrName, std::vector<uint8_t>& memory, uint32_t* ptrNumRows,
                          uint32_t* ptrNumColumns, uint32_t* ptrNumBytesPerElement);

    virtual void SaveFile(const char* fileName, bool shouldAppend, std::string name, void* ptrMemory, uint32_t numRows, uint32_t numColumns);
};
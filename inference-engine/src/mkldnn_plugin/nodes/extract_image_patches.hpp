// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "base.hpp"
#include "caseless.hpp"
#include "ie_parallel.hpp"

#include <string>
#include <vector>
#include <set>
#include <cmath>
// algo is for debug purposes
#include <algorithm>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

struct jit_eximpat_params {
    int OB, IC, IH, IW; // from input shape
    int OH, OW; //from out shape
    int KH, KW; // kernel sizes
    int SH, SW; // strides
    int RH, RW; // rates
    int PL, PT; // determine padding
    int dtype_size; // byte size of the datatype
    int block_size; // num of dtype units in the supported vector instruction set
};

struct jit_eximpat_args {
    int64_t h_lo_pad;
    int64_t h_hi_pad;
    int64_t w_lo_pad;
    int64_t w_hi_pad;
    const void* src;
    void* dst; // const?
};

struct jit_uni_eximpat_kernel {
    void (*ker_)(const jit_eximpat_args *);

    void operator()(const jit_eximpat_args *args) { assert(ker_); ker_(args); }

    jit_eximpat_params jpp;

    virtual void create_ker() = 0;

    explicit jit_uni_eximpat_kernel(jit_eximpat_params jpp) : ker_(nullptr), jpp(jpp) {}
    virtual ~jit_uni_eximpat_kernel() {}
};

using details::CaselessEq;

class ExtractImagePatchesImpl : public ExtLayerBase {
public:
    explicit ExtractImagePatchesImpl(const CNNLayer*);
    StatusCode execute(std::vector<Blob::Ptr>&, std::vector<Blob::Ptr>&, ResponseDesc*) noexcept override;

    template<typename T>
    void process_data(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs) noexcept {
        const T* src_data = inputs[0]->cbuffer().as<const T*>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T* dst_data = outputs[0]->buffer().as<T*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto& inDims = inputs[0]->getTensorDesc().getDims();
        const size_t inDimsSize = inDims.size(); // Must always be 4 according to the specs.

        const size_t BATCH = 0, CHANNEL = 1, HIGHT = 0, WIDTH = 1;

        const int64_t IC = inDims[CHANNEL];
        const int64_t IH = inDims[inDimsSize - 2];
        const int64_t IW = inDims[inDimsSize - 1];

        const auto& outDims = outputs[0]->getTensorDesc().getDims();
        const size_t outDimsSize = outDims.size(); // Must always be 4 according to the specs.

        const int64_t OB = outDims[BATCH];
        //const int64_t OC = outDims[CHANNEL]; // Must always be KH * KW * IC according to the specs.
        const int64_t OH = outDims[outDimsSize - 2];
        const int64_t OW = outDims[outDimsSize - 1];

        const int64_t KH = _ksizes[HIGHT];
        const int64_t KW = _ksizes[WIDTH];
        const int64_t SH = _strides[HIGHT];
        const int64_t SW = _strides[WIDTH];
        const int64_t RH = _rates[HIGHT];
        const int64_t RW = _rates[WIDTH];
        const int64_t PL = _pads[HIGHT];
        const int64_t PT = _pads[WIDTH];
        const std::vector<int64_t> ostrides = {KH * KW * IC * OH * OW, KW * IC * OH * OW, IC * OH * OW, OH * OW};
        const std::vector<int64_t> istrides = {IC * IH * IW, IH * IW, IW};

        // for debug purposes
        //for(int64_t i=0; i < OW * OH * IC * KW * KH * OB ; i++)
        //    dst_data[i] = -1;
        std::fill(dst_data, dst_data + OW * OH * IC * KW * KH * OB, -1);
        if (eximpat_kernel){
        //if (0){
            //auto src_ptr = reinterpret_cast<const char *>(src_data);
            //auto dst_ptr = reinterpret_cast<char *>(dst_data);

            auto thread_body = [&](const int64_t ob, const int64_t kh, const int64_t kw, const int64_t ic) {
                const int64_t ih_start = kh * RH - PT;
                const int64_t iw_start = kw * RW - PL;
                const int64_t ih_lpad = ih_start >= 0 ? 0 : std::ceil(- 1.f * ih_start / SH);
                const int64_t iw_lpad = iw_start >= 0 ? 0 : std::ceil(- 1.f * iw_start / SW);
                const int64_t ih_hpad = std::ceil((IH - 1.f * ih_start) / SH) > OH ? OH : std::ceil((IH + -1.f * ih_start) / SH);
                const int64_t iw_hpad = std::ceil((IW - 1.f * iw_start) / SW) > OW ? OW : std::ceil((IW - 1.f * iw_start) / SW);

                int64_t dst_offset = ob * ostrides[0] + kh * ostrides[1] + kw * ostrides[2] + ic * ostrides[3];
                int64_t src_offset = ob * istrides[0] + ic * istrides[1] + ih_start * istrides[2] + iw_start + ih_lpad * SH * IW;
                //const int64_t ioffset = ob * istrides[0] + ic * istrides[1] + ih_start * istrides[2] + iw_start;

                auto args = jit_eximpat_args();
                args.src = reinterpret_cast<const char *>(src_data + src_offset);
                args.dst = reinterpret_cast<char *>(dst_data + dst_offset);
                args.h_lo_pad = ih_lpad;
                args.h_hi_pad = ih_hpad;
                args.w_lo_pad = iw_lpad;
                args.w_hi_pad = iw_hpad;
                (*eximpat_kernel)(&args);
            };
            parallel_for4d(OB, KH, KW, IC, thread_body);
        } else {
            auto thread_body = [&](const int64_t ob, const int64_t kh, const int64_t kw, const int64_t ic) {
                const int64_t iw_start = kw * RW - PL;
                //const int64_t iw_stop = iw_start + OW * SW;
                const int64_t ih_start = kh * RH - PT;
                //const int64_t ih_stop = ih_start + OH * SH;


                const int64_t ih_lpad = ih_start >= 0 ? 0 : std::ceil(- 1.f * ih_start / SH);
                const int64_t iw_lpad = iw_start >= 0 ? 0 : std::ceil(- 1.f * iw_start / SW);

                const int64_t ih_hpad = std::ceil((IH - 1.f * ih_start) / SH) > OH ? OH : std::ceil((IH + -1.f * ih_start) / SH);
                const int64_t iw_hpad = std::ceil((IW - 1.f * iw_start) / SW) > OW ? OW : std::ceil((IW - 1.f * iw_start) / SW);

                int64_t dst_idx = ob * ostrides[0] + kh * ostrides[1] + kw * ostrides[2] + ic * ostrides[3];


                /*
                for (int64_t ih = 0; ih < ih_lpad; ih++)
                    for (int64_t iw = 0; iw < OW; iw++)
                            dst_data[dst_idx++] = T(0);
                */

                for (int64_t i = 0; i < ih_lpad * OW; i++)
                        dst_data[dst_idx++] = T(0);

                const int64_t ioffset = ob * istrides[0] + ic * istrides[1] + ih_start * istrides[2] + iw_start;

                for (int64_t ishift = ioffset + ih_lpad * SH * IW; ishift < ioffset + ih_hpad * SH * IW; ishift += SH * IW) {
                    for (int64_t iw = 0; iw < iw_lpad; iw++)
                        dst_data[dst_idx++] = T(0);
                    /*
                    for (int64_t iw = iw_lpad; iw < iw_hpad; iw++)
                        dst_data[dst_idx++] = src_data[ishift + iw * SW + ih * SH * IW];
                    */
                    for (int64_t src_idx = ishift + iw_lpad * SW; src_idx < ishift + iw_hpad * SW; src_idx += SW)
                        dst_data[dst_idx++] = src_data[src_idx];

                    for (int64_t i = 0; i < (OW - iw_hpad); i++)
                        dst_data[dst_idx++] = T(0);
                }

                for (int64_t i = 0; i < (OH - ih_hpad) * OW; i++)
                        dst_data[dst_idx++] = T(0);
                /*
                for (int64_t ih = ih_hpad; ih < OH; ih++)
                    for (int64_t iw = 0; iw < OW; iw++)
                        dst_data[dst_idx++] = T(0);
                */

            };
            /*
            auto thread_body = [&](const int64_t ob, const int64_t kh, const int64_t kw, const int64_t ic) {
                const int64_t iw_start = kw * RW - PL;
                const int64_t iw_stop = iw_start + OW * SW;
                const int64_t ih_start = kh * RH - PT;
                const int64_t ih_stop = ih_start + OH * SH;
                int64_t dst_idx = ob * ostrides[0] + kh * ostrides[1] + kw * ostrides[2] + ic * ostrides[3];
                int64_t ishift = ob * istrides[0] + ic * istrides[1] + ih_start * istrides[2];
                for (int64_t ih = ih_start; ih < ih_stop; ih += SH, ishift += SH * IW) {
                    for (int64_t iw = iw_start; iw < iw_stop; iw += SW, dst_idx++) {
                        if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                            dst_data[dst_idx] = T(0);
                        } else {
                            dst_data[dst_idx] = src_data[ishift + iw];
                        }
                    }
                }
            };
             */
            parallel_for4d(OB, KH, KW, IC, thread_body);
        }
        std::cout << "\n======================\n\n";
        for(int i=0; i < IH; i++){
            for(int j=0; j < IW; j++) {
                std::cout << src_data[ i * IW + j ] << " ";
            }
            std::cout << "\n";
        }
        // NB! works only for IC=1
        std::cout << "-------------------------\n";
        for(int kh=0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                std::cout << "KH = " << kh << " KW = " << kw << "\n";
                for (int i = 0; i < OH; i++) {
                    for (int j = 0; j < OW; j++) {
                        std::cout << dst_data[kh * KW * OH * OW + kw * OH * OW + i * OW + j] << " ";
                    }
                    std::cout << "\n";
                }
            }
        }
    }

private:
    std::vector<int64_t> _ksizes;
    std::vector<int64_t> _strides;
    std::vector<int64_t> _rates;
    std::vector<int64_t> _pads;
    std::string _auto_pad;

    std::shared_ptr<jit_uni_eximpat_kernel> eximpat_kernel;
    static const std::set<size_t> _supported_precisions_sizes;
    inline void set_pads(const std::string & pad_str, const std::vector<int64_t> & params);
};

const std::set<size_t> ExtractImagePatchesImpl::_supported_precisions_sizes = {1, 2, 4, 8};

REG_FACTORY_FOR(ExtractImagePatchesImpl, ExtractImagePatches);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine

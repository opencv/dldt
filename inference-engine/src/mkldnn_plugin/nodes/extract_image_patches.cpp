// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches.hpp"
#include "list.hpp"
#include <cpu/x64/jit_generator.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using details::CaselessEq;

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_eximpat_args, field)

template <cpu_isa_t isa>
struct jit_uni_eximpat_kernel_f32 : public jit_uni_eximpat_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eximpat_kernel_f32)

    explicit jit_uni_eximpat_kernel_f32(jit_eximpat_params jpp) : jit_uni_eximpat_kernel(jpp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);

        //loop(jpp.n);  //loop over input and output to get the work done

        this->postamble();
    }
    // some additional functions to facilitate the main loop
    /*
    void load(const Xbyak::Xmm &xmm, const Xbyak::Address &addr) {
        switch (jpp.data_size) {
            case 16: movups(xmm, addr); break;
            case 8: movsd(xmm, addr); break;
            case 4: movss(xmm, addr); break;
            case 2: pinsrw(xmm, addr, 0x0); break;
            case 1: pinsrb(xmm, addr, 0x0); break;
        }
    }

    void store(const Xbyak::Address &addr, const Xbyak::Xmm &xmm) {
        switch (jpp.data_size) {
            case 16: movups(addr, xmm); break;
            case 8: movsd(addr, xmm); break;
            case 4: movss(addr, xmm); break;
            case 2: pextrw(addr, xmm, 0x0); break;
            case 1: pextrb(addr, xmm, 0x0); break;
        }
    }
    */
    // The main loop where all the work is done
    /*
    void loop(int n) {
        mov(reg_work_amount, jpp.dst_block_dims[n]);

        Xbyak::Label main_loop_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label exit_label;

        if (n + 1 == jpp.ndims) {
            if (jpp.src_strides[n] == jpp.dst_strides[n] == 1) {
                uint32_t step = vlen / jpp.data_size;

                L(main_loop_label);
                {
                    cmp(reg_work_amount, step);
                    jl(tail_loop_label, T_NEAR);

                    uni_vmovups(vmm, ptr[reg_src]);
                    uni_vmovups(ptr[reg_dst], vmm);

                    add(reg_src, step * jpp.data_size);
                    add(reg_dst, step * jpp.data_size);
                    sub(reg_work_amount, step);

                    jmp(main_loop_label, T_NEAR);
                }
            }
        }

        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            je(exit_label, T_NEAR);

            if (n + 1 == jpp.ndims) {
                load(xmm, ptr[reg_src]);
                store(ptr[reg_dst], xmm);
            } else {
                aux_reg_src = reg_src;
                aux_reg_dst = reg_dst;
                push(aux_reg_src);
                push(aux_reg_dst);
                push(reg_work_amount);
                loop(n + 1);
                pop(reg_work_amount);
                pop(reg_dst);
                pop(reg_src);
            }

            add(reg_src, jpp.src_strides[n] * jpp.data_size);
            add(reg_dst, jpp.dst_strides[n] * jpp.data_size);
            sub(reg_work_amount, 1);

            jmp(tail_loop_label, T_NEAR);
        }

        L(exit_label);
    }
    */

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    reg64_t reg_src = r8;
    reg64_t reg_dst = r9;
    reg64_t reg_work_amount = r10;
    reg64_t reg_params = abi_param1;

    Vmm vmm = Vmm(0);
    Xbyak::Xmm xmm = Xbyak::Xmm(0);
};

ExtractImagePatchesImpl::ExtractImagePatchesImpl(const CNNLayer* layer) {
    try {
        std::string errorPrefix = std::string("Layer ") + layer->type + " with name '" + layer->name + "' ";
        if (details::CaselessEq<std::string>()("ExtractImagePatchesLayer", layer->type))
            IE_THROW() << errorPrefix << "is not an instance of ExtractImagePatchesLayer class";

        if (layer->insData.size() != 1 || layer->outData.size() != 1)
            IE_THROW() << errorPrefix << "has incorrect number of input or output edges!"
                << " Input: " << layer->insData.size() << "; Output: " << layer->outData.size();

        auto inData = layer->insData[0].lock();
        if (inData == nullptr)
            IE_THROW() << errorPrefix << "has nullable input data";

        if (inData->getTensorDesc().getDims().size() != 4)
            IE_THROW() << errorPrefix << "must have 4D input tensor. Actual: " << inData->getTensorDesc().getDims().size();

        if (layer->outData[0]->getTensorDesc().getDims().size() != 4)
            IE_THROW() << errorPrefix << "must have 4D output tensor. Actual: " << layer->outData[0]->getTensorDesc().getDims().size();

        if (inData->getLayout() != NCHW)
            IE_THROW() << errorPrefix << "has unsupported layout: " << inData->getLayout();

        const auto precision = inData->getTensorDesc().getPrecision();
        if (_supported_precisions_sizes.find(precision.size()) == _supported_precisions_sizes.end())
            IE_THROW() << errorPrefix << "has unsupported precision: " << precision.name();

        auto ksizes = layer->GetParamAsUInts("sizes");
        auto strides = layer->GetParamAsUInts("strides");
        auto rates = layer->GetParamAsUInts("rates");
        _auto_pad = layer->GetParamAsString("auto_pad");
        if (!CaselessEq<std::string>()(_auto_pad, "valid")
                && !CaselessEq<std::string>()(_auto_pad, "same_upper")
                && !CaselessEq<std::string>()(_auto_pad, "same_lower"))
            IE_THROW() <<  errorPrefix << "has unsupported auto_pad value: " << _auto_pad;
        if (ksizes.size() != 2 || strides.size() != 2 || rates.size() != 2)
            IE_THROW() << errorPrefix << "must have the following attributes with shape {2}: sizes, strides, rates.";

        _ksizes.clear();
        _strides.clear();
        _rates.clear();
        for (size_t i = 0; i < ksizes.size(); i++)
            _ksizes.push_back((int64_t)ksizes[i]);
        for (size_t i = 0; i < strides.size(); i++)
            _strides.push_back((int64_t)strides[i]);
        for (size_t i = 0; i < rates.size(); i++)
            _rates.push_back((int64_t)rates[i]);

        LayerConfig config;

        DataConfig inConfig;
        inConfig.desc = inData->getTensorDesc();
        config.inConfs.push_back(inConfig);

        DataConfig outConfig;
        outConfig.desc = layer->outData[0]->getTensorDesc();
        outConfig.desc.setPrecision(inConfig.desc.getPrecision());
        outConfig.desc.setLayout(inConfig.desc.getLayout());
        config.outConfs.push_back(outConfig);

        config.dynBatchSupport = false;
        confs.push_back(config);
    } catch (InferenceEngine::Exception &ex) {
        errorMsg = ex.what();
    }
}

StatusCode ExtractImagePatchesImpl::execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
    switch (inputs[0]->getTensorDesc().getPrecision().size()) {
        case 1: {
            process_data<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs);
            break;
        }
        case 2: {
            process_data<PrecisionTrait<Precision::U16>::value_type>(inputs, outputs);
            break;
        }
        case 4: {
            process_data<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs);
            break;
        }
        case 8: {
            process_data<PrecisionTrait<Precision::U64>::value_type>(inputs, outputs);
            break;
        }
        default: {
            if (resp) {
                std::string errorMsg = "ExtractImagePatches layer does not support precision '"
                        + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
    }

    return OK;
}

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine

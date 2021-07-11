// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_uni_kernel.hpp"

using namespace MKLDNNPlugin;
using namespace mkldnn::impl::cpu;

#define GET_OFF(field) offsetof(gatherJitExecArgs, field)

template <x64::cpu_isa_t isa>
jitUniGatherKernel<isa>::jitUniGatherKernel(jGatherConfParams jcp) :
        jitGatherKernelBase(jcp), x64::jit_generator() {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    elPerVec = vlen / jcp.dataTypeSize;
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::create_ker() {
    x64::jit_generator::create_kernel();
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::generate() {
    this->preamble();

    mov(regSrc, ptr[regParams + GET_OFF(src)]);
    mov(regDst, ptr[regParams + GET_OFF(dst)]);
    mov(regIndices, ptr[regParams + GET_OFF(indices)]);
    mov(regIdxIter, ptr[regParams + GET_OFF(idxIter)]);

    mov(regAux1, ptr[regParams + GET_OFF(dataTypeSize)]);
    uni_vpbroadcastd(vmmDictTypeSize, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(idxTypeSize)]);
    uni_vpbroadcastd(vmmAux3, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
    uni_vpbroadcastd(vmmAxisDim, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(vecLen)]);
    uni_vpbroadcastd(vmmVecLen, ptr[regAux1]);

//    mov(regSpecIndicesSize, ptr[regParams + GET_OFF(specIndicesSizeInBytes)]);

    mov(regAux1, ptr[regParams + GET_OFF(specIndicesSizePtr)]);
    uni_vpbroadcastd(vmmSpecIdxSize, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(batchIndices)]);
    uni_vmovups(vmmIdxBatchSum, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(betweenBatchAndAxisIdx)]);
    uni_vmovups(vmmBeforeAxisSum, ptr[regAux1]);
    mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSize)]);
    uni_vpbroadcastd(vmmAxisAndAfterAxisSize, ptr[regAux1]);
    uni_vpmulld(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAxisAndAfterAxisSize);
    mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeInBytes)]);
    uni_vpbroadcastd(vmmAux0, ptr[regAux1]);
    uni_vpmulld(vmmAux0, vmmAux0, vmmIdxBatchSum);
    uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAux0);

    uni_vpmulld(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSize);

    mov(regAux1, ptr[regParams + GET_OFF(specIndices)]);
    uni_vmovups(vmmSpecIndices, ptr[regAux1]);
    uni_vpmulld(vmmSpecIndices, vmmSpecIndices, vmmAux3);

    mov(regBetweenBatchAndAxisIter, ptr[regParams + GET_OFF(beforeAxisCounter)]);
    mov(regBetweenBatchAndAxisSize, ptr[regParams + GET_OFF(betweenBatchAndAxisSize)]);

    mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

//        mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
//        mov(regAux2, ptr[regParams + GET_OFF(retVal)]);

    if (isa == x64::avx512_common) {
        vpcmpub(kMaskOnes, vmmGatherMask, vmmGatherMask, 0);
    }

    uni_vpxor(vmmZeros, vmmZeros, vmmZeros);

    mov(regAux1, ptr[regParams + GET_OFF(afterAxisBlockSize)]);
    Xbyak::Label lBlock_N;
    cmp(regAux1, 1);
    jg(lBlock_N, T_NEAR);
    {
        if (jcp_.dataTypeSize == 4) {
            Xbyak::Label lLessThanVector, lEnd4;
            mov(regAux1, ptr[regParams + GET_OFF(specIndicesSizeInBytes)]);
            cmp(regAux1, vlen);
            jl(lLessThanVector, T_NEAR);
                gatherLongIdx();
                jmp(lEnd4, T_NEAR);
            L(lLessThanVector);
                mov(regAux1, ptr[regParams + GET_OFF(permIdx)]);
                uni_vmovups(vmmPermIdx, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
                uni_vmovups(vmmBeforeAxisDiff, ptr[regAux1]);
                uni_vpmulld(vmmBeforeAxisDiff, vmmBeforeAxisDiff, vmmDictTypeSize);
                mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeInBytes)]);
                uni_vpbroadcastd(vmmSrcAfterBatchSize, ptr[regAux1]);

                gatherShortIdx();
//                jmp(lEnd4, T_NEAR);
            L(lEnd4);
        } else if (jcp_.dataTypeSize == 2) {
            Xbyak::Label lLessThanVector, lEnd;
            mov(regAux1, ptr[regParams + GET_OFF(specIndicesSizeInBytes)]);
            cmp(regAux1, vlen);
            jl(lLessThanVector, T_NEAR);
                gatherLongIdx16();
                jmp(lEnd2, T_NEAR);
            L(lLessThanVector);
                mov(regAux1, ptr[regParams + GET_OFF(permIdx)]);
                uni_vmovups(vmmPermIdx, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
                uni_vmovups(vmmBeforeAxisDiff, ptr[regAux1]);
                uni_vpmulld(vmmBeforeAxisDiff, vmmBeforeAxisDiff, vmmDictTypeSize);
                mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeInBytes)]);
                uni_vpbroadcastd(vmmSrcAfterBatchSize, ptr[regAux1]);

                gatherShortIdx16();
//                jmp(lEnd4, T_NEAR);
            L(lEnd);
        } else if (jcp_.dataTypeSize == 1) {
            auto& vmmShufMask = vmmAux8;

            mov(regAux1, ptr[regParams + GET_OFF(shufMask8bitUni)]);
            uni_vmovups(vmmShufMask, ptr[regAux1]);

            auto& vmmPermMask = vmmAux9;
            if (isa == x64::avx512_common) {
                mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA5)]);
            } else {
                mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA2)]);
            }
            uni_vmovups(vmmPermMask, ptr[regAux1]);

            Xbyak::Label lDstIdxLoop, lTail;
            L(lDstIdxLoop);
            {
                cmp(regWorkAmount, elPerVec);
                jl(lTail, T_NEAR);

                gatherAndGroup(vmmDst, vmmShufMask);
                gatherAndGroup(vmmAux4, vmmShufMask);
                vshufps(vmmDst, vmmDst, vmmAux4, 0);

                gatherAndGroup(vmmAux4, vmmShufMask);
                gatherAndGroup(vmmAux5, vmmShufMask);
                vshufps(vmmAux4, vmmAux4, vmmAux5, 0);

                vshufps(vmmDst, vmmDst, vmmAux4, 0x88);
                vpermd(vmmDst, vmmPermMask, vmmDst);

                uni_vmovups(ptr[regDst], vmmDst);

                add(regDst, vlen);
                sub(regWorkAmount, elPerVec);

                jmp(lDstIdxLoop, T_NEAR);
            }
            L(lTail);
            tail();
        }
    }
    L(lBlock_N);
    // else if (jcp_.afterAxisSize >= elPerVec) {
//            Xbyak::Label lDstIdxLoop, lTail;
//            L(lDstIdxLoop);
//            {
//                cmp(regWorkAmount, elPerVec);
//                jl(lTail, T_NEAR);
//
//                vpGatherDD(vmmDst);
//                uni_vmovups(ptr[regDst], vmmDst);
//
//                add(regDst, vlen);
//                sub(regWorkAmount, elPerVec);
//
//                jmp(lDstIdxLoop, T_NEAR);
//            }
//            L(lTail);
//            tail();
//    } else {
    {
//        Xbyak::Label lDstIdxLoop, lTail;
//        L(lDstIdxLoop);
//        {
//            cmp(regWorkAmount, elPerVec);
//            jl(lTail, T_NEAR);
//
//            vpGatherDDBlk(vmmDst);
//            uni_vmovups(ptr[regDst], vmmDst);
//
//            add(regDst, vlen);
//            sub(regWorkAmount, elPerVec);
//
//            jmp(lDstIdxLoop, T_NEAR);
//        }
//        L(lTail);
//        tail();
    }

    this->postamble();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::tail() {
    Xbyak::Label lFinish;
    Xbyak::Reg32 reg32Aux3(regAux3.getIdx());
    Xbyak::Reg16 reg16Aux3(regAux3.getIdx());
    Xbyak::Reg8  reg8Aux3(regAux3.getIdx());

    mov(regAux1, regWorkAmount);
    mov(reg32Aux3, 0xFFFFFFFF);
    uni_vmovups(vmmOnes, vmmZeros);
    uni_vmovups(vmmAux0, vmmZeros);

    if (isa == x64::avx512_common) {
        Xbyak::Label l1;
        for (uint8_t i = 0; i < elPerVec; i++) {
            cmp(regAux1, 0);
            je(l1, T_NEAR);

            if (i < 4) {
                vpinsrd(xmmOnes, xmmOnes, reg32Aux3, i);
            } else if (i < 8) {
                vpinsrd(xmmAux0, xmmAux0, reg32Aux3, i);
                vinserti128(vmmOnes, vmmOnes, xmmAux0, 2);
            } else if (i < 12) {
                if (i == 8)
                    uni_vmovups(xmmAux0, vmmZeros);
                vpinsrd(xmmAux0, xmmAux0, reg32Aux3, i);
                vinserti32x4(vmmOnes, vmmOnes, xmmAux0, 3);
            } else {
                if (i == 12)
                    uni_vmovups(xmmAux0, vmmZeros);
                vpinsrd(xmmAux0, xmmAux0, reg32Aux3, i);
                vinserti32x4(vmmOnes, vmmOnes, xmmAux0, 4);
            }
            sub(regAux1, 1);
        }
        L(l1);
    } else {
        Xbyak::Label l1;
        for (uint8_t i = 0; i < elPerVec; i++) {
            cmp(regAux1, 0);
            je(l1, T_NEAR);

            if (i < 4) {
                vpinsrd(xmmOnes, xmmOnes, reg32Aux3, i);
            } else if (i < 8) {
                vpinsrd(xmmAux0, xmmAux0, reg32Aux3, i);
                vinserti128(vmmOnes, vmmOnes, xmmAux0, 2);
            } else if (i < 12) {
//                if (i == 8)
//                    uni_vmovups(xmmAux0, vmmZeros);
//                vpinsrd(xmmAux0, xmmAux0, reg32Aux3, i);
//                vinserti32x4(vmmOnes, vmmOnes, xmmAux0, 3);
            } else {
//                if (i == 12)
//                    uni_vmovups(xmmAux0, vmmZeros);
//                vpinsrd(xmmAux0, xmmAux0, reg32Aux3, i);
//                vinserti32x4(vmmOnes, vmmOnes, xmmAux0, 4);
            }
            sub(regAux1, 1);
        }
        L(l1);
    }

    uni_vcvtdq2ps(vmmAux0, vmmBeforeAxisSum);
    uni_vcvtdq2ps(vmmAux1, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, vmmAux1);
    vroundps(vmmAux0, vmmAux0, 0x1B);
    uni_vcvtps2dq(vmmAux0, vmmAux0);
    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSize);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndices);

    uni_vmovups(vmmSrcShifts, vmmOnes);
    vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
    // Compensate negative indices.
    vpcmpgtd(vmmAux0, vmmZeros, vmmAux1);
    vpand(vmmAux0, vmmAux0, vmmAxisDim);
    uni_vpaddd(vmmAux1, vmmAux1, vmmAux0);
    // Check boundaries
    vpcmpgtd(vmmGatherMask, vmmAxisDim, vmmAux1);
    vpcmpgtd(vmmAux0, vmmZeros, vmmAux1);
    vpandn(vmmGatherMask, vmmAux0, vmmGatherMask);
    vpand(vmmGatherMask, vmmGatherMask, vmmSrcShifts);
    // Gather data
    uni_vpmulld(vmmAux1, vmmAux1, vmmDictTypeSize);
    uni_vpaddd(vmmAux1, vmmAux1, vmmBeforeAxisSum);
    vpgatherdd(vmmDst, ptr[regSrc + vmmAux1], vmmGatherMask);

    if (isa == x64::avx512_common) {
        for (int i = 0; i < 2; i++) {
            vextracti32x8(ymmAux0, zmmDst, i);
            for (int j = 0; j < 2; j++) {
                vextracti128(xmmAux1, ymmAux0, j);
                for (int k = 0; k < 4; k++) {
                    cmp(regWorkAmount, 0);
                    je(lFinish, T_NEAR);

                    vpextrd(reg32Aux3, xmmAux1, k);

                    if (jcp_.dataTypeSize == 4) {
                        mov(ptr[regDst], reg32Aux3);
                    } else if (jcp_.dataTypeSize == 2) {
                        mov(ptr[regDst], reg16Aux3);
                    } else if (jcp_.dataTypeSize == 1) {
                        mov(ptr[regDst], reg8Aux3);
                    }

                    add(regDst, jcp_.dataTypeSize);
                    sub(regWorkAmount, 1);
                }
            }
        }
    } else {
        for (int j = 0; j < 2; j++) {
            vextracti128(xmmAux1, vmmDst, j);
            for (int k = 0; k < 4; k++) {
                cmp(regWorkAmount, 0);
                je(lFinish, T_NEAR);

                vpextrd(reg32Aux3, xmmAux1, k);

                if (jcp_.dataTypeSize == 4) {
                    mov(ptr[regDst], reg32Aux3);
                } else if (jcp_.dataTypeSize == 2) {
                    mov(ptr[regDst], reg16Aux3);
                } else if (jcp_.dataTypeSize == 1) {
                    mov(ptr[regDst], reg8Aux3);
                }

                add(regDst, jcp_.dataTypeSize);
                sub(regWorkAmount, 1);
            }
        }
    }
    L(lFinish);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillIndicies(Xbyak::Xmm& dst, Xbyak::Xmm& mask) {
//    Xbyak::Label lPerElements, lExit;
//
//    mov(regAux1, ptr[regParams + GET_OFF(specIndicesSizeInBytes)]);
//    mov(regAux2, regAux1);
//    sub(regAux2, vlenXmm);
//    cmp(regIdxIter, regAux2);
//    jg(lPerElements, T_NEAR);
//        uni_vmovups(dst, ptr[regIndices + regIdxIter]);
//        uni_vpmulld(dst, dst, xmmDictTypeSize);
//        // Check boundaries
//        vpcmpgtd(mask, dst, xmmMinusOne);
//        vpcmpgtd(xmmAux1, xmmAxDim, dst);
//        vpand(mask, mask, xmmAux1);
//
//        uni_vpaddd(dst, dst, xmmAxDimSum);
//        add(regIdxIter, vlenXmm);
//    cmp(regIdxIter, regAux1);
//    jl(lExit, T_NEAR);
//        uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
//        mov(regIdxIter, 0);
//    jmp(lExit, T_NEAR);
//
//    L(lPerElements);
//    for (uint8_t i = 0; i < 4; i++) {
//        Xbyak::Label insertLabel;
//
//        cmp(regIdxIter, regAux1);
//        jl(insertLabel, T_NEAR);
//            mov(regIdxIter, 0);
//            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
//
//        L(insertLabel);
//        uni_vpbroadcastd(xmmAux1, ptr[regIndices + regIdxIter]);
//        uni_vpmulld(xmmAux1, xmmAux1, xmmDictTypeSize);
//        vpcmpgtd(xmmAux3, xmmAux1, xmmMinusOne);
//        vpcmpgtd(xmmAux7, xmmAxDim, xmmAux1);
//        vpand(xmmAux3, xmmAux3, xmmAux7);
//        uni_vpaddd(xmmAux1, xmmAux1, xmmAxDimSum);
//        vinsertps(dst, dst, xmmAux1, i << 4);
//        vinsertps(mask, mask, xmmAux3, i << 4);
//        add(regIdxIter, sizeof(int));
//    }
//    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillIndicies(Xbyak::Ymm& dstIndices, Xbyak::Ymm& mask) {
//    Xbyak::Label lPerXmm, lExit;
//
//    mov(regAux1, ptr[regParams + GET_OFF(specIndicesSizeInBytes)]);
//    mov(regAux2, regAux1);
//    sub(regAux2, vlenYmm);
//    cmp(regIdxIter, regAux2);
//    jg(lPerXmm, T_NEAR);
//        uni_vmovups(dstIndices, ptr[regIndices + regIdxIter]);
//        uni_vpmulld(dstIndices, dstIndices, vmmDictTypeSize);
//        // Check boundaries
//        vpcmpgtd(mask, dstIndices, vmmMinusOne);
//        vpcmpgtd(vmmAux1, vmmAxDim, dstIndices);
//        vpand(mask, mask, vmmAux1);
//
//        uni_vpaddd(dstIndices, dstIndices, vmmAxDimSum);
//        add(regIdxIter, vlenYmm);
//    cmp(regIdxIter, regAux1);
//    jl(lExit, T_NEAR);
//        uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
//        mov(regIdxIter, 0);
//    jmp(lExit, T_NEAR);
//    L(lPerXmm);
//        for (int i = 0; i < 2; i++) {
//            fillIndicies(xmmAux0, xmmAux2);
//            vinsertf128(dstIndices, dstIndices, xmmAux0, i);
//            vinsertf128(mask, mask, xmmAux2, i);
//        }
//    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillIndicies(Xbyak::Zmm& dst, Xbyak::Opmask& mask) {
//    Xbyak::Label lPerYmm, lExit;
//
//    cmp(regIdxIter, jcp_.indicesSize - vlen);
//    jg(lPerYmm, T_NEAR);
//        uni_vmovups(dst, ptr[regIndices + regIdxIter]);
//        uni_vpmulld(dst, dst, vmmDictTypeSize);
//    vpcmpgtd(mask, dst, vmmMinusOne);
//    vpcmpgtd(kMaskAux2, vmmAxDim, dst);
//    kandd(mask, mask, kMaskAux2);
//        uni_vpaddd(dst, dst, vmmAxDimSum);
//        add(regIdxIter, vlen);
//    cmp(regIdxIter, jcp_.indicesSize);
//    jl(lExit, T_NEAR);
//        uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
//        mov(regIdxIter, 0);
//    jmp(lExit, T_NEAR);
//    L(lPerYmm);
//        for (int i = 0; i < 2; i++) {
//            fillIndicies(ymmAux2, ymmAux10);
//            vinsertf32x8(dst, dst, ymmAux2, i);
//            vinsertf32x8(vmmAux11, vmmAux11, ymmAux10, i);
//        }
//        vpmovd2m(mask, ymmAux10);
//    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillIndiciesLongIdx(Xbyak::Ymm& dstIndices, Xbyak::Ymm& mask) {
    Xbyak::Label lIdxStride, lExit;
    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    uni_vpaddd(vmmSpecIndices, vmmSpecIndices, vmmVecLen);

    mov(regAux1, ptr[regParams + GET_OFF(specIndicesSizeInBytes)]);
    add(regIdxIter, vlen);
    cmp(regIdxIter, regAux1);
    jge(lIdxStride, T_NEAR);
        // Gather spec indices
        uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndices);
        vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes); // could be movups here
//uni_vmovups(dstIndices, vmmAux1);
        // Compensate negative indices.
        vpcmpgtd(mask, vmmZeros, vmmAux1);
        vpand(mask, mask, vmmAxisDim);
        uni_vpaddd(vmmAux1, vmmAux1, mask);
        // Check boundaries
        vpcmpgtd(mask, vmmAxisDim, vmmAux1);
        vpcmpgtd(dstIndices, vmmZeros, vmmAux1);
        vpandn(mask, dstIndices, mask);

//uni_vmovups(dstIndices, vmmAux1);
        uni_vpmulld(vmmAux1, vmmAux1, vmmDictTypeSize);
        uni_vpaddd(dstIndices, vmmBeforeAxisSum, vmmAux1);
    jmp(lExit, T_NEAR);
    L(lIdxStride);
        sub(regIdxIter, regAux1);
        // increment indices
        vpcmpgtd(vmmAux0, vmmSpecIdxSize, vmmSpecIndices);
        vpandn(vmmAux0, vmmAux0, vmmOnes);
        uni_vpand(vmmAux1, vmmAux0, vmmSpecIdxSize);
//uni_vmovups(dstIndices, vmmAux1);
        uni_vpsubd(dstIndices, vmmSpecIndices, vmmAux1);
        uni_vpsubd(vmmSpecIndices, vmmSpecIndices, vmmSpecIdxSize);
//uni_vmovups(dstIndices, vmmSpecIndices);

        Xbyak::Label l1, l2;
        inc(regBetweenBatchAndAxisIter);
        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
        jl(l1, T_NEAR);
            mov(regBetweenBatchAndAxisIter, 0);
            uni_vpand(vmmAux1, vmmAux0, vmmSpecIdxSize);
            uni_vpaddd(vmmAux1, vmmIdxBatchSum, vmmAux1);
//uni_vmovups(dstIndices, vmmAux1);
            uni_vpaddd(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSize);
            uni_vpaddd(vmmAux1, vmmAux1, dstIndices);
            jmp(l2, T_NEAR);
        L(l1);
//uni_vmovups(dstIndices, vmmIdxBatchSum);
            uni_vpaddd(vmmAux1, vmmIdxBatchSum, dstIndices);
        L(l2);

//uni_vmovups(dstIndices, vmmOnes);
        vpgatherdd(dstIndices, ptr[regIndices + vmmAux1], vmmOnes);
//uni_vmovups(dstIndices, dstIndices);
        // Compensate negative indices.
        vpcmpgtd(mask, vmmZeros, dstIndices);
        vpand(mask, mask, vmmAxisDim);
        uni_vpaddd(dstIndices, dstIndices, mask);
        // Check boundaries
        vpcmpgtd(mask, vmmAxisDim, dstIndices);
        vpcmpgtd(vmmAux1, vmmZeros, dstIndices);
        vpandn(mask, vmmAux1, mask);

        uni_vpmulld(vmmAux1, dstIndices, vmmDictTypeSize);
//uni_vmovups(dstIndices, dstIndices);

        uni_vpand(vmmAux0, vmmAux0, vmmAxisAndAfterAxisSize);
        uni_vpaddd(vmmAux0, vmmAux0, vmmBeforeAxisSum);
//uni_vmovups(dstIndices, vmmAux0);
        uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAxisAndAfterAxisSize);
//uni_vmovups(dstIndices, mask);

        uni_vpaddd(dstIndices, vmmAux0, vmmAux1);
    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillIndiciesShortIdx(Xbyak::Ymm& dstIndices, Xbyak::Ymm& mask) {
    vpermd(vmmSpecIndices, vmmPermIdx, vmmSpecIndices);
    uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmBeforeAxisDiff);
    vpermd(vmmBeforeAxisDiff, vmmPermIdx, vmmBeforeAxisDiff);
    uni_vcvtdq2ps(vmmAux0, vmmBeforeAxisSum);
    uni_vcvtdq2ps(vmmAux1, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, vmmAux1);
    vroundps(vmmAux0, vmmAux0, 0x1B);
    uni_vcvtps2dq(vmmAux0, vmmAux0);
    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSize);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndices);
//uni_vmovups(dstIndices, vmmAux0);

    // Gather indices
    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
//uni_vmovups(dstIndices, vmmAux1);
    // Compensate negative indices.
    vpcmpgtd(mask, vmmZeros, vmmAux1);
    vpand(mask, mask, vmmAxisDim);
    uni_vpaddd(vmmAux1, vmmAux1, mask);
    // Check boundaries
    vpcmpgtd(mask, vmmAxisDim, vmmAux1);
    vpcmpgtd(dstIndices, vmmZeros, vmmAux1);
    vpandn(mask, dstIndices, mask);
//uni_vmovups(dstIndices, vmmAux1);

    uni_vpmulld(vmmAux1, vmmAux1, vmmDictTypeSize);
//uni_vmovups(dstIndices, vmmBeforeAxisSum);

    uni_vpaddd(dstIndices, vmmBeforeAxisSum, vmmAux1);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillIndiciesBlk(Xbyak::Ymm& dst, Xbyak::Ymm& mask) {
//    Xbyak::Label lPerXmm, lExit;
//
//    cmp(regIdxIter, jcp_.indicesSize - vlenYmm);
//    jg(lPerXmm, T_NEAR);
//        uni_vmovups(dst, ptr[regIndices + regIdxIter]);
//        uni_vpmulld(dst, dst, vmmDictTypeSize);
//        // Check boundaries
//        vpcmpgtd(mask, dst, vmmMinusOne);
//        vpcmpgtd(vmmAux1, vmmAxDim, dst);
//        vpand(mask, mask, vmmAux1);
//
//        uni_vpaddd(dst, dst, vmmAxDimSum);
//        add(regIdxIter, vlenYmm);
//    cmp(regIdxIter, jcp_.indicesSize);
//    jl(lExit, T_NEAR);
//        uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
//        mov(regIdxIter, 0);
//    jmp(lExit, T_NEAR);
//    L(lPerXmm);
//        for (int i = 0; i < 2; i++) {
//            fillIndicies(xmmAux0, xmmAux2);
//            vinsertf128(dst, dst, xmmAux0, i);
//            vinsertf128(mask, mask, xmmAux2, i);
//        }
//    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::vpGatherDD(const Xbyak::Ymm& dst) {
    fillIndicies(vmmSrcShifts, vmmGatherMask);
//mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
//uni_vmovups(ptr[regAux1], vmmSrcShifts);
    vpgatherdd(dst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::vpGatherDD(const Xbyak::Zmm& dst) {
    fillIndicies(vmmSrcShifts, kMaskAux1);
    vpgatherdd(dst | kMaskAux1, ptr[regSrc + vmmSrcShifts]);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::vpGatherDDBlk(const Xbyak::Ymm& dst) {
    fillIndiciesBlk(vmmSrcShifts, vmmGatherMask);
    vpgatherdd(dst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherAndGroup(const Xbyak::Ymm& dst, const Xbyak::Ymm& shufMask) {
    vpGatherDD(dst);
    vpshufb(dst, dst, shufMask);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherAndGroup(const Xbyak::Zmm& dst, const Xbyak::Zmm& shufMask) {
    vpGatherDD(dst);
    vpshufb(dst | kMaskOnes, dst, shufMask);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherLongIdx() {
    Xbyak::Label lDstIdxLoop, lTail, l1;
    cmp(regWorkAmount, elPerVec);
    jl(lTail, T_NEAR);

    // First iteration
    uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndices);
    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
//uni_vmovups(ptr[regDst], vmmBeforeAxisSum);
    // Compensate negative indices.
    vpcmpgtd(vmmAux0, vmmZeros, vmmAux1);
    vpand(vmmAux0, vmmAux0, vmmAxisDim);
    uni_vpaddd(vmmAux1, vmmAux1, vmmAux0);
    // Check boundaries.
    vpcmpgtd(vmmGatherMask, vmmAxisDim, vmmAux1);
    vpcmpgtd(vmmAux0, vmmZeros, vmmAux1);
    vpandn(vmmGatherMask, vmmAux0, vmmGatherMask);
    // Gather data
    uni_vpmulld(vmmAux1, vmmAux1, vmmDictTypeSize);
    uni_vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
    vpgatherdd(vmmDst, ptr[regSrc + vmmAux0], vmmGatherMask);
    uni_vmovups(ptr[regDst], vmmDst);

    add(regIdxIter, vlen);
    mov(regAux1, ptr[regParams + GET_OFF(specIndicesSizeInBytes)]);
    cmp(regIdxIter, regAux1);
    jl(l1, T_NEAR);
        uni_vpaddd(vmmAux0, vmmSpecIndices, vmmVecLen);
        vpcmpgtd(vmmAux1, vmmSpecIdxSize, vmmAux0);
        vpandn(vmmAux0, vmmAux1, vmmSpecIdxSize);
        uni_vpsubd(vmmSpecIndices, vmmSpecIndices, vmmAux0);
        sub(regIdxIter, regAux1);

        vpandn(vmmAux0, vmmAux1, vmmAxisAndAfterAxisSize);
        uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAux0);

        inc(regBetweenBatchAndAxisIter);
        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
        jl(l1, T_NEAR);
            mov(regBetweenBatchAndAxisIter, 0);
            uni_vpaddd(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSize);
    L(l1);

    add(regDst, vlen);
    sub(regWorkAmount, elPerVec);

    L(lDstIdxLoop);
    {
        cmp(regWorkAmount, elPerVec);
        jl(lTail, T_NEAR);

        fillIndiciesLongIdx(vmmSrcShifts, vmmGatherMask);
        vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        uni_vmovups(ptr[regDst], vmmDst);
//uni_vmovups(ptr[regDst], vmmSrcShifts);

        add(regDst, vlen);
        sub(regWorkAmount, elPerVec);

        jmp(lDstIdxLoop, T_NEAR);
    }

    L(lTail);
    tail();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherShortIdx16() {
    Xbyak::Label lDstIdxLoop1, lTail;
    cmp(regWorkAmount, elPerVec);
    jl(lTail, T_NEAR);

    auto& vmmShufMask = vmmAux8;
    mov(regAux1, ptr[regParams + GET_OFF(shufMask16bitUni)]);
    uni_vmovups(vmmShufMask, ptr[regAux1]);

    auto& vmmPermMask = vmmOnes;

    // First iteration
    uni_vcvtdq2ps(vmmAux0, vmmBeforeAxisSum);
    uni_vcvtdq2ps(vmmAux1, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, vmmAux1);
    vroundps(vmmAux0, vmmAux0, 0x1B);
    uni_vcvtps2dq(vmmAux0, vmmAux0);
    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSize);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndices);
//uni_vmovups(ptr[regDst], vmmAux0);
    // Gather spec indices.
    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
//uni_vmovups(ptr[regDst], vmmAux1);
    // Compensate negative indices.
    vpcmpgtd(vmmGatherMask, vmmZeros, vmmAux1);
    vpand(vmmGatherMask, vmmGatherMask, vmmAxisDim);
    uni_vpaddd(vmmAux1, vmmAux1, vmmGatherMask);
    // Check boundaries
    vpcmpgtd(vmmGatherMask, vmmAxisDim, vmmAux1);
    vpcmpgtd(vmmAux0, vmmZeros, vmmAux1);
    vpandn(vmmGatherMask, vmmAux0, vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmAux1);
    // Gather data
    uni_vpmulld(vmmAux1, vmmAux1, vmmDictTypeSize);
    uni_vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
    vpgatherdd(vmmDst, ptr[regSrc + vmmAux0], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmBeforeAxisSum);
    vpshufb(vmmDst, vmmDst, vmmShufMask);

    fillIndiciesShortIdx(vmmSrcShifts, vmmGatherMask);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
    vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
    vpshufb(vmmAux0, vmmAux0, vmmShufMask);

    vshufps(vmmDst, vmmDst, vmmAux0, 0x44);
    if (isa == x64::avx512_common) {
        mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA5)]);
    } else {
        mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA2)]);
    }
    uni_vmovups(vmmPermMask, ptr[regAux1]);
    vpermd(vmmDst, vmmPermMask, vmmDst);

    uni_vmovups(ptr[regDst], vmmDst);

    add(regDst, vlen);
    sub(regWorkAmount, elPerVec);

    L(lDstIdxLoop1);
    {
        cmp(regWorkAmount, elPerVec);
        jl(lTail, T_NEAR);

        fillIndiciesShortIdx(vmmSrcShifts, vmmGatherMask);
        vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
//add(regDst, vlen);
        vpshufb(vmmDst, vmmDst, vmmShufMask);

        fillIndiciesShortIdx(vmmSrcShifts, vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
        vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        vpshufb(vmmAux0, vmmAux0, vmmShufMask);

        vshufps(vmmDst, vmmDst, vmmAux0, 0x44);
        if (isa == x64::avx512_common) {
            mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA5)]);
        } else {
            mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA2)]);
        }
        uni_vmovups(vmmPermMask, ptr[regAux1]);
        vpermd(vmmDst, vmmPermMask, vmmDst);

        uni_vmovups(ptr[regDst], vmmDst);
//uni_vmovups(ptr[regDst], vmmSrcShifts);

        add(regDst, vlen);
        sub(regWorkAmount, elPerVec);

        jmp(lDstIdxLoop1, T_NEAR);
    }

    L(lTail);
    tail();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherLongIdx16() {
    Xbyak::Label lDstIdxLoop, lTail, l1;
    cmp(regWorkAmount, elPerVec);
    jl(lTail, T_NEAR);

    auto& vmmShufMask = vmmAux3;
    mov(regAux1, ptr[regParams + GET_OFF(shufMask16bitUni)]);
    uni_vmovups(vmmShufMask, ptr[regAux1]);

    auto& vmmPermMask = vmmOnes;

    // First iteration
    // Gather spec indices
    uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndices);
    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
//uni_vmovups(ptr[regDst], vmmAux1);
    // Compensate negative indices.
    vpcmpgtd(vmmAux0, vmmZeros, vmmAux1);
    vpand(vmmAux0, vmmAux0, vmmAxisDim);
    uni_vpaddd(vmmAux1, vmmAux1, vmmAux0);
    // Check boundaries.
    vpcmpgtd(vmmGatherMask, vmmAxisDim, vmmAux1);
    vpcmpgtd(vmmAux0, vmmZeros, vmmAux1);
    vpandn(vmmGatherMask, vmmAux0, vmmGatherMask);
    // Gather data
//uni_vmovups(ptr[regDst], vmmAux1);
    uni_vpmulld(vmmAux1, vmmAux1, vmmDictTypeSize);
    uni_vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
    vpgatherdd(vmmDst, ptr[regSrc + vmmAux0], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmAux0);
    vpshufb(vmmDst, vmmDst, vmmShufMask);
//uni_vmovups(ptr[regDst], vmmDst);

    add(regIdxIter, vlen);
    mov(regAux1, ptr[regParams + GET_OFF(specIndicesSizeInBytes)]);
    cmp(regIdxIter, regAux1);
    jl(l1, T_NEAR);
        sub(regIdxIter, regAux1);
        uni_vpaddd(vmmAux0, vmmSpecIndices, vmmVecLen);
        vpcmpgtd(vmmAux1, vmmSpecIdxSize, vmmAux0);
        vpandn(vmmAux0, vmmAux1, vmmSpecIdxSize);
        uni_vpsubd(vmmSpecIndices, vmmSpecIndices, vmmAux0);

        vpandn(vmmAux0, vmmAux1, vmmAxisAndAfterAxisSize);
        uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAux0);

        inc(regBetweenBatchAndAxisIter);
        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
        jl(l1, T_NEAR);
            mov(regBetweenBatchAndAxisIter, 0);
            uni_vpaddd(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSize);
    L(l1);

    fillIndiciesLongIdx(vmmSrcShifts, vmmGatherMask);
    vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
    vpshufb(vmmAux0, vmmAux0, vmmShufMask);

    vshufps(vmmDst, vmmDst, vmmAux0, 0x44);
    if (isa == x64::avx512_common) {
        mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA5)]);
    } else {
        mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA2)]);
    }
    uni_vmovups(vmmPermMask, ptr[regAux1]);
    vpermd(vmmDst, vmmPermMask, vmmDst);

    uni_vmovups(ptr[regDst], vmmDst);

    add(regDst, vlen);
    sub(regWorkAmount, elPerVec);

    L(lDstIdxLoop);
    {
        cmp(regWorkAmount, elPerVec);
        jl(lTail, T_NEAR);

        fillIndiciesLongIdx(vmmSrcShifts, vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
//add(regDst, vlen);
        vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        vpshufb(vmmDst, vmmDst, vmmShufMask);

        fillIndiciesLongIdx(vmmSrcShifts, vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
        vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        vpshufb(vmmAux0, vmmAux0, vmmShufMask);

        vshufps(vmmDst, vmmDst, vmmAux0, 0x44);
        if (isa == x64::avx512_common) {
            mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA5)]);
        } else {
            mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA2)]);
        }
        uni_vmovups(vmmPermMask, ptr[regAux1]);
        vpermd(vmmDst, vmmPermMask, vmmDst);

        uni_vmovups(ptr[regDst], vmmDst);

        add(regDst, vlen);
        sub(regWorkAmount, elPerVec);

        jmp(lDstIdxLoop, T_NEAR);
    }

    L(lTail);
    tail();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherShortIdx() {
    Xbyak::Label lDstIdxLoop1, lTail;
    cmp(regWorkAmount, elPerVec);
    jl(lTail, T_NEAR);

    // First iteration
    uni_vcvtdq2ps(vmmAux0, vmmBeforeAxisSum);
    uni_vcvtdq2ps(vmmAux1, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, vmmAux1);
    vroundps(vmmAux0, vmmAux0, 0x1B);
    uni_vcvtps2dq(vmmAux0, vmmAux0);
    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSize);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndices);
//uni_vmovups(ptr[regDst], vmmAux0);
    // Gather spec indices.
    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
    // Compensate negative indices.
    vpcmpgtd(vmmGatherMask, vmmZeros, vmmAux1);
    vpand(vmmGatherMask, vmmGatherMask, vmmAxisDim);
    uni_vpaddd(vmmAux1, vmmAux1, vmmGatherMask);
    // Check boundaries
    vpcmpgtd(vmmGatherMask, vmmAxisDim, vmmAux1);
    vpcmpgtd(vmmAux0, vmmZeros, vmmAux1);
    vpandn(vmmGatherMask, vmmAux0, vmmGatherMask);
    // Gather data
    uni_vpmulld(vmmAux1, vmmAux1, vmmDictTypeSize);
    uni_vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
    vpgatherdd(vmmDst, ptr[regSrc + vmmAux0], vmmGatherMask);
    uni_vmovups(ptr[regDst], vmmDst);

//    add(regIdxIter, vlen);
    add(regDst, vlen);
    sub(regWorkAmount, elPerVec);

    L(lDstIdxLoop1);
    {
        cmp(regWorkAmount, elPerVec);
        jl(lTail, T_NEAR);

        fillIndiciesShortIdx(vmmSrcShifts, vmmGatherMask);
        vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        uni_vmovups(ptr[regDst], vmmDst);
//        uni_vmovups(ptr[regDst], vmmSrcShifts);

        add(regDst, vlen);
        sub(regWorkAmount, elPerVec);

        jmp(lDstIdxLoop1, T_NEAR);
    }

    L(lTail);
    tail();
}

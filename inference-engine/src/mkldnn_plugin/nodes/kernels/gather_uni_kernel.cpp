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
    if (jcp.dataTypeSize == 2)
        dataTypeShift = 1;
    else if (jcp.dataTypeSize == 4)
        dataTypeShift = 2;
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
    mov(regSpecIdxSizeInBytes, ptr[regParams + GET_OFF(specIndicesSizeInBytes)]);

    mov(regAux1, ptr[regParams + GET_OFF(idxTypeSize)]);
    uni_vpbroadcastd(vmmAux3, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
    uni_vpbroadcastd(vmmAxisDim, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(vecLen)]);
    uni_vpbroadcastd(vmmVecLen, ptr[regAux1]);

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

    if (isa == x64::avx512_common) {
        vpcmpub(kMaskOnes, vmmGatherMask, vmmGatherMask, 0);
    }

    uni_vpxor(vmmZeros, vmmZeros, vmmZeros);

    mov(regAux1, ptr[regParams + GET_OFF(afterAxisBlockSize)]);
    Xbyak::Label lBlock_N;
    cmp(regAux1, 1);
    jg(lBlock_N, T_NEAR);
    {
        Xbyak::Label lLessThanVector, lTail, lEnd;
        cmp(regWorkAmount, elPerVec);
        jl(lTail, T_NEAR);

        cmp(regSpecIdxSizeInBytes, vlen);
        jl(lLessThanVector, T_NEAR);
            if (jcp_.dataTypeSize == 4)
                gatherLongIdx32();
            if (jcp_.dataTypeSize == 2)
                gatherLongIdx16();
            if (jcp_.dataTypeSize == 1)
                gatherLongIdx8();
            jmp(lEnd, T_NEAR);
        L(lLessThanVector);
            mov(regAux1, ptr[regParams + GET_OFF(permIdx)]);
            uni_vmovups(vmmPermIdx, ptr[regAux1]);
            mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
            uni_vmovups(vmmBeforeAxisDiff, ptr[regAux1]);
            if (jcp_.dataTypeSize != 1)
                vpslld(vmmBeforeAxisDiff, vmmBeforeAxisDiff, dataTypeShift); // multiply by type size
            mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeInBytes)]);
            uni_vpbroadcastd(vmmSrcAfterBatchSize, ptr[regAux1]);

            if (jcp_.dataTypeSize == 4)
                gatherShortIdx32();
            if (jcp_.dataTypeSize == 2)
                gatherShortIdx16();
            if (jcp_.dataTypeSize == 1)
                gatherShortIdx8();
            jmp(lEnd, T_NEAR);
        L(lTail);
            if (jcp_.dataTypeSize == 4)
                tail32();
            if (jcp_.dataTypeSize == 2)
                tail16();
            if (jcp_.dataTypeSize == 1)
                tail8();
        L(lEnd);
    }
    L(lBlock_N);

    this->postamble();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::tail32() {
    Xbyak::Label lFinish, l1;
    cmp(regWorkAmount, 0);
    je(lFinish, T_NEAR);

    mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeInBytes)]);
    uni_vpbroadcastd(vmmSrcAfterBatchSize, ptr[regAux1]);
    mov(regAux1, regWorkAmount);
    Xbyak::Reg32 reg32Aux3(regAux3.getIdx());
    mov(reg32Aux3, 0xFFFFFFFF);
    uni_vmovups(vmmOnes, vmmZeros);

    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(regAux1, 0);
        je(l1, T_NEAR);

        if (i % 4 == 0)
            uni_vmovups(vmmAux0, vmmZeros);

        vpinsrd(xmmAux0, xmmAux0, reg32Aux3, i % 4);
        if (isa == x64::avx2) {
            vinserti128(vmmOnes, vmmOnes, xmmAux0, i / 4);
        } else if (isa == x64::avx512_common) {
            vinserti64x2(vmmOnes, vmmOnes, xmmAux0, i / 4);
        }
        sub(regAux1, 1);
    }
    L(l1);
    uni_vcvtdq2ps(vmmAux0, vmmBeforeAxisSum);
    uni_vcvtdq2ps(vmmAux1, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, vmmAux1);
    if (isa == x64::avx2) {
        vroundps(vmmAux0, vmmAux0, 0x1);
    } else if (isa == x64::avx512_common) {
        vrndscaleps(vmmAux0, vmmAux0, 0x1);
    }
    uni_vcvtps2dq(vmmAux0, vmmAux0);
    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSize);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndices);
    uni_vmovups(vmmDst, vmmOnes);
    if (isa == x64::avx2) {
        vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
        normalizeRawIndices(vmmAux1, vmmGatherMask, vmmAux0);
    } else if (isa == x64::avx512_common) {
        vpmovd2m(kMaskOnes, vmmOnes);
        vpgatherdd(vmmAux1 | kMaskOnes, ptr[regIndices + vmmAux0]);
        normalizeRawIndices(zmmAux1, kGatherMask, kMaskAux1);
    }

    uni_vpaddd(vmmSrcShifts, vmmBeforeAxisSum, vmmAux1);
    if (isa == x64::avx2) {
        uni_vmovups(vmmOnes, vmmDst);
        vpand(vmmGatherMask, vmmGatherMask, vmmOnes);
        vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
    } else if (isa == x64::avx512_common) {
        vpmovd2m(kMaskAux1, vmmDst);
        kandd(kGatherMask, kGatherMask, kMaskAux1);
        vpgatherdd(vmmDst | kGatherMask, ptr[regSrc + vmmSrcShifts]);
    }

//    for (int i = 0; i < vlen / vlenYmm; i++) {
//        if (isa == x64::avx512_common)
//            vextracti32x8(ymmAux0, zmmDst, i);
//        else
//            uni_vmovups(ymmAux0, vmmDst);

        for (int j = 0; j < vlen / vlenXmm; j++) {
            if (isa == x64::avx2)
                vextracti128(xmmAux0, vmmDst, j);
            else if (isa == x64::avx512_common)
                vextracti64x2(xmmAux0, vmmDst, j);

            for (int k = 0; k < 4; k++) {
                cmp(regWorkAmount, 0);
                je(lFinish, T_NEAR);

                vpextrd(reg32Aux3, xmmAux0, k);
                mov(ptr[regDst], reg32Aux3);

                add(regDst, jcp_.dataTypeSize);
                sub(regWorkAmount, 1);
            }
        }
//    }
    L(lFinish);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::tail16() {
    auto& vmmShufMask = vmmAux8;
    mov(regAux1, ptr[regParams + GET_OFF(shufMask16bitUni)]);
    uni_vmovups(vmmShufMask, ptr[regAux1]);

    auto& vmmPermMask = vmmAux4;
    if (isa == x64::avx512_common) {
        mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA5)]);
    } else {
        mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA2)]);
    }
    uni_vmovups(vmmPermMask, ptr[regAux1]);

    Xbyak::Label lLessThanVector, l1, l2, l3, l4;

    Xbyak::Label lFinish;
    Xbyak::Reg32 reg32Aux3(regAux3.getIdx());
    Xbyak::Reg16 reg16Aux3(regAux3.getIdx());

    mov(regAux1, regWorkAmount);
    mov(reg32Aux3, 0xFFFFFFFF);
    uni_vmovups(vmmOnes, vmmZeros);

    cmp(regSpecIdxSizeInBytes, vlen);
    jl(lLessThanVector, T_NEAR);
        calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
        jmp(l1, T_NEAR);
    L(lLessThanVector);
        calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
    L(l1);
//    for (uint8_t i = 0; i < elPerVec / 2; i++) {
//        cmp(regAux1, 0);
//        je(l2, T_NEAR);
//
//        if (i % 4 == 0)
//            uni_vmovups(vmmAux0, vmmZeros);
//
//        vpinsrd(xmmAux0, xmmAux0, reg32Aux3, i % 4);
//        if (isa == x64::avx2) {
//            vinserti128(vmmOnes, vmmOnes, xmmAux0, i / 4);
//        } else if (isa == x64::avx512_common) {
//            if (i % 8 == 0)
//                uni_vmovups(ymmAux1, vmmZeros);
//            vinserti128(ymmAux1, ymmAux1, xmmAux0, i / 4);
//            vinserti32x4(vmmOnes, vmmOnes, ymmAux1, 4);
//        }
//        sub(regAux1, 1);
//    }
//    L(l2);
//    vpand(vmmGatherMask, vmmGatherMask, vmmOnes);
//    vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//    vpshufb(vmmDst, vmmDst, vmmShufMask);

    cmp(regSpecIdxSizeInBytes, vlen);
    jl(lLessThanVector, T_NEAR);
//        calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
        jmp(l3, T_NEAR);
    L(lLessThanVector);
        calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
    L(l3);

//    uni_vmovups(vmmOnes, vmmZeros);
//
//    for (uint8_t i = 0; i < elPerVec / 2; i++) {
//        cmp(regAux1, 0);
//        je(l4, T_NEAR);
//
//        if (i % 4 == 0)
//            uni_vmovups(vmmAux0, vmmZeros);
//
//        vpinsrd(xmmAux0, xmmAux0, reg32Aux3, i % 4);
//        if (isa == x64::avx2) {
//            vinserti128(vmmOnes, vmmOnes, xmmAux0, i / 4);
//        } else if (isa == x64::avx512_common) {
//            if (i % 8 == 0)
//                uni_vmovups(ymmAux1, vmmZeros);
//            vinserti128(ymmAux1, ymmAux1, xmmAux0, i / 4);
//            vinserti32x4(vmmOnes, vmmOnes, ymmAux1, 4);
//        }
//        sub(regAux1, 1);
//    }
//    L(l4);
//    vpand(vmmGatherMask, vmmGatherMask, vmmOnes);
//    vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//    vpshufb(vmmAux0, vmmAux0, vmmShufMask);
//
//    vshufps(vmmDst, vmmDst, vmmAux0, 0x44);
//    vpermd(vmmDst, vmmPermMask, vmmDst);

//    if (isa == x64::avx512_common) {
//        for (int i = 0; i < 2; i++) {
//            vextracti32x8(ymmAux0, zmmDst, i);
//            for (int j = 0; j < 2; j++) {
//                vextracti128(xmmAux1, ymmAux0, j);
//                for (int k = 0; k < 4; k++) {
//                    cmp(regWorkAmount, 0);
//                    je(lFinish, T_NEAR);
//
//                    vpextrd(reg32Aux3, xmmAux1, k);
//                    mov(ptr[regDst], reg16Aux3);
//
//                    add(regDst, jcp_.dataTypeSize);
//                    sub(regWorkAmount, 1);
//                }
//            }
//        }
//    } else {
//        for (int j = 0; j < 2; j++) {
//            vextracti128(xmmAux1, vmmDst, j);
//            for (int k = 0; k < 4; k++) {
//                cmp(regWorkAmount, 0);
//                je(lFinish, T_NEAR);
//
//                vpextrd(reg32Aux3, xmmAux1, k);
//                mov(ptr[regDst], reg16Aux3);
//
//                add(regDst, jcp_.dataTypeSize);
//                sub(regWorkAmount, 1);
//            }
//        }
//    }
    L(lFinish);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::tail8() {
    Xbyak::Label lLessThanVector, l1;
    cmp(regSpecIdxSizeInBytes, vlen);
    jl(lLessThanVector, T_NEAR);
        calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
        jmp(l1, T_NEAR);
    L(lLessThanVector);
        calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
    L(l1);

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
//        uint8_t j = 0;
        for (uint8_t i = 0; i < elPerVec; i++) {
            cmp(regAux1, 0);
            je(l1, T_NEAR);

            vpinsrd(xmmAux0, xmmAux0, reg32Aux3, i % 4);
            if ((i + 1) % 4 == 0) {
                vinserti128(vmmOnes, vmmOnes, xmmAux0, i / 4);
            }
            sub(regAux1, 1);
        }
        L(l1);
        vpand(vmmGatherMask, vmmGatherMask, vmmOnes);
        vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
    }

    if (isa == x64::avx512_common) {
        for (int i = 0; i < 2; i++) {
            vextracti32x8(ymmAux0, zmmDst, i);
            for (int j = 0; j < 2; j++) {
                vextracti128(xmmAux1, ymmAux0, j);
                for (int k = 0; k < 4; k++) {
                    cmp(regWorkAmount, 0);
                    je(lFinish, T_NEAR);

                    vpextrd(reg32Aux3, xmmAux1, k);
                    mov(ptr[regDst], reg8Aux3);

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
                mov(ptr[regDst], reg8Aux3);

                add(regDst, jcp_.dataTypeSize);
                sub(regWorkAmount, 1);
            }
        }
    }
    L(lFinish);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftLong(Xbyak::Ymm& dstIndices, Xbyak::Ymm& dstMask, Xbyak::Ymm& idxMask) {
    Xbyak::Label lIdxStride, lExit;
//    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    uni_vpaddd(vmmSpecIndices, vmmSpecIndices, vmmVecLen);

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeInBytes);
    jge(lIdxStride, T_NEAR);
        uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndices);
        vpgatherdd(dstIndices, ptr[regIndices + vmmAux0], idxMask); // TODO: could be movups here
        normalizeRawIndices(dstIndices, dstMask, vmmAux0);
        uni_vpaddd(dstIndices, dstIndices, vmmBeforeAxisSum);
    jmp(lExit, T_NEAR);
    L(lIdxStride);
        sub(regIdxIter, regSpecIdxSizeInBytes);
        vpcmpgtd(vmmAux0, vmmSpecIdxSize, vmmSpecIndices);
        vpandn(vmmAux1, vmmAux0, vmmSpecIdxSize);
        uni_vpsubd(dstIndices, vmmSpecIndices, vmmAux1);
        uni_vpsubd(vmmSpecIndices, vmmSpecIndices, vmmSpecIdxSize);

        Xbyak::Label l1;
        inc(regBetweenBatchAndAxisIter);
        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
        uni_vpaddd(vmmAux1, vmmIdxBatchSum, dstIndices);
        jl(l1, T_NEAR);
            mov(regBetweenBatchAndAxisIter, 0);
            vpandn(dstIndices, vmmAux0, vmmSpecIdxSize);
            uni_vpaddd(vmmAux1, vmmAux1, dstIndices);
            uni_vpaddd(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSize);
        L(l1);

        vpgatherdd(dstIndices, ptr[regIndices + vmmAux1], idxMask);
        normalizeRawIndices(dstIndices, dstMask, vmmAux1);

        vpandn(vmmAux0, vmmAux0, vmmAxisAndAfterAxisSize);
        uni_vpaddd(vmmAux0, vmmAux0, vmmBeforeAxisSum);
        uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAxisAndAfterAxisSize);

        uni_vpaddd(dstIndices, dstIndices, vmmAux0);
    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftLong(Xbyak::Zmm& dstIndices, Xbyak::Opmask& dstMask, Xbyak::Opmask& idxMask) {
    Xbyak::Label lIdxStride, lExit;
//    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    vpaddd(vmmSpecIndices, vmmSpecIndices, vmmVecLen);

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeInBytes);
    jge(lIdxStride, T_NEAR);
        vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndices);
        vpgatherdd(dstIndices | idxMask, ptr[regIndices + vmmAux0]); // TODO: could be vmovups here
        normalizeRawIndices(dstIndices, dstMask, kMaskAux1);
        vpaddd(dstIndices, dstIndices, vmmBeforeAxisSum);
    jmp(lExit, T_NEAR);
    L(lIdxStride);
        sub(regIdxIter, regSpecIdxSizeInBytes);
        vpcmpd(kMaskAux1, vmmSpecIdxSize, vmmSpecIndices, 2); // 2 - LE
        vpaddd(vmmAux1, vmmIdxBatchSum, vmmSpecIndices);
        vpsubd(vmmAux1 | kMaskAux1, vmmAux1, vmmSpecIdxSize);
        vpsubd(vmmSpecIndices, vmmSpecIndices, vmmSpecIdxSize);

        Xbyak::Label l1;
        inc(regBetweenBatchAndAxisIter);
        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
        jl(l1, T_NEAR);
            mov(regBetweenBatchAndAxisIter, 0);
            vpaddd(vmmAux1 | kMaskAux1, vmmAux1, vmmSpecIdxSize);
            vpaddd(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSize);
        L(l1);

        vpgatherdd(dstIndices | idxMask, ptr[regIndices + vmmAux1]);
        normalizeRawIndices(dstIndices, dstMask, kMaskAux2);

        vpaddd(dstIndices, dstIndices, vmmBeforeAxisSum);
        vpaddd(dstIndices | kMaskAux1, dstIndices, vmmAxisAndAfterAxisSize);
        vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAxisAndAfterAxisSize);
    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftShort(Xbyak::Ymm& dstIndices, Xbyak::Ymm& dstMask) {
    vpermd(vmmSpecIndices, vmmPermIdx, vmmSpecIndices);
    uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmBeforeAxisDiff);
    vpermd(vmmBeforeAxisDiff, vmmPermIdx, vmmBeforeAxisDiff);
    // Calculate batch indices
    uni_vcvtdq2ps(vmmAux0, vmmBeforeAxisSum);
    uni_vcvtdq2ps(dstIndices, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, dstIndices);
    if (isa == x64::avx2) {
        vroundps(vmmAux0, vmmAux0, 0x1);
    } else if (isa == x64::avx512_common) {
        vrndscaleps(vmmAux0, vmmAux0, 0x1);
    }
    uni_vcvtps2dq(vmmAux0, vmmAux0);

    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSize);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndices);

    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    vpgatherdd(dstIndices, ptr[regIndices + vmmAux0], vmmOnes);
    normalizeRawIndices(dstIndices, dstMask, vmmAux0);
    uni_vpaddd(dstIndices, dstIndices, vmmBeforeAxisSum);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftShort(Xbyak::Zmm& dstIndices, Xbyak::Opmask& dstMask) {
    vpermd(vmmSpecIndices, vmmPermIdx, vmmSpecIndices);
    uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmBeforeAxisDiff);
    vpermd(vmmBeforeAxisDiff, vmmPermIdx, vmmBeforeAxisDiff);
    // Calculate batch indices
    uni_vcvtdq2ps(vmmAux0, vmmBeforeAxisSum);
    uni_vcvtdq2ps(dstIndices, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, dstIndices);
    if (isa == x64::avx2) {
        vroundps(vmmAux0, vmmAux0, 0x1);
    } else if (isa == x64::avx512_common) {
        vrndscaleps(vmmAux0, vmmAux0, 0x1);
    }
    uni_vcvtps2dq(vmmAux0, vmmAux0);

    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSize);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndices);

    vpcmpb(kMaskOnes, vmmOnes, vmmOnes, 0);
    vpgatherdd(dstIndices | kMaskOnes, ptr[regIndices + vmmAux0]);
    normalizeRawIndices(dstIndices, dstMask, kMaskAux1);
    uni_vpaddd(dstIndices, dstIndices, vmmBeforeAxisSum);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::normalizeRawIndices(Xbyak::Ymm& rawIndices, Xbyak::Ymm& dstMask, Xbyak::Ymm& aux) {
    // Compensate negative indices.
    vpcmpgtd(aux, vmmZeros, rawIndices);
    vpand(aux, aux, vmmAxisDim);
    uni_vpaddd(rawIndices, rawIndices, aux);
    // Check boundaries.
    vpcmpgtd(dstMask, vmmAxisDim, rawIndices);
    vpcmpgtd(aux, vmmZeros, rawIndices);
    vpandn(dstMask, aux, dstMask);
    // Multiply by type size.
    if (jcp_.dataTypeSize != 1)
        vpslld(rawIndices, rawIndices, dataTypeShift);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::normalizeRawIndices(Xbyak::Zmm& rawIndices, Xbyak::Opmask& dstMask, Xbyak::Opmask& auxMask) {
    // Compensate negative indices.
    vpcmpgtd(auxMask, vmmZeros, rawIndices);
    vpaddd(rawIndices | auxMask, rawIndices, vmmAxisDim);
    // Check boundaries.
    vpcmpgtd(auxMask, vmmAxisDim, rawIndices);
    vpcmpd(dstMask | auxMask, vmmZeros, rawIndices, 2); // 2 - LE
    // Multiply by type size.
    if (jcp_.dataTypeSize != 1)
        vpslld(rawIndices, rawIndices, dataTypeShift);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherLongIdx32() {
    Xbyak::Label lDstIdxLoop, lTail, l1, lEnd;

    // First iteration
    uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndices);
    if (isa == x64::avx2) {
        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
        normalizeRawIndices(vmmAux1, vmmGatherMask, vmmAux0);
        uni_vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
        vpgatherdd(vmmDst, ptr[regSrc + vmmAux0], vmmGatherMask);
    } else if (isa == x64::avx512_common) {
        vpcmpb(kMaskOnes, vmmOnes, vmmOnes, 0);
        vpgatherdd(vmmAux1 | kMaskOnes, ptr[regIndices + vmmAux0]);
        normalizeRawIndices(zmmAux1, kGatherMask, kMaskAux1);
        vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
        vpgatherdd(vmmDst | kGatherMask, ptr[regSrc + vmmAux0]);
    }
    uni_vmovups(ptr[regDst], vmmDst);

    add(regDst, vlen);
    sub(regWorkAmount, elPerVec);

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeInBytes);
    jl(l1, T_NEAR);
        uni_vpaddd(vmmAux0, vmmSpecIndices, vmmVecLen);
        if (isa == x64::avx2) {
            vpcmpgtd(vmmAux1, vmmSpecIdxSize, vmmAux0);
            vpandn(vmmAux0, vmmAux1, vmmSpecIdxSize);
            uni_vpsubd(vmmSpecIndices, vmmSpecIndices, vmmAux0);
            vpandn(vmmAux0, vmmAux1, vmmAxisAndAfterAxisSize);
            uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAux0);
        } else if (isa == x64::avx512_common) {
            vpcmpd(kMaskAux1, vmmSpecIdxSize, vmmAux0, 2); // 2 - LE
            vpsubd(vmmSpecIndices | kMaskAux1, vmmSpecIndices, vmmSpecIdxSize);
            vpaddd(vmmBeforeAxisSum | kMaskAux1, vmmBeforeAxisSum, vmmAxisAndAfterAxisSize);
        }

        sub(regIdxIter, regSpecIdxSizeInBytes);
        inc(regBetweenBatchAndAxisIter);
        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
        jl(l1, T_NEAR);
            mov(regBetweenBatchAndAxisIter, 0);
            uni_vpaddd(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSize);
    L(l1);

    L(lDstIdxLoop);
    {
        cmp(regWorkAmount, elPerVec);
        jl(lTail, T_NEAR);

        if (isa == x64::avx2) {
            uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
            calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
            vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        } else if (isa == x64::avx512_common) {
            vpcmpb(kMaskOnes, vmmOnes, vmmOnes, 0);
            calcSrcShiftLong(zmmSrcShifts, kGatherMask, kMaskOnes);
            vpgatherdd(vmmDst | kGatherMask, ptr[regSrc + vmmSrcShifts]);
        }

        uni_vmovups(ptr[regDst], vmmDst);

        add(regDst, vlen);
        sub(regWorkAmount, elPerVec);

        jmp(lDstIdxLoop, T_NEAR);
    }

    L(lTail);
        cmp(regWorkAmount, 0);
        je(lEnd, T_NEAR);
        Xbyak::Label l2;
        uni_vpaddd(vmmSpecIndices, vmmSpecIndices, vmmVecLen);

        add(regIdxIter, vlen);
        cmp(regIdxIter, regSpecIdxSizeInBytes);
        jl(l2, T_NEAR);
            sub(regIdxIter, regSpecIdxSizeInBytes);
            if (isa == x64::avx2) {
                vpcmpgtd(vmmAux0, vmmSpecIdxSize, vmmSpecIndices);
                vpandn(vmmAux1, vmmAux0, vmmSpecIdxSize);
                uni_vpsubd(vmmSpecIndices, vmmSpecIndices, vmmAux1);
                vpandn(vmmAux0, vmmAux0, vmmAxisAndAfterAxisSize);
                uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmAux0);
            } else if (isa == x64::avx512_common) {
                vpcmpd(kMaskAux1, vmmSpecIdxSize, vmmSpecIndices, 2); // 2 - LE
                vpsubd(vmmSpecIndices | kMaskAux1, vmmSpecIndices, vmmSpecIdxSize);
                vpaddd(vmmBeforeAxisSum | kMaskAux1, vmmBeforeAxisSum, vmmAxisAndAfterAxisSize);
            }
        L(l2);
        tail32();
    L(lEnd);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherLongIdx16() {
    Xbyak::Label lDstIdxLoop, lTail, l1;
    cmp(regWorkAmount, elPerVec);
    jl(lTail, T_NEAR);

    auto& vmmShufMask = vmmAux3;
    mov(regAux1, ptr[regParams + GET_OFF(shufMask16bitUni)]);
    uni_vmovups(vmmShufMask, ptr[regAux1]);

    auto& vmmPermMask = vmmAux4;
    if (isa == x64::avx512_common) {
        mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA5)]);
    } else {
        mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA2)]);
    }
    uni_vmovups(vmmPermMask, ptr[regAux1]);

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
    vpslld(vmmAux1, vmmAux1, 1); // multiply by type size
    uni_vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
    vpgatherdd(vmmDst, ptr[regSrc + vmmAux0], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmAux0);
    vpshufb(vmmDst, vmmDst, vmmShufMask);
//uni_vmovups(ptr[regDst], vmmDst);

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeInBytes);
    jl(l1, T_NEAR);
        sub(regIdxIter, regSpecIdxSizeInBytes);
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

    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
    vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
    vpshufb(vmmAux0, vmmAux0, vmmShufMask);

    vshufps(vmmDst, vmmDst, vmmAux0, 0x44);
    vpermd(vmmDst, vmmPermMask, vmmDst);

    uni_vmovups(ptr[regDst], vmmDst);

    add(regDst, vlen);
    sub(regWorkAmount, elPerVec);

    L(lDstIdxLoop);
    {
        cmp(regWorkAmount, elPerVec);
        jl(lTail, T_NEAR);

        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
//add(regDst, vlen);
        vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        vpshufb(vmmDst, vmmDst, vmmShufMask);

        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
        vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        vpshufb(vmmAux0, vmmAux0, vmmShufMask);

        vshufps(vmmDst, vmmDst, vmmAux0, 0x44);
        vpermd(vmmDst, vmmPermMask, vmmDst);

        uni_vmovups(ptr[regDst], vmmDst);

        add(regDst, vlen);
        sub(regWorkAmount, elPerVec);

        jmp(lDstIdxLoop, T_NEAR);
    }

    L(lTail);
//    tail16();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherLongIdx8() {
    Xbyak::Label lDstIdxLoop, lTail, l1;
    cmp(regWorkAmount, elPerVec);
    jl(lTail, T_NEAR);

    auto& vmmShufMask = vmmAux4;
    mov(regAux1, ptr[regParams + GET_OFF(shufMask8bitUni)]);
    uni_vmovups(vmmShufMask, ptr[regAux1]);

    auto& vmmPermMask = vmmOnes;

    // First iteration
    // Gather spec indices
    uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndices);
    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
//uni_vmovups(ptr[regDst], vmmAux0);
//   Compensate negative indices.
    vpcmpgtd(vmmAux0, vmmZeros, vmmAux1);
    vpand(vmmAux0, vmmAux0, vmmAxisDim);
    uni_vpaddd(vmmAux1, vmmAux1, vmmAux0);
    // Check boundaries.
    vpcmpgtd(vmmGatherMask, vmmAxisDim, vmmAux1);
    vpcmpgtd(vmmAux0, vmmZeros, vmmAux1);
    vpandn(vmmGatherMask, vmmAux0, vmmGatherMask);
    // Gather data
//uni_vmovups(ptr[regDst], vmmAux1);
    uni_vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
    vpgatherdd(vmmDst, ptr[regSrc + vmmAux0], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmAux0);
    vpshufb(vmmDst, vmmDst, vmmShufMask);
//uni_vmovups(ptr[regDst], vmmDst);

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeInBytes);
    jl(l1, T_NEAR);
        sub(regIdxIter, regSpecIdxSizeInBytes);
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

    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
    vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
    vpshufb(vmmAux0, vmmAux0, vmmShufMask);

    vshufps(vmmDst, vmmDst, vmmAux0, 0x0);

    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
    vpgatherdd(vmmAux3, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
    vpshufb(vmmAux3, vmmAux3, vmmShufMask);

    uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
    calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
    vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
    vpshufb(vmmAux0, vmmAux0, vmmShufMask);

    vshufps(vmmAux3, vmmAux3, vmmAux0, 0x0);

    vshufps(vmmDst, vmmDst, vmmAux3, 0x88);

    if (isa == x64::avx512_common) {
        mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA5)]);
    } else {
        mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA2)]);
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

        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
//add(regDst, vlen);
        vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        vpshufb(vmmDst, vmmDst, vmmShufMask);

        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
//add(regDst, vlen);
        vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        vpshufb(vmmAux0, vmmAux0, vmmShufMask);

        vshufps(vmmDst, vmmDst, vmmAux0, 0x0);

        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
        vpgatherdd(vmmAux3, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
//add(regDst, vlen);
        vpshufb(vmmAux3, vmmAux3, vmmShufMask);

        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        calcSrcShiftLong(vmmSrcShifts, vmmGatherMask, vmmOnes);
        vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
        vpshufb(vmmAux0, vmmAux0, vmmShufMask);

        vshufps(vmmAux3, vmmAux3, vmmAux0, 0x0);

        vshufps(vmmDst, vmmDst, vmmAux3, 0x88);

        if (isa == x64::avx512_common) {
            mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA5)]);
        } else {
            mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA2)]);
        }
        uni_vmovups(vmmPermMask, ptr[regAux1]);
        vpermd(vmmDst, vmmPermMask, vmmDst);

        uni_vmovups(ptr[regDst], vmmDst);

        add(regDst, vlen);
        sub(regWorkAmount, elPerVec);

        jmp(lDstIdxLoop, T_NEAR);
    }

    L(lTail);
    tail8();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherShortIdx32() {
    Xbyak::Label lDstIdxLoop, lTail, lEnd;

    // First iteration
    uni_vcvtdq2ps(vmmAux0, vmmBeforeAxisSum);
    uni_vcvtdq2ps(vmmAux1, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, vmmAux1);
    if (isa == x64::avx2) {
        vroundps(vmmAux0, vmmAux0, 0x1);
    } else if (isa == x64::avx512_common) {
        vrndscaleps(vmmAux0, vmmAux0, 0x1);
    }
    uni_vcvtps2dq(vmmAux0, vmmAux0);
    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSize);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndices);
    // Gather spec indices.
    if (isa == x64::avx2) {
        uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        vpgatherdd(vmmAux1, ptr[regIndices + vmmAux0], vmmOnes);
        normalizeRawIndices(vmmAux1, vmmGatherMask, vmmAux0);
    } else if (isa == x64::avx512_common) {
        vpcmpb(kMaskOnes, vmmOnes, vmmOnes, 0);
        vpgatherdd(vmmAux1 | kMaskOnes, ptr[regIndices + vmmAux0]);
        normalizeRawIndices(zmmAux1, kGatherMask, kMaskAux1);
    }
    // Gather data
    uni_vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
    if (isa == x64::avx2) {
        vpgatherdd(vmmDst, ptr[regSrc + vmmAux0], vmmGatherMask);
    } else if (isa == x64::avx512_common) {
        vpgatherdd(vmmDst | kGatherMask, ptr[regSrc + vmmAux0]);
    }
    uni_vmovups(ptr[regDst], vmmDst);

    add(regDst, vlen);
    sub(regWorkAmount, elPerVec);

    L(lDstIdxLoop);
    {
        cmp(regWorkAmount, elPerVec);
        jl(lTail, T_NEAR);

        if (isa == x64::avx2) {
            calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
            vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        } else if (isa == x64::avx512_common) {
            calcSrcShiftShort(zmmSrcShifts, kGatherMask);
            vpgatherdd(vmmDst | kGatherMask, ptr[regSrc + vmmSrcShifts]);
        }
        uni_vmovups(ptr[regDst], vmmDst);

        add(regDst, vlen);
        sub(regWorkAmount, elPerVec);

        jmp(lDstIdxLoop, T_NEAR);
    }

    L(lTail);
    cmp(regWorkAmount, 0);
    je(lEnd, T_NEAR);
        vpermd(vmmSpecIndices, vmmPermIdx, vmmSpecIndices);
        uni_vpaddd(vmmBeforeAxisSum, vmmBeforeAxisSum, vmmBeforeAxisDiff);
        vpermd(vmmBeforeAxisDiff, vmmPermIdx, vmmBeforeAxisDiff);
        tail32();
    L(lEnd);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherShortIdx16() {
    Xbyak::Label lDstIdxLoop1, lTail;
    cmp(regWorkAmount, elPerVec);
    jl(lTail, T_NEAR);

    auto& vmmShufMask = vmmAux8;
    mov(regAux1, ptr[regParams + GET_OFF(shufMask16bitUni)]);
    uni_vmovups(vmmShufMask, ptr[regAux1]);

    auto& vmmPermMask = vmmAux4;
    if (isa == x64::avx512_common) {
        mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA5)]);
    } else {
        mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA2)]);
    }
    uni_vmovups(vmmPermMask, ptr[regAux1]);

    // First iteration
    uni_vcvtdq2ps(vmmAux0, vmmBeforeAxisSum);
    uni_vcvtdq2ps(vmmAux1, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, vmmAux1);
    if (isa == x64::avx2) {
        vroundps(vmmAux0, vmmAux0, 0x1);
    } else if (isa == x64::avx512_common) {
        vrndscaleps(vmmAux0, vmmAux0, 0x1);
    }
    uni_vcvtps2dq(vmmAux0, vmmAux0);
    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSize);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndices);
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
    vpslld(vmmAux1, vmmAux1, 1); // multiply by type size
    uni_vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
    vpgatherdd(vmmDst, ptr[regSrc + vmmAux0], vmmGatherMask);
    vpshufb(vmmDst, vmmDst, vmmShufMask);

    calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
    vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
    vpshufb(vmmAux0, vmmAux0, vmmShufMask);

    vshufps(vmmDst, vmmDst, vmmAux0, 0x44);
    vpermd(vmmDst, vmmPermMask, vmmDst);

    uni_vmovups(ptr[regDst], vmmDst);

    add(regDst, vlen);
    sub(regWorkAmount, elPerVec);

    L(lDstIdxLoop1);
    {
        cmp(regWorkAmount, elPerVec);
        jl(lTail, T_NEAR);

        calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
        vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
//add(regDst, vlen);
        vpshufb(vmmDst, vmmDst, vmmShufMask);

        calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
        vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        vpshufb(vmmAux0, vmmAux0, vmmShufMask);

        vshufps(vmmDst, vmmDst, vmmAux0, 0x44);
        vpermd(vmmDst, vmmPermMask, vmmDst);

        uni_vmovups(ptr[regDst], vmmDst);
//uni_vmovups(ptr[regDst], vmmSrcShifts);

        add(regDst, vlen);
        sub(regWorkAmount, elPerVec);

        jmp(lDstIdxLoop1, T_NEAR);
    }

    L(lTail);
//    tail16();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::gatherShortIdx8() {
    Xbyak::Label lDstIdxLoop1, lTail;
    cmp(regWorkAmount, elPerVec);
    jl(lTail, T_NEAR);

    auto& vmmShufMask = vmmAux4;
    mov(regAux1, ptr[regParams + GET_OFF(shufMask8bitUni)]);
    uni_vmovups(vmmShufMask, ptr[regAux1]);

    auto& vmmPermMask = vmmOnes;

    // First iteration
    uni_vcvtdq2ps(vmmAux0, vmmBeforeAxisSum);
    uni_vcvtdq2ps(vmmAux1, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, vmmAux1);
    if (isa == x64::avx2) {
        vroundps(vmmAux0, vmmAux0, 0x1);
    } else if (isa == x64::avx512_common) {
        vrndscaleps(vmmAux0, vmmAux0, 0x1);
    }
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
    uni_vpaddd(vmmAux0, vmmAux1, vmmBeforeAxisSum);
    vpgatherdd(vmmDst, ptr[regSrc + vmmAux0], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmDst);
    vpshufb(vmmDst, vmmDst, vmmShufMask);

    calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
    vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmAux0);
    vpshufb(vmmAux0, vmmAux0, vmmShufMask);

    vshufps(vmmDst, vmmDst, vmmAux0, 0x0);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmDst);

    calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
    vpgatherdd(vmmAux8, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmAux8);
    vpshufb(vmmAux8, vmmAux8, vmmShufMask);

    calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
    vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmAux0);
    vpshufb(vmmAux0, vmmAux0, vmmShufMask);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmAux0);

    vshufps(vmmAux8, vmmAux8, vmmAux0, 0x0);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmAux8);

    vshufps(vmmDst, vmmDst, vmmAux8, 0x88);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmDst);

    if (isa == x64::avx512_common) {
        mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA5)]);
    } else {
        mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA2)]);
    }
    uni_vmovups(vmmPermMask, ptr[regAux1]);
    vpermd(vmmDst, vmmPermMask, vmmDst);
//add(regDst, vlen);
//uni_vmovups(ptr[regDst], vmmDst);

    uni_vmovups(ptr[regDst], vmmDst);

    add(regDst, vlen);
    sub(regWorkAmount, elPerVec);

    L(lDstIdxLoop1);
    {
        cmp(regWorkAmount, elPerVec);
        jl(lTail, T_NEAR);

        calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
        vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
//add(regDst, vlen);
        vpshufb(vmmDst, vmmDst, vmmShufMask);

        calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
//add(regDst, vlen);
        vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        vpshufb(vmmAux0, vmmAux0, vmmShufMask);

        vshufps(vmmDst, vmmDst, vmmAux0, 0x0);

        calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
//add(regDst, vlen);
        vpgatherdd(vmmAux8, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        vpshufb(vmmAux8, vmmAux8, vmmShufMask);

        calcSrcShiftShort(vmmSrcShifts, vmmGatherMask);
//uni_vmovups(ptr[regDst], vmmSrcShifts);
        vpgatherdd(vmmAux0, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
        vpshufb(vmmAux0, vmmAux0, vmmShufMask);

        vshufps(vmmAux8, vmmAux8, vmmAux0, 0x0);

        vshufps(vmmDst, vmmDst, vmmAux8, 0x88);

        if (isa == x64::avx512_common) {
            mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA5)]);
        } else {
            mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA2)]);
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
    tail8();
}

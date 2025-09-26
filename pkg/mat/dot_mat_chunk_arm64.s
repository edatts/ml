//go:build arm64

#include "textflag.h"
#include "go_asm.h"

// func DotMatChunk8(A1, B1, B2, B3, B4, out []float32)
TEXT ·DotMatChunk8(SB),NOSPLIT,$0-144
    // A1 is part of our A Matrix row, each of B1-4 is part of one of our B
    // matrix rows, and out is a slice of the corresponding output row.

    // Move our slice pointers into their own registers
    MOVD A1w+0(FP), R0
    MOVD B1+24(FP), R1
    MOVD B2+48(FP), R2
    MOVD B3+72(FP), R3
    MOVD B4+96(FP), R4
    MOVD out+120(FP), R5
    
    // Load our vector registers
    VLD1 (R0), [V0.S4]
    VLD1 (R1), [V1.S4, V2.S4]
    VLD1 (R2), [V3.S4, V4.S4]
    VLD1 (R3), [V5.S4, V6.S4]
    VLD1 (R4), [V7.S4, V8.S4]
    VLD1 (R5), [V9.S4, V10.S4]

    WORD $0x4F801029 // VFMLA V0.S[0], V1.S4, V9.S4
    WORD $0x4FA01069 // VFMLA V0.S[1], V3.S4, V9.S4
    WORD $0x4F8018A9 // VFMLA V0.S[2], V5.S4, V9.S4
    WORD $0x4FA018E9 // VFMLA V0.S[3], V7.S4, V9.S4

    WORD $0x4F80104A // VFMLA V0.S[0], V2.S4, V10.S4
    WORD $0x4FA0108A // VFMLA V0.S[1], V4.S4, V10.S4
    WORD $0x4F8018CA // VFMLA V0.S[2], V6.S4, V10.S4
    WORD $0x4FA0190A // VFMLA V0.S[3], V8.S4, V10.S4

    VST1 [V9.S4, V10.S4], (R5)
    
    RET

// func DotMatChunk4(A1, B1, B2, B3, B4, out []float32)
TEXT ·DotMatChunk4(SB),NOSPLIT,$0-144
    // Move our slice pointers into their own registers
    MOVD A1w+0(FP), R0
    MOVD B1+24(FP), R1
    MOVD B2+48(FP), R2
    MOVD B3+72(FP), R3
    MOVD B4+96(FP), R4
    MOVD out+120(FP), R5
    
    // Load our vector registers
    VLD1 (R0), [V0.S4]
    VLD1 (R1), [V1.S4]
    VLD1 (R2), [V2.S4]
    VLD1 (R3), [V3.S4]
    VLD1 (R4), [V4.S4]
    VLD1 (R5), [V5.S4]
    
    WORD $0x4F801025 // VFMLA V0.S[0], V1.S4, V5.S4
    WORD $0x4FA01045 // VFMLA V0.S[1], V2.S4, V5.S4
    WORD $0x4F801865 // VFMLA V0.S[2], V3.S4, V5.S4
    WORD $0x4FA01885 // VFMLA V0.S[3], V4.S4, V5.S4

    VST1 [V5.S4], (R5)
    
    RET

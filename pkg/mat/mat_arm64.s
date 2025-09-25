//go:build arm64

#include "textflag.h"
#include "go_asm.h"

// func DotVec4F64(A []float64, B1, B2, B3, B4 float64) float64
TEXT ·DotVec4F64(SB),NOSPLIT,$0-64

    // Move our A slice base pointer into R0
    MOVD A+0(FP), R0

    // Load 256 bits from the A slice into V0 and V1 
    VLD1 (R0), [V0.D2, V1.D2]

    // Load our other floats into V2 and V3
    MOVD $B1+24(FP), R1
    VLD1 (R1), [V2.D2, V3.D2]

    // Multiply our vectors together and accumulate the results
    VMOV ZR, V4.D2 // Zero the V4 register
    VFMLA V0.D2, V2.D2, V4.D2
    VFMLA V1.D2, V3.D2, V4.D2

    // Add our vector elements together
    VMOV V4.D[0], R2
    VMOV V4.D[1], R3
    FMOVD R2, F0
    FMOVD R3, F1
    FADDD F0, F1, F2
    FMOVD F2, ret+56(FP)
    RET

// func DotVec4F32(A []float32, B1, B2, B3, B4 float32) float32
TEXT ·DotVec4F32(SB),NOSPLIT,$0-44
    // Move our A slice base pointer into R0
    MOVD A+0(FP), R0

    // Load 128 bits from the A slice into V0
    VLD1 (R0), [V0.S4]

    // Load our other floats into V1
    MOVD $B1+24(FP), R1
    VLD1 (R1), [V1.S4]

    // Multiply our vectors together and accumulate the results
    VMOV ZR, V2.S4 // Zero the V2 register
    VFMLA V0.S4, V1.S4, V2.S4

    // Add our vector elements together
    WORD $0x4EB1B842; // addv s2, v2.4s
    // VMOV V4.S[0], R2
    // VMOV V4.S[1], R3
    // VMOV V4.S[2], R4
    // VMOV V4.S[3], R5
    // FMOVS R2, F0
    // FMOVS R3, F1
    // FMOVS R4, F2
    // FMOVS R5, F3
    // FADDS F0, F1, F4
    // FADDS F2, F3, F5
    // FADDS F4, F5, F6
    FMOVS F2, ret+40(FP)
    RET

// func DotVec4F32Slc(A, B []float32) float32
TEXT ·DotVec4F32Slc(SB),NOSPLIT,$0-52
    // Move alice pointers to R0 and R1
    MOVD A+0(FP), R0
    MOVD B+24(FP), R1

    // Load floats into V0 and V1 
    VLD1 (R0), [V0.S4]
    VLD1 (R1), [V1.S4]

    // Multiply our vectors together and accumulate the results
    VMOV ZR, V2.S4 // Zero the V2 register
    VFMLA V0.S4, V1.S4, V2.S4

    // Add our vector elements together
    // WORD $0x4EB1B842; // addv s2, v2.4s // Doesn't work for some reason...
    VMOV V2.S[0], R2
    VMOV V2.S[1], R3
    VMOV V2.S[2], R4
    VMOV V2.S[3], R5
    FMOVS R2, F0
    FMOVS R3, F1
    FMOVS R4, F2
    FMOVS R5, F3
    FADDS F0, F1, F4
    FADDS F2, F3, F5
    FADDS F4, F5, F6
    FMOVS F6, ret+48(FP)
    RET

// func DotVec4F32NoSlc(A1, A2, A3, A4, B1, B2, B3, B4 float32) float32
TEXT ·DotVec4F32NoSlc(SB),NOSPLIT,$0-36
    // We will always have 4 floats in the slice so can ignore the length
    
    // Move FP address to R0
    MOVD $A+0(FP), R0

    // Load 256 bits into V0 and V1 
    VLD1 (R0), [V0.S4, V1.S4]
    

    // Multiply our vectors together and accumulate the results
    VMOV ZR, V2.S4 // Zero the V2 register
    VFMLA V0.S4, V1.S4, V2.S4

    // Add our vector elements together
    // VADDV V2.S4, F2
    WORD $0x4EB1B842; // addv s2, v2.4s
    VMOV V2.S[0], R2
    VMOV V2.S[1], R3
    VMOV V2.S[2], R4
    VMOV V2.S[3], R5
    FMOVS R2, F0
    FMOVS R3, F1
    FMOVS R4, F2
    FMOVS R5, F3
    FADDS F0, F1, F4
    FADDS F2, F3, F5
    FADDS F4, F5, F6
    FMOVS F6, ret+32(FP)
    RET

// func DotVec8F32(A []float32, B1, B2, B3, B4, B5, B6, B7, B8 float32) float32
TEXT ·DotVec8F32(SB),NOSPLIT,$0-60
    // We will always have 4 floats in the slice so can ignore the length
    
    // Move our A slice base pointer into R0
    MOVD A+0(FP), R0

    // Load 256 bits from the A slice into V0 and V1 
    VLD1 (R0), [V0.S4, V1.S4]

    // Load our other floats into V2 and V3
    MOVD $B1+24(FP), R1
    VLD1 (R1), [V2.S4, V3.S4]

    // Multiply our vectors together and accumulate the results
    VMOV ZR, V4.S4 // Zero the V4 register
    VFMLA V0.S4, V2.S4, V4.S4
    VFMLA V1.S4, V3.S4, V4.S4

    // Add our vector elements together
    // VADDV V2.S4, F2
    // WORD $0x4EB1B882; // addv s2, v4.4s // Not working :(
    VMOV V4.S[0], R2
    VMOV V4.S[1], R3
    VMOV V4.S[2], R4
    VMOV V4.S[3], R5
    FMOVS R2, F0
    FMOVS R3, F1
    FMOVS R4, F2
    FMOVS R5, F3
    FADDS F0, F1, F4
    FADDS F2, F3, F5
    FADDS F4, F5, F6
    FMOVS F6, ret+56(FP)
    RET

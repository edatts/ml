//go:build arm64

#include "textflag.h"
#include "go_asm.h"

// func DotF16MatChunk16(A1, B1, B2, B3, B4, B5, B6, B7, B8, out []Float16)
TEXT ·DotF16MatChunk16(SB),NOSPLIT,$0-240
    // A1 is part of our A Matrix row, each of B1-8 is part of one of our B
    // matrix rows, and out is a slice of the corresponding output row.

    // Move our slice pointers into their own registers
    MOVD A1+0(FP), R0
    MOVD B1+24(FP), R1
    MOVD B2+48(FP), R2
    MOVD B3+72(FP), R3
    MOVD B4+96(FP), R4
    MOVD B5+120(FP), R5
    MOVD B6+144(FP), R6
    MOVD B7+168(FP), R7
    MOVD B8+192(FP), R8
    MOVD out+216(FP), R9

    // Load our vector registers
    VLD1 (R0), [V0.H8]
    VLD1 (R1), [V1.H8, V2.H8]
    VLD1 (R2), [V3.H8, V4.H8]
    VLD1 (R3), [V5.H8, V6.H8]
    VLD1 (R4), [V7.H8, V8.H8]
    VLD1 (R5), [V9.H8, V10.H8]
    VLD1 (R6), [V11.H8, V12.H8]
    VLD1 (R7), [V13.H8, V14.H8]
    VLD1 (R8), [V15.H8, V16.H8]
    VLD1 (R9), [V17.H8, V18.H8]

    WORD $0x4f001031 // fmla.8h v17, v1, v0[0] -> VFMLA V0.H[0], V1.H8, V17.H8
    WORD $0x4f101071 // fmla.8h v17, v3, v0[1] -> VFMLA V0.H[1], V3.H8, V17.H8
    WORD $0x4f2010b1 // fmla.8h v17, v5, v0[2] -> VFMLA V0.H[2], V5.H8, V17.H8
    WORD $0x4f3010f1 // fmla.8h v17, v7, v0[3] -> VFMLA V0.H[3], V7.H8, V17.H8
    WORD $0x4f001931 // fmla.8h v17, v9, v0[4] -> VFMLA V0.H[4], V9.H8, V17.H8
    WORD $0x4f101971 // fmla.8h v17, v11, v0[5] -> VFMLA V0.H[5], V11.H8, V17.H8
    WORD $0x4f2019b1 // fmla.8h v17, v13, v0[6] -> VFMLA V0.H[6], V13.H8, V17.H8
    WORD $0x4f3019f1 // fmla.8h v17, v15, v0[7] -> VFMLA V0.H[7], V15.H8, V17.H8
    WORD $0x4f001052 // fmla.8h v18, v2, v0[0] -> VFMLA V0.H[0], V2.H8, V18.H8
    WORD $0x4f101092 // fmla.8h v18, v4, v0[1] -> VFMLA V0.H[1], V4.H8, V18.H8
    WORD $0x4f2010d2 // fmla.8h v18, v6, v0[2] -> VFMLA V0.H[2], V6.H8, V18.H8
    WORD $0x4f301112 // fmla.8h v18, v8, v0[3] -> VFMLA V0.H[3], V8.H8, V18.H8
    WORD $0x4f001952 // fmla.8h v18, v10, v0[4] -> VFMLA V0.H[4], V10.H8, V18.H8
    WORD $0x4f101992 // fmla.8h v18, v12, v0[5] -> VFMLA V0.H[5], V12.H8, V18.H8
    WORD $0x4f2019d2 // fmla.8h v18, v14, v0[6] -> VFMLA V0.H[6], V14.H8, V18.H8
    WORD $0x4f301a12 // fmla.8h v18, v16, v0[7] -> VFMLA V0.H[7], V16.H8, V18.H8

    VST1 [V17.H8, V18.H8], (R9)
    
    RET

// func DotF16MatChunk8(A1, B1, B2, B3, B4, B5, B6, B7, B8, out []Float16)
TEXT ·DotF16MatChunk8(SB),NOSPLIT,$0-240
    // Move our slice pointers into their own registers
    MOVD A1+0(FP), R0
    MOVD B1+24(FP), R1
    MOVD B2+48(FP), R2
    MOVD B3+72(FP), R3
    MOVD B4+96(FP), R4
    MOVD B5+120(FP), R5
    MOVD B6+144(FP), R6
    MOVD B7+168(FP), R7
    MOVD B8+192(FP), R8
    MOVD out+216(FP), R9
    
    // Load our vector registers
    VLD1 (R0), [V0.H8]
    VLD1 (R1), [V1.H8]
    VLD1 (R2), [V2.H8]
    VLD1 (R3), [V3.H8]
    VLD1 (R4), [V4.H8] 
    VLD1 (R5), [V5.H8]
    VLD1 (R6), [V6.H8]
    VLD1 (R7), [V7.H8]
    VLD1 (R8), [V8.H8]
    VLD1 (R9), [V9.H8]
    
    WORD $0x4f001029 // fmla.8h v9, v1, v0[0] -> VFMLA V0.H[0], V1.H8, V9.H8
    WORD $0x4f101049 // fmla.8h v9, v2, v0[1] -> VFMLA V0.H[1], V2.H8, V9.H8
    WORD $0x4f201069 // fmla.8h v9, v3, v0[2] -> VFMLA V0.H[2], V3.H8, V9.H8
    WORD $0x4f301089 // fmla.8h v9, v4, v0[3] -> VFMLA V0.H[3], V4.H8, V9.H8
    WORD $0x4f0018a9 // fmla.8h v9, v5, v0[4] -> VFMLA V0.H[4], V5.H8, V9.H8
    WORD $0x4f1018c9 // fmla.8h v9, v6, v0[5] -> VFMLA V0.H[5], V6.H8, V9.H8
    WORD $0x4f2018e9 // fmla.8h v9, v7, v0[6] -> VFMLA V0.H[6], V7.H8, V9.H8
    WORD $0x4f301909 // fmla.8h v9, v8, v0[7] -> VFMLA V0.H[7], V8.H8, V9.H8

    VST1 [V9.H8], (R9)
    
    RET

// func DotF16MatChunk4(A1, B1, B2, B3, B4, B5, B6, B7, B8, out []Float16)
TEXT ·DotF16MatChunk4(SB),NOSPLIT,$0-240
    // Move our slice pointers into their own registers
    MOVD A1+0(FP), R0
    MOVD B1+24(FP), R1
    MOVD B2+48(FP), R2
    MOVD B3+72(FP), R3
    MOVD B4+96(FP), R4
    MOVD B5+120(FP), R5
    MOVD B6+144(FP), R6
    MOVD B7+168(FP), R7
    MOVD B8+192(FP), R8
    MOVD out+216(FP), R9
    
    // Load our vector registers
    VLD1 (R0), [V0.H8]
    VLD1 (R1), [V1.H4]
    VLD1 (R2), [V2.H4]
    VLD1 (R3), [V3.H4]
    VLD1 (R4), [V4.H4] 
    VLD1 (R5), [V5.H4]
    VLD1 (R6), [V6.H4]
    VLD1 (R7), [V7.H4]
    VLD1 (R8), [V8.H4]
    VLD1 (R9), [V9.H4]
    
    WORD $0x0f001029 // fmla.4h v9, v1, v0[0] -> VFMLA V0.H[0], V1.H4, V9.H4
    WORD $0x0f101049 // fmla.4h v9, v2, v0[1] -> VFMLA V0.H[1], V2.H4, V9.H4
    WORD $0x0f201069 // fmla.4h v9, v3, v0[2] -> VFMLA V0.H[2], V3.H4, V9.H4
    WORD $0x0f301089 // fmla.4h v9, v4, v0[3] -> VFMLA V0.H[3], V4.H4, V9.H4
    WORD $0x0f0018a9 // fmla.4h v9, v5, v0[4] -> VFMLA V0.H[4], V5.H4, V9.H4
    WORD $0x0f1018c9 // fmla.4h v9, v6, v0[5] -> VFMLA V0.H[5], V6.H4, V9.H4
    WORD $0x0f2018e9 // fmla.4h v9, v7, v0[6] -> VFMLA V0.H[6], V7.H4, V9.H4
    WORD $0x0f301909 // fmla.4h v9, v8, v0[7] -> VFMLA V0.H[7], V8.H4, V9.H4

    VST1 [V9.H4], (R9)
    
    RET

// func DotF16RowCol(A, B, out []Float16, idx, stride, outIdx int)
TEXT ·DotF16RowCol(SB),NOSPLIT,$0-96
    // Get slice pointers
    MOVD A+0(FP), R0
    MOVD B+24(FP), R1
    MOVD out+48(FP), R2

    // Get other params
    MOVD idx+72(FP), R3
    MOVD stride+80(FP), R4
    MOVD outIdx+88(FP), R5

    // Advance the output slice pointer by outIdx multiplied
    // by the number of bytes (2) in each element.
    LSL $1, R5, R5 
    ADD R5, R2, R2

    // Multiply stride by number of bytes in an element (2).
    LSL $1, R4, R4

    // Increment B data pointer by our start index multiplied by 2
    LSL $1, R3, R3
    ADD R3, R1, R1

    // Load row data
    VLD1 (R0), [V0.H8]

    // Load column data
    //
    // VLD1.P (R6)(R11), [V31.D1]        <=>      ld1 {v31.1d}, [x6], x11
    //
    VLD1.P (R1)(R4), V1.H[0]
    VLD1.P (R1)(R4), V2.H[0]
    VLD1.P (R1)(R4), V3.H[0]
    VLD1.P (R1)(R4), V4.H[0]
    VLD1.P (R1)(R4), V5.H[0]
    VLD1.P (R1)(R4), V6.H[0]
    VLD1.P (R1)(R4), V7.H[0]
    VLD1.P (R1)(R4), V8.H[0]

    // Use INS instruction to insert elements into our desired reg
    WORD $0x6e060441 // ins v1.h[1], v2.h[0]
    WORD $0x6e0a0461 // ins v1.h[2], v3.h[0]
    WORD $0x6e0e0481 // ins v1.h[3], v4.h[0]
    WORD $0x6e1204a1 // ins v1.h[4], v5.h[0]
    WORD $0x6e1604c1 // ins v1.h[5], v6.h[0]
    WORD $0x6e1a04e1 // ins v1.h[6], v7.h[0]
    WORD $0x6e1e0501 // ins v1.h[7], v8.h[0]

    // Multiply
    WORD $0x6e401c29 // fmul.8h v9, v1, v0

    // Now we need to reduce across the vector to sum the data.
    WORD $0x6e491529 // faddp.8h v9, v9, v9 // result in first 4 lanes
    WORD $0x2e491529 // faddp.4h v9, v9, v9 // result in first 2 lanes
    WORD $0x5e30d921 // faddp h1, v9.2h 
    
    FMOVS F1, R6
    MOVH R6, (R2)
    RET

// func DotF16Chunk4x8(A1, B1, B2, B3, B4, out []Float16)
TEXT ·DotF16Chunk4x8(SB),NOSPLIT,$0-144
    // Move our slice pointers into their own registers
    MOVD A1+0(FP), R0
    MOVD B1+24(FP), R1
    MOVD B2+48(FP), R2
    MOVD B3+72(FP), R3
    MOVD B4+96(FP), R4
    MOVD out+120(FP), R5
    
    // Load our vector registers
    VLD1 (R0), [V0.H4]
    VLD1 (R1), [V1.H8]
    VLD1 (R2), [V2.H8]
    VLD1 (R3), [V3.H8]
    VLD1 (R4), [V4.H8] 
    VLD1 (R5), [V5.H8]
    
    WORD $0x4f001025 // fmla.8h v5, v1, v0[0] -> VFMLA V0.H[0], V1.H8, V9.H8
    WORD $0x4f101045 // fmla.8h v5, v2, v0[1] -> VFMLA V0.H[1], V2.H8, V9.H8
    WORD $0x4f201065 // fmla.8h v5, v3, v0[2] -> VFMLA V0.H[2], V3.H8, V9.H8
    WORD $0x4f301085 // fmla.8h v5, v4, v0[3] -> VFMLA V0.H[3], V4.H8, V9.H8
    
    VST1 [V5.H8], (R5)
    
    RET

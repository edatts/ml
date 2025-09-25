//go:build arm64

#include "textflag.h"
#include "go_asm.h"

// func DotVec4(row, B []float32, k, j, n int) float32
TEXT ·DotVec4(SB),NOSPLIT,$0-76
    // Move our A slice pointers into R0 and R1
    MOVD row+0(FP), R0
    MOVD B+24(FP), R1

    // Move values of k and j into R2 and R3
    MOVD k+48(FP), R2
    MOVD j+56(FP), R3
    MOVD n+64(FP), R4
    MOVD $4, R5

    // We need to index our slice with k:k+3
    MUL R2, R5, R6 // K * 4 bytes to get offset for row
    ADD R0, R6, R0 // Add offset to row pointer
    // Load 128 bits from our row into V0
    VLD1 (R0), [V0.S4]

    // Load our other floats into V1
    MUL R6, R4, R7 // Offset for column, we want column j, rows k : k+3 

    // We want to add 4 * j, then add K * 4 * NumCols, then add 4 * NumCols 3 times
    MUL R3, R5, R6 // j * 4
    ADD R1, R6, R1 
    MUL R2, R5, R6 // K * 4
    MUL R6, R4, R6 // (K * 4) * NumCols
    ADD R1, R6, R1
    VLD1 (R1), V1.S[0]
    MUL R4, R5, R6 // NumCols * 4
    ADD R1, R6, R1
    VLD1 (R1), V1.S[1]
    ADD R1, R6, R1
    VLD1 (R1), V1.S[2]
    ADD R1, R6, R1
    VLD1 (R1), V1.S[3]

    // Multiply our vectors together and accumulate the results
    VMOV ZR, V2.S4 // Zero the V2 register
    VFMLA V0.S4, V1.S4, V2.S4

    // VMOV V2.S[3], R12
    // SCVTFS R0, F0
    // FMOVS R12, F0
    // FMOVS F0, ret+72(FP)
    // RET

    // Add our vector elements together
    // FMOVS ZR, F2
    // WORD $0x4EB1B842; // addv s2, v2.4s
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
    // FMOVS F2, R10
    // MOVW R2, ret+72(FP)
    FMOVS F6, ret+72(FP)
    RET

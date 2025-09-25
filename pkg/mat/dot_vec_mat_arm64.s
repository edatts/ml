//go:build arm64

#include "textflag.h"
#include "go_asm.h"

// func DotVecMat(row, B, out []float32)
TEXT ·DotVecMat(SB),NOSPLIT,$0-72
    // Okay so arugments are an input row, our column data, and our output row

    // Move our slice pointers into R0, R1, and R2
    MOVD row+0(FP), R0
    MOVD B+24(FP), R1
    MOVD out+48(FP), R3

    // Num bytes per index
    MOVD $4, R4
    
    // Number of columns is len of output row
    MOVD out+56(FP), R5
    
    // Input row length
    MOVD row+8(FP), R6
    
    // Num iterations (len(row)/4)
    SDIV R4, R6, R7
    
    // Remainder
    // msub <Xd>, <Xn>, <Xm>, <Xa>
    // MSUB <Rm>, <Ra>, <Rn>, <Rd>
    MSUB R4, R6, R7, R8 // R8 is remainder

    // Num inner iterations (len(out_row)/4)
    SDIV R4, R5, R9

    // Inner remainder
    MSUB R4, R5, R9, R10 // R10 is remainder

    // MOVD.P R7, +8(R3)
    // MOVD.P R8, +8(R3)
    // RET

    // We need to loop through our input row for each column and multiply
    // our row values with those for the column. There are a number of
    // things we need to keep track of here. Number of iterations left.
    // The remainder after the last iteration (so we can loop again). The
    // index for the values in each column. etc.

loop: // In this loop we want to take 4 elements of our input row and
      // multiply each element through the corresponding row of our B
      // matrix, accumulating the results in a set of vector registers.
    
    // If counter is zero break out to remainder loop
    CMP R7, ZR
    BEQ remainder_loop

    VLD1.P (R0), [V0.S4]
  
inner_loop: // This loop needs to accumulate values for each row of the
            // B matrix rows being operated on.
    CMP R9, ZR


    
    B inner_loop

inner_end:
    // TODO: Finish outer loop

remainder_loop: // In this one mul one float per iter
    CMP R8, ZR
    BEQ return

    B remainder_loop

return:
    RET


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

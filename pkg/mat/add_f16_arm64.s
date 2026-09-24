//go:build arm64

#include "textflag.h"
#include "go_asm.h"

// func addF16(A, B Float16) Float16
TEXT ·archAddF16(SB),NOSPLIT,$0-10
    // Move halfword values to registers
    MOVHU A+0(FP), R0
    MOVHU B+2(FP), R1

    // Then to float registers
    FMOVS R0, F0
    FMOVS R1, F1

    WORD $0x1ee02822 // fadd h2, h1, h0

    // Move result to general purpose registers then return
    FMOVS F2, R2
    MOVH R2, ret+8(FP)
    RET

// func addF16s(A, B, out []Float16)
TEXT ·archAddF16s(SB),NOSPLIT,$0-72
    // We're going to assume that all of the lengths are the
    // same so this must be checked before calling the function.

    // Get the pointers for the data
    MOVD A+0(FP), R0
    MOVD B+24(FP), R1
    MOVD out+48(FP), R2

    // Get the lengths
    MOVD A_len+8(FP), R3
    MOVD B_len+32(FP), R4
    MOVD out_len+56(FP), R5

    // Get number of main iterations. Logical shift right by 3 is
    // equivalent to dividing by 8.
    LSR $3, R3, R6
    CBZ R6, remainder_setup // Compare and branch on zero

main_loop:

    // Load vector registers
    VLD1 (R0), [V0.H8]
    VLD1 (R1), [V1.H8]

    // Add vector registers
    WORD $0x4e401422 // VFADD V0.H8, V1.H8, V2.H8 -> fadd.8h v2, v1, v0

    // Store the result
    VST1 [V2.H8], (R2)

    // Increment the pointer addresses
    ADD $16, R0, R0
    ADD $16, R1, R1
    ADD $16, R2, R2

    SUB $1, R6, R6
    CBNZ R6, main_loop // Compare and branch on non-zero.


remainder_setup:
    
    // Get number of iterations
    //
    // If we are dividing by a power of 2 (call it n) here then we can use
    // the AND instruction with n-1 to perform a modulo opertion. This works
    // because we are essentially zeroing all bits that contribute more than
    // the value of n to our integer.
    AND $7, R3, R6
    CBZ R6, done // If no elements left jump to done

remainder_loop:
    // This loop is for adding the remaining elements that don't fit
    // in a single vector register.

    // Load half words into scalar registers
    MOVHU (R0), R7 // MOVHU -> LDRH (Load Register Halfword)
    MOVHU (R1), R8

    // Load bits into floating point registers
    FMOVS R7, F0
    FMOVS R8, F1

    // Add halfword values
    WORD $0x1ee02822 // fadd h2, h1, h0

    // Move halfword result to general purpose register then store
    // the value in our results slice
    FMOVS F2, R9
    MOVH R9, (R2)

    // Increment the pointer addresses
    ADD $2, R0, R0
    ADD $2, R1, R1
    ADD $2, R2, R2

    SUB $1, R6, R6
    CBNZ R6, remainder_loop // If not zero continue iterating

done:
    RET


//go:build arm64

package mat

const haveArchF16 = true

func archAddF16s(A, B, out []Float16)

func archMulF16s(A, B, out []Float16)

func archAddF16(A, B Float16) Float16

func archMulF16(A, B Float16) Float16

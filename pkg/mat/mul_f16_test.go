package mat

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/x448/float16"
)

func TestMulF16s(t *testing.T) {
	var (
		A        = make([]Float16, 10)
		B        = make([]Float16, 10)
		out      = make([]Float16, 10)
		expected = make([]Float16, 10)
		numsA    = []float32{
			1, 2, 3, 4, 5, 6, 7, 8, 9,
		}
		numsB = []float32{
			9, 8, 7, 6, 5, 4, 3, 2, 1,
		}
		expectedNums = []float32{
			9, 16, 21, 24, 25, 24, 21, 16, 9,
		}
	)

	for i := range numsA {
		A[i] = float16.Fromfloat32(numsA[i])
		B[i] = float16.Fromfloat32(numsB[i])
		expected[i] = float16.Fromfloat32(expectedNums[i])
	}

	mulF16s(A, B, out)
	for i, num := range out {
		require.Equal(t, expected[i], num)
	}
}

func TestMulF16(t *testing.T) {
	var (
		A        = float16.Fromfloat32(3)
		B        = float16.Fromfloat32(4)
		expected = float16.Fromfloat32(12)
	)

	require.Equal(t, expected, mulF16(A, B))
}

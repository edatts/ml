package mat

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/x448/float16"
)

func TestAddF16s(t *testing.T) {
	var (
		A        = make([]Float16, 10)
		B        = make([]Float16, 10)
		out      = make([]Float16, 10)
		expected = make([]Float16, 10)
		numsA    = []float32{
			1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
		}
		numsB = []float32{
			0, 9, 8, 7, 6, 5, 4, 3, 2, 1,
		}
		expectedNums = []float32{
			1, 11, 11, 11, 11, 11, 11, 11, 11, 1,
		}
	)

	for i := range 10 {
		A[i] = float16.Fromfloat32(numsA[i])
		B[i] = float16.Fromfloat32(numsB[i])
		expected[i] = float16.Fromfloat32(expectedNums[i])
	}

	addF16s(A, B, out)
	for i, num := range out {
		require.Equal(t, expected[i], num)
	}
}

func TestAddF16(t *testing.T) {
	var (
		A        = float16.Fromfloat32(3)
		B        = float16.Fromfloat32(4)
		expected = float16.Fromfloat32(7)
	)

	require.Equal(t, expected, addF16(A, B))
}

func BenchmarkAddFloats(b *testing.B) {
	b.Run("F32", func(b *testing.B) {
		A := float32(4)
		B := float32(5)
		for b.Loop() {
			_ = A + B
		}

	})

	b.Run("F16", func(b *testing.B) {
		A := float16.Fromfloat32(4)
		B := float16.Fromfloat32(5)
		for b.Loop() {
			_ = addF16(A, B)

		}
	})
}

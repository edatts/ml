package mat_test

import (
	"testing"

	"github.com/edatts/ml/pkg/mat"
	"github.com/stretchr/testify/require"
	"github.com/x448/float16"
)

func TestF16MatrixMul(t *testing.T) {
	var (
		AData = [][]float32{
			{1, 2, 3, 4, 5, 1, 1, 1, 3, 6, 1},
			{1, 2, 3, 4, 5, 1, 1, 1, 3, 6, 1},
			{1, 2, 3, 4, 5, 1, 1, 1, 3, 6, 1},
			{1, 2, 3, 4, 5, 1, 1, 1, 3, 6, 1},
			{1, 2, 3, 4, 5, 1, 1, 1, 3, 6, 1},
			{1, 2, 3, 4, 5, 1, 1, 1, 3, 6, 1},
			{1, 2, 3, 4, 5, 1, 1, 1, 3, 6, 1},
			{1, 2, 3, 4, 5, 1, 1, 1, 3, 6, 1},
			{1, 2, 3, 4, 5, 1, 1, 1, 3, 6, 1},
		}
		BData = [][]float32{
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		}
		expectedData = [][]float32{
			{28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28},
			{28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28},
			{28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28},
			{28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28},
			{28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28},
			{28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28},
			{28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28},
			{28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28},
			{28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28},
		}
	)

	A, err := mat.NewF16FromSlices(toF16(AData))
	require.NoError(t, err)

	B, err := mat.NewF16FromSlices(toF16(BData))
	require.NoError(t, err)

	expected, err := mat.NewF16FromSlices(toF16(expectedData))
	require.NoError(t, err)

	out := mat.NewF16(9, 19)

	require.NoError(t, out.Mul(A, B))
	for i := range out.NumRows() {
		for j := range out.NumCols() {
			require.Equal(t, expected.Row(i)[j], out.Row(i)[j])
		}
	}
}

func BenchmarkF16MatrixMul(b *testing.B) {
	var (
		A     = randMatrixFromShape([2]int{64, 141})
		B     = randMatrixFromShape([2]int{141, 153})
		A_F32 = toF32(A)
		B_F32 = toF32(B)
		A_F16 = toF16(A_F32)
		B_F16 = toF16(B_F32)
	)

	matAF32, err := mat.NewFromSlices(A_F32)
	require.NoError(b, err)
	matBF32, err := mat.NewFromSlices(B_F32)
	require.NoError(b, err)
	outF32 := mat.New(64, 153)

	matAF16, err := mat.NewF16FromSlices(A_F16)
	require.NoError(b, err)
	matBF16, err := mat.NewF16FromSlices(B_F16)
	require.NoError(b, err)
	outF16 := mat.NewF16(64, 153)

	b.Run("F32", func(b *testing.B) {
		for b.Loop() {
			err := outF32.Mul(matAF32, matBF32)
			require.NoError(b, err)
		}

	})

	b.Run("F16", func(b *testing.B) {
		for b.Loop() {
			err := outF16.Mul(matAF16, matBF16)
			require.NoError(b, err)
		}
	})
}

func toF16(nums [][]float32) [][]mat.Float16 {
	var out = make([][]mat.Float16, len(nums))
	for i, row := range nums {
		out[i] = make([]mat.Float16, len(row))
		for j, elem := range row {
			out[i][j] = float16.Fromfloat32(elem)
		}
	}
	return out
}

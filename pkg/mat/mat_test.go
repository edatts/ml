package mat_test

import (
	"log/slog"
	"math/rand"
	"testing"

	"github.com/edatts/ml/pkg/mat"
	"github.com/stretchr/testify/require"
	gmat "gonum.org/v1/gonum/mat"
)

func TestMultiply(t *testing.T) {
	t.Run("output shape", func(t *testing.T) {
		var tests = []struct {
			shapeA    [2]int
			shapeB    [2]int
			shouldErr bool
			shapeY    [2]int
		}{
			{[2]int{2, 3}, [2]int{3, 4}, false, [2]int{2, 4}},
			{[2]int{3, 2}, [2]int{2, 4}, false, [2]int{3, 4}},
			{[2]int{4, 1}, [2]int{1, 5}, false, [2]int{4, 5}},
			{[2]int{2, 2}, [2]int{3, 4}, true, [2]int{0, 0}},
		}

		for _, test := range tests {
			A := matrixFromShape(test.shapeA)
			B := matrixFromShape(test.shapeB)

			Y, err := mat.Mul(A, B)
			require.Equal(t, test.shouldErr, err != nil)
			require.Equal(t, test.shapeY, shapeOf(Y))
		}
	})

	t.Run("output values", func(t *testing.T) {
		var (
			A = [][]float64{
				{1, 2, 3, 4, 5},
				{1, 2, 3, 4, 5},
			}
			B = [][]float64{
				{1, 2, 3, 4, 5},
				{1, 2, 3, 4, 5},
				{1, 2, 3, 4, 5},
				{1, 2, 3, 4, 5},
				{1, 2, 3, 4, 5},
			}
			Y_expected = [][]float64{
				{15, 30, 45, 60, 75},
				{15, 30, 45, 60, 75},
			}
		)

		for _, fn := range []func([][]float64, [][]float64) ([][]float64, error){
			mat.Mul,
			mat.MulUnroll,
			mat.MulConcurrent,
			mat.MulUnrollConcurrent,
			mat.MulSIMD,
			mat.MulConcurrentSIMD,
		} {
			// slog.Info("test index", "idx", idx, "A", A)
			Y, err := fn(A, B)
			require.NoError(t, err)
			require.Equal(t, shapeOf(Y_expected), shapeOf(Y))
			for i, row := range Y_expected {
				// slog.Info("testinfo", "i", i, "row", Y[i])
				for j, elem := range row {
					require.Equal(t, elem, Y[i][j])
				}
			}
		}
	})
}

func TestAddMatVecRows(t *testing.T) {
	var (
		A = [][]float64{
			{1, 1, 1},
			{1, 2, 3},
			{3, 2, 1},
			{1, 2, 1},
		}
		B = []float64{
			1, 2, 3,
		}
		Y_expected = [][]float64{
			{2, 3, 4},
			{2, 4, 6},
			{4, 4, 4},
			{2, 4, 4},
		}
	)

	Y, err := mat.AddMatVecRows(A, B)
	require.NoError(t, err)

	for i := range len(Y_expected) {
		for j := range len(Y_expected[0]) {
			require.Equal(t, Y_expected[i][j], Y[i][j])
		}
	}
}

func TestMatrixMul(t *testing.T) {
	var (
		// A = mat.NewFromSlices([][]float32{
		// 	{1, 2, 3, 4, 5, 1},
		// 	{1, 2, 3, 4, 5, 1},
		// 	{1, 2, 3, 4, 5, 1},
		// 	{1, 2, 3, 4, 5, 1},
		// })
		// B = mat.NewFromSlices([][]float32{
		// 	{1, 1, 1, 1, 1},
		// 	{1, 1, 1, 1, 1},
		// 	{1, 1, 1, 1, 1},
		// 	{1, 1, 1, 1, 1},
		// 	{1, 1, 1, 1, 1},
		// 	{1, 1, 1, 1, 1},
		// })
		// out      = mat.New(4, 5)
		// expected = mat.NewFromSlices([][]float32{
		// 	{16, 16, 16, 16, 16},
		// 	{16, 16, 16, 16, 16},
		// 	{16, 16, 16, 16, 16},
		// 	{16, 16, 16, 16, 16},
		// })
		A = mat.NewFromSlices([][]float32{
			{1, 2, 3, 4, 5, 1, 1, 1, 1},
			{1, 2, 3, 4, 5, 1, 1, 1, 1},
			{1, 2, 3, 4, 5, 1, 1, 1, 1},
			{1, 2, 3, 4, 5, 1, 1, 1, 1},
		})
		B = mat.NewFromSlices([][]float32{
			{1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 1},
		})
		out      = mat.New(4, 7)
		expected = mat.NewFromSlices([][]float32{
			{19, 19, 19, 19, 19, 19, 19},
			{19, 19, 19, 19, 19, 19, 19},
			{19, 19, 19, 19, 19, 19, 19},
			{19, 19, 19, 19, 19, 19, 19},
		})
	)

	require.NoError(t, out.Mul(A, B))
	for i := range out.NumRows() {
		slog.Info("out", "row", out.Row(i))
		for j := range out.NumCols() {
			// _, _ = expected, j
			// slog.Info("indices", "i", i, "j", j)
			require.Equal(t, expected.Row(i)[j], out.Row(i)[j])
		}
	}
}

func TestDotVec(t *testing.T) {
	var (
		A        = []float32{1, 2, 3, 4}
		B        = []float32{1, 2, 3, 4}
		expected = float32(30)
	)

	require.Equal(t, expected, mat.DotVec4F32Slc(A, B))
}

func BenchmarkMultiply(b *testing.B) {
	var (
		A     = randMatrixFromShape([2]int{64, 128})
		B     = randMatrixFromShape([2]int{128, 128})
		A_f   = flatten(A)
		B_f   = flatten(B)
		A_g   = gmat.NewDense(64, 128, A_f)
		B_g   = gmat.NewDense(128, 128, B_f)
		A_F32 = toF32(A)
		B_F32 = toF32(B)

		dense     = gmat.NewDense(64, 128, nil)
		matrixA   = mat.NewFromSlices(A_F32)
		matrixB   = mat.NewFromSlices(B_F32)
		outMatrix = mat.New(64, 128)
	)

	b.Run("sequential", func(b *testing.B) {
		for b.Loop() {
			_, err := mat.Mul(A, B)
			require.NoError(b, err)
		}
	})

	b.Run("unrolled", func(b *testing.B) {
		for b.Loop() {
			_, err := mat.MulUnroll(A, B)
			require.NoError(b, err)
		}
	})

	b.Run("concurrent", func(b *testing.B) {
		for b.Loop() {
			_, err := mat.MulConcurrent(A, B)
			require.NoError(b, err)
		}
	})

	b.Run("unrolledConcurrent", func(b *testing.B) {
		for b.Loop() {
			_, err := mat.MulUnrollConcurrent(A, B)
			require.NoError(b, err)
		}
	})

	// b.Run("sequentialSIMD", func(b *testing.B) {
	// 	for b.Loop() {
	// 		_, err := mat.MulSIMD(A, B)
	// 		require.NoError(b, err)
	// 	}
	// })

	// b.Run("concurrentSIMD", func(b *testing.B) {
	// 	for b.Loop() {
	// 		_, err := mat.MulConcurrentSIMD(A, B)
	// 		require.NoError(b, err)
	// 	}
	// })

	b.Run("concurrentSIMD", func(b *testing.B) {
		for b.Loop() {
			require.NoError(b, outMatrix.Mul(matrixA, matrixB))
		}
	})

	b.Run("gonum", func(b *testing.B) {
		for b.Loop() {
			dense.Mul(A_g, B_g)
		}
	})
}

func BenchmarkDotVec(b *testing.B) {
	var (
		A     = make([]float64, 8)
		B     = make([]float64, 8)
		A_F32 = make([]float32, 8)
		B_F32 = make([]float32, 8)
	)

	for i := range 8 {
		num, num2 := rand.Float64(), rand.Float64()
		A[i], A_F32[i] = num, float32(num)
		B[i], B_F32[i] = num2, float32(num2)
	}

	b.Run("DotVec4_F64", func(b *testing.B) {
		for b.Loop() {
			for i := 0; i < 8; i += 4 {
				mat.DotVec4F64(A[i:i+4], B[i+0], B[i+1], B[i+2], B[i+3])
			}
		}
	})

	b.Run("DotVec4_F32", func(b *testing.B) {
		for b.Loop() {
			for i := 0; i < 8; i += 4 {
				mat.DotVec4F32(A_F32[i:i+4], B_F32[i+0], B_F32[i+1], B_F32[i+2], B_F32[i+3])
			}
		}
	})

	b.Run("DotVec4_F32_Slice", func(b *testing.B) {
		for b.Loop() {
			for i := 0; i < 8; i += 4 {
				mat.DotVec4F32Slc(A_F32, B_F32)
			}
		}
	})

	b.Run("DotVec4_F32_NoSlice", func(b *testing.B) {
		for b.Loop() {
			for i := 0; i < 8; i += 4 {
				mat.DotVec4F32NoSlc(A_F32[i], A_F32[i+1], A_F32[i+2], A_F32[i+3], B_F32[i+0], B_F32[i+1], B_F32[i+2], B_F32[i+3])
			}
		}
	})

	b.Run("DotVec8_F32", func(b *testing.B) {
		for b.Loop() {
			mat.DotVec8F32(A_F32, B_F32[0], B_F32[1], B_F32[2], B_F32[3], B_F32[4], B_F32[5], B_F32[6], B_F32[7])
		}
	})

	b.Run("DotVec8_F32_Naive", func(b *testing.B) {
		for b.Loop() {
			var num float32
			for i := range 8 {
				num += A_F32[i] * B_F32[i]
			}
			_ = num
		}
	})

}

func randMatrixFromShape(s [2]int) [][]float64 {
	var out = make([][]float64, s[0])
	for i := range out {
		var row = make([]float64, s[1])
		for j := range row {
			row[j] = float64(rand.Intn(9))
		}
		out[i] = row
	}
	return out
}

func matrixFromShape(s [2]int) [][]float64 {
	var out = make([][]float64, s[0])
	for i := range out {
		out[i] = make([]float64, s[1])
	}
	return out
}

func shapeOf(m [][]float64) [2]int {
	if len(m) == 0 {
		return [2]int{0, 0}
	}

	return [2]int{len(m), len(m[0])}
}

func flatten(in [][]float64) []float64 {
	var out = make([]float64, 0)
	for i := range len(in) {
		out = append(out, in[i]...)
	}
	return out
}

func toF32(X [][]float64) [][]float32 {
	var out = make([][]float32, len(X))
	for i := range len(X) {
		out[i] = make([]float32, len(X[0]))
		for j := range len(X[0]) {
			out[i][j] = float32(X[i][j])
		}
	}
	return out
}

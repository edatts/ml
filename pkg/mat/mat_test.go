package mat_test

import (
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
				{1, 2, 3},
				{1, 2, 3},
			}
			B = [][]float64{
				{1, 2, 3, 4, 5},
				{1, 2, 3, 4, 5},
				{1, 2, 3, 4, 5},
			}
			Y_expected = [][]float64{
				{6, 12, 18, 24, 30},
				{6, 12, 18, 24, 30},
			}
		)

		for _, fn := range []func([][]float64, [][]float64) ([][]float64, error){
			mat.Mul,
			mat.MulParallel,
		} {
			Y, err := fn(A, B)
			require.NoError(t, err)
			require.Equal(t, shapeOf(Y_expected), shapeOf(Y))
			for i, row := range Y_expected {
				for j, elem := range row {
					require.Equal(t, elem, Y[i][j])
				}
			}
		}
	})
}

func BenchmarkMultiply(b *testing.B) {
	var (
		A   = randMatrixFromShape([2]int{32, 64})
		B   = randMatrixFromShape([2]int{64, 64})
		A_f = flatten(A)
		B_f = flatten(B)
		A_g = gmat.NewDense(32, 64, A_f)
		B_g = gmat.NewDense(64, 64, B_f)
	)

	b.Run("sequential", func(b *testing.B) {
		for b.Loop() {
			_, err := mat.Mul(A, B)
			require.NoError(b, err)
		}
	})

	b.Run("parallel", func(b *testing.B) {
		for b.Loop() {
			_, err := mat.MulParallel(A, B)
			require.NoError(b, err)
		}
	})

	b.Run("gonum", func(b *testing.B) {
		for b.Loop() {
			gmat.NewDense(32, 64, nil).Mul(A_g, B_g)
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

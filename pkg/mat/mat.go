package mat

import (
	"fmt"
	"sync"
)

// The matrix product is the dot product between the row vectors of A
// and the column vectors of B, hence, the number of columns in A must
// be euqal to the number of columns in B.
//
// For the input and output dimensions:
// (r,c)*(c,w) = (r,w)
func Mul(A [][]float64, B [][]float64) ([][]float64, error) {
	if err := mulValidate(A, B); err != nil {
		return nil, err
	}

	var out = make([][]float64, len(A))
	for i, row := range A {
		out[i] = make([]float64, len(B[0]))
		for j := range len(B[0]) {
			// Here we take the dot product of rows of A with columns of B
			for k, elem := range row {
				out[i][j] += elem * B[k][j] // A[i][k] * B[k][j]
			}

		}
	}

	return out, nil
}

func MulUnroll(A [][]float64, B [][]float64) ([][]float64, error) {
	if err := mulValidate(A, B); err != nil {
		return nil, err
	}

	// Determine number of iterations for unrolling
	quotient := len(A[0]) / 4

	// Here we want to try to unroll each vector into parts to take
	// advantage of CPU pipelining for (hopefully) better performance.
	var out = make([][]float64, len(A))
	for i := range len(A) {
		out[i] = make([]float64, len(B[0]))
		for j := range len(B[0]) {
			// Unrolling the loop into chunks of 4
			for k := 0; k < quotient*4; k += 4 {
				sum1 := A[i][k] * B[k][j]
				sum2 := A[i][k+1] * B[k+1][j]
				sum3 := A[i][k+2] * B[k+2][j]
				sum4 := A[i][k+3] * B[k+3][j]
				out[i][j] += sum1 + sum2 + sum3 + sum4
			}

			// Handle any remaining elements
			for k := quotient * 4; k < len(A[0]); k++ {
				out[i][j] += A[i][k] * B[k][j]
			}

		}
	}

	return out, nil
}

func MulConcurrent(A [][]float64, B [][]float64) ([][]float64, error) {
	if err := mulValidate(A, B); err != nil {
		return nil, err
	}

	var wg sync.WaitGroup
	var out = make([][]float64, len(A))
	for i, row := range A {
		wg.Add(1)
		go func(row []float64) {
			outRow := make([]float64, len(B[0]))
			for j := range len(B[0]) {
				for k, elem := range row {
					// Here we need to take the dot of A_i...N and B_j...M
					outRow[j] += elem * B[k][j] // A[i][k] * B[k][j]
				}
			}
			out[i] = outRow
			wg.Done()
		}(row)
	}

	wg.Wait()

	return out, nil
}

func MulUnrollConcurrent(A [][]float64, B [][]float64) ([][]float64, error) {
	if err := mulValidate(A, B); err != nil {
		return nil, err
	}

	quotient := len(A[0]) / 4

	var wg sync.WaitGroup
	var out = make([][]float64, len(A))
	for i := range len(A) {
		wg.Add(1)
		go func() {
			outRow := make([]float64, len(B[0]))
			for j := range len(B[0]) {
				// runtime.LockOSThread()
				// We actually see worse performance here unless we lock this
				// goroutine to the OS thread for the duration of this loop. My
				// assumption is that the goroutine scheduling interferes with the
				// CPU pipelining and causes issues with cache locality since
				// goroutines will end up executing across multiple OS threads.
				for k := 0; k < quotient*4; k += 4 {
					sum1 := A[i][k] * B[k][j]
					sum2 := A[i][k+1] * B[k+1][j]
					sum3 := A[i][k+2] * B[k+2][j]
					sum4 := A[i][k+3] * B[k+3][j]
					outRow[j] += sum1 + sum2 + sum3 + sum4
				}
				// runtime.UnlockOSThread()

				// Handle any remaining elements
				for k := quotient * 4; k < len(A[0]); k++ {
					outRow[j] += A[i][k] * B[k][j]
				}
			}
			out[i] = outRow
			wg.Done()
		}()
	}

	wg.Wait()

	return out, nil
}

func MulSIMD(A [][]float64, B [][]float64) ([][]float64, error) {
	if err := mulValidate(A, B); err != nil {
		return nil, err
	}

	// Determine number of iterations.
	quotient := len(A[0]) / 4

	var out = make([][]float64, len(A))
	for i := range len(A) {
		out[i] = make([]float64, len(B[0]))
		for j := range len(B[0]) {
			// Utilizing Go ASM to achieve SIMD
			for k := 0; k < quotient*4; k += 4 {
				// dotVec(A[i][k:k+4], )
				panic("MulSIMD is unimplemented")
			}

			// Handle any remaining elements
			for k := quotient * 4; k < len(A[0]); k++ {
				out[i][j] += A[i][k] * B[k][j]
			}

		}
	}

	return out, nil
}

func mulValidate(A [][]float64, B [][]float64) error {
	if len(A) == 0 || len(A[0]) == 0 {
		return fmt.Errorf("0 length rows or columns provided")
	}

	if len(A[0]) != len(B) {
		return fmt.Errorf("matrices are incompatible shapes for multiplication")
	}

	return nil
}

// This function does not perfom a matrix addition, instead the value
// of the vector is broadcast over each row of the matrix for addition.
func AddMatVecRows(A [][]float64, B []float64) ([][]float64, error) {
	if len(A) == 0 || len(A[0]) != len(B) {
		return nil, fmt.Errorf("incompatible shapes for addition, len(A)=%d, len(B)=%d", len(A), len(B))
	}

	var out = make([][]float64, len(A))
	for i := range len(A) {
		out[i] = make([]float64, len(A[0]))
		for j := range len(B) {
			out[i][j] = A[i][j] + B[j]
		}
	}

	return out, nil
}

func ApplyActivation(A [][]float64, fn func([]float64) []float64) [][]float64 {
	var out = make([][]float64, len(A))
	for i, row := range A {
		out[i] = fn(row)
	}

	return out
}

func Hadamard(A [][]float64, B [][]float64) [][]float64 {
	var out = make([][]float64, len(A))
	for i := range len(A) {
		var row = make([]float64, len(A[0]))
		for j := range len(A[0]) {
			row[j] = A[i][j] * B[i][j]
		}
		out[i] = row
	}
	return out
}

func AvgCols(A [][]float64) []float64 {
	if len(A) == 0 {
		return make([]float64, 0)
	}

	var out = make([]float64, len(A[0]))
	for j := range len(A[0]) {
		for i := range len(A) {
			out[j] += A[i][j]
		}
		out[j] /= float64(len(A))
	}
	return out
}

func MulScalar(A [][]float64, B float64) [][]float64 {
	var out = make([][]float64, len(A))
	for i := range len(A) {
		var row = make([]float64, len(A[0]))
		for j := range len(A[0]) {
			row[j] = A[i][j] * B
		}
		out[i] = row
	}
	return out
}

func Transpose(A [][]float64) [][]float64 {
	if len(A) == 0 {
		return make([][]float64, 0)
	}

	var out = make([][]float64, len(A[0]))
	for i := range len(A) {
		for j := range len(A[0]) {
			if len(out[j]) == 0 {
				out[j] = make([]float64, len(A))
			}
			out[j][i] = A[i][j]
		}
	}

	return out
}

func MulVecScalar(A []float64, B float64) []float64 {
	var out = make([]float64, len(A))
	for i := range len(A) {
		out[i] = A[i] * B
	}
	return out
}

func AddVec(A []float64, B []float64) []float64 {
	var out = make([]float64, len(A))
	for i := range len(A) {
		out[i] = A[i] + B[i]
	}
	return out
}

func Add(A [][]float64, B [][]float64) [][]float64 {
	var out = make([][]float64, len(A))
	for i := range len(A) {
		out[i] = make([]float64, len(A[0]))
		for j := range len(A[0]) {
			out[i][j] = A[i][j] + B[i][j]
		}
	}
	return out
}

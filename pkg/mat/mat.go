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
	if len(A) == 0 || len(A[0]) == 0 {
		return nil, fmt.Errorf("0 length rows or columns provided")
	}

	if len(A[0]) != len(B) {
		return nil, fmt.Errorf("matrices are incompatible shapes for multiplication")
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

func MulParallel(A [][]float64, B [][]float64) ([][]float64, error) {
	if len(A[0]) != len(B) {
		return nil, fmt.Errorf("matrices are incompatible shapes for multiplication")
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

// This function does not perfom a matrix addition, instead the value
// of the vector is broadcast over each column of the matrix for addition.
func AddVec(A [][]float64, B []float64) ([][]float64, error) {
	if len(A) != len(B) {
		return nil, fmt.Errorf("incompatible shapes for addition")
	}

	var out = make([][]float64, len(A))
	for i := range len(A) {
		var row = make([]float64, len(A[0]))
		for j := range len(B) {
			row[j] = A[i][j] + B[i]
		}
		out[i] = row
	}

	return out, nil
}

func ApplyActivation(A [][]float64, fn func(float64) float64) [][]float64 {
	var out = make([][]float64, len(A))
	for i := range len(A) {
		row := make([]float64, len(A[i]))
		for j := range len(A[0]) {
			row[j] = fn(A[i][j])
		}
		out[i] = row
	}

	return out
}

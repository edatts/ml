package mat

import (
	"fmt"
	"sync"
)

type shape [2]int

func (s shape) NumRows() int {
	return s[0]
}

func (s shape) NumCols() int {
	return s[1]
}

type Matrix struct {
	data []float32
	shape
}

func New(numRows, numCols int) *Matrix {
	return &Matrix{
		data:  make([]float32, numRows*numCols),
		shape: [2]int{numRows, numCols},
	}
}

func NewFromSlices(orig [][]float32) (*Matrix, error) {
	if len(orig) == 0 {
		return nil, fmt.Errorf("zero length slice provided")
	}

	for _, row := range orig[1:] {
		if len(row) != len(orig[0]) {
			return nil, fmt.Errorf("not all rows are the same length")
		}
	}

	var data = make([]float32, len(orig)*len(orig[0]))
	for i := range len(orig) {
		for j := range len(orig[0]) {
			data[i*len(orig[0])+j] = orig[i][j]
		}
	}

	return &Matrix{
		data:  data,
		shape: [2]int{len(orig), len(orig[0])},
	}, nil
}

func NewFromData(numRows, numCols int, data []float32) (*Matrix, error) {
	if len(data) != numRows*numCols {
		return nil, fmt.Errorf("length of data is incompatible with specified dimensions")
	}

	return &Matrix{
		data:  data,
		shape: [2]int{numRows, numCols},
	}, nil
}

func (m *Matrix) Data() []float32 {
	return m.data
}

func (m *Matrix) Row(i int) []float32 {
	start := i * m.NumCols()
	return m.data[start : start+m.NumCols()]
}

func (m *Matrix) Traspose() *Matrix {
	if len(m.data) == 0 {
		return m
	}

	var outData = make([]float32, len(m.data))
	for i := range m.NumRows() {
		for j := range m.NumCols() {
			remainder := ((i * m.NumCols()) + j) % m.NumCols()
			outData[(m.NumRows()*remainder)+i] = m.data[(i*m.NumCols())+j]
		}
	}

	return &Matrix{
		data:  outData,
		shape: [2]int{m.NumCols(), m.NumRows()},
	}
}

func (m *Matrix) AddToRows(vec []float32) error {
	if len(vec) != m.NumCols() {
		return fmt.Errorf("vector and matrix are incompatible shapes")
	}

	for i := range m.NumRows() {
		for j := range m.Row(i) {
			m.Row(i)[j] += vec[j]
		}
	}

	return nil
}

func (m *Matrix) ApplyActivation(fn func(float32) float32) *Matrix {
	var out = New(m.NumRows(), m.NumCols())
	for i, datum := range m.data {
		out.data[i] = fn(datum)
	}
	return out
}

func (m *Matrix) Hadamard(A, B *Matrix) error {
	if A.shape != B.shape || m.shape != A.shape {
		return fmt.Errorf("input matrices are incompatible shapes")
	}

	for i := range m.data {
		m.data[i] = A.data[i] * B.data[i]
	}

	return nil
}

func (m *Matrix) AvgCols() []float32 {
	var out = make([]float32, m.NumCols())
	for i := range m.NumRows() {
		for j, elem := range m.Row(i) {
			out[j] += elem
		}
	}

	for i := range m.NumCols() {
		out[i] /= float32(m.NumRows())
	}

	return out
}

// Mul calculates the dot product between input matrices A and B and stores
// the result in the matrix m. It returns an error if any of the matrices
// are not compatible shapes for the operation.
func (m *Matrix) Mul(A, B *Matrix) error {
	// TODO: If we want to use this in our NN implementations then we need to
	// 		 zero the output matrix before starting to accmulate results...
	if err := m.mulValidate(A, B); err != nil {
		return err
	}

	var (
		AN         = A.NumCols()
		BN         = B.NumCols()
		AQuotient  = AN / 4
		BQuotient  = BN / 8
		BRemainder = BN % 8
	)

	var wg sync.WaitGroup
	for i := range A.NumRows() {
		i_AN := i * AN
		i_BN := i * BN
		var row = A.data[i_AN : (i_AN)+AN]
		var outRow = m.data[i_BN : (i_BN)+BN]
		wg.Add(1)
		go func() {
			// Here we chunk our data before passing it into to the ASM func, we
			// use a 1 x 4 slice from a row of matrix A and a 4 x 8 chunk from 4
			// rows of matrix B. We accumulate the results into a 1 x 4 slice of
			// the corresponding row of the output matrix. We use the quotients
			// of our input and output row lengths divided by their coresponding
			// chunk dimensions to create the chunking loops. This way we don't
			// have any out of bounds array accesses. Any remaining elements are
			// handled in separate loops after the chunking loops.
			for j := 0; j < AQuotient; j++ {
				var A1 = row[(j * 4) : (j*4)+4]
				for k := 0; k < BQuotient && BN >= 8; k++ {
					// slog.Info("indices", "j", j*4, "k", k*4)
					kIdx := k * 8
					chunkIdx := kIdx + ((j * 4) * BN)
					B1 := B.data[chunkIdx+(0*BN) : chunkIdx+(0*BN)+8]
					B2 := B.data[chunkIdx+(1*BN) : chunkIdx+(1*BN)+8]
					B3 := B.data[chunkIdx+(2*BN) : chunkIdx+(2*BN)+8]
					B4 := B.data[chunkIdx+(3*BN) : chunkIdx+(3*BN)+8]

					// This is our GoASM func that handles the muliplicaitons and
					// additions for each chunk, the results are accumulated in the
					// output slice in an additive fashion.
					DotMatChunk8(A1, B1, B2, B3, B4, outRow[kIdx:kIdx+8])

				}

				// If remainder is >= 4 then process a 4 x 4 chunk here then add
				// 4 to the start index of the remainder loop.
				var extraIndex int
				if BRemainder >= 4 {
					k := BQuotient * 8
					chunkIdx := k + ((j * 4) * BN)
					B1 := B.data[chunkIdx+(0*BN) : chunkIdx+(0*BN)+4]
					B2 := B.data[chunkIdx+(1*BN) : chunkIdx+(1*BN)+4]
					B3 := B.data[chunkIdx+(2*BN) : chunkIdx+(2*BN)+4]
					B4 := B.data[chunkIdx+(3*BN) : chunkIdx+(3*BN)+4]

					DotMatChunk4(A1, B1, B2, B3, B4, outRow[k:k+4])
					extraIndex += 4
				}

				// Handle remainder
				for k := BQuotient*8 + extraIndex; k < BN; k++ {
					// slog.Info("indices", "j", j*4, "k", k)
					chunkIdx := k + ((j * 4) * BN)
					// slog.Info("indices", "cIdx", chunkIdx)
					R1 := A1[0] * B.data[chunkIdx+(0*BN)]
					R2 := A1[1] * B.data[chunkIdx+(1*BN)]
					R3 := A1[2] * B.data[chunkIdx+(2*BN)]
					R4 := A1[3] * B.data[chunkIdx+(3*BN)]
					outRow[k] += R1 + R2 + R3 + R4
				}
			}

			// Handle remainder
			for j := AQuotient * 4; j < AN; j++ {
				for k := range BN {
					outRow[k] += row[j] * B.data[(j*BN)+k]
				}
			}

			wg.Done()
		}()
	}

	wg.Wait()
	return nil
}

func (m *Matrix) mulValidate(A, B *Matrix) error {
	if m == nil || A == nil || B == nil {
		return fmt.Errorf("one of more matrices are nil")
	}

	if A.NumRows() == 0 || A.NumCols() == 0 {
		return fmt.Errorf("0 length rows or columns provided")
	}

	if A.NumCols() != B.NumRows() {
		return fmt.Errorf("matrices are incompatible shapes for multiplication")
	}

	if m.NumRows() != A.NumRows() || m.NumCols() != B.NumCols() {
		return fmt.Errorf("output matrix is not the correct dimensions")
	}

	return nil
}

func DotMatChunk8(A1, B1, B2, B3, B4, out []float32)

func DotMatChunk4(A1, B1, B2, B3, B4, out []float32)

// The matrix product is the dot product between the row vectors of A
// and the column vectors of B, hence, the number of columns in A must
// be euqal to the number of columns in B.
//
// For the input and output dimensions:
// (r,c)*(c,w) = (r,w)
func Mul(A [][]float64, B [][]float64) ([][]float64, error) {
	if err := multiplyValidate(A, B); err != nil {
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
	if err := multiplyValidate(A, B); err != nil {
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
	if err := multiplyValidate(A, B); err != nil {
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
	if err := multiplyValidate(A, B); err != nil {
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
				for k := 0; k < quotient*4; k += 4 {
					sum1 := A[i][k] * B[k][j]
					sum2 := A[i][k+1] * B[k+1][j]
					sum3 := A[i][k+2] * B[k+2][j]
					sum4 := A[i][k+3] * B[k+3][j]
					outRow[j] += sum1 + sum2 + sum3 + sum4
				}

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
	if err := multiplyValidate(A, B); err != nil {
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
				out[i][j] += float64(DotVec4F64(A[i][k:k+3], B[k][j], B[k+1][j], B[k+2][j], B[k+3][j]))
			}

			// Handle any remaining elements
			for k := quotient * 4; k < len(A[0]); k++ {
				out[i][j] += A[i][k] * B[k][j]
			}

		}
	}

	return out, nil
}

func MulConcurrentSIMD(A [][]float64, B [][]float64) ([][]float64, error) {
	if err := multiplyValidate(A, B); err != nil {
		return nil, err
	}

	// Determine number of iterations.
	quotient := len(A[0]) / 4

	var wg sync.WaitGroup
	var out = make([][]float64, len(A))
	for i := range len(A) {
		wg.Add(1)
		go func() {
			out[i] = make([]float64, len(B[0]))
			for j := range len(B[0]) {
				// Utilizing Go ASM to implement SIMD
				for k := 0; k < quotient*4; k += 4 {
					out[i][j] += DotVec4F64(A[i][k:k+3], B[k][j], B[k+1][j], B[k+2][j], B[k+3][j])
				}

				// Handle any remaining elements
				for k := quotient * 4; k < len(A[0]); k++ {
					out[i][j] += A[i][k] * B[k][j]
				}
			}
			wg.Done()
		}()
	}

	wg.Wait()

	return out, nil
}

func MulConcurrentSIMDF32(A [][]float32, B [][]float32) ([][]float32, error) {
	if err := multiplyValidate(A, B); err != nil {
		return nil, err
	}

	// Determine number of iterations.
	quotient := len(A[0]) / 8

	var wg sync.WaitGroup
	var out = make([][]float32, len(A))
	for i := range len(A) {
		wg.Add(1)
		go func() {
			out[i] = make([]float32, len(B[0]))
			for j := range len(B[0]) {
				// Utilizing Go ASM to implement SIMD
				for k := 0; k < quotient*8; k += 8 {
					out[i][j] += DotVec8F32(
						A[i][k:k+8],
						B[k][j], B[k+1][j], B[k+2][j], B[k+3][j],
						B[k+4][j], B[k+5][j], B[k+6][j], B[k+7][j],
					)
				}

				// Handle any remaining elements
				for k := quotient * 8; k < len(A[0]); k++ {
					out[i][j] += A[i][k] * B[k][j]
				}
			}
			wg.Done()
		}()
	}

	wg.Wait()

	return out, nil
}

func DotVec4F64(A []float64, B1, B2, B3, B4 float64) float64

func DotVec4F32(A []float32, B1, B2, B3, B4 float32) float32

func DotVec4F32Slc(A, B []float32) float32

func DotVec4F32NoSlc(A1, A2, A3, A4, B1, B2, B3, B4 float32) float32

func DotVec8F32(A []float32, B1, B2, B3, B4, B5, B6, B7, B8 float32) float32

func multiplyValidate[AT, BT any](A [][]AT, B [][]BT) error {
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

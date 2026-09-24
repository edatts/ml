package mat

import (
	"fmt"
	"sync"

	"github.com/x448/float16"
)

type Float16 = float16.Float16

type F16Matrix struct {
	data []Float16
	shape
}

func NewF16(numRows, numCols int) *F16Matrix {
	return &F16Matrix{
		data:  make([]Float16, numRows*numCols),
		shape: [2]int{numRows, numCols},
	}
}

func NewF16FromSlices(orig [][]Float16) (*F16Matrix, error) {
	if len(orig) == 0 {
		return nil, fmt.Errorf("zero length slice provided")
	}

	for _, row := range orig[1:] {
		if len(row) != len(orig[0]) {
			return nil, fmt.Errorf("not all rows are the same length")
		}
	}

	var data = make([]Float16, len(orig)*len(orig[0]))
	for i := range len(orig) {
		for j := range len(orig[0]) {
			data[i*len(orig[0])+j] = orig[i][j]
		}
	}

	return &F16Matrix{
		data:  data,
		shape: [2]int{len(orig), len(orig[0])},
	}, nil
}

func NewF16FromData(numRows, numCols int, data []Float16) (*F16Matrix, error) {
	if len(data) != numRows*numCols {
		return nil, fmt.Errorf("length of data is incompatible with specified dimensions")
	}

	return &F16Matrix{
		data:  data,
		shape: [2]int{numRows, numCols},
	}, nil
}

func (m *F16Matrix) AsSlices() [][]Float16 {
	var out = make([][]Float16, m.NumRows())
	for i := range m.NumRows() {
		out[i] = m.Row(i)
	}
	return out
}

func (m *F16Matrix) Data() []Float16 {
	return m.data
}

func (m *F16Matrix) Row(i int) []Float16 {
	start := i * m.NumCols()
	return m.data[start : start+m.NumCols()]
}

func (m *F16Matrix) Rows() [][]Float16 {
	var out = make([][]Float16, m.NumRows())
	for i := range m.NumRows() {
		out[i] = m.Row(i)
	}
	return out
}

func (m *F16Matrix) Transpose() *F16Matrix {
	if len(m.data) == 0 {
		return m
	}

	var outData = make([]Float16, len(m.data))
	for i := range m.NumRows() {
		for j := range m.NumCols() {
			remainder := ((i * m.NumCols()) + j) % m.NumCols()
			outData[(m.NumRows()*remainder)+i] = m.data[(i*m.NumCols())+j]
		}
	}

	return &F16Matrix{
		data:  outData,
		shape: [2]int{m.NumCols(), m.NumRows()},
	}
}

func (m *F16Matrix) AddToRows(vec []Float16) error {
	if len(vec) != m.NumCols() {
		return fmt.Errorf("vector and matrix are incompatible shapes")
	}

	for _, row := range m.Rows() {
		// On older ARM CPUs this will compile successfully but will fail
		// at runtime...
		// TODO: Gate this functionality and add fallback for older ARM CPUs
		addF16s(row, vec, row)
	}

	return nil
}

func (m *F16Matrix) ApplyActivation(fn func(Float16) Float16) *F16Matrix {
	var out = NewF16(m.NumRows(), m.NumCols())
	for i, datum := range m.data {
		out.data[i] = fn(datum)
	}
	return out
}

func (m *F16Matrix) Hadamard(A, B *F16Matrix) error {
	if A.shape != B.shape || m.shape != A.shape {
		return fmt.Errorf("input matrices are incompatible shapes")
	}

	mulF16s(A.data, B.data, m.data)
	return nil
}

func (m *F16Matrix) AvgCols() []Float16 {
	var out = make([]Float16, m.NumCols())
	for i := range m.NumRows() {
		for j, elem := range m.Row(i) {
			out[j] = addF16(out[j], elem)
		}
	}

	for i := range m.NumCols() {
		out[i] = divF16(out[i], float16.Fromfloat32(float32(m.NumRows())))
	}

	return out
}

// Mul calculates the dot product between input matrices A and B and stores
// the result in the matrix m. It returns an error if any of the matrices
// are not compatible shapes for the operation.
func (m *F16Matrix) Mul(A, B *F16Matrix) error {
	if err := m.mulValidate(A, B); err != nil {
		return err
	}

	// Zero data slice before we start to accumulate results
	for i := range m.data {
		m.data[i] = 0
	}

	var (
		AN         = A.NumCols()
		BN         = B.NumCols()
		AQuotient  = AN / 8
		ARemainder = AN % 8
		BQuotient  = BN / 16
		BRemainder = BN % 16
	)

	var wg sync.WaitGroup
	for i := range A.NumRows() {
		i_AN := i * AN
		i_BN := i * BN
		var ARem = ARemainder
		var row = A.data[i_AN : (i_AN)+AN]
		var outRow = m.data[i_BN : (i_BN)+BN]
		wg.Add(1)
		go func() {
			// Here we chunk our data before passing it into to the ASM func, we
			// use a 1 x 8 slice from a row of matrix A and an 8 x 16 chunk from 4
			// rows of matrix B. We accumulate the results into a 1 x 8 slice of
			// the corresponding row of the output matrix. We use the quotients
			// of our input and output row lengths divided by their coresponding
			// chunk dimensions to create the chunking loops. This way we don't
			// have any out of bounds array accesses. Any remaining elements are
			// handled in separate loops after the chunking loops.
			for j := 0; j < AQuotient; j++ {
				var BRem = BRemainder
				var A1 = row[(j * 8) : (j*8)+8]
				for k := 0; k < BQuotient && BN >= 16; k++ {
					// slog.Info("indices", "j", j*4, "k", k*4)
					kIdx := k * 16
					chunkIdx := kIdx + ((j * 8) * BN)
					B1 := B.data[chunkIdx+(0*BN) : chunkIdx+(0*BN)+16]
					B2 := B.data[chunkIdx+(1*BN) : chunkIdx+(1*BN)+16]
					B3 := B.data[chunkIdx+(2*BN) : chunkIdx+(2*BN)+16]
					B4 := B.data[chunkIdx+(3*BN) : chunkIdx+(3*BN)+16]
					B5 := B.data[chunkIdx+(4*BN) : chunkIdx+(4*BN)+16]
					B6 := B.data[chunkIdx+(5*BN) : chunkIdx+(5*BN)+16]
					B7 := B.data[chunkIdx+(6*BN) : chunkIdx+(6*BN)+16]
					B8 := B.data[chunkIdx+(7*BN) : chunkIdx+(7*BN)+16]

					// This is our GoASM func that handles the muliplications and
					// additions for each chunk, the results are accumulated in the
					// output slice in an additive fashion.
					DotF16MatChunk16(A1, B1, B2, B3, B4, B5, B6, B7, B8, outRow[kIdx:kIdx+16])
				}

				// If remainder is >= 8 then process an 8 x 8 chunk here then add
				// 8 to the start index of the remainder loop.
				var extraBIndex int
				if BRem >= 8 {
					k := BQuotient * 16
					chunkIdx := k + ((j * 8) * BN)
					B1 := B.data[chunkIdx+(0*BN) : chunkIdx+(0*BN)+8]
					B2 := B.data[chunkIdx+(1*BN) : chunkIdx+(1*BN)+8]
					B3 := B.data[chunkIdx+(2*BN) : chunkIdx+(2*BN)+8]
					B4 := B.data[chunkIdx+(3*BN) : chunkIdx+(3*BN)+8]
					B5 := B.data[chunkIdx+(4*BN) : chunkIdx+(4*BN)+8]
					B6 := B.data[chunkIdx+(5*BN) : chunkIdx+(5*BN)+8]
					B7 := B.data[chunkIdx+(6*BN) : chunkIdx+(6*BN)+8]
					B8 := B.data[chunkIdx+(7*BN) : chunkIdx+(7*BN)+8]

					DotF16MatChunk8(A1, B1, B2, B3, B4, B5, B6, B7, B8, outRow[k:k+8])
					extraBIndex += 8
					BRem -= 8
				}

				// If remainder is >= 4 then process an 8 x 4 chunk here then add
				// 4 to the start index of the remainder loop.
				if BRem >= 4 {
					k := BQuotient*16 + extraBIndex
					chunkIdx := k + ((j * 8) * BN)
					B1 := B.data[chunkIdx+(0*BN) : chunkIdx+(0*BN)+4]
					B2 := B.data[chunkIdx+(1*BN) : chunkIdx+(1*BN)+4]
					B3 := B.data[chunkIdx+(2*BN) : chunkIdx+(2*BN)+4]
					B4 := B.data[chunkIdx+(3*BN) : chunkIdx+(3*BN)+4]
					B5 := B.data[chunkIdx+(4*BN) : chunkIdx+(4*BN)+4]
					B6 := B.data[chunkIdx+(5*BN) : chunkIdx+(5*BN)+4]
					B7 := B.data[chunkIdx+(6*BN) : chunkIdx+(6*BN)+4]
					B8 := B.data[chunkIdx+(7*BN) : chunkIdx+(7*BN)+4]

					DotF16MatChunk4(A1, B1, B2, B3, B4, B5, B6, B7, B8, outRow[k:k+4])
					extraBIndex += 4
					BRem -= 4
				}

				// Handle remainder
				//
				// This loop iterates the remaining columns in B that didn't fit into
				// any of the previous chunks. For this part we use an assembly func
				// that fetches the matrix B data using the pointer to B.data and an
				// offset (BN), then uses a single vector FMUL to multiply the values
				// before using multiple FADDP instructions to sum the values across
				// the vector.
				for k := BQuotient*16 + extraBIndex; k < BN; k++ {
					chunkIdx := k + ((j * 8) * BN)
					DotF16RowCol(A1, B.data, outRow, chunkIdx, BN, k)
				}
			}

			// Handle remaining chunk of A
			//
			// Here we try to process a 4x8 chunk since there are not enough
			// columns left in A to do a 4xN chunk.
			var extraAIndex int
			if ARem >= 4 {
				var A1 = row[(AQuotient * 8) : (AQuotient*8)+4]
				for k := 0; k < BN/8 && BN >= 8; k++ {
					kIdx := k * 8
					chunkIdx := kIdx + ((AQuotient * 8) * BN)
					B1 := B.data[chunkIdx+(0*BN) : chunkIdx+(0*BN)+8]
					B2 := B.data[chunkIdx+(1*BN) : chunkIdx+(1*BN)+8]
					B3 := B.data[chunkIdx+(2*BN) : chunkIdx+(2*BN)+8]
					B4 := B.data[chunkIdx+(3*BN) : chunkIdx+(3*BN)+8]
					DotF16Chunk4x8(A1, B1, B2, B3, B4, outRow[kIdx:kIdx+8])
				}

				extraAIndex += 4
			}

			// Handle remainder
			//
			// This loop iterates the remaining columns in A that don't fit into
			// one of the chunks above and processes it naively against each
			// corresponding row in B. In reality we should still be able to
			// make use of some vector instructions here to speed up this part.
			//
			for j := AQuotient*8 + extraAIndex; j < AN; j++ {
				for k := range BN {
					outRow[k] = addF16(outRow[k], mulF16(row[j], B.data[(j*BN)+k]))
				}
			}

			wg.Done()
		}()
	}

	wg.Wait()
	return nil
}

func (m *F16Matrix) mulValidate(A, B *F16Matrix) error {
	if m == nil || A == nil || B == nil {
		return fmt.Errorf("one or more matrices are nil")
	}

	if A.NumRows() == 0 || A.NumCols() == 0 {
		return fmt.Errorf("0 length rows or columns provided")
	}

	if A.NumCols() != B.NumRows() {
		return fmt.Errorf("matrices are incompatible shapes for multiplication, shapes(A=%v, B=%v)", A.shape, B.shape)
	}

	if m.NumRows() != A.NumRows() || m.NumCols() != B.NumCols() {
		return fmt.Errorf("output matrix is not the correct dimensions, expected %d rows and %d cols, got %d rows and %d cols", A.NumRows(), B.NumCols(), m.NumRows(), m.NumCols())
	}

	return nil
}

func DotF16MatChunk16(A1, B1, B2, B3, B4, B5, B6, B7, B8, out []Float16)

func DotF16MatChunk8(A1, B1, B2, B3, B4, B5, B6, B7, B8, out []Float16)

func DotF16MatChunk4(A1, B1, B2, B3, B4, B5, B6, B7, B8, out []Float16)

func DotF16RowCol(A, B, out []Float16, idx, stride, outIdx int)

func DotF16Chunk4x8(A1, B1, B2, B3, B4, out []Float16)

func addF16s(A, B, out []Float16) {
	if haveArchF16 {
		archAddF16s(A, B, out)
		return
	}

	for i, elem := range A {
		out[i] = float16.Fromfloat32(elem.Float32() + B[i].Float32())
	}
}

func mulF16s(A, B, out []Float16) {
	if haveArchF16 {
		archMulF16s(A, B, out)
		return
	}

	for i, elem := range A {
		out[i] = float16.Fromfloat32(elem.Float32() * B[i].Float32())
	}
}

func addF16(A, B Float16) Float16 {
	if haveArchF16 {
		return archAddF16(A, B)
	}

	return float16.Fromfloat32(A.Float32() + B.Float32())
}

func mulF16(A, B Float16) Float16 {
	if haveArchF16 {
		return archMulF16(A, B)
	}

	return float16.Fromfloat32(A.Float32() * B.Float32())
}

func divF16(A, B Float16) Float16 {
	return float16.Fromfloat32(A.Float32() / B.Float32())
}

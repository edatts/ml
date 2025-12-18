package cnn

import (
	"fmt"
	"log/slog"

	"github.com/edatts/ml/pkg/mat"
)

type Sample struct {
	data []float32 // The dimensons (H, W C) are all flattened into the data slice.
	Shape
}

func NewSample(shape Shape) Sample {
	return Sample{
		data:  make([]float32, shape.Height()*shape.Width()*shape.Channels()),
		Shape: shape,
	}
}

func NewSampleFromData(shape Shape, data []float32) (Sample, error) {
	if len(data) != shape.Height()*shape.Width()*shape.Channels() {
		return Sample{}, fmt.Errorf("length of data does not match provided dimensions")
	}

	return Sample{
		data:  data,
		Shape: shape,
	}, nil
}

func NewBatch(batchSize int, sampleShape Shape) []Sample {
	var out = make([]Sample, batchSize)
	for i := range batchSize {
		out[i] = NewSample(sampleShape)
	}
	return out
}

// Creates a new batch from a *mat.Matrix where each row of the matrix
// is converted to a Sample in the batch.
//
// Should we copy the data instead of re-slicing it? This way if the
// original matrix is modified we won't modify the batch data...
//
// For now re-slice the data because I don't think we'll be keeping
// the matrices we convert into slices...
//
// Note that the provided shape is the shape of the Sample not the
// shape of the batch. This might change later if we implement a
// proper batch type.
func NewBatchFromMatrix(shape Shape, matrix *mat.Matrix) ([]Sample, error) {
	if matrix == nil {
		return nil, fmt.Errorf("nil matrix provided")
	}

	if len(matrix.Data()) != matrix.NumRows()*shape.Width()*shape.Height()*shape.Channels() {
		return nil, fmt.Errorf("matrix data is incompatible with shape")
	}

	var err error
	var out = make([]Sample, matrix.NumRows())
	for i := range matrix.NumRows() {
		out[i], err = NewSampleFromData(shape, matrix.Row(i))
		if err != nil {
			return nil, fmt.Errorf("new sample error: %w", err)
		}
	}

	return out, nil
}

func (s Sample) ImageSize() int {
	return s.Height() * s.Width()
}

func (s Sample) ApplyActivation(actFn func(float32) float32) Sample {
	var out = Sample{
		data:  make([]float32, len(s.data)),
		Shape: s.Shape,
	}
	for i, datum := range s.data {
		out.data[i] = actFn(datum)
	}
	return out
}

func (s Sample) Hadamard(s2 Sample) (Sample, error) {
	if s.Shape != s2.Shape {
		return Sample{}, fmt.Errorf("samples are incompatible shapes for hadamard product")
	}

	var out = Sample{
		data:  make([]float32, len(s.data)),
		Shape: s.Shape,
	}

	for i := range len(s.data) {
		out.data[i] = s.data[i] * s2.data[i]
	}

	return out, nil
}

// This function converts the batch into a matrix for use in fully connected
// layers. This can be implemented as a receiver on the Batch type if we
// implement it later. This operation will also be more efficient if we
// store all the batch data in a contiguous slice of memory.
func BatchToMatrix(batch []Sample) (*mat.Matrix, error) {
	if len(batch) == 0 {
		return nil, fmt.Errorf("batch length is zero")
	}

	var sampleLen = len(batch[0].data)
	var outData = make([]float32, len(batch)*sampleLen)
	for i, sample := range batch {
		for j := range sample.data {
			outData[i*sampleLen+j] = sample.data[j]
		}
	}

	return mat.NewFromData(len(batch), sampleLen, outData)
}

func (s Sample) Im2Col(stride, kernelHeight, kernelWidth int) (*mat.Matrix, error) {
	if stride != 1 {
		panic("only stride of 1 currently supported...")
	}

	// Formula for output size of convolution
	//
	// H_out = floor((H−K1+2P)/S)+1
	// W_out = floor((W−K2+2P)/S)+1
	//
	// P = 0 // No padding
	// S = 1 // Stride of 1
	//
	// Convolution between the input feature map of dimension H × W and
	// the weight kernel of dimension k1 × k2 produces an output feature
	// map of size (H − k1 + 1) by (W − k2 + 1) when there is no padding
	// and the stride is 1
	//
	H_sMat := (s.Height() - kernelHeight + 1)
	W_sMat := (s.Width() - kernelWidth + 1)
	numStrides := H_sMat * W_sMat

	slog.Info("stride info", "numStrides", numStrides, "H_sMat", H_sMat, "W_sMat", W_sMat)

	// The total size of the sample matrix is the number of elements in one
	// column multiplied by the number of strides. The number of elements in
	// one column is the kernel size multiplied by the number of channels.
	kernelLen := kernelHeight * kernelWidth
	sampleMatNumRows := kernelLen * s.Channels()
	rowLen := numStrides

	var sampleMatDataColMajor []float32
	for i := range H_sMat {
		for j := range W_sMat {
			col := make([]float32, sampleMatNumRows)
			for n := range s.Channels() {
				image := s.data[n*s.ImageSize() : n*s.ImageSize()+s.ImageSize()]
				for k := range kernelHeight {
					copy(col[(n*kernelLen)+(k*kernelWidth):(n*kernelLen)+(k*kernelWidth)+kernelWidth], image[(i*s.Width())+j+(k*s.Width()):(i*s.Width())+j+(k*s.Width())+kernelWidth])
				}
			}
			slog.Info("col", "col", col)
			sampleMatDataColMajor = append(sampleMatDataColMajor, col...)
		}
	}

	// Parsing as the transpose of the desired matrix for simplicity
	sampleMatrixColMajor, err := mat.NewFromData(rowLen, sampleMatNumRows, sampleMatDataColMajor)
	if err != nil {
		return nil, fmt.Errorf("failed instantiating colomn major sample matrix: %w", err)
	}

	return sampleMatrixColMajor.Traspose(), nil
}

func ToData(batch []Sample) [][]float32 {
	var out = make([][]float32, len(batch))
	for i, sample := range batch {
		out[i] = sample.data
	}
	return out
}

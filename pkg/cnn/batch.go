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
		return Sample{}, fmt.Errorf("data is not compatible with provided shape, len(data)=%d, shape=%v", len(data), shape)
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

func NewBatchFromSlices(shape Shape, slices [][]float32) ([]Sample, error) {
	if slices == nil {
		return nil, fmt.Errorf("input slice is nil")
	}

	var err error
	var out = make([]Sample, len(slices))
	for i, row := range slices {
		out[i], err = NewSampleFromData(shape, row)
		if err != nil {
			return nil, fmt.Errorf("failed instantiating sample: %w", err)
		}
	}

	return out, nil
}

func AsSlices(batch []Sample) [][]float32 {
	var out = make([][]float32, len(batch))
	for i, sample := range batch {
		out[i] = sample.data
	}
	return out
}

// func (s Sample) ImageSize() int {
// 	return s.Height() * s.Width()
// }

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

// Im2Col applies the Im2Col operation to the sample, copying the data to
// a new *mat.Matrix before returning it. This function only works for 2D
// convolution since it is assumed that the kernel depth is the same as
// the number of input channels.
//
// Assume we have the following input volume with three channels and a 3x3 kernel:
//
// |  + --- + --- + --- + --- +  + --- + --- + --- + --- +  + --- + --- + --- + --- +
// |  | 111 | 121 | 131 | 141 |  | 112 | 122 | 132 | 142 |  | 113 | 123 | 133 | 143 |
// |  + --- + --- + --- + --- +  + --- + --- + --- + --- +  + --- + --- + --- + --- +
// |  | 211 | 221 | 231 | 241 |  | 212 | 222 | 232 | 242 |  | 213 | 223 | 233 | 243 |
// |  + --- + --- + --- + --- +  + --- + --- + --- + --- +  + --- + --- + --- + --- +
// |  | 311 | 321 | 331 | 341 |  | 312 | 322 | 332 | 342 |  | 313 | 323 | 333 | 343 |
// |  + --- + --- + --- + --- +  + --- + --- + --- + --- +  + --- + --- + --- + --- +
// |  | 411 | 421 | 431 | 441 |  | 412 | 422 | 432 | 442 |  | 413 | 423 | 433 | 443 |
// |  + --- + --- + --- + --- +  + --- + --- + --- + --- +  + --- + --- + --- + --- +
//
// The resulting columns for the first three kernel strides would be:
//
//	[ 111, 121, 131, 211, 221, 231, 311, 321, 331,
//	  112, 122, 132, 212, 222, 232, 312, 322, 332,
//	  113, 123, 133, 213, 223, 233, 313, 323, 333 ]
//
//	[ 121, 131, 141, 221, 231, 241, 321, 331, 341,
//	  122, 132, 142, 222, 232, 242, 322, 332, 342,
//	  123, 133, 143, 223, 233, 243, 323, 333, 343 ]
//
//	[ 211, 221, 231, 311, 321, 331, 411, 421, 431,
//	  212, 222, 232, 312, 322, 332, 412, 422, 432,
//	  213, 223, 233, 313, 323, 333, 413, 423, 433 ]
//
// If we consider that the data is stored in an array, then we know
// it is contiguous and can be accessed with an index into the array.
// Considering that we will be reading 3 contiguous elements at a time
// then we would need to read and write at the following indexes:
//
// In this case, the write indexes should be:
//
//	[ 0,  3,  6,  9, 12, 15, 18, 21, 24... ]
//
// And the read indexes should be:
//
//	[ 0,  4,  8, 16, 20, 24, 32, 36, 40,
//	  1,  5,  9, 17, 21, 25, 33, 37, 41,
//	  4,  8, 12, 20, 24, 28, 36, 40, 44,
//	  5,  9, 13, 21, 25, 29, 37, 41, 45... ]
//
// When preserving the number of input channels we need to process each
// channel separately, this means that we would need more columns and
// more strides. The 3x3 kernel must have only 1 channel otherwise the
// matrices will be incompatible shapes. In this case, the columns for
// the first three strides would be:
//
//	[ 111, 121, 131, 211, 221, 231, 311, 321, 331 ]
//	[ 112, 122, 132, 212, 222, 232, 312, 322, 332 ]
//	[ 113, 123, 133, 213, 223, 233, 313, 323, 333 ]
func (s Sample) Im2Col(stride, kernelHeight, kernelWidth int, padding Padding, preserveChannels bool) (*mat.Matrix, error) {
	if stride != 1 {
		panic("only stride of 1 currently supported...")
	}

	P_h, P_w, err := getPadding(padding, kernelHeight, kernelWidth)
	if err != nil {
		return nil, fmt.Errorf("failed getting padding: %w", err)
	}

	// Full formula for output size of convolution:
	//
	// H_out = floor((H_in + P_t + P_b - D_h * (K_h - 1) - 1)/S_h)) + 1
	// W_out = floor((W_in + P_l + P_r - D_w * (K_w - 1) - 1)/S_h)) + 1
	//
	// We are ignoring dilation (D_h, D_w) for now so assume this is 1.
	// We also assume the padding is the same for top and bottom as well
	// as for left and right: (P_h = P_t + P_b), (P_w = P_l + P_r).
	// Currently we also enforce that stride is 1 for both height and
	// width. As such, the formula can be simplified to:
	//
	// H_out = floor( ( H_in + P_h − K_h ) / S_h ) + 1
	// W_out = floor( ( W_in + P_w − K_w ) / S_w ) + 1
	//
	// We can further simplify the formulas in the case where padding is
	// 0 and stride is 1. In this case the dimensions of the resulting
	// feature map can be found like so:
	//
	// P = 0 // No padding
	// S = 1 // Stride of 1
	//
	// H_out = H_in − K_h + 1
	// W_out = W_in − K_w + 1
	//
	// In all of these formulas we ignore the depth of the kernel because
	// for a 2D conv the depth of the kernel is the same as the number of
	// input channels.
	//
	// We use this formula to determine how many times we need to iterate
	// to capture the overlap between the kernel and the input volume, this
	// allows us to index the elements of the input volume in the same way
	// they would be indexed during a convolution operation.
	//
	H_out := ((s.Height()-kernelHeight+P_h)/stride + 1)
	W_out := ((s.Width()-kernelWidth+P_w)/stride + 1)
	numStrides := H_out * W_out

	// The total size of the sample matrix is the number of elements in one
	// column multiplied by the number of strides. The number of elements in
	// one column is the kernel volume.
	kernelArea := kernelHeight * kernelWidth

	slog.Info("im2col info", "numStrides", numStrides, "H_out", H_out, "W_out", W_out, "P_h", P_h, "P_w", P_w, "len(data)", kernelArea*s.Channels()*numStrides)

	// Iteration variables.
	var writeIdx int
	var sampleMatDataColMajor = make([]float32, kernelArea*s.Channels()*numStrides)
	var data = s.data

	// If we have padding we need to write it into the output in the
	// correct locations in the underlying storage slice. Let's take
	// an example:
	//
	// Original input:
	//
	//  + -- + -- +
	//  | 11 | 12 |
	//  + -- + -- +
	//  | 21 | 22 |
	//  + -- + -- +
	//
	// Padded Input (P_h = 2, P_w = 2):
	//
	// 	+ -- + -- + -- + -- +
	// 	| 00 | 00 | 00 | 00 |
	// 	+ -- + -- + -- + -- +
	// 	| 00 | 22 | 23 | 00 |
	// 	+ -- + -- + -- + -- +
	// 	| 00 | 32 | 33 | 00 |
	// 	+ -- + -- + -- + -- +
	// 	| 00 | 00 | 00 | 00 |
	// 	+ -- + -- + -- + -- +
	//
	// In this case the indices of the actual data map like so:
	//	- (1, 1) -> (2, 2)
	//	- (1, 2) -> (2, 3)
	//	- (2, 1) -> (3, 2)
	//	- (2, 2) -> (3, 3)
	//
	// Padded Input (P_h = 4, P_w = 4):
	//
	// 	+ -- + -- + -- + -- + -- + -- +
	// 	| 00 | 00 | 00 | 00 | 00 | 00 |
	// 	+ -- + -- + -- + -- + -- + -- +
	// 	| 00 | 00 | 00 | 00 | 00 | 00 |
	// 	+ -- + -- + -- + -- + -- + -- +
	// 	| 00 | 00 | 33 | 34 | 00 | 00 |
	// 	+ -- + -- + -- + -- + -- + -- +
	// 	| 00 | 00 | 43 | 44 | 00 | 00 |
	// 	+ -- + -- + -- + -- + -- + -- +
	//  | 00 | 00 | 00 | 00 | 00 | 00 |
	//  + -- + -- + -- + -- + -- + -- +
	//  | 00 | 00 | 00 | 00 | 00 | 00 |
	//  + -- + -- + -- + -- + -- + -- +
	//
	// In this exmaple the indices of the actual data map like so:
	//	- (1, 1) -> (3, 3)
	//	- (1, 2) -> (3, 4)
	//	- (2, 1) -> (4, 3)
	//	- (2, 2) -> (4, 4)
	//
	// So it looks like we just need to add half of the total
	// padding to the index for each dimension.
	//
	// Idx_hp = Idx_h + P_h / 2
	// Idx_wp = Idx_w + P_w / 2
	//
	// Ideally we would be able to allocate the padding only at
	// the destination but it will probably be a lot easier to
	// implement if we allocate the zero padding at the source
	// then copy the data over as we do now. The downside here
	// is that it seems wasteful...
	//
	// If we have padding we should re-allocate the underlying data into a new
	// slice with the added padding elements, this way we can easily copy the
	// data for both the padded and non-padded case.
	if padding != Valid {
		// Current length of data is H_in * W_in * numChannels, the length of
		// the padded data is (H_in+P_h) * (W_in+P_w) * numChannels.
		var strides = s.Shape.Strides()
		var paddedStrides = Shape{s.Channels(), s.Height() + P_h, s.Width() + P_w}.Strides()
		var paddedData = make([]float32, (s.Height()+P_h)*(s.Width()*P_w)*s.Channels())
		for i := 0; i < len(s.data); i += s.Width() {
			// Copy the data into the new slice, transforming the original
			// indices into corresponding indices in the padded slice.
			//
			// Idx_hp = Idx_h + P_h / 2
			// Idx_wp = Idx_w + P_w / 2
			//
			indices := strides.Indices(i)
			indices[1] = indices.H() + P_h/2
			indices[2] = indices.W() + P_w/2
			idx := paddedStrides.FlatIndex(indices)

			copy(paddedData[idx:idx+s.Width()], s.data[i:i+s.Width()])
		}
		data = paddedData
	}

	// Normally each column is the full area masked by the kernel, this
	// results in summation across spatial dimensions and channels. If we
	// want to preserve the channels we need each channel to be represented
	// in a separate column of the Im2Col sample matrix.
	//
	// So now we iterate through the strides and channels to copy the masked
	// portions of the input volume into the output matrix, we're going to
	// copy the data into a slice in column major format and then transpose
	// it before returning as this is conceptually easier to work with.
	if preserveChannels {
		// If we want to preserve the channels we need to stride across each
		// channel individually so we iterate the channels first, this changes
		// the read order of the data so that each channel is parsed into it's
		// own separate column from the other channels.
		for k := range s.Channels() {
			for i := range H_out {
				for j := range W_out {
					for l := range kernelHeight {
						// This should be the index for where to read the data from.
						readIdx := (i * stride * (s.Width() + P_w)) + (j * stride) + (k * (s.Height() + P_h) * (s.Width() + P_w)) + (l * (s.Width() + P_w))

						copy(sampleMatDataColMajor[writeIdx:writeIdx+kernelWidth], data[readIdx:readIdx+kernelWidth])
						writeIdx += kernelWidth
					}
				}
			}
		}
	} else {
		for i := range H_out {
			for j := range W_out {
				for k := range s.Channels() {
					for l := range kernelHeight {
						// This should be the index for where to read the data from.
						readIdx := (i * stride * (s.Width() + P_w)) + (j * stride) + (k * (s.Height() + P_h) * (s.Width() + P_w)) + (l * (s.Width() + P_w))

						copy(sampleMatDataColMajor[writeIdx:writeIdx+kernelWidth], data[readIdx:readIdx+kernelWidth])
						writeIdx += kernelWidth
					}
				}
			}
		}
	}

	// Parse into column major matrix and transpose before returning.
	colMajorRows, colMajorCols := numStrides, kernelArea*s.Channels()
	if preserveChannels {
		colMajorRows *= s.Channels()
		colMajorCols /= s.Channels()
	}

	slog.Info("im2col info", "numStrides", numStrides, "H_out", H_out, "W_out", W_out)
	slog.Info("im2col matrix", "height", colMajorCols, "width", colMajorRows)
	slog.Info("data len", "len", len(sampleMatDataColMajor))

	sampleMatrixColMajor, err := mat.NewFromData(colMajorRows, colMajorCols, sampleMatDataColMajor)
	if err != nil {
		return nil, fmt.Errorf("failed instantiating colomn major sample matrix: %w", err)
	}

	return sampleMatrixColMajor.Transpose(), nil
}

func getPadding(padding Padding, K_h, K_w int) (int, int, error) {
	if K_h%2 != 1 {
		return 0, 0, fmt.Errorf("kernel height must be odd")
	}

	if K_w%2 != 1 {
		return 0, 0, fmt.Errorf("kernel width must be odd")
	}

	// When considering padding it is convenient to assert that the kernel uses an
	// odd number of elements. This is because it allows us to always pad with the
	// same number of rows above and below the input and the same number of columns
	// to the left and right of the input.
	//
	// In this case the formula for total "Same" height and width padding is:
	//
	// P_h = (K_h - 1)
	// P_w = (K_w - 1)
	//
	// While the formula for total "Full" height and width padding is:
	//
	// P_h = (K_h - 1) * 2
	// P_w = (K_w - 1) * 2
	//
	switch padding {
	case Valid:
		return 0, 0, nil
	case Same:
		return (K_h - 1), (K_w - 1), nil
	case Full:
		return (K_h - 1) * 2, (K_w - 1) * 2, nil
	default:
		return 0, 0, fmt.Errorf("unsupported padding type '%s'", padding)
	}
}

func ToData(batch []Sample) [][]float32 {
	var out = make([][]float32, len(batch))
	for i, sample := range batch {
		out[i] = sample.data
	}
	return out
}

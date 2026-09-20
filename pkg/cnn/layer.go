package cnn

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"slices"
	"strings"

	"github.com/edatts/ml/pkg/mat"
	"github.com/edatts/ml/pkg/model"
	"golang.org/x/sync/errgroup"
)

type layerType string

// Originally I thought that the kernels were 2D but actually they are
// 3D volumes where, for 2D Conv laters, the depth (channels) must be
// equal to the input depth for the layer.
type Kernel struct {
	weights []float32
	dCdW    []float32

	// We're not goint to zero these as we want to keep the previous
	// value to use in the calculation of the next velocity term.
	//
	// vt = β * vt_prev + (1 - β) * dCdW
	velocities []float32

	bias float32
	dCdB float32

	shape Shape // Should enforce (size, size, len(input))
	// size int // Only square for now
}

func (k *Kernel) zeroGrads() {
	for i := range k.dCdW {
		k.dCdW[i] = 0
	}
	k.dCdB = 0
}

type Layer interface {
	Forward(isTest bool) error
	Backward() error
	update(lr, lambda, beta float64)

	// For 2D Conv layers we usually have shape (N, C, H, W) (batch_size, channels,
	// height, width) where channels starts off as the number of input channels (eg;
	// 1 for greyscale and 3 for RGB) and becomes the number of kernels/filters
	// after a pass through a convolutional later.
	Shape() Shape

	Name() string
	init(batchSize int)
	batch() []Sample
	activationDeltas() []Sample
	sumSquaredWeights() float32
	parameters() map[string]model.Tensor
	loadParameters(model.Tensor) error
}

// type LayerType int

// const (
// 	Input LayerType = iota
// 	Conv2D
// 	Pooling2D
// 	FullyConnected
// 	Output
// )

// func (c *CNN) newLayer(lType LayerType, prev Layer, shape Shape) (Layer, error) {
// 	switch lType {
// 	case Input:
// 		return &input{shape: shape}, nil
// 	case Conv2D:
// 		return newConv2D(prev, 32), nil
// 	case Pooling2D:
// 		return newPool2D(prev), nil
// 	case FullyConnected:
// 		return newFullyConnected(shape.Channels(), prev), nil
// 	case Output:
// 		return newOutput(shape, prev), nil
// 	}

// 	return nil, fmt.Errorf("unknown layer type '%d'", lType)
// }

var _ Layer = &input{}

// This layer simply holds the samples for a single batch. It has no kernels
// and no activation functions, hence the activations of this layer are
// simply the input values.
type input struct {
	name string

	// Assume Height * Width * Channels are flattened into the inner
	// slices, each element of the outer slice is a sample in the batch.
	// acts  [][]float32
	acts []Sample

	shape Shape
}

func (c *CNN) newInput(shape Shape) *input {
	return &input{
		name:  c.assignName("input"),
		shape: shape,
	}
}

func (l *input) Forward(_ bool) error {
	return nil
}

func (l *input) Backward() error {
	return nil
}

// NoOp. Nothing to update in this layer...
func (l *input) update(_, _, _ float64) {}

func (l *input) Shape() Shape {
	return l.shape
}

func (l *input) init(batchSize int) {
	l.acts = make([]Sample, batchSize)
}

func (l *input) batch() []Sample {
	return l.acts
}

func (l *input) activationDeltas() []Sample {
	return nil
}

func (l *input) sumSquaredWeights() float32 {
	return 0
}

func (l *input) Name() string {
	return l.name
}

func (l *input) parameters() map[string]model.Tensor {
	return nil
}

func (l *input) loadParameters(_ model.Tensor) error {
	return errors.New("input layers have no parameters to load")
}

var _ Layer = &conv2D{}

type conv2D struct {
	name  string
	prev  Layer
	actFn ActivationFunc

	logits []Sample
	dCdZ   []Sample

	acts []Sample
	dCdA []Sample

	// The weights here are actually just the kernels. We have the same
	// numer of kernels as output channels for the layer
	// weights [][]float32
	// dCdW    [][]float32
	kernels      []*Kernel
	kernelMatrix *mat.Matrix

	shape      Shape
	kernelSize int // For now just support square kernels
	stride     int // For now just support 1?
	batchSize  int
}

// TODO: Update the layer to support variable padding...
func (c *CNN) newConv2D(prev Layer, kernelSize, numKernels int) *conv2D {
	// Need some logic to ensure that the kernel shape, input shape, and
	// output shape are all compatible. This will need to take into
	// account the stride and padding for the layer as well.
	// H_out = floor((H−K1+2P)/S)+1
	// W_out = floor((W−K2+2P)/S)+1
	H_out := prev.Shape().Height() - kernelSize + 1
	W_out := prev.Shape().Width() - kernelSize + 1

	if H_out < 1 {
		panic("conv2d: previous height can not be less than the kernel height")
	}

	if W_out < 1 {
		panic("conv2d: previous width can not be less than the kernel width")
	}

	layer := &conv2D{
		name:  c.assignName("conv2d"),
		prev:  prev,
		actFn: ReLU{},

		shape:      Shape{numKernels, H_out, W_out},
		kernelSize: kernelSize,
		stride:     1, // Only 1 for now
	}

	layer.initKernels()

	return layer
}

func (l *conv2D) init(batchSize int) {
	// if !l.initialized {
	// 	l.initKernels()
	// 	l.initialized = true
	// }

	if batchSize != l.batchSize {
		// Update batch size
		l.batchSize = batchSize
		l.logits = NewBatch(batchSize, l.shape)
		l.acts = NewBatch(batchSize, l.shape)
	}

	// Zero kernel matrix
	l.kernelMatrix = nil

	// Zero gradients
	l.initGrads(batchSize)
	for _, k := range l.kernels {
		k.zeroGrads()
	}
}

func (l *conv2D) initKernels() {
	numInputs := l.prev.Shape().Channels() * l.kernelSize * l.kernelSize
	l.kernels = make([]*Kernel, l.Shape().Channels())
	for i := range len(l.kernels) {
		l.kernels[i] = &Kernel{
			weights:    make([]float32, numInputs),
			dCdW:       make([]float32, numInputs),
			velocities: make([]float32, numInputs),

			bias: float32(rand.NormFloat64() * 0.05),
			dCdB: 0,

			shape: [3]int{l.prev.Shape().Channels(), l.kernelSize, l.kernelSize},
			// size: l.kernelSize,
		}

		for j := range l.kernels[i].weights {
			// Kaiming He initialization
			l.kernels[i].weights[j] = float32(rand.NormFloat64() * math.Sqrt(float64(2)/float64(numInputs)))
		}
	}
}

func (l *conv2D) initGrads(batchSize int) {
	l.dCdZ = NewBatch(batchSize, l.shape)
	l.dCdA = NewBatch(batchSize, l.shape)
}

func (l *conv2D) getKernelMatrix() (*mat.Matrix, error) {
	// We need to remember to set the matrix to nil after every batch for this
	// to work properly, otherwise the weights will never update...
	if l.kernelMatrix != nil {
		return l.kernelMatrix, nil
	}

	if len(l.kernels) == 0 {
		return nil, fmt.Errorf("layer is not initialized, no kernels found")
	}

	// kernelVol can also just be len(kernel.weights)...
	kernelVol := l.kernelSize * l.kernelSize * l.prev.Shape().Channels()
	// Each kernel is unrolled into a row so the size of the kernel matrix is
	// the number of kernels multiplied by the kernel volume.
	var kernelMatrixData = make([]float32, len(l.kernels)*kernelVol)
	for i, kernel := range l.kernels {
		copy(kernelMatrixData[i*kernelVol:(i*kernelVol)+kernelVol], kernel.weights)
	}

	var err error
	l.kernelMatrix, err = mat.NewFromData(len(l.kernels), kernelVol, kernelMatrixData)
	if err != nil {
		return nil, fmt.Errorf("failed instantiating kernel matrix: %w", err)
	}

	return l.kernelMatrix, nil
}

// getdCdZMatrix builds a matrix from the logit deltas to be used in a convolution
// operation in the backwards pass.
func (l *conv2D) getdCdZMatrix(dCdZ []float32) (*mat.Matrix, error) {
	if len(dCdZ) == 0 {
		return nil, fmt.Errorf("layer is not initialized, no dCdZ found")
	}

	// slog.Info("dCdZ", "len", len(dCdZ))
	// slog.Info("dCdZ", "shape", l.dCdZ[0].Shape)
	// slog.Info("prev shape", "shape", l.prev.Shape())

	// The number of feature maps is equal to the number of kernels in the layer
	numFeatureMaps := len(l.kernels)
	featureMapArea := l.shape.Height() * l.shape.Width()

	dCdZMatrix, err := mat.NewFromData(numFeatureMaps, featureMapArea, dCdZ)
	if err != nil {
		return nil, fmt.Errorf("failed instantiating activation derivative matrix: %w", err)
	}

	// for i := range dCdZ {
	// 	for j := range l.prev.Shape().Channels() {
	// 		copy(dCdZMatrixData[(i*numCols)+(j*derivLen):(i*numCols)+(j*derivLen)+derivLen], dCdZ[i*derivLen:i*derivLen+derivLen])
	// 	}
	// }

	// derivLen := l.prev.Shape().Height() * l.prev.Shape().Width()
	// numCols := derivLen * l.prev.Shape().Channels()
	// var dCdZMatrixData = make([]float32, l.shape.Channels()*numCols)
	// for i := range dCdZ {
	// 	for j := range l.prev.Shape().Channels() {
	// 		copy(dCdZMatrixData[(i*numCols)+(j*derivLen):(i*numCols)+(j*derivLen)+derivLen], dCdZ[i*derivLen:i*derivLen+derivLen])
	// 	}
	// }

	// dCdZMatrix, err := mat.NewFromData(l.shape.Channels(), numCols, dCdZMatrixData)
	// if err != nil {
	// 	return nil, fmt.Errorf("failed instantiating activation derivative matrix: %w", err)
	// }

	// slog.Info("dCdz matrix data", "data", dCdZMatrixData)

	return dCdZMatrix, nil
}

func (l *conv2D) Forward(_ bool) error {
	// The way convolutions are usually handled is by converting the masked
	// areas of the input into columns using the im2col method and unrolling
	// the kernels into rows. This results in two matrices that can then be
	// mulitplied together to produce the output feature map. If the input
	// has multiple channels, we then calculate an elementwise sum of the
	// result of the matrix multiplications to calculate our feature map.

	kernelMatrix, err := l.getKernelMatrix()
	if err != nil {
		return fmt.Errorf("failed getting kernel matrix: %w", err)
	}

	// So we can actually use im2col to generate one very large matrix for
	// our entire sample volume. To do this, we unroll each kernel into a
	// vector and each one of these vectors is used as a row in a matrix.
	// Next, we take the input volume and for each stride we capture the
	// sub-volume that is masked by the kernel. We unroll each sub-volume
	// into a column of another matrix. We can then take the dot product
	// between the two matrices and the resulting matrix corresponds to
	// our feature map where each row is an unrolled feature.
	//
	// For example: Assume kernel shape of (4, 3, 3) with 8 kernels and an
	// input shape of (4, 6, 6).
	//
	// Each column would be 36 (4 * 3 * 3) elements long and there would be
	// 16 columns (one column for each stride). So we would have dims of
	// (36, 16) for the activations matrix.
	//
	// For the kernel matrix we would have dims of (8, 36) because we have
	// 8 kernels and each kernel volume gets unrolled into a single row.
	//
	// When we multiply the two matrices the dims of the output activations
	// are (8, 16) where each row is a flattened output feature map.
	//

	// Can we roll the entire batch into a single matrix? Currently the Im2Col
	// matrix holds a single input sample, this means that each column is a
	// single stride over the sample. For kernels that are similar size to the
	// input this means there are not many strides and hence not many columns.
	// Our matrix multiplcations will be most efficient for matrices with 16
	// or more columns (or 32 or more if we switch to FP16) so we should try
	// to pack more (or all) samples into one matrix.
	//
	// We can also parallelize the Im2Col operation by applying it to multiple
	// samples in parallel.
	//
	var eg errgroup.Group
	for i, sample := range l.prev.batch() {
		eg.Go(func() error {
			sampleMatrix, err := sample.Im2Col(l.stride, l.kernelSize, l.kernelSize, Valid, false)
			if err != nil {
				return fmt.Errorf("im2col error: %w", err)
			}

			// slog.Info("sampleMatrix", "height", sampleMatrix.NumRows(), "width", sampleMatrix.NumCols())
			// slog.Info("kernel matrix", "height", kernelMatrix.NumRows(), "width", kernelMatrix.NumCols())

			featureMapsMatrix := mat.New(kernelMatrix.NumRows(), sampleMatrix.NumCols())
			if err := featureMapsMatrix.Mul(kernelMatrix, sampleMatrix); err != nil {
				return fmt.Errorf("failed multiplying kernel and sample matrices: %w", err)
			}

			// Store logits in layer for backward pass. Each row in our feature maps
			// matrix is a single feature map.
			l.logits[i].data = featureMapsMatrix.Data()

			// Apply bias to feature map for each kernel
			for j := range l.logits[i].data {
				l.logits[i].data[j] += l.kernels[j%featureMapsMatrix.NumRows()].bias
			}

			// Apply activation func and store activations for backwards pass
			// l.acts[i].data = featureMapsMatrix.ApplyActivation().Data()
			l.acts[i] = l.logits[i].ApplyActivation(l.actFn.Forward)
			return nil
		})
	}

	return eg.Wait()
}

// Currently we are using vanilla Gradient Descent. We would like to add momentum
// to try and improve convergence characteristics.
//
// Velocity Update∶
//
// vt=β⋅vt−1+(1−β)⋅∇L(θt)
//
// Weight Update:
//
// W_1 = W_0 - lr * vt
//
// So we need up modify our weight update and also save previous gradients instead
// of zeroing them every batch...
func (l *conv2D) Backward() error {
	// Here we need to backpropagate our loss backwards through the conv
	// layer. To do this we carry out another convolution operation, but
	// instead of using the kernel, we use the derivative of the cost
	// with respect to the logits in our feature map. The output
	// of this operation is the derivative of the cost with respect to
	// the weights in our original input kernel.

	// // Check that grads are zero'd
	// for _, kernel := range l.kernels {
	// 	slog.Info("weight grads", "grads", kernel.dCdW)
	// 	slog.Info("bias grad", "grad", kernel.dCdB)
	// }

	// dCdz = actFn'(z) * dC/dA
	var err error
	for i, logitSample := range l.logits {
		actPrimeSample := logitSample.ApplyActivation(l.actFn.Backward)
		l.dCdZ[i], err = actPrimeSample.Hadamard(l.dCdA[i])
		if err != nil {
			return fmt.Errorf("failed calculating logit delta: %w", err)
		}
	}

	// slog.Info("dCdA", "data", l.dCdA[0])
	// slog.Info("dCdz", "data", l.dCdZ[0])

	// dCdW = dz/dW * dC/dz
	// dCdW = prevA * dC/dz (Using the convoluton operation)
	for i, sample := range l.prev.batch() {
		dCdZMatrix, err := l.getdCdZMatrix(l.dCdZ[i].data)
		if err != nil {
			return fmt.Errorf("failed getting derivative matrix for activations: %w", err)
		}

		// slog.Info("sample shape", "shape", sample.Shape)
		// slog.Info("dCdz shape", "shape", l.dCdZ[i].Shape)
		// slog.Info("dCdz matrix", "rows", dCdZMatrix.NumRows(), "cols", dCdZMatrix.NumCols())

		// Since we use the deltas of our feature maps (logits) as kernels for
		// another convolution operation, we need to perform another im2col
		// operation on our previous activations to generate a new sample matrix.
		//
		// The issue with this is that for the backwards pass convolution we
		// must not sum across channels. The feature map gradients are 2D and
		// so we need to convolve each feature map gradient across each input
		// channel separately to preserve the channel dimensions and ensure
		// that we get the correct dimensions for the weight gradients.
		sampleMatrix, err := sample.Im2Col(l.stride, l.shape.Height(), l.shape.Width(), Valid, true)
		if err != nil {
			return fmt.Errorf("sample im2col error: %w", err)
		}

		// slog.Info("prev acts matrix", "rows", sampleMatrix.NumRows(), "cols", sampleMatrix.NumCols())

		// Weight derivatives
		dCdWMatrix := mat.New(dCdZMatrix.NumRows(), sampleMatrix.NumCols())
		if err := dCdWMatrix.Mul(dCdZMatrix, sampleMatrix); err != nil {
			return fmt.Errorf("failed multiplying derivative and sample matrices: %w", err)
		}

		// slog.Info("dCdW Matrix", "rows", dCdWMatrix.NumRows(), "cols", dCdWMatrix.NumCols())

		for j, kernel := range l.kernels {
			// Store weight derivatives for update
			for k := range kernel.dCdW {
				kernel.dCdW[k] += dCdWMatrix.Row(j)[k]
			}

			// dCdB = dz/dB * dC/dz = 1 * dC/dz
			for _, dCdZ := range dCdZMatrix.Row(j) {
				kernel.dCdB += dCdZ
			}
		}
	}

	// Average weight and bias deltas over batch
	batchLen := len(l.prev.batch())
	for _, kernel := range l.kernels {
		for k := range kernel.dCdW {
			kernel.dCdW[k] /= float32(batchLen)
		}

		kernel.dCdB /= float32(batchLen)
	}

	if l.prev.activationDeltas() == nil { // Only if prev is input...
		return nil
	}

	var eg errgroup.Group
	var weightMatrices = make([]*mat.Matrix, len(l.kernels))
	for i, kernel := range l.kernels {
		eg.Go(func() error {
			kSample, err := NewSampleFromData(kernel.shape, kernel.weights)
			if err != nil {
				return fmt.Errorf("failed getting sample from weights: %w", err)
			}

			weightMatrices[i], err = kSample.Im2Col(l.stride, l.shape.Height(), l.shape.Width(), Full, true)
			if err != nil {
				return fmt.Errorf("failed im2col on weights: %w", err)
			}
			return nil
		})
	}

	if err := eg.Wait(); err != nil {
		return err
	}

	eg = errgroup.Group{}
	for i := range l.prev.batch() {
		eg.Go(func() error {
			dCdZMatrix, err := l.getdCdZMatrix(l.dCdZ[i].data)
			if err != nil {
				return fmt.Errorf("failed getting derivative matrix for activations: %w", err)
			}

			// dC/prevA = dz/dprevA * dC/dz
			// dC/prevA = W * dC/dz
			// Here we need to do another convolution operation using the kernel
			// weights in place of the input feature maps and the logit grads in
			// place of the kernel. This operation should be a "full" convolution.
			// This means we need to add some padding to the input depending on
			// the difference in size between the input (kernel weights) and the
			// kernel (in this case the logit grads). TBD on this exact formula
			// but it needs to result in each element of the logit grads being
			// convolved with every element of the input.
			//
			// For example: Given input spatial dimensions of 3x3 and kernel
			// spatial dimensions of 2x2, we would have logit spatial dimensions
			// of 2x2. To fully convolve the 2x2 logit grads with the 2x2 kernel
			// weights we would pad the weights with zeros up to dimensions of
			// 4x4. We would then proceed with the convolution operation.
			//
			// Padded Weights (Im2Col input):
			//
			// 	+ -- + -- + -- + -- +
			// 	| 00 | 00 | 00 | 00 |
			// 	+ -- + -- + -- + -- +
			// 	| 00 | 11 | 12 | 00 |
			// 	+ -- + -- + -- + -- +
			// 	| 00 | 21 | 22 | 00 |
			// 	+ -- + -- + -- + -- +
			// 	| 00 | 00 | 00 | 00 |
			// 	+ -- + -- + -- + -- +
			//
			// Logit Grads:
			//
			// 	+ -- + -- +
			// 	| 11 | 12 |
			// 	+ -- + -- +
			// 	| 21 | 22 |
			// 	+ -- + -- +
			//
			// Resulting Prev Activation Grads:
			//
			// 	+ -- + -- + -- +
			// 	| 11 | 12 | 13 |
			// 	+ -- + -- + -- +
			// 	| 21 | 22 | 23 |
			// 	+ -- + -- + -- +
			// 	| 31 | 32 | 33 |
			// 	+ -- + -- + -- +
			//
			// If we consider the channels... Assume we had 2 channels on the
			// original input and hence 2 channels on the kernel. Assume we
			// had 4 output channels and, hence, 4 kernels. In order to arrive
			// at the correct number of activation channels in backprop we
			// should convolve each logit grad feature map with the specific
			// kernel that produced it in the forward pass. This means we
			// also need to process each channel of each kernel separately
			// and then sum the results for each channel across all kernels.
			//
			// For this to work properly we also need to rotate the logit
			// gradients by 180 degrees so that the individual grads are
			// mulitplied with the respective weights that originally
			// produced their corresponding logits.

			// We will need to operate separately on each kernel and in order
			// to arrive at the correct dimensions for the activation gradients
			// we will need to preserve the channels of the kernels by only
			// summing across the spatial dimensions of each kernel. Then, we
			// will need to sum the results across all kernels for each of the
			// desired Activation gradient volumes.
			//
			// Generate kernel Im2Col

			for j, weightMatrix := range weightMatrices {

				// We need to take a single row of the dCdZ matrix here because we
				// are operating on each kernel individually. Later on we might be
				// able to operate on all kernels at once...
				//
				// TODO: See if we can roll this all up into one matrix...
				//
				// We also need to rotate the logit grads by 180 degrees so that
				// we backprop the loss into the correct weights. for a 180 degree
				// rotation we can simple reverse the backing array...
				dCdZSubMatrix, err := mat.NewFromData(1, dCdZMatrix.NumCols(), reverseSlice(dCdZMatrix.Row(j)))
				if err != nil {
					return fmt.Errorf("failed getting dCdZ sub-matrix: %w", err)
				}

				dCdAMatrix := mat.New(dCdZSubMatrix.NumRows(), weightMatrix.NumCols())
				if err := dCdAMatrix.Mul(dCdZSubMatrix, weightMatrix); err != nil {
					return fmt.Errorf("failed multiplying logit grads with weights: %w", err)
				}

				// slog.Info("dCdZSubMatrix shape", "shape", []int{dCdZSubMatrix.NumCols(), dCdZSubMatrix.NumRows()})
				// slog.Info("dCdAMatrix shape", "shape", []int{dCdAMatrix.NumCols(), dCdAMatrix.NumRows()})
				// slog.Info("prev activations shape", "shape", l.prev.activationDeltas()[i].Shape)
				// slog.Info("prev activation deltas", "len", len(l.prev.activationDeltas()[i].data))

				// Sum the outputs across kernels for each sample.
				for k, datum := range dCdAMatrix.Data() {
					l.prev.activationDeltas()[i].data[k] += datum
				}

			}
			return nil
		})

		// // weightMatrix := mat.New(dCdWMatrix.NumRows(), dCdWMatrix.NumCols())
		// // for i, k := range l.kernels {
		// // 	for j, weight := range k.weights {
		// // 		weightMatrix.Data()[(len(k.weights)*i)+j] = weight
		// // 	}
		// // }

		// dCdZSample, err := NewSampleFromData(l.shape, l.dCdZ[i].data)
		// if err != nil {
		// 	return fmt.Errorf("failed getting sample from dCdZ data: %w", err)
		// }

		// dCdZIm2Col, err := dCdZSample.Im2Col(l.stride, 1, 1, false)
		// if err != nil {
		// 	return fmt.Errorf("failed applying Im2Col to dCdZ sample: %w", err)
		// }

		// slog.Info("dCdZ Im2Col", "rows", dCdZIm2Col.NumRows(), "cols", dCdZIm2Col.NumRows())

		// slog.Info("dCdZ", "rows", dCdZMatrix.NumRows(), "cols", dCdZMatrix.NumCols())
		// slog.Info("kernelMatrix", "rows", kernelMatrix.NumRows(), "cols", kernelMatrix.NumCols())
		// // slog.Info("weight matrix", "rows", weightMatrix.NumRows(), "cols", weightMatrix.NumCols())
		// slog.Info("prev acts", "shape", l.prev.activationDeltas()[0].Shape)

		// // Logit grads shape: (256, 1, 1)
		// // Weights shape: (256, 3, 3)
		// // Prev acts shape: (128, 3, 3)

		// dCdAMatrix := mat.New(dCdZMatrix.NumRows(), kernelMatrix.NumRows())
		// if err := dCdAMatrix.Mul(dCdZMatrix, kernelMatrix.Transpose()); err != nil {
		// 	// dCdAMatrix := mat.New(weightMatrix.NumCols(), dCdZMatrix.NumCols())
		// 	// if err := dCdAMatrix.Mul(weightMatrix.Transpose(), dCdZMatrix); err != nil {
		// 	return fmt.Errorf("failed multiplying dCdz and transpose weights: %w", err)
		// }

		// prevActDeltas, err := NewSampleFromData(l.prev.Shape(), dCdAMatrix.Data())
		// if err != nil {
		// 	return fmt.Errorf("failed instantiating prev layers activation delta sample: %w", err)
		// }

		// l.prev.activationDeltas()[i] = prevActDeltas
	}

	return eg.Wait()
}

func (l *conv2D) update(lr, lambda, beta float64) {
	for _, kernel := range l.kernels {
		for j, weight := range kernel.weights {
			// kernel.weights[j] -= float32(lr) * (kernel.dCdW[j] + float32(lambda)*weight)

			// Calculate velocity terms:
			// vt = β * vt_prev + (1 - β) * dCdW
			kernel.velocities[j] = kernel.velocities[j]*float32(beta) + (1-float32(beta))*kernel.dCdW[j]

			// Update weights with velocity term. In our implementation the weight
			// decay is decoupled from momentum.
			kernel.weights[j] -= float32(lr) * (kernel.velocities[j] + float32(lambda)*weight)
		}

		kernel.bias -= float32(lr) * kernel.dCdB
	}
}

func (l *conv2D) Shape() Shape {
	return l.shape
}

func (l *conv2D) batch() []Sample {
	return l.acts
}

func (l *conv2D) activationDeltas() []Sample {
	return l.dCdA
}

func (l *conv2D) sumSquaredWeights() float32 {
	var sum float32
	for _, k := range l.kernels {
		for _, w := range k.weights {
			sum += w * w
		}
	}
	return sum
}

func (l *conv2D) biases() []float32 {
	var out = make([]float32, len(l.kernels))
	for i, k := range l.kernels {
		out[i] = k.bias
	}
	return out
}

func (l *conv2D) Name() string {
	return l.name
}

func (l *conv2D) parameters() map[string]model.Tensor {
	return map[string]model.Tensor{
		fmt.Sprintf("%s.weights", l.name): {
			Name:  fmt.Sprintf("%s.weights", l.name),
			Shape: model.Shape{l.shape.Channels(), l.prev.Shape().Channels(), l.kernelSize, l.kernelSize},
			Data:  l.kernelMatrix.Data(), // This should only ever be nil briefly during Forward()...
		},
		fmt.Sprintf("%s.biases", l.name): {
			Name:  fmt.Sprintf("%s.biases", l.name),
			Shape: model.Shape{0, 0, 0, l.shape.Channels()},
			Data:  l.biases(),
		},
	}
}

func (l *conv2D) loadParameters(tensor model.Tensor) error {
	if strings.HasSuffix(tensor.Name, ".weights") {
		var kernelVol = l.kernelSize * l.kernelSize * l.prev.Shape().Channels()
		for i, k := range l.kernels {
			k.weights = tensor.Data[i*kernelVol : i*kernelVol+kernelVol]
		}
		return nil
	}

	if strings.HasSuffix(tensor.Name, ".biases") {
		for i, bias := range tensor.Data {
			l.kernels[i].bias = bias
		}
		return nil
	}

	return fmt.Errorf("unknown parameter type, expected weights or biases")
}

var _ Layer = &pool2D{}

type pool2D struct {
	name string
	prev Layer

	acts       []Sample
	dCdA       []Sample
	maxIndices [][]int

	P_h         int
	P_w         int
	windowSize  int // For now only square
	shape       Shape
	initialized bool
	batchSize   int
}

func (c *CNN) newPool2D(prev Layer) *pool2D {
	// Ensure input shape and window size are compatible. We should probably only
	// specify the window size and then infer the output shape from the input
	// shape, if it is compatible with the window size.

	var P_h int
	var P_w int

	if prev.Shape().Height()%2 != 0 {
		// Going to have to use padding...
		slog.Warn("pool2d incompatible height, adding height padding...")
		P_h = 1
		// slog.Error("input shape is not compatible with pooling parameters", "inputShape", prev.Shape(), "poolingWindow", 2, "poolingStride", 2)
		// panic("input height is not compatible with window size and stride of 2")
	}

	if prev.Shape().Width()%2 != 0 {
		slog.Warn("pool2d incompatible width, adding width padding...")
		P_w = 1
		// slog.Error("input shape is not compatible with pooling parameters", "inputShape", prev.Shape(), "poolingWindow", 2, "poolingStride", 2)
		// panic("input width is not compatible with window size and stride of 2")
	}

	// H_out = floor((H−K1+2P)/S)+1
	// W_out = floor((W−K2+2P)/S)+1
	H_out := (prev.Shape().Height()-2+P_h)/2 + 1
	W_out := (prev.Shape().Width()-2+P_w)/2 + 1

	return &pool2D{
		name:       c.assignName("pool2d"),
		prev:       prev,
		windowSize: 2, // Window size is the same as stride...
		P_h:        P_h,
		P_w:        P_w,
		shape:      Shape{prev.Shape().Channels(), H_out, W_out},
	}
}

func (l *pool2D) init(batchSize int) {
	if !l.initialized {
		l.acts = NewBatch(batchSize, l.shape)
		l.initialized = true
	}

	if batchSize != l.batchSize {
		l.batchSize = batchSize
		l.acts = NewBatch(batchSize, l.shape)
	}

	l.maxIndices = make([][]int, batchSize)
	// Zero grads
	l.dCdA = NewBatch(batchSize, l.shape)
}

// TODO: Tests for forward pass shape and values...
// TODO: Review padding and formula for output size...
func (l *pool2D) Forward(_ bool) error {
	// The forward pass of the pooling layer calculates the max value for
	// a particular area of each input channels. It operates individually
	// on each input channel, preserving depth of the input. The stride
	// is typically equal to the size of the pooling window so that each
	// element of the input is only operated on once.

	if l.windowSize != 2 {
		return fmt.Errorf("pool2d must be re-implemented to support a window size greater than 2")
	}

	// In the case below we replace 2P with P because we are only padding
	// the rightmost column and bottom row with zeros...
	// H_out = floor((H−K1+2P)/S)+1
	// W_out = floor((W−K2+2P)/S)+1
	H_in := l.prev.Shape().Height()
	W_in := l.prev.Shape().Width()
	H_out := ((H_in - l.windowSize + l.P_h) / l.windowSize) + 1
	W_out := ((W_in - l.windowSize + l.P_w) / l.windowSize) + 1
	// imageLen := H_in * W_in
	// paddedLen := (H_in + l.P_h) * (W_in + l.P_w)
	outLen := H_out * W_out

	// slog.Info("dims", "H_in", H_in, "paddedLen", paddedLen)

	var strides = l.prev.Shape().Strides()
	var paddedShape = Shape{l.prev.Shape().Channels(), l.prev.Shape().Height() + l.P_h, l.prev.Shape().Width() + l.P_w}
	var paddedStrides = paddedShape.Strides()
	for n, sample := range l.prev.batch() {
		var paddedData = make([]float32, paddedShape.Volume())
		for i := 0; i < len(sample.data); i += 2 {
			// Because we're only padding one row on the right and one row on
			// the bottom the individual indices of the data elements remain
			// the same, but the flat index increases by one for each row.
			indices := strides.Indices(i)
			idx := paddedStrides.FlatIndex(indices)
			copy(paddedData[idx:idx+2], sample.data[i:i+2])
		}

		var data = sample.data
		var outData = make([]float32, outLen*sample.Channels())
		l.maxIndices[n] = make([]int, sample.Channels()*H_out*W_out)
		for c := range sample.Channels() {
			for h := range H_out {
				for w := range W_out {
					s := l.windowSize // stride (only 2 for now)
					idx_0 := paddedStrides.FlatIndex(Indices{c, h * s, w * s})
					idx_1 := paddedStrides.FlatIndex(Indices{c, h * s, w*s + 1})
					idx_2 := paddedStrides.FlatIndex(Indices{c, h*s + 1, w * s})
					idx_3 := paddedStrides.FlatIndex(Indices{c, h*s + 1, w*s + 1})

					maximum := max(data[idx_0], data[idx_1], data[idx_2], data[idx_3])
					maxPaddedIdx := idx_0 + slices.Index(data[idx_0:idx_3+1], maximum)

					// Indices are the same for real data elements because we
					// only pad the right and bottom.
					maxPaddedIndices := paddedStrides.Indices(maxPaddedIdx)
					maxIdx := strides.FlatIndex(maxPaddedIndices)

					datumIdx := l.shape.Strides().FlatIndex(Indices{c, h, w})
					l.maxIndices[n][datumIdx] = maxIdx
					outData[datumIdx] = maximum
				}
			}
		}

		// var outData = make([]float32, outLen*sample.Channels())
		// l.maxIndices[n] = []int{}
		// for m := range sample.Channels() {
		// 	image := l.padImage(paddedLen, sample.data[m*imageLen:m*imageLen+imageLen])
		// 	for i := range H_out {
		// 		for j := range W_out {
		// 			var chunk []float32
		// 			for k := range l.windowSize {
		// 				chunk = append(chunk, image[(i*W_in)+(k*W_in)+j:(i*W_in)+(k*W_in)+j+l.windowSize]...)
		// 			}
		// 			maximum := slices.Max(chunk)
		// 			maxIdx := slices.Index(image, maximum) + m*imageLen
		// 			outData[(m*outLen)+(i*H_out)+j] = maximum
		// 			l.maxIndices[n] = append(l.maxIndices[n], maxIdx)
		// 		}
		// 	}
		// }

		outSample, err := NewSampleFromData([3]int{sample.Channels(), H_out, W_out}, outData)
		if err != nil {
			return fmt.Errorf("failed instantiting new output sample: %w", err)
		}
		l.acts[n] = outSample
	}

	return nil
}

// TODO: Write test for padImage...
// func (l *pool2D) padImage(paddedLen int, data []float32) []float32 {
// 	if paddedLen == len(data) {
// 		return data
// 	}

// 	var out = make([]float32, paddedLen)
// 	var x int
// 	for i, datum := range data {
// 		out[i+x] = datum
// 		if (i+1)%l.prev.Shape().Width() == 0 {
// 			x++
// 		}
// 	}
// 	return out
// }

func (l *pool2D) Backward() error {
	// In the backward pass there are no learnable parameters in this layer
	// so we are simply propagating the gradients backwards through the
	// original maximum elements for each section of the input covered
	// by the sliding window.

	// This part would probably be easiest if we just store the indices of
	// the maximum values for each window during the forwards pass

	for i, gradsSample := range l.dCdA {
		// // Zero the previous grads before backpropagating
		// l.prev.activationDeltas()[i] = NewSample(l.prev.Shape())

		for j, grad := range gradsSample.data {
			// This is the index of the padded volume so we need to convert
			// it into the non-padded index for use with the gradient data.
			idx := l.maxIndices[i][j]
			l.prev.activationDeltas()[i].data[idx] = grad
		}
	}

	return nil
}

// NoOp. Nothing to update in this layer...
func (l *pool2D) update(_, _, _ float64) {}

func (l *pool2D) Shape() Shape {
	return l.shape
}

func (l *pool2D) batch() []Sample {
	return l.acts
}

func (l *pool2D) activationDeltas() []Sample {
	return l.dCdA
}

func (l *pool2D) sumSquaredWeights() float32 {
	return 0
}

func (l *pool2D) Name() string {
	return l.name
}

func (l *pool2D) parameters() map[string]model.Tensor {
	return nil
}

func (l *pool2D) loadParameters(_ model.Tensor) error {
	return errors.New("pool2D layers have no parameters to load")
}

var _ Layer = &fullyConnected{}

type fullyConnected struct {
	name  string
	prev  Layer
	actFn ActivationFunc

	logits *mat.Matrix
	dCdZ   *mat.Matrix

	prevActivations *mat.Matrix
	// acts            *mat.Matrix
	// dCdA            *mat.Matrix
	acts []Sample
	dCdA []Sample

	// The weights here are connections to the prev layer
	weights *mat.Matrix
	dCdW    *mat.Matrix

	biases []float32
	dCdB   []float32

	shape     Shape
	batchSize int
}

func (c *CNN) newFullyConnected(numNeurons int, prev Layer) *fullyConnected {
	l := &fullyConnected{
		name:  c.assignName("full"),
		prev:  prev,
		actFn: ReLU{},

		shape: [3]int{1, 1, numNeurons},
	}

	l.initWeights()
	l.initBiases()

	return l
}

func (l *fullyConnected) init(batchSize int) {
	// if !l.initialized {
	// 	l.initWeights()
	// 	l.initBiases()
	// 	l.initialized = true
	// }

	if batchSize != l.batchSize {
		l.batchSize = batchSize
		l.logits = mat.New(batchSize, l.shape.Width())
		l.acts = NewBatch(batchSize, l.shape)
	}

	// Zero grads
	l.dCdZ = mat.New(batchSize, l.shape.Width())
	l.dCdA = NewBatch(batchSize, l.shape)
	l.dCdW = mat.New(l.dCdW.NumRows(), l.dCdW.NumCols())
	l.dCdB = make([]float32, len(l.dCdB))

}

func (l *fullyConnected) initWeights() {
	// This assumes height and width of prev layer are both 1.
	l.weights = mat.New(l.prev.Shape().Volume(), l.shape.Width())
	l.dCdW = mat.New(l.prev.Shape().Volume(), l.shape.Width())

	numInputs := l.prev.Shape().Volume()
	for i := range l.weights.Data() {
		// Kaiming He initialization
		l.weights.Data()[i] = float32(rand.NormFloat64() * math.Sqrt(float64(2)/float64(numInputs)))
	}
}

func (l *fullyConnected) initBiases() {
	l.biases = make([]float32, l.shape.Width())
	l.dCdB = make([]float32, l.shape.Width())
	for i := range l.biases {
		l.biases[i] = float32(rand.NormFloat64() * 0.05)
	}
}

func (l *fullyConnected) Forward(_ bool) error {
	// We need to flatten each sample of the previous activations so that
	// we can represent the entire batch as a matrix.

	var prevActivations = make([][]float32, len(l.prev.batch()))
	for i, sample := range l.prev.batch() {
		prevActivations[i] = sample.data
	}

	var err error
	l.prevActivations, err = mat.NewFromSlices(prevActivations)
	if err != nil {
		return fmt.Errorf("failed instantiating previous activaitons matrix: %w", err)
	}

	// Do we need to zero the logits first? Maybe at the start of a new batch so
	// that the last batch doesn't contain old values if it is smaller than the
	// other batches. (Now that I think of it we probs shouldn't support incomplete
	// batches, we can just skip any remaining data that doesn't make up a whole
	// batch during the training process...)
	if err := l.logits.Mul(l.prevActivations, l.weights); err != nil {
		return fmt.Errorf("failed multiplying previous activations with weights: %w", err)
	}

	if err := l.logits.AddToRows(l.biases); err != nil {
		return fmt.Errorf("failed adding biases to logits: %w", err)
	}

	l.acts, err = NewBatchFromMatrix(l.shape, l.logits.ApplyActivation(l.actFn.Forward))
	if err != nil {
		return fmt.Errorf("failed creating new batch of activations: %w", err)
	}

	return nil
}

func (l *fullyConnected) Backward() error {
	// dCdz = actFn'(z) * dC/dA
	dCdAMat, err := BatchToMatrix(l.dCdA)
	if err != nil {
		return fmt.Errorf("failed converting batch to matrix: %w", err)
	}

	actPrime := l.logits.ApplyActivation(l.actFn.Backward)
	if err := l.dCdZ.Hadamard(actPrime, dCdAMat); err != nil {
		return fmt.Errorf("failed calculating logit gradients: %w", err)
	}

	// dCdB = sum(1 * dC/dz)/batchSize
	// dCdB is averaged over the batch for each neuron
	l.dCdB = l.dCdZ.AvgCols()

	// dCdW = sum(prevA * dC/dz)/batchSize
	// dCdW is also averaged over the batch
	if err := l.dCdW.Mul(l.prevActivations.Transpose(), l.dCdZ); err != nil {
		return fmt.Errorf("failed multiplying prev activations with logit grads: %w", err)
	}

	// Using this receiver to do a scalar multiplication becaue I'm lazy...
	l.dCdW = l.dCdW.ApplyActivation(func(in float32) float32 { return in / float32(l.logits.NumRows()) })

	// dCdA_(L-1) = W * dCdz
	// l.prev.dCdA, err = mat.MulConcurrent(l.dCdz, mat.Transpose(l.weights))
	dCdA_prevMat := mat.New(l.dCdZ.NumRows(), l.weights.NumRows())
	if err := dCdA_prevMat.Mul(l.dCdZ, l.weights.Transpose()); err != nil {
		return fmt.Errorf("failed multiplying logit deltas by transpose weights: %w", err)
	}

	// slog.Info("dCdA_prevMat", "data", dCdA_prevMat.Row(0))

	if l.prev.activationDeltas() != nil {
		for i := range dCdA_prevMat.NumRows() {
			for j, grad := range dCdA_prevMat.Row(i) {
				l.prev.activationDeltas()[i].data[j] = grad
			}
			// l.prev.activationDeltas()[i] = Sample{data: dCdA_prevMat.Row(i), Shape: l.prev.Shape()}
		}
	}

	return nil
}

func (l *fullyConnected) update(lr, lambda, _ float64) {
	// Regularize and update weights
	for i, weight := range l.weights.Data() {
		l.weights.Data()[i] -= float32(lr) * (l.dCdW.Data()[i] + (weight * float32(lambda)))
	}

	// Update biases
	for i := range l.biases {
		l.biases[i] -= float32(lr) * l.dCdB[i]
	}
}

func (l *fullyConnected) Shape() Shape {
	return l.shape
}

func (l *fullyConnected) batch() []Sample {
	return l.acts
}

func (l *fullyConnected) activationDeltas() []Sample {
	return l.dCdA
}

func (l *fullyConnected) sumSquaredWeights() float32 {
	var sum float32
	for _, w := range l.weights.Data() {
		sum += w * w
	}
	return sum
}

func (l *fullyConnected) Name() string {
	return l.name
}

func (l *fullyConnected) parameters() map[string]model.Tensor {
	return map[string]model.Tensor{
		fmt.Sprintf("%s.weights", l.name): {
			Name:  fmt.Sprintf("%s.weights", l.name),
			Shape: model.Shape{0, 0, l.prev.Shape().Width(), l.shape.Width()},
			Data:  l.weights.Data(),
		},
		fmt.Sprintf("%s.biases", l.name): {
			Name:  fmt.Sprintf("%s.biases", l.name),
			Shape: model.Shape{0, 0, 0, l.shape.Width()},
			Data:  l.biases,
		},
	}
}

// TODO: Should probably validate the shape...
func (l *fullyConnected) loadParameters(tensor model.Tensor) error {
	if strings.HasSuffix(tensor.Name, ".weights") {
		var err error
		if l.weights, err = mat.NewFromData(l.weights.NumRows(), l.weights.NumCols(), tensor.Data); err != nil {
			return fmt.Errorf("failed re-creating weights matrix: %w", err)
		}
		return nil
	}

	if strings.HasSuffix(tensor.Name, ".biases") {
		l.biases = tensor.Data
		return nil
	}

	return fmt.Errorf("unknown parameter type, expected weights or biases")
}

func (c *CNN) newOutput(numOutputs int, prev Layer) *output {
	return &output{
		fullyConnected: c.newFullyConnected(numOutputs, prev),
		actFn:          SoftMax{},
		shape:          [3]int{0, 0, numOutputs},
	}
}

type output struct {
	*fullyConnected

	actFn SoftMax

	shape Shape
}

// This receiver overrides the one from the embedded fully connected layer
// this is so that we can apply the softmax activation to the logits
func (l *output) batch() []Sample {
	var out = make([]Sample, len(l.fullyConnected.batch()))
	for i, sample := range l.fullyConnected.batch() {
		out[i] = Sample{
			data:  l.actFn.Forward(sample.data),
			Shape: l.shape,
		}
	}
	return out
}

var _ Layer = &flatten{}

type flatten struct {
	name  string
	prev  Layer
	shape Shape
}

func (c *CNN) newFlatten(prev Layer) *flatten {
	numInputs := prev.Shape().Channels() * prev.Shape().Height() * prev.Shape().Width()
	return &flatten{
		name: c.assignName("flatten"),
		prev: prev,

		// TODO: Update Shape implementation to support variable rank...
		shape: Shape{1, 1, numInputs},
	}
}

func (l *flatten) init(_ int) {}

// We can leave this as a no-op because all this layer does
// is pass through the previous data with different shape.
func (l *flatten) Forward(_ bool) error {
	return nil
}

// We should be able to leave this as a no-op since we just
// pass the grads to the previous layer in activationDeltas.
func (l *flatten) Backward() error {

	return nil
}

func (l *flatten) update(_, _, _ float64) {}

func (l *flatten) Shape() Shape {
	return l.shape
}

func (l *flatten) batch() []Sample {
	// This should just return the previous layers
	// activations unchanged but flattened.
	var acts = make([]Sample, len(l.prev.batch()))
	for i, ps := range l.prev.batch() {
		acts[i] = Sample{
			data:  ps.data,
			Shape: l.shape,
		}
	}

	return acts
}

func (l *flatten) activationDeltas() []Sample {
	// This technically returns an unexpected shape to the caller. This could
	// turn into a subtle and sneaky bug later...
	// TODO: Refactor flatten layer...
	return l.prev.activationDeltas()
}

func (l *flatten) sumSquaredWeights() float32 {
	return 0
}

func (l *flatten) Name() string {
	return l.name
}

func (l *flatten) parameters() map[string]model.Tensor {
	return nil
}

func (l *flatten) loadParameters(_ model.Tensor) error {
	return errors.New("flatten layers have no parameters to load")
}

func reverseSlice(slc []float32) []float32 {
	var out = make([]float32, len(slc))
	for i, elem := range slc {
		out[len(slc)-1-i] = elem
	}
	return out
}

var _ Layer = &dropout{}

type dropout struct {
	name         string
	prev         Layer
	dropoutRatio float64

	dropoutMask []Sample
	dCdA        []Sample

	shape Shape
}

func (c *CNN) newDropout(prev Layer, ratio float64) *dropout {
	return &dropout{
		name:         c.assignName("dropout"),
		prev:         prev,
		dropoutRatio: ratio,
		shape:        prev.Shape(),
	}
}

func (l *dropout) init(batchSize int) {
	l.dropoutMask = NewBatch(batchSize, l.shape)
	l.dCdA = NewBatch(batchSize, l.shape)
}

// Dropout layers zero the activations for a number of neurons and then
// scales up the rest of the activations proportionally to the number
// of dropped activations.
func (l *dropout) Forward(isTest bool) error {
	if isTest {
		for i, sample := range l.prev.batch() {
			for j := range sample.data {
				l.dropoutMask[i].data[j] = 1
			}
		}

		return nil
	}

	// fmt.Printf("before: %v\n", l.prev.batch()[0].data[:20])
	var dropoutThreshold = int(l.dropoutRatio * 1000)
	for i, sample := range l.prev.batch() {
		for j := range sample.data {
			n := rand.Intn(1000)
			if n < dropoutThreshold {
				sample.data[j] = 0
			} else {
				sample.data[j] *= float32(1 / (1 - l.dropoutRatio))
				l.dropoutMask[i].data[j] = 1
			}
		}
	}
	// fmt.Printf("after: %v\n", l.prev.batch()[0].data[:20])
	return nil
}

// During the backwards pass we need to zero out the grads for all of
// the elements that were dropped while passing back all other grads
// unchanged.
func (l *dropout) Backward() error {
	// fmt.Printf("before: %v\n", l.dCdA[0].data[:15])
	// fmt.Printf("dropout mask: %v\n", l.dropoutMask[0].data[:15])
	var err error
	for i, sample := range l.dCdA {
		l.prev.activationDeltas()[i], err = sample.Hadamard(l.dropoutMask[i])
		if err != nil {
			return fmt.Errorf("failed applying dropout mask to grads: %w", err)
		}
	}
	// fmt.Printf("after: %v\n", l.prev.activationDeltas()[0].data[:15])
	return nil
}

func (l *dropout) update(_, _, _ float64) {}

func (l *dropout) Shape() Shape {
	return l.shape
}

func (l *dropout) batch() []Sample {
	// This should return the previous layers activations with the
	// dropped elements set to zero and the remaining elemnts scaled
	// up appropriately.
	return l.prev.batch()
}

func (l *dropout) activationDeltas() []Sample {
	return l.dCdA
}

func (l *dropout) sumSquaredWeights() float32 {
	return 0
}

func (l *dropout) Name() string {
	return l.name
}

func (l *dropout) parameters() map[string]model.Tensor {
	return nil
}

func (l *dropout) loadParameters(_ model.Tensor) error {
	return errors.New("dropout layers have no parameters to load")
}

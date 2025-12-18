package cnn

import (
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"slices"

	"github.com/edatts/ml/pkg/mat"
)

type Kernel struct {
	weights []float32
	dCdW    []float32

	bias float32
	dCdB float32

	// shape int
	size int // Only square for now
}

func (k *Kernel) zeroGrads() {
	for i := range k.dCdW {
		k.dCdW[i] = 0
	}
	k.dCdB = 0
}

type Layer interface {
	Forward() error
	Backward() error
	Update(lr float32)

	// For CNNs we usually have shape (i, j, k, l) (batch_size, height, width, channels)
	// where channels starts off as the number of input channels (eg; 1 for greyscale
	// and 3 for RGB) and becomes the number of kernels/filters after a pass through a
	// convolutional later.
	Shape() Shape

	init(batchSize int)
	batch() []Sample
	activationDeltas() []Sample
	sumSquaredWeights() float32
}

type LayerType int

const (
	Input LayerType = iota
	Conv2D
	Pooling2D
	FullyConnected
	Output
)

func newLayer(lType LayerType, prev Layer, shape Shape, lambda float32) (Layer, error) {
	switch lType {
	case Input:
		return &input{shape: shape}, nil
	case Conv2D:
		return newConv2D(prev, 32, lambda), nil
	case Pooling2D:
		return newPool2D(prev), nil
	case FullyConnected:
		return newFullyConnected(shape.Channels(), prev, lambda), nil
	case Output:
		return newOutput(shape, prev, lambda), nil
	}

	return nil, fmt.Errorf("unknown layer type '%d'", lType)
}

// This layer simply holds the samples for a single batch. It has no kernels
// and no activation functions, hence the activations of this layer are
// simply the input values.
type input struct {

	// Assume Height * Width * Channels are flattened into the inner
	// slices, each element of the outer slice is a sample in the batch.
	// acts  [][]float32
	acts []Sample

	shape Shape
}

func newInput(shape Shape) *input {
	return &input{
		shape: shape,
	}
}

func (l *input) Forward() error {
	return nil
}

func (l *input) Backward() error {
	return nil
}

// NoOp. Nothing to update in this layer...
func (l *input) Update(_ float32) {}

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

type conv2D struct {
	prev   Layer
	lambda float32
	actFn  ActivationFunc

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

	shape       Shape
	kernelSize  int // For now just support square kernels
	stride      int // For now just support 1?
	initialized bool
	batchSize   int
}

func newConv2D(prev Layer, numKernels int, lambda float32) *conv2D {
	// Need some logic to ensure that the kernel shape, input shape, and
	// output shape are all compatible. This will need to take into
	// account the stride and padding for the layer as well.
	// H_out = floor((H−K1+2P)/S)+1
	// W_out = floor((W−K2+2P)/S)+1
	H_out := prev.Shape().Height() - 3 + 1
	W_out := prev.Shape().Width() - 3 + 1

	layer := &conv2D{
		prev:   prev,
		lambda: lambda,
		actFn:  ReLU{},

		shape:      Shape{H_out, W_out, numKernels},
		kernelSize: 3,
		stride:     1, // Only 1 for now
	}

	return layer
}

func (l *conv2D) init(batchSize int) {
	if !l.initialized {
		l.initKernels()
		l.initialized = true
	}

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
	kernelLen := l.kernelSize * l.kernelSize
	numInputs := l.shape.Channels() * l.shape.Height() * l.shape.Width() * kernelLen

	l.kernels = make([]*Kernel, l.Shape().Channels())
	for i := range len(l.kernels) {
		l.kernels[i] = &Kernel{
			weights: make([]float32, l.kernelSize*l.kernelSize),
			dCdW:    make([]float32, l.kernelSize*l.kernelSize),

			bias: float32(rand.NormFloat64()*0.05 - 0.025),
			dCdB: 0,

			size: l.kernelSize,
		}

		for j := range l.kernels[i].weights {
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

	kernelLen := l.kernelSize * l.kernelSize
	numCols := kernelLen * l.prev.Shape().Channels()
	var kernelMatrixData = make([]float32, len(l.kernels)*numCols)
	for i, kernel := range l.kernels {
		for j := range l.prev.Shape().Channels() {
			copy(kernelMatrixData[(i*numCols)+(j*kernelLen):(i*numCols)+(j*kernelLen)+kernelLen], kernel.weights)
		}
	}

	var err error
	l.kernelMatrix, err = mat.NewFromData(len(l.kernels), numCols, kernelMatrixData)
	if err != nil {
		return nil, fmt.Errorf("failed instantiating kernel matrix: %w", err)
	}

	return l.kernelMatrix, nil
}

func (l *conv2D) getdCdZMatrix(dCdZ []float32) (*mat.Matrix, error) {
	// We need to remember to set the matrix to nil after every batch for this
	// to work properly, otherwise the weights will never update...
	if len(dCdZ) == 0 {
		return nil, fmt.Errorf("layer is not initialized, no dCdA found")
	}

	derivLen := l.shape.Height() * l.shape.Width()
	numCols := derivLen * l.prev.Shape().Channels()
	var dCdZMatrixData = make([]float32, l.shape.Channels()*numCols)
	for i := range dCdZ {
		for j := range l.prev.Shape().Channels() {
			copy(dCdZMatrixData[(i*numCols)+(j*derivLen):(i*numCols)+(j*derivLen)+derivLen], dCdZ[i*derivLen:i*derivLen+derivLen])
		}
	}

	dCdZMatrix, err := mat.NewFromData(l.shape.Channels(), numCols, dCdZMatrixData)
	if err != nil {
		return nil, fmt.Errorf("failed instantiating activation derivative matrix: %w", err)
	}

	return dCdZMatrix, nil
}

func (l *conv2D) Forward() error {
	// The way convolutions are usually handled is by converting the masked
	// areas of the input into columns using the in2col method and unrolling
	// the kenels into rows. This results in two matrices that can then be
	// mulitplied together to produce the output feature map. If the input
	// has multiple channels, we then calculate an elementwise sum of the
	// result of the matrix multiplications to calculate our feature map.

	kernelMatrix, err := l.getKernelMatrix()
	if err != nil {
		return fmt.Errorf("failed getting kernel matrix: %w", err)
	}

	// So we can actually use im2col to generate one very alrge matrix for
	// our entire sample volume. To do this, we unroll each kernel into a
	// vector and concatenate it onto iteself once for each channel - 1.
	// Each one of these vectors becomes a row in a matrix. Next, we take
	// the input volume and unroll each masked sub-volume into a column.
	// This creates our second matrix. We can then take the dot product
	// between the two matrices and the resulting matrix corresponds to
	// our feature map where each row is an unrolled feature.

	for i, sample := range l.prev.batch() {
		sampleMatrix, err := sample.Im2Col(l.stride, l.kernelSize, l.kernelSize)
		if err != nil {
			return fmt.Errorf("im2col error: %w", err)
		}

		featureMapsMatrix := mat.New(kernelMatrix.NumRows(), sampleMatrix.NumCols())
		if err := featureMapsMatrix.Mul(kernelMatrix, sampleMatrix); err != nil {
			return fmt.Errorf("failed multiplying kernel and sample matrices: %w", err)
		}

		// Store logits in layer for backward pass. Each row in our feature maps
		// matrix is a single feature map.
		l.logits[i].data = featureMapsMatrix.Data()

		// Apply bias to feature map for each kernel
		for j := range l.logits[i].data {
			l.logits[i].data[j] += l.kernels[j%featureMapsMatrix.NumCols()].bias
		}

		// Apply activation func and store activations for backwards pass
		// l.acts[i].data = featureMapsMatrix.ApplyActivation().Data()
		l.acts[i] = l.logits[i].ApplyActivation(l.actFn.Backward)
	}

	return nil
}

func (l *conv2D) Backward() error {
	// Here we need to backpropagate our loss backwards through the conv
	// layer. To do with we carry out another convolution operation, but
	// instead of using the kernel, we use the derivative of the cost
	// with respect to the logits in our feature map. The output
	// of this operation is the derivative of the cost with respect to
	// the weights in our original input kernel.

	// dCdz = actFn'(z) * dC/dA
	var err error
	for i, logitSample := range l.logits {
		actPrimeSample := logitSample.ApplyActivation(l.actFn.Backward)
		l.dCdZ[i], err = actPrimeSample.Hadamard(l.dCdA[i])
		if err != nil {
			return fmt.Errorf("failed calculating logit delta: %w", err)
		}
	}

	// dCdW = dz/dW * dC/dz
	// dCdW = prevA * dC/dz (Using the convoluton operation)
	for i, sample := range l.prev.batch() {
		// Since we use the deltas of our feature maps (logits) as kernels for
		// another convolution operation, we need to perform another im2col
		// operation on our previous activations to generate a new sample matrix.
		dCdZMatrix, err := l.getdCdZMatrix(l.dCdZ[i].data)
		if err != nil {
			return fmt.Errorf("failed getting derivative matrix for activations: %w", err)
		}

		sampleMatrix, err := sample.Im2Col(l.stride, l.shape.Height(), l.shape.Width())
		if err != nil {
			return fmt.Errorf("im2col error: %w", err)
		}

		// Weight derivatives
		dCdWMatrix := mat.New(dCdZMatrix.NumRows(), sampleMatrix.NumCols())
		if err := dCdWMatrix.Mul(dCdZMatrix, sampleMatrix); err != nil {
			return fmt.Errorf("failed muiltiplying derivative and sample matrices: %w", err)
		}

		for j, kernel := range l.kernels {
			// Store weight derivatives for update
			for k := range kernel.dCdW {
				kernel.dCdW[k] += dCdWMatrix.Row(j)[k]
				// dCdB = dz/dB * dC/dz = 1 * dC/dz
				kernel.dCdB += dCdZMatrix.Row(j)[k]
			}
		}

		// dC/prevA = dz/dprevA * dC/dz
		// dC/prevA = W * dC/dz
		kernelMatrix, err := l.getKernelMatrix()
		if err != nil {
			return fmt.Errorf("failed getting kernel matrix: %w", err)
		}

		dCdAMatrix := mat.New(dCdZMatrix.NumRows(), kernelMatrix.NumRows())
		if err := dCdAMatrix.Mul(dCdZMatrix, kernelMatrix.Traspose()); err != nil {
			return fmt.Errorf("faileded mulitplyig dCdz and weights: %w", err)
		}

		prevActDeltas, err := NewSampleFromData(l.prev.Shape(), dCdAMatrix.Data())
		if err != nil {
			return fmt.Errorf("failed instantiating prev layers activation delta sample: %w", err)
		}

		l.prev.activationDeltas()[i] = prevActDeltas
	}

	// Average weight and bias deltas over batch
	batchLen := len(l.prev.batch())
	for _, kernel := range l.kernels {
		for k := range kernel.dCdW {
			kernel.dCdW[k] /= float32(batchLen)
		}
		kernel.dCdB /= float32(batchLen)
	}

	return nil
}

func (l *conv2D) Update(lr float32) {
	for i, kernel := range l.kernels {
		for j, weight := range kernel.weights {
			kernel.weights[i] += (l.lambda * weight) + (-lr * kernel.dCdW[j])
		}

		kernel.bias += -lr * kernel.dCdB
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

}

type pool2D struct {
	prev Layer

	acts       []Sample
	dCdA       []Sample
	maxIndices [][]int

	H_pad       int
	W_pad       int
	windowSize  int // For now only square
	shape       Shape
	initialized bool
	batchSize   int
}

func newPool2D(prev Layer) *pool2D {
	// Ensure input shape and window size are compatible. We should probably only
	// specify the window size and then infer the output shape from the input
	// shape, if it is compatible with the window size.

	var H_pad int
	var W_pad int

	if prev.Shape().Height()%2 != 0 {
		// Going to have to use padding...
		slog.Warn("incompatible height, adding height padding...")
		H_pad = 1
		// slog.Error("input shape is not compatible with pooling parameters", "inputShape", prev.Shape(), "poolingWindow", 2, "poolingStride", 2)
		// panic("input height is not compatible with window size and stride of 2")
	}

	if prev.Shape().Width()%2 != 0 {
		slog.Warn("incompatible width, adding width padding...")
		W_pad = 1
		// slog.Error("input shape is not compatible with pooling parameters", "inputShape", prev.Shape(), "poolingWindow", 2, "poolingStride", 2)
		// panic("input width is not compatible with window size and stride of 2")
	}

	// H_out = floor((H−K1+2P)/S)+1
	// W_out = floor((W−K2+2P)/S)+1
	H_out := (prev.Shape().Height()-2+H_pad)/2 + 1
	W_out := (prev.Shape().Width()-2+W_pad)/2 + 1

	return &pool2D{
		prev:       prev,
		windowSize: 2, // Window size is the same as stride...
		H_pad:      H_pad,
		W_pad:      W_pad,
		shape:      Shape{H_out, W_out, prev.Shape().Channels()},
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

	// Zero grads
	l.dCdA = NewBatch(batchSize, l.shape)

}

// TODO: Tests for forward pass shape and values...
func (l *pool2D) Forward() error {
	// The forward pass of the pooling layer calculates the max value for
	// a particular area of each input channels. It operates individually
	// on each input channel, preserving depth of the input. The stride
	// is typically equal to the size of the pooling window so that each
	// element of the input is only operated on once.

	// Initially let's omit padding and enforce that the input dimensions
	// must be compatible with the window size.

	// In the case below we replace 2P with P because we are only padding
	// the rightmost column and bottom row with zeros...
	// H_out = floor((H−K1+2P)/S)+1
	// W_out = floor((W−K2+2P)/S)+1
	H_in := l.prev.batch()[0].Height()
	W_in := l.prev.batch()[0].Width()
	H_out := ((H_in - l.windowSize + l.H_pad) / l.windowSize) + 1
	W_out := ((W_in - l.windowSize + l.W_pad) / l.windowSize) + 1
	imageLen := H_in * W_in
	paddedLen := (H_in + l.H_pad) * (W_in + l.W_pad)
	outLen := H_out * W_out

	for n, sample := range l.prev.batch() {
		var outData = make([]float32, outLen*sample.Channels())
		l.maxIndices[n] = []int{}
		for m := range sample.Channels() {
			image := l.padImage(paddedLen, sample.data[m*imageLen:m*imageLen+imageLen])
			for i := range H_out {
				for j := range W_out {
					var chunk []float32
					for k := range l.windowSize {
						chunk = append(chunk, image[(i*W_in)+(k*W_in)+j:(i*W_in)+(k*W_in)+j+l.windowSize]...)
					}
					maximum := slices.Max(chunk)
					maxIdx := slices.Index(image, maximum) + m*imageLen
					outData[(m*outLen)+(i*H_out)+j] = maximum
					l.maxIndices[n] = append(l.maxIndices[n], maxIdx)
				}
			}
		}
		outSample, err := NewSampleFromData([3]int{H_out, W_out, sample.Channels()}, outData)
		if err != nil {
			return fmt.Errorf("failed instantiting new output sample: %w", err)
		}
		l.acts[n] = outSample
	}

	return nil
}

// TODO: Write test for padImage...
func (l *pool2D) padImage(paddedLen int, data []float32) []float32 {
	var out = make([]float32, paddedLen)
	var x int
	for i, datum := range data {
		out[i+x] = datum
		if (i+1)%l.prev.batch()[0].Width() == 0 {
			x++
		}
	}
	return out
}

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
			idx := l.maxIndices[i][j]
			l.prev.activationDeltas()[i].data[idx] = grad
		}
	}

	return nil
}

// NoOp. Nothing to update in this layer...
func (l *pool2D) Update(_ float32) {}

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

type fullyConnected struct {
	prev   Layer
	lambda float32
	actFn  ActivationFunc

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

	shape Shape
}

func newFullyConnected(numNeurons int, prev Layer, lambda float32) *fullyConnected {
	// For now we're going to consider the shape of this type of layer to be
	// (1, 1, C). We will omit a flattening layer and implement a flatten
	// method on the Sample, this way we can represent the entirety of the
	// batch of previous layer activations as a single matrix to use in
	// the forward pass of this layer. We will probably have to implement
	// the reverse of the flatten operation as well so that we can easily
	// backpropagate the gradients into the previous layer.

	return &fullyConnected{
		prev:   prev,
		lambda: lambda,
		actFn:  ReLU{},

		shape: [3]int{1, 1, numNeurons},
	}
}

func (l *fullyConnected) init(batchSize int) {
	panic("fully connected init is unimplemented")
	if !l.initialized {
		l.initWeights()
		l.initBiases()
		l.initialized = true
	}

	if batchSize != l.batchSize {
		l.batchSize = batchSize
		l.logits
		l.acts
	}

	// Zero grads
	l.dCdZ = mat.New(l.dCdZ.NumRows(), l.dCdZ.NumCols())
	l.dCdA = NewBatch(batchSize, l.shape)
	l.dCdW = mat.New(l.dCdW.NumRows(), l.dCdW.NumCols())
	l.dCdB = make([]float32, len(l.dCdB))

}

func (l *fullyConnected) Forward() error {
	// We need to flatten each sample of the previous activations so that
	// we can represent the entire batch as a metrix.

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
	if err := l.dCdW.Mul(l.prevActivations.Traspose(), l.dCdZ); err != nil {
		return fmt.Errorf("failed multiplying prev activations with logit grads: %w", err)
	}

	// Using this receiver to do a scalar multiplication becaue I'm lazy...
	l.dCdW = l.dCdW.ApplyActivation(func(in float32) float32 { return in / float32(l.logits.NumRows()) })

	// dCdA_(L-1) = W * dCdz
	// l.prev.dCdA, err = mat.MulConcurrent(l.dCdz, mat.Transpose(l.weights))
	dCdA_prevMat := mat.New(l.dCdZ.NumRows(), l.weights.NumRows())
	if err := dCdA_prevMat.Mul(l.dCdZ, l.weights.Traspose()); err != nil {
		return fmt.Errorf("failed multiplying logit deltas by transpose weights: %w", err)
	}

	for i := range dCdA_prevMat.NumRows() {
		l.prev.activationDeltas()[i] = Sample{data: dCdA_prevMat.Row(i), Shape: l.prev.Shape()}
	}

	return nil
}

func (l *fullyConnected) Update(lr float32) {
	// Regularize and update weights
	for i, weight := range l.weights.Data() {
		l.weights.Data()[i] += (weight * l.lambda) + (-lr * l.dCdW.Data()[i])
	}

	// Update biases
	for i := range l.biases {
		l.biases[i] += -lr * l.dCdB[i]
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

}

func newOutput(shape Shape, prev Layer, lambda float32) *output {
	return &output{
		fullyConnected: newFullyConnected(shape.Channels(), prev, lambda),
		actFn:          SoftMax{},
		shape:          shape,
	}
}

type output struct {
	*fullyConnected

	actFn SoftMax

	shape Shape
}

// This receiver overrides the one from the embedded fully conencted layer
// this is so that we can apply the softmax activation to the
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

// type output struct {
// 	prev Layer

// 	logits []Sample
// 	dCdZ   []Sample

// 	acts []Sample
// 	dCdA []Sample

// 	// The weights here are connections to the prev layer
// 	weights []Sample
// 	dCdW    []Sample

// 	biases Sample
// 	dCdB   Sample

// 	shape Shape
// }

// func (l *output) Forward() error {

// 	return nil
// }

// func (l *output) Backward() error {
// 	return nil
// }

// func (l *output) Shape() Shape {
// 	return l.shape
// }

// func (l *output) batch() []Sample {
// 	return l.acts
// }

// func (l *output) activationDeltas() []Sample {
// 	return l.dCdA
// }

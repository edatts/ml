package mlp

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/edatts/ml/pkg/mat"
	"github.com/edatts/ml/pkg/shape"
	"github.com/edatts/ml/pkg/types"
)

type Shape = shape.Shape

type Layer interface {
	Backward() error
	Forward() error
	Shape() Shape
	Type() LayerType

	width() int
	activations() *mat.F32Matrix
	grads() *mat.F32Matrix
	update(lr, lambda, beta float64)
	init(batchSize int)
	sumSquaredWeights() float64
	data() [2]types.Tensor
}

type LayerType string

const (
	Input  LayerType = "input"
	Hidden LayerType = "hidden"
	Output LayerType = "output"
)

var _ Layer = &input{}

type input struct {
	acts *mat.F32Matrix

	size  int
	shape Shape
}

func (i *input) Backward() error {
	return nil
}

func (i *input) Forward() error {
	return nil
}

func (i *input) activations() *mat.F32Matrix {
	return i.acts
}

func (i *input) grads() *mat.F32Matrix {
	// No grads for this layer
	return &mat.F32Matrix{}
}

func (i *input) update(_, _, _ float64) {}

func (i *input) init(batchSize int) {
	i.acts = mat.New(batchSize, i.Shape().Width())
}

func (i *input) sumSquaredWeights() float64 {
	return 0
}

func (i *input) data() [2]types.Tensor {
	return [2]types.Tensor{}
}

func (i *input) Shape() Shape {
	return i.shape
}

func (i *input) Type() LayerType {
	return Input
}

func (i *input) width() int {
	return i.size
}

var _ Layer = &layer{}

func (l *layer) width() int {
	return 0
}

type layer struct {
	prev  Layer
	actFn ActivationFunc
	// initialized bool
	batchSize      int
	size           int
	isOutput       bool
	classification bool

	// weights [][]float64
	// dCdW    [][]float64
	weights    *mat.F32Matrix
	dCdW       *mat.F32Matrix
	velocities [][]float32

	// biases []float64
	// dCdB   []float64
	biases []float32
	dCdB   []float32

	// logits [][]float64
	// dCdz   [][]float64
	logits *mat.F32Matrix
	dCdz   *mat.F32Matrix

	// activations [][]float64
	// dCdA        [][]float64
	acts *mat.F32Matrix
	dCdA *mat.F32Matrix

	shape Shape
}

func (m *MLP) newLayer(lType LayerType, n int, prev Layer) Layer {
	if lType == Input {
		return &input{
			size:  n,
			shape: shape.New(n),
		}
	}

	l := &layer{
		prev:  prev,
		actFn: m.getActivationFunc(lType),
		shape: shape.New(n),
	}

	// In case of classification we embed the final layer to overwrite it's
	// activations() receiver to apply the Softmax activation function.
	if lType == Output {
		l.isOutput = true
	}

	if m.classification {
		l.classification = true
	}

	l.weights, l.dCdW, l.velocities = initWeights(l.actFn, l.prev.Shape().Width(), l.Shape().Width())
	l.biases, l.dCdB = initBiases(l.Shape().Width())

	return l
}

func (l *layer) grads() *mat.F32Matrix {
	// if l.Type() == Output {
	// 	slog.Info("grads", "grads", l.dCdA.Data()[:4])
	// }
	return l.dCdA
}

func (l *layer) activations() *mat.F32Matrix {
	if l.isOutput && l.classification {
		// Override the Activations() receiver of the embedded layer so that we can
		// apply the softmax activation when reading the output activations.
		for i := range l.logits.NumRows() {
			for j, act := range (SoftMax{}).Forward(l.logits.Row(i)) {
				l.acts.Row(i)[j] = act
			}
		}
	}

	// if l.Type() == Output {
	// slog.Info("acts", "acts", l.acts.Data()[:4])
	// slog.Info("weights", "weights", l.weights.Data()[:10])
	// }

	return l.acts
}

func (l *layer) init(batchSize int) {
	if l.prev == nil {
		return
	}

	if batchSize != l.batchSize {
		// Update batch size
		l.batchSize = batchSize
		// Resize logits, and activations for new batch size
		l.logits, l.acts = initActivations(batchSize, l.Shape().Width())
		l.dCdz, l.dCdA = initGradients(batchSize, l.Shape().Width())
	}
}

func (l *layer) Forward() error {
	// Forward required us to, for each neuron, take the activations of
	// the previous layer, multiply them by the weights, then add the bias.
	// In our activations matrix each neuron is indexed by row and each
	// batch sample is indexed by column. In our weights matrix each neuron
	// in the current layer is indexed by row and connection (weight) to a
	// previous layer's neuron is indexed by column.

	// z = actFn(dot(A,W) + b)

	// Imagine we want to pipeline the execution of our layers, with each
	// layer executing on a different goroutine.
	//
	// We can only run the forward pass once the result from any prior
	// Forward() calls has been fully read. So we need to reader to set
	// a flag 'consumed' on the layer. If 'consumed', we can run the
	// next batch

	if err := l.logits.Mul(l.prev.activations(), l.weights); err != nil {
		return fmt.Errorf("failed multiplying activations and weights: %w", err)
	}

	if err := l.logits.AddToRows(l.biases); err != nil {
		return fmt.Errorf("failed adding biases: %w", err)
	}

	l.acts = l.logits.ApplyActivation(l.actFn.Forward)
	return nil
}

func (l *layer) Backward() error {
	// There are three main derivatves we need here, all three are the derivative of
	// the loss, but with respect to different components, the weights, the bias, and
	// the previous activations. The derivative of the previous activation needs to be
	// a summation of all of the downstream partial derivatives with respect to the
	// downstream activations.

	// slog.Info("grads shape", "rows", l.dCdA.NumRows(), "cols", l.dCdA.NumCols())
	// slog.Info("grads", "grads", l.dCdA.Row(0)[0:10])

	// dCdz = actFn'(z) * dC/dA
	actPrime := l.logits.ApplyActivation(l.actFn.Backward)
	if err := l.dCdz.Hadamard(actPrime, l.dCdA); err != nil {
		return fmt.Errorf("failed hadamard: %w", err)
	}

	// dCdB = sum(1 * dC/dz)/batchSize
	l.dCdB = l.dCdz.AvgCols()

	// dCdW = sum(prevA * dC/dz)/batchSize
	if err := l.dCdW.Mul(l.prev.activations().Transpose(), l.dCdz); err != nil {
		return fmt.Errorf("failed multiplying acts transpose with logit grads: %w", err)
	}

	// Using this receiver to do a scalar multiplication becaue I'm lazy...
	l.dCdW = l.dCdW.ApplyActivation(func(in float32) float32 { return in / float32(l.batchSize) })

	// dCdA_(L-1) = W * dCdz
	if l.prev.grads() != nil && l.prev.grads().Data() != nil { // Exclude input layer
		if err := l.prev.grads().Mul(l.dCdz, l.weights.Transpose()); err != nil {
			return fmt.Errorf("failed multiplying logit grads with weights transpose: %w", err)
		}
	}

	return nil
}

func (l *layer) update(lr, lambda, beta float64) {
	// Regularize and update
	for i := range l.weights.NumRows() {
		for j, w := range l.weights.Row(i) {
			// To support SGD with momentum we need to calculate the velocity
			// term and use it in place of the weight gradient. When beta is
			// 0 then the velocity term is simply the most recent weight grad.
			//
			// // vt = β * vt_prev + (1 - β) * dCdW
			l.velocities[i][j] = l.velocities[i][j]*float32(beta) + (1-float32(beta))*l.dCdW.Row(i)[j]

			l.weights.Row(i)[j] -= float32(lr) * (l.velocities[i][j] + (float32(lambda) * w))
		}
	}

	for i := range l.biases {
		l.biases[i] += float32(-lr) * l.dCdB[i]
	}
}

func (l *layer) data() [2]types.Tensor {
	// panic(fmt.Sprintf("numRows='%d' numCols='%d'", l.weights.NumRows(), l.weights.NumCols()))
	return [2]types.Tensor{
		{
			Data:     &types.F32Data{Data: l.weights.Data()},
			DataType: types.F32,
			Shape:    shape.New(l.weights.NumRows(), l.weights.NumCols()),
		},
		{
			Data:     &types.F32Data{Data: l.biases},
			DataType: types.F32,
			Shape:    shape.New(len(l.biases)),
		},
	}
}

func (l *layer) Shape() Shape {
	return l.shape
}

func (l *layer) Type() LayerType {
	if l.isOutput {
		return Output
	}
	return Hidden
}

func initWeights(act ActivationFunc, prevWidth, width int) (*mat.F32Matrix, *mat.F32Matrix, [][]float32) {
	var (
		weights    = mat.New(prevWidth, width)
		dCdW       = mat.New(prevWidth, width)
		velocities = make([][]float32, prevWidth)
	)

	for i := range int(prevWidth) {
		velocities[i] = make([]float32, width)
		for j := range width {
			switch act.(type) {
			case Sigmoid:
				// Xavier Normal initialization: W ~ N(0, sqrt(2/n_in + n_out))
				weights.Row(i)[j] = float32(rand.NormFloat64() * math.Sqrt(float64(2)/float64(prevWidth+width)))
			case ReLU, LeakyReLU:
				// Kaiming He intialization: W ~ N(0, sqrt(2/n_inputs))

				// Here we modify the n_inputs term to be max(n_inputs, 10), this is to
				// prevent exploding gradients when the number of input features is very
				// small. This was observed with n_inputs = 1 when testing regression.
				weights.Row(i)[j] = float32(rand.NormFloat64() * math.Sqrt(float64(2)/float64(max(prevWidth, 10))))
			default:
				// Kaiming He intialization: W ~ N(0, sqrt(2/n_inputs))
				weights.Row(i)[j] = float32(rand.NormFloat64() * math.Sqrt(float64(2)/float64(max(prevWidth, 10))))
			}
		}
	}

	return weights, dCdW, velocities
}

func initBiases(width int) ([]float32, []float32) {
	var (
		biases = make([]float32, width)
		dCdB   = make([]float32, width)
	)

	for i := range width {
		biases[i] = float32(rand.NormFloat64() * 0.05)
	}

	return biases, dCdB
}

func initGradients(batchSize int, width int) (*mat.F32Matrix, *mat.F32Matrix) {
	return mat.New(batchSize, width), mat.New(batchSize, width)
}

func initActivations(batchSize int, width int) (*mat.F32Matrix, *mat.F32Matrix) {
	return mat.New(batchSize, width), mat.New(batchSize, width)
}

func (m *MLP) getActivationFunc(lType LayerType) ActivationFunc {
	switch lType {
	case Input, Output:
		// Used for output because we override with softmax in case of classification
		return NoOp{}
	case Hidden:
		fallthrough
	default:
		// return ReLU{}
		return LeakyReLU{}
	}
}

func (l *layer) sumSquaredWeights() float64 {
	var sum float64
	for i := range l.weights.NumRows() {
		for _, w := range l.weights.Row(i) {
			sum += float64(w * w)
		}
	}
	return sum
}

// dz/dW = prevA
//
// dA/dz = derivActiv(z)
//
// dC/dA = 2(y_bar - y) for MSE
//
// dz/dprevA = W
//
// Weight
// dC/dW = dz/dW * dA/dz * dC/dA = prevA * derivActiv(z) * 2(y_bar - y)
//
// Bias
// dC/dB = dz/dB * dA/dz * dC/dA = 1 * derivActiv(z) * 2(y_bar - y)
//
// Prev A
// dC/dprevA = dz/dprevA * dA/dz * dC/dA
// 			 = W * derivActiv(z) * 2(y_bar - y)
//
// Prev weight
// dC/dprevW = dprevz/dprevW * dprevA/dprevz 	 * dC/dprevA
//			 = dprevz/dprevW * dprevA/dprevz 	 * dz/dprevA * dA/dz * dC/dA
// 			 = prevprevA 	 * derivActiv(prevz) * W * derivActiv(z) * 2(y_bar - y)
//
// Prev Bias
// dC/dprevB = dprevZ/dprevB * dprevA/dprevz 	 * dz/dprevA * dA/dz 		 * dC/dA
// 			 = 1 			 * derivActiv(prevz) * W 		 * derivActiv(z) * 2(y_bar - y)
//
// Prev prev A
// dC/dprevprevA = dprevz/dprevprevA * dprevA/dprevz  	  * dC/dprevA
//				 = prevW			 * derivActive(prevz) * W * derivActiv(z) * 2(y_bar - y)

package mlp

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/edatts/ml/pkg/mat"
)

type Layer interface {
	Backward() error
	Forward() error
	width() int
	activations() *mat.Matrix
	grads() *mat.Matrix
	update(lr, lambda float64)
	init(batchSize int)
	sumSquaredWeights() float64
}

type LayerType int

const (
	Input LayerType = iota
	Hidden
	Output
)

type input struct {
	// TODO: Add shape to all layer types
	// shape Shape

	acts *mat.Matrix

	size int
}

func (i *input) Backward() error {
	return nil
}

func (i *input) Forward() error {
	return nil
}

func (i *input) width() int {
	return i.size
}

func (i *input) activations() *mat.Matrix {
	return i.acts
}

func (i *input) grads() *mat.Matrix {
	// No grads for this layer
	return &mat.Matrix{}
}

func (i *input) update(_, _ float64) {}

func (i *input) init(batchSize int) {
	i.acts = mat.New(batchSize, i.size)
}

func (i *input) sumSquaredWeights() float64 {
	return 0
}

type layer struct {
	prev        Layer
	actFn       ActivationFunc
	initialized bool
	batchSize   int
	size        int

	// weights [][]float64
	// dCdW    [][]float64
	weights *mat.Matrix
	dCdW    *mat.Matrix

	// biases []float64
	// dCdB   []float64
	biases []float32
	dCdB   []float32

	// logits [][]float64
	// dCdz   [][]float64
	logits *mat.Matrix
	dCdz   *mat.Matrix

	// activations [][]float64
	// dCdA        [][]float64
	acts *mat.Matrix
	dCdA *mat.Matrix
}

type output struct {
	*layer

	actFn SoftMax
}

// Override the Activations() receiver of the embedded layer so that we can
// apply the softmax activation when reading the output activations.
func (o *output) activations() *mat.Matrix {
	for i := range o.logits.NumRows() {
		for j, act := range o.actFn.Forward(o.logits.Row(i)) {
			o.acts.Row(i)[j] = act
		}
	}

	return o.acts
}

func (m *MLP) newLayer(lType LayerType, n int, prev Layer) Layer {
	if lType == Input {
		return &input{
			size: n,
		}
	}

	l := &layer{
		prev:  prev,
		actFn: m.getActivationFunc(lType),
		size:  n,
	}

	// In case of classification we embed the final layer to overwrite it's
	// activations() receiver to apply the Softmax activation function.
	if lType == Output && m.classification {
		out := &output{
			layer: l,
		}
		return out
	}

	return l
}

func (l *layer) width() int {
	return l.size
}

func (l *layer) grads() *mat.Matrix {
	return l.dCdA
}

func (l *layer) activations() *mat.Matrix {
	return l.acts
}

func (l *layer) init(batchSize int) {
	if l.prev == nil {
		return
	}

	if !l.initialized {
		// Init
		l.weights, l.dCdW = initWeights(l.actFn, l.prev.width(), l.width())
		l.biases, l.dCdB = initBiases(l.width())
		l.initialized = true
	}

	if batchSize != l.batchSize {
		// Update batch size
		l.batchSize = batchSize
		// Resize logits, and activations for new batch size
		l.logits, l.acts = initActivations(batchSize, l.width())
		l.dCdz, l.dCdA = initGradients(batchSize, l.width())
	}

	// // Do we need to zero these every time?
	// l.logits, l.acts = initActivations(batchSize, l.width())
	// // Zero grads
	// l.dCdz, l.dCdA = initGradients(batchSize, l.width())
	// l.dCdB = make([]float32, l.width())
	// l.dCdW = mat.New(l.prev.width(), l.width())
}

func (l *layer) Forward() error {
	// Forward required us to, for each neuron, take the activations of
	// the previous layer, multiply them by the weights, then add the bias.
	// In our activations matrix each neuron is indexed by row and each
	// batch sample is indexed by column. In our weights matrix each neuron
	// in the current layer is indexed by row and connection (weight) to a
	// previous layer's neuron is indexed by column.

	// z = actFn(dot(A,W) + b)

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
	if err := l.dCdW.Mul(l.prev.activations().Traspose(), l.dCdz); err != nil {
		return fmt.Errorf("failed multiplying acts transpose with logit grads: %w", err)
	}

	// Using this receiver to do a scalar multiplication becaue I'm lazy...
	l.dCdW = l.dCdW.ApplyActivation(func(in float32) float32 { return in / float32(l.batchSize) })

	// dCdA_(L-1) = W * dCdz
	if l.prev.grads() != nil && l.prev.grads().Data() != nil { // Exclude input layer
		if err := l.prev.grads().Mul(l.dCdz, l.weights.Traspose()); err != nil {
			return fmt.Errorf("failed multiplying logit grads with weights transpose: %w", err)
		}
	}

	return nil
}

func (l *layer) update(lr, lambda float64) {
	// Regularize and update
	for i := range l.weights.NumRows() {
		for j, w := range l.weights.Row(i) {
			l.weights.Row(i)[j] -= float32(lr) * (l.dCdW.Row(i)[j] + (float32(lambda) * w))
		}
	}

	for i := range l.biases {
		l.biases[i] += float32(-lr) * l.dCdB[i]
	}

}

func initWeights(act ActivationFunc, prevWidth, width int) (*mat.Matrix, *mat.Matrix) {
	var (
		weights = mat.New(prevWidth, width)
		dCdW    = mat.New(prevWidth, width)
	)

	for i := range int(prevWidth) {
		for j := range width {
			switch act.(type) {
			case Sigmoid:
				// Xavier Normal initialization: W ~ N(0, sqrt(2/n_in + n_out))
				weights.Row(i)[j] = float32(rand.NormFloat64() * math.Sqrt(float64(2)/float64(prevWidth+width)))
			case ReLU:
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

	return weights, dCdW
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

func initGradients(batchSize int, width int) (*mat.Matrix, *mat.Matrix) {
	return mat.New(batchSize, width), mat.New(batchSize, width)
}

func initActivations(batchSize int, width int) (*mat.Matrix, *mat.Matrix) {
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
		return ReLU{}
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

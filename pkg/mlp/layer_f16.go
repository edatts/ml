package mlp

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/edatts/ml/pkg/mat"
	"github.com/edatts/ml/pkg/shape"
	"github.com/edatts/ml/pkg/types"
	"github.com/x448/float16"
)

type F16Layer interface {
	Backward() error
	Forward() error
	Shape() Shape
	Type() LayerType

	width() int
	activations() *mat.F16Matrix
	grads() *mat.F16Matrix
	update(lr, lambda float64, gradScalingFactor float32)
	init(batchSize int)
	sumSquaredWeights() float64
	data() [2]types.Tensor
}

var _ F16Layer = &f16Input{}

type f16Input struct {
	acts *mat.F16Matrix

	size  int
	shape Shape
}

func (i *f16Input) Backward() error {
	return nil
}

func (i *f16Input) Forward() error {
	return nil
}

func (i *f16Input) activations() *mat.F16Matrix {
	return i.acts
}

func (i *f16Input) grads() *mat.F16Matrix {
	// No grads for this layer
	return &mat.F16Matrix{}
}

func (i *f16Input) update(_, _ float64, _ float32) {}

func (i *f16Input) init(batchSize int) {
	i.acts = mat.NewF16(batchSize, i.Shape().Width())
}

func (i *f16Input) sumSquaredWeights() float64 {
	return 0
}

func (i *f16Input) data() [2]types.Tensor {
	return [2]types.Tensor{}
}

func (i *f16Input) Shape() Shape {
	return i.shape
}

func (i *f16Input) Type() LayerType {
	return Input
}

func (i *f16Input) width() int {
	return i.size
}

var _ F16Layer = &f16Layer{}

func (l *f16Layer) width() int {
	return 0
}

type f16Layer struct {
	prev      F16Layer
	actFn     F16ActivationFunc
	batchSize int
	size      int
	isOutput  bool

	masterWeights []float32
	weights       *mat.F16Matrix
	dCdW          *mat.F16Matrix

	masterBiases []float32
	biases       []types.Float16
	dCdB         []types.Float16

	logits *mat.F16Matrix
	dCdz   *mat.F16Matrix

	acts *mat.F16Matrix
	dCdA *mat.F16Matrix

	shape Shape
}

func (m *F16MLP) newLayer(lType LayerType, n int, prev F16Layer) F16Layer {
	if lType == Input {
		return &f16Input{
			size:  n,
			shape: shape.New(n),
		}
	}

	l := &f16Layer{
		prev:  prev,
		actFn: m.getActivationFunc(lType),
		shape: shape.New(n),
	}

	// In case of classification we embed the final layer to overwrite it's
	// activations() receiver to apply the Softmax activation function.
	if lType == Output && m.classification {
		l.isOutput = true
	}

	l.weights, l.dCdW = initF16Weights(l.actFn, l.prev.Shape().Width(), l.Shape().Width())
	l.biases, l.dCdB = initF16Biases(l.Shape().Width())
	l.masterWeights, l.masterBiases = initMasterWeightsAndBiases(l.weights, l.biases)

	return l
}

func (l *f16Layer) grads() *mat.F16Matrix {
	return l.dCdA
}

func (l *f16Layer) activations() *mat.F16Matrix {
	if l.isOutput {
		// Override the Activations() receiver of the embedded layer so that we can
		// apply the softmax activation when reading the output activations.
		for i := range l.logits.NumRows() {
			for j, act := range (F16SoftMax{}).Forward(l.logits.Row(i)) {
				l.acts.Row(i)[j] = act
			}
		}

		// fmt.Printf("output acts: %v\n\n", l.acts.Data()[:10])
	}

	return l.acts
}

func (l *f16Layer) init(batchSize int) {
	if l.prev == nil {
		return
	}

	if batchSize != l.batchSize {
		// Update batch size
		l.batchSize = batchSize
		// Resize logits, and activations for new batch size
		l.logits, l.acts = initF16Activations(batchSize, l.Shape().Width())
		l.dCdz, l.dCdA = initF16Gradients(batchSize, l.Shape().Width())
	}
}

func (l *f16Layer) Forward() error {
	// Forward required us to, for each neuron, take the activations of
	// the previous layer, multiply them by the weights, then add the bias.
	// In our activations matrix each neuron is indexed by row and each
	// batch sample is indexed by column. In our weights matrix each neuron
	// in the current layer is indexed by row and connection (weight) to a
	// previous layer's neuron is indexed by column.

	// z = actFn(dot(A,W) + b)

	// fmt.Printf("prec activations: %v\n\n", l.prev.activations().Data()[:10])

	if err := l.logits.Mul(l.prev.activations(), l.weights); err != nil {
		return fmt.Errorf("failed multiplying activations and weights: %w", err)
	}

	if err := l.logits.AddToRows(l.biases); err != nil {
		return fmt.Errorf("failed adding biases: %w", err)
	}

	// fmt.Printf("logits: %v\n\n", l.logits.Data()[:10])

	l.acts = l.logits.ApplyActivation(l.actFn.Forward)

	// fmt.Printf("acts: %v\n\n", l.acts.Data()[:10])

	return nil
}

func (l *f16Layer) Backward() error {
	// There are three main derivatves we need here, all three are the derivative of
	// the loss, but with respect to different components, the weights, the bias, and
	// the previous activations. The derivative of the previous activation needs to be
	// a summation of all of the downstream partial derivatives with respect to the
	// downstream activations.

	// slog.Info("grads shape", "rows", l.dCdA.NumRows(), "cols", l.dCdA.NumCols())
	// slog.Info("grads", "grads", l.dCdA.Row(0)[0:10])

	// fmt.Printf("activation grads: %v\n", l.dCdA.Data()[:10])

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
	l.dCdW = l.dCdW.ApplyActivation(func(in types.Float16) types.Float16 { return float16.Fromfloat32(in.Float32() / float32(l.batchSize)) })

	// dCdA_(L-1) = W * dCdz
	if l.prev.grads() != nil && l.prev.grads().Data() != nil { // Exclude input layer
		if err := l.prev.grads().Mul(l.dCdz, l.weights.Transpose()); err != nil {
			return fmt.Errorf("failed multiplying logit grads with weights transpose: %w", err)
		}
	}

	return nil
}

func (l *f16Layer) update(lr, lambda float64, scalingFactor float32) {
	// Regularize and update
	for i, w := range l.masterWeights {
		reg := float32(lambda) * w
		scaledGrad := l.dCdW.Data()[i].Float32() / scalingFactor
		newWeight := w - (float32(lr) * (scaledGrad + reg))
		// if i%350 == 0 {
		// 	slog.Info("weights", "delta", -(float32(lr) * (scaledGrad + reg)))
		// }
		l.masterWeights[i] = newWeight
		l.weights.Data()[i] = float16.Fromfloat32(newWeight)
	}

	for i, b := range l.masterBiases {
		newBias := b - (float32(lr) * (l.dCdB[i].Float32() / scalingFactor))
		l.masterBiases[i] = newBias
		l.biases[i] = float16.Fromfloat32(newBias)
	}
}

func (l *f16Layer) data() [2]types.Tensor {
	// panic(fmt.Sprintf("numRows='%d' numCols='%d'", l.weights.NumRows(), l.weights.NumCols()))
	return [2]types.Tensor{
		{
			Data:     &types.F16Data{Data: l.weights.Data()},
			DataType: types.F16,
			Shape:    shape.New(l.weights.NumRows(), l.weights.NumCols()),
		},
		{
			Data:     &types.F16Data{Data: l.biases},
			DataType: types.F16,
			Shape:    shape.New(len(l.biases)),
		},
	}
}

func (l *f16Layer) Shape() Shape {
	return l.shape
}

func (l *f16Layer) Type() LayerType {
	if l.isOutput {
		return Output
	}
	return Hidden
}

func initMasterWeightsAndBiases(weights *mat.F16Matrix, biases []types.Float16) ([]float32, []float32) {
	var (
		masterWeights = make([]float32, len(weights.Data()))
		masterBiases  = make([]float32, len(biases))
	)

	for i, elem := range weights.Data() {
		masterWeights[i] = elem.Float32()
	}

	for i, elem := range biases {
		masterBiases[i] = elem.Float32()
	}

	return masterWeights, masterBiases
}

func initF16Weights(act F16ActivationFunc, prevWidth, width int) (*mat.F16Matrix, *mat.F16Matrix) {
	var (
		weights = mat.NewF16(prevWidth, width)
		dCdW    = mat.NewF16(prevWidth, width)
	)

	for i := range int(prevWidth) {
		for j := range width {
			switch act.(type) {
			case F16Sigmoid:
				// Xavier Normal initialization: W ~ N(0, sqrt(2/n_in + n_out))
				weights.Row(i)[j] = float16.Fromfloat32(float32(rand.NormFloat64() * math.Sqrt(float64(2)/float64(prevWidth+width))))
			case F16ReLU:
				// Kaiming He intialization: W ~ N(0, sqrt(2/n_inputs))

				// Here we modify the n_inputs term to be max(n_inputs, 10), this is to
				// prevent exploding gradients when the number of input features is very
				// small. This was observed with n_inputs = 1 when testing regression.
				weights.Row(i)[j] = float16.Fromfloat32(float32(rand.NormFloat64() * math.Sqrt(float64(2)/float64(max(prevWidth, 10)))))
			default:
				// Kaiming He intialization: W ~ N(0, sqrt(2/n_inputs))
				weights.Row(i)[j] = float16.Fromfloat32(float32(rand.NormFloat64() * math.Sqrt(float64(2)/float64(max(prevWidth, 10)))))
			}
		}
	}

	return weights, dCdW
}

func initF16Biases(width int) ([]types.Float16, []types.Float16) {
	var (
		biases = make([]types.Float16, width)
		dCdB   = make([]types.Float16, width)
	)

	for i := range width {
		biases[i] = float16.Fromfloat32(float32(rand.NormFloat64() * 0.05))
	}

	return biases, dCdB
}

func initF16Gradients(batchSize int, width int) (*mat.F16Matrix, *mat.F16Matrix) {
	return mat.NewF16(batchSize, width), mat.NewF16(batchSize, width)
}

func initF16Activations(batchSize int, width int) (*mat.F16Matrix, *mat.F16Matrix) {
	return mat.NewF16(batchSize, width), mat.NewF16(batchSize, width)
}

func (m *F16MLP) getActivationFunc(lType LayerType) F16ActivationFunc {
	switch lType {
	case Input, Output:
		// Used for output because we override with softmax in case of classification
		return F16NoOp{}
	case Hidden:
		fallthrough
	default:
		return F16ReLU{}
	}
}

func (l *f16Layer) sumSquaredWeights() float64 {
	var sum float64
	for i := range l.weights.NumRows() {
		for _, w := range l.weights.Row(i) {
			sum += float64(w.Float32() * w.Float32())
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

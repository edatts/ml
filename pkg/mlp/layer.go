package mlp

import (
	"fmt"
	"log/slog"
	"math"
	"math/rand"

	"github.com/edatts/ml/pkg/mat"
)

type LayerType int

const (
	Input LayerType = iota
	Hidden
	Output
)

type layer struct {
	prev        *layer
	actFn       ActivationFunc
	lambda      float64 // regularization factor for updating weigts
	initialized bool
	batchSize   int
	width       int16

	weights [][]float64
	dCdW    [][]float64

	biases []float64
	dCdB   []float64

	logits [][]float64
	dCdz   [][]float64

	activations [][]float64
	dCdA        [][]float64
}

func (m *MLP) newLayer(lType LayerType, n int16, prev *layer, lambda float64) *layer {
	return &layer{
		prev:   prev,
		actFn:  m.getActivationFunc(lType),
		lambda: lambda,
		width:  n,
	}
}

func (l *layer) init(batchSize int) {
	if l.prev == nil {
		return
	}

	if !l.initialized {
		// Init
		l.weights, l.dCdW = initWeights(l.actFn, l.prev.width, l.width)
		l.biases, l.dCdB = initBiases(l.width)
		l.dCdz, l.dCdA = initGradients(batchSize, l.width)
		l.initialized = true
	}

	if batchSize != l.batchSize {
		// Update batch size
		l.batchSize = batchSize
		// We don't change the shape of the logit and activation matrices
		// yet because they are re-assigned during the forward pass.
	}

	// Zero grads
	l.dCdz, l.dCdA = initGradients(batchSize, l.width)
	l.dCdB = make([]float64, l.width)
	l.dCdW = make([][]float64, l.prev.width)
	for i := range l.prev.width {
		l.dCdW[i] = make([]float64, l.width)
	}
}

func (l *layer) Forward() {
	// Forward required us to, for each neuron, take the activations of
	// the previous layer, multiply them by the weights, then add the bias.
	// In our activaitons matrix each neuron is indexed by row and each
	// batch sample is indexed by column. In our weights matrix each neuron
	// in the current layer is indexed by row and connection (weight) to a
	// previous layer's neuron is indexed by column.

	// z = actFn(dot(A,W) + b)
	product, err := mat.MulConcurrent(l.prev.activations, l.weights)
	if err != nil {
		panic(fmt.Sprintf("forward: %s", err))
	}

	l.logits, err = mat.AddMatVecRows(product, l.biases)
	if err != nil {
		panic(fmt.Sprintf("forward: %s", err))
	}

	l.activations = mat.ApplyActivation(l.logits, l.actFn.Forward)
}

func (l *layer) Backward() {
	// There are three main derivatves we need here, all three are the derivative of
	// the loss, but with respect to different components, the weights, the bias, and
	// the previous activations. The derivative of the previous activation needs to be
	// a summation of all of the downstream partial derivatives with respect to the
	// downstream activations.

	// dCdz = actFn'(z) * dC/dA
	actPrime := mat.ApplyActivation(l.logits, l.actFn.Backward)
	l.dCdz = mat.Hadamard(actPrime, l.dCdA)

	// dCdB = sum(1 * dC/dz)/batchSize
	l.dCdB = mat.AvgCols(l.dCdz)

	// dCdW = sum(prevA * dC/dz)/batchSize
	var err error
	l.dCdW, err = mat.MulConcurrent(mat.Transpose(l.prev.activations), l.dCdz)
	if err != nil {
		panic(fmt.Sprintf("backward: %s", err))
	}
	l.dCdW = mat.MulScalar(l.dCdW, float64(1)/float64(l.batchSize))

	// dCdA_(L-1) = W * dCdz
	l.prev.dCdA, err = mat.MulConcurrent(l.dCdz, mat.Transpose(l.weights))
	if err != nil {
		panic(fmt.Sprintf("backward: %s", err))
	}
}

func (l *layer) Update(lr float64) {
	// l.biases = mat.AddVec(l.biases, mat.MulVecScalar(l.dCdB, -lr))

	// l.weights = mat.Add(l.weights, mat.MulScalar(l.weights, l.lambda))
	// l.weights = mat.Add(l.weights, mat.MulScalar(l.dCdW, -lr))

	// Regularize and update
	for i := range len(l.weights) {
		for j := range len(l.weights[0]) {
			l.biases[j] += -lr * l.dCdB[j]
			l.weights[i][j] += (l.weights[i][j] * l.lambda) + (-lr * l.dCdW[i][j])
		}
	}
}

// type Layer struct {
// 	prev    *Layer
// 	lType   LayerType
// 	Neurons []*Neuron
// }

// // func newLayer(lType LayerType, n int16, prev *Layer, lambda float64) *Layer {
// // 	return &Layer{
// // 		prev:    prev,
// // 		lType:   lType,
// // 		Neurons: newNeurons(n, prev, lType, lambda),
// // 	}
// // }

// func (l *Layer) Forward(newBatch bool, batchLen int) {
// 	for _, neuron := range l.Neurons {
// 		neuron.Forward(newBatch, batchLen)
// 	}
// }

// func (l *Layer) Backward() {
// 	for _, neuron := range l.Neurons {
// 		neuron.Backward()
// 	}
// }

// func (l *Layer) Update(lr float64) {
// 	for _, neuron := range l.Neurons {
// 		neuron.Update(lr)
// 	}
// }

func (l layer) logWweights() {
	slog.Info("weights", "weights", l.weights[:1])
}

func (l *layer) logBiases() {
	slog.Info("biases", "biases", l.biases[:1])
}

func (l *layer) logGrads() {
	slog.Info("grads", "grads", l.dCdz)
}

// func (l *Layer) logActivations() {
// 	var a []float64
// 	for _, neuron := range l.Neurons {
// 		a = append(a, neuron.activations...)
// 	}
// 	slog.Info("activations", "acts", a)
// }

func (l *layer) logLossDeriv() {
	slog.Info("Loss deriv", "dCdA", l.dCdA)
}

// func (l *Layer) logLogits() {
// 	var logits []float64
// 	for _, neuron := range l.Neurons {
// 		logits = append(logits, neuron.logits...)
// 	}
// 	slog.Info("Logits", "logits", logits)
// }

func initWeights(act ActivationFunc, prevWidth, width int16) ([][]float64, [][]float64) {
	var (
		weights = make([][]float64, prevWidth)
		dCdW    = make([][]float64, prevWidth)
	)

	for i := range prevWidth {
		weights[i] = make([]float64, width)
		dCdW[i] = make([]float64, width)
		for j := range width {
			switch act.(type) {
			case Sigmoid:
				// Xavier Normal initialization: W ~ N(0, sqrt(2/n_in + n_out))
				weights[i][j] = rand.NormFloat64() * math.Sqrt(float64(2)/float64(prevWidth+width))
			case ReLU:
				// Kaiming He intialization: W ~ N(0, sqrt(2/n_inputs))
				weights[i][j] = rand.NormFloat64() * math.Sqrt(float64(2)/float64(prevWidth))
			default:
				weights[i][j] = rand.Float64()*0.1 - 0.05
			}
		}
	}

	return weights, dCdW
}

func initBiases(width int16) ([]float64, []float64) {
	var (
		biases = make([]float64, width)
		dCdB   = make([]float64, width)
	)

	for i := range width {
		biases[i] = rand.Float64()*0.05 - 0.025
	}

	return biases, dCdB
}

func initGradients(batchSize int, width int16) ([][]float64, [][]float64) {
	var (
		dCdz = make([][]float64, batchSize)
		dCdA = make([][]float64, batchSize)
	)

	for i := range batchSize {
		dCdz[i] = make([]float64, width)
		dCdA[i] = make([]float64, width)
	}

	return dCdz, dCdA
}

func (m *MLP) getActivationFunc(lType LayerType) ActivationFunc {
	switch lType {
	case Input:
		return NoOp{}
	case Output:
		if m.classification {
			return SoftMax{}
		}
		return NoOp{}
	case Hidden:
		fallthrough
	default:
		return ReLU{}
	}
}

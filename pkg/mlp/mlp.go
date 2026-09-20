package mlp

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/edatts/ml/pkg/mat"
	"github.com/edatts/ml/pkg/model"
)

var _ model.Model = &MLP{}

type MLP struct {
	Layers         []Layer
	classification bool
	identity       string
}

type Option func(*MLP)

func WithClassifcation() Option {
	return func(o *MLP) {
		o.classification = true
	}
}

func WithHiddenLayer(width int) Option {
	return func(m *MLP) {
		m.Layers = append(m.Layers, m.newLayer(Hidden, width, m.Layers[len(m.Layers)-1]))
	}
}

func WithHiddenLayers(widths ...int) Option {
	return func(m *MLP) {
		for _, width := range widths {
			m.Layers = append(m.Layers, m.newLayer(Hidden, width, m.Layers[len(m.Layers)-1]))
		}
	}
}

// Assume the layers are sequential and densely connected.
func New(inputSize, outputSize int, opts ...Option) (*MLP, error) {
	m := &MLP{Layers: make([]Layer, 1)}
	m.Layers[0] = m.newLayer(Input, inputSize, nil)
	for _, optFn := range opts {
		optFn(m)
	}

	if len(m.Layers) == 1 {
		return nil, fmt.Errorf("no hidden layers specified, hidden layers are required")
	}

	m.Layers = append(m.Layers, m.newLayer(Output, outputSize, m.Layers[len(m.Layers)-1]))

	return m, nil
}

func (m *MLP) inputLayer() Layer {
	return m.Layers[0]
}

func (m *MLP) outputLayer() Layer {
	return m.Layers[len(m.Layers)-1]
}

func (m *MLP) inputLen() int {
	return int(m.inputLayer().width())
}

func (m *MLP) outputLen() int {
	return int(m.outputLayer().width())
}

func (m *MLP) SumSquaredWeights() float64 {
	var sum float64
	for _, layer := range m.Layers[1:] {
		sum += layer.sumSquaredWeights()
	}
	return sum
}

func (m *MLP) Forward(batch [][]float32, _ bool) ([][]float32, error) {
	if len(m.Layers) < 2 {
		return nil, ErrNotEnoughLayers
	}

	if len(batch) == 0 {
		return nil, ErrEmptyInputData
	}

	if len(batch[0]) != m.inputLen() {
		return nil, fmt.Errorf("len(batch)=%d, len(row)=%d: %w", len(batch), len(batch[0]), ErrInvalidInputSize)
	}

	// Set activations of input layer
	m.inputLayer().init(len(batch))
	for i := range m.inputLayer().activations().NumRows() {
		for j := range m.inputLayer().activations().Row(i) {
			m.inputLayer().activations().Row(i)[j] = batch[i][j]
		}
	}

	// Forward
	for i, l := range m.Layers[1:] {
		l.init(len(batch))
		if err := l.Forward(); err != nil {
			return nil, fmt.Errorf("layer %d failed forward pass: %w", i, err)
		}
	}

	// Collect outputs
	return m.outputLayer().activations().AsSlices(), nil

}

func (m *MLP) Backward(dCdA [][]float32, lr, lambda, _ float64) error {
	// Set gradients of output layer
	for i := range len(dCdA) {
		for j := range len(dCdA[0]) {
			m.outputLayer().grads().Row(i)[j] = dCdA[i][j]
		}
	}

	// Exclude input layer from backwards pass
	for i := len(m.Layers) - 1; i >= 1; i-- {
		if err := m.Layers[i].Backward(); err != nil {
			return fmt.Errorf("layer %d failed backwards pass: %w", i+1, err)
		}
	}

	for i := len(m.Layers) - 1; i >= 1; i-- {
		m.Layers[i].update(lr, lambda)
	}

	return nil
}

func (m *MLP) learnableLayers() []*layer {
	var out []*layer
	for _, l := range m.Layers {
		if learnable, ok := l.(*layer); ok {
			out = append(out, learnable)
		}
	}
	return out
}

func (m *MLP) Weights() []model.Tensor {
	var out []model.Tensor
	for i, l := range m.learnableLayers() {
		d := l.data()

		d[0].Name = fmt.Sprintf("hidden.%d.weights", i)
		out = append(out, d[0])

		d[1].Name = fmt.Sprintf("hidden.%d.biases", i)
		out = append(out, d[1])
	}

	return out
}

func (m *MLP) LoadWeights(parameters []model.Tensor) error {
	var numLearnable = len(m.learnableLayers())
	if len(parameters) != numLearnable*2 {
		return fmt.Errorf("unexpected number of weight tensors, expected %d, got %d", numLearnable*2, len(parameters))
	}

	for i, l := range m.learnableLayers() {
		l.init(0) // To ensure we don't overwrite loaded weights later.
		weights := parameters[i*2]
		biases := parameters[i*2+1]

		if l.prev.width() != weights.Shape.Height() {
			return fmt.Errorf("invalid height for weights in learnable layer %d, expected %d, got %d", i, l.prev.width(), weights.Shape.Height())
		}

		if l.width() != weights.Shape.Width() {
			return fmt.Errorf("invalid width for weights in learnable layer %d, expected %d, got %d", i, l.width(), weights.Shape.Width())
		}

		if l.width() != biases.Shape.Width() {
			return fmt.Errorf("invalid width for biases in learnable layer %d, expected %d, got %d", i, len(l.biases), biases.Shape.Width())
		}

		var err error
		if l.weights, err = mat.NewFromData(l.prev.width(), l.width(), weights.Data); err != nil {
			return fmt.Errorf("failed loading weights: %w", err)
		}

		l.biases = biases.Data
	}

	return nil
}

// Identity returns the hex encoded sha256 hash of the concatenation of the
// layer type and the layer shape bytes for all layers.
func (m *MLP) Identity() string {
	if m.identity != "" {
		return m.identity
	}

	var b []byte
	for _, l := range m.Layers {
		b = append(b, []byte(l.Type())...)
		b = append(b, l.Shape().Bytes()...)
	}

	h := sha256.New()
	_, _ = h.Write(b)
	m.identity = hex.EncodeToString(h.Sum(nil))
	return m.identity
}

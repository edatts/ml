package mlp

import (
	"fmt"

	"github.com/edatts/ml/pkg/model"
)

var _ model.Model = &MLP{}

type MLP struct {
	Layers []Layer

	classification bool
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

func (m *MLP) Forward(batch [][]float32) ([][]float32, error) {
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

func (m *MLP) Backward(dCdA [][]float32, lr, lambda float64) error {
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

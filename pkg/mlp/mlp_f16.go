package mlp

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/edatts/ml/pkg/mat"
	"github.com/edatts/ml/pkg/model"
	"github.com/edatts/ml/pkg/types"
	"github.com/x448/float16"
)

var _ model.Model = &F16MLP{}

type F16MLP struct {
	Layers            []F16Layer
	classification    bool
	identity          string
	outputs           [][]float32
	gradScalingFactor float32
}

// Assume the layers are sequential and densely connected.
func NewF16(inputSize, outputSize int, optFns ...Option) (*F16MLP, error) {
	panic("F16 MLP is not fully functional. See comments on mat.F16Matrix.")

	o := &opts{}
	for _, fn := range optFns {
		fn(o)
	}

	if len(o.layerWidths) == 0 {
		return nil, fmt.Errorf("no hidden layers specified, hidden layers are required")
	}

	m := &F16MLP{
		Layers:            make([]F16Layer, len(o.layerWidths)+2),
		classification:    o.classification,
		gradScalingFactor: 1024,
	}
	m.Layers[0] = m.newLayer(Input, inputSize, nil)
	for i, width := range o.layerWidths {
		m.Layers[i+1] = m.newLayer(Hidden, width, m.Layers[i])
	}
	m.Layers[len(m.Layers)-1] = m.newLayer(Output, outputSize, m.Layers[len(m.Layers)-2])
	return m, nil
}

func (m *F16MLP) inputLayer() F16Layer {
	return m.Layers[0]
}

func (m *F16MLP) outputLayer() F16Layer {
	return m.Layers[len(m.Layers)-1]
}

func (m *F16MLP) inputLen() int {
	return m.inputLayer().Shape().Width()
}

func (m *F16MLP) outputLen() int {
	return m.outputLayer().Shape().Width()
}

func (m *F16MLP) SumSquaredWeights() float64 {
	var sum float64
	for _, layer := range m.Layers[1:] {
		sum += layer.sumSquaredWeights()
	}
	return sum
}

func (m *F16MLP) Forward(batch [][]float32, _ bool) ([][]float32, error) {
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
			m.inputLayer().activations().Row(i)[j] = float16.Fromfloat32(batch[i][j])
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
	if len(batch) != len(m.outputs) {
		m.outputs = make([][]float32, len(batch))
		for i, row := range m.outputLayer().activations().Rows() {
			m.outputs[i] = make([]float32, len(row))
			for j, elem := range row {
				m.outputs[i][j] = elem.Float32()
			}
			// fmt.Printf("%v\n", m.outputs[i])
		}
	} else {
		for i, row := range m.outputLayer().activations().Rows() {
			for j, elem := range row {
				m.outputs[i][j] = elem.Float32()
			}
		}
	}

	return m.outputs, nil

}

func (m *F16MLP) Backward(dCdA [][]float32, lr, lambda, _ float64) error {
	// Set gradients of output layer with scaling.
	for i := range len(dCdA) {
		for j := range len(dCdA[0]) {
			m.outputLayer().grads().Row(i)[j] = float16.Fromfloat32(dCdA[i][j] * m.gradScalingFactor)
		}
	}

	// Exclude input layer from backwards pass
	for i := len(m.Layers) - 1; i >= 1; i-- {
		if err := m.Layers[i].Backward(); err != nil {
			return fmt.Errorf("layer %d failed backwards pass: %w", i+1, err)
		}
	}

	for i := len(m.Layers) - 1; i >= 1; i-- {
		m.Layers[i].update(lr, lambda, m.gradScalingFactor)
	}

	return nil
}

func (m *F16MLP) learnableLayers() []*f16Layer {
	var out []*f16Layer
	for _, l := range m.Layers {
		if learnable, ok := l.(*f16Layer); ok {
			out = append(out, learnable)
		}
	}
	return out
}

func (m *F16MLP) Weights() []types.Tensor {
	var out []types.Tensor
	for i, l := range m.learnableLayers() {
		d := l.data()

		d[0].Name = fmt.Sprintf("hidden.%d.weights", i)
		out = append(out, d[0])

		d[1].Name = fmt.Sprintf("hidden.%d.biases", i)
		out = append(out, d[1])
	}

	return out
}

func (m *F16MLP) LoadWeights(parameters []types.Tensor) error {
	var numLearnable = len(m.learnableLayers())
	if len(parameters) != numLearnable*2 {
		return fmt.Errorf("unexpected number of weight tensors, expected %d, got %d", numLearnable*2, len(parameters))
	}

	for i, l := range m.learnableLayers() {
		l.init(0) // To ensure we don't overwrite loaded weights later.
		weights := parameters[i*2]
		biases := parameters[i*2+1]

		if l.prev.Shape().Width() != weights.Shape.Height() {
			return fmt.Errorf("invalid height for weights in learnable layer %d, expected %d, got %d", i, l.prev.Shape().Width(), weights.Shape.Height())
		}

		if l.Shape().Width() != weights.Shape.Width() {
			return fmt.Errorf("invalid width for weights in learnable layer %d, expected %d, got %d", i, l.Shape().Width(), weights.Shape.Width())
		}

		if l.Shape().Width() != biases.Shape.Width() {
			return fmt.Errorf("invalid width for biases in learnable layer %d, expected %d, got %d", i, len(l.biases), biases.Shape.Width())
		}

		var ok bool
		weightData, ok := weights.FP16Data()
		if !ok {
			return fmt.Errorf("unexpected data type, expected 'FP32Data', got '%T'", biases.Data)
		}

		var err error
		if l.weights, err = mat.NewF16FromData(l.prev.Shape().Width(), l.Shape().Width(), weightData); err != nil {
			return fmt.Errorf("failed loading weights: %w", err)
		}

		l.biases, ok = biases.FP16Data()
		if !ok {
			return fmt.Errorf("unexpected data type, expected 'FP32Data', got '%T'", biases.Data)
		}
	}

	return nil
}

// Identity returns the hex encoded sha256 hash of the concatenation of the
// layer type and the layer shape bytes for all layers.
func (m *F16MLP) Identity() string {
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

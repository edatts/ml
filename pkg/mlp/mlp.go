package mlp

import (
	"errors"
	"fmt"
	"slices"
)

type MLP struct {
	lambda float64 // Regularization factor

	Layers []*layer

	classification bool
}

type Option func(*MLP)

func WithClassifcation() Option {
	return func(o *MLP) {
		o.classification = true
	}
}

func WithHiddenLayer() Option {
	return func(m *MLP) {

	}
}

// Assume the layers are sequential and densely connected.
func New(inputSize, outputSize, numHidden int, opts ...Option) (*MLP, error) {
	if numHidden <= 0 {
		return nil, errors.New("hidden layers must be positive or zero")
	}

	m := &MLP{lambda: 2.5e-5, Layers: make([]*layer, numHidden+2)}
	for _, optFn := range opts {
		optFn(m)
	}

	m.Layers[0] = m.newLayer(Input, int16(inputSize), nil, m.lambda)

	for i := range numHidden {
		l := m.newLayer(Hidden, 512, m.Layers[i], m.lambda)
		m.Layers[i+1] = l
	}

	m.Layers[len(m.Layers)-1] = m.newLayer(Output, int16(outputSize), m.Layers[len(m.Layers)-2], m.lambda)

	return m, nil
}

func (m *MLP) inputLayer() *layer {
	return m.Layers[0]
}

func (m *MLP) outputLayer() *layer {
	return m.Layers[len(m.Layers)-1]
}

func (m *MLP) inputLen() int {
	return int(m.inputLayer().width)
}

func (m *MLP) outputLen() int {
	return int(m.outputLayer().width)
}

func (m *MLP) sumSquaredWeights() float64 {
	var sum float64
	for _, layer := range m.Layers[1:] {
		for _, row := range layer.weights {
			for _, w := range row {
				sum += w * w
			}
		}
	}
	return sum
}

func (m *MLP) Forward(batch [][]float64) ([][]float64, error) {
	if len(m.Layers) < 2 {
		return nil, ErrNotEnoughLayers
	}

	if len(batch) == 0 {
		return nil, ErrEmptyInputData
	}

	if len(batch[0]) != m.inputLen() {
		return nil, fmt.Errorf("len(batch)=%d, len(row)=%d: %w", len(batch), len(batch[0]), ErrInvalidInputSize)
	}

	// Set activaitons of input layer
	m.inputLayer().activations = batch

	// Forward
	for _, l := range m.Layers[1:] {
		l.init(len(batch))
		l.Forward()
	}

	// Collect outputs
	return m.outputLayer().activations, nil

	// // TODO: Add layer init func and zero the gradients.
	// //		 We could also check the previous batch size and
	// //		 update it if it has changed mid-trainiing.
	// for _, n := range m.inputNeurons() {
	// 	n.logits = make([]float64, len(batch))
	// 	n.activations = make([]float64, len(batch))
	// 	n.dCdz = make([]float64, len(batch))
	// 	n.dCdA = make([]float64, len(batch))
	// }

	// // TODO: Move the softmax activation function inside the
	// //		 output layer and handle the option that enables
	// //		 classification.
	// //
	// // Note: This is more complicated for classificatioon as
	// //		 we have to pass the whole batch into the actFn.
	// //		 We need to modify layers to apply the actFn to
	// //		 all of the outputs for a sample at once...
	// var outputs = make([][]float64, len(batch))
	// for i, data := range batch {
	// 	for j, neuron := range m.inputNeurons() {
	// 		neuron.Output = data[j]
	// 		neuron.activations[i] = data[j]
	// 	}

	// 	for _, layer := range m.Layers[1:] {
	// 		layer.Forward(i == 0, len(batch))
	// 	}

	// 	var output = make([]float64, m.outputLen())
	// 	for j, neuron := range m.outputNeurons() {
	// 		output[j] = neuron.Output
	// 	}

	// 	if m.classification {
	// 		outputs[i] = SoftMax{}.Forward(output)
	// 	} else {
	// 		outputs[i] = output
	// 	}

	// 	for j, datum := range outputs[i] {
	// 		m.outputNeurons()[j].activations[i] = datum
	// 	}
	// }

	// return outputs, nil
}

func (m *MLP) Regress(batch [][]float64, y [][]float64) ([][]float64, float64, float64, error) {
	outputs, err := m.Forward(batch)
	if err != nil {
		return nil, 0, 0, err
	}

	loss, err := MeanSquaredError(outputs, y)
	if err != nil {
		return nil, 0, 0, err
	}

	// Derivative of our loss with respect to the predictions is 2(ŷ - y)
	// Set the activations of output layer to the deriv of loss for backwards pass
	for i, pred := range outputs {
		for j, actual := range y[i] {
			m.outputLayer().dCdA[i][j] = 2 * (pred[j] - actual)
			// neuron := m.outputNeurons()[j]
			// neuron.dCdA[i] = 2 * (pred[j] - actual)
		}
	}

	// Regularize the loss with L2 regularization
	// 0.5 * lambda * sum(W^2)
	regLoss := 0.5 * m.lambda * m.sumSquaredWeights()

	return outputs, loss, loss + regLoss, nil
}

func (m *MLP) Classify(batch [][]float64, y [][]int) ([][]float64, float64, float64, error) {
	outputs, err := m.Forward(batch)
	if err != nil {
		return nil, 0, 0, err
	}

	loss, err := CategoricalCrossEntropy(outputs, y)
	if err != nil {
		return nil, 0, 0, err
	}

	// Derivative of our loss with respect to the predicitons is ŷ - y, this
	// includes the derivative of SoftMax in the derivation so that is not
	// accounted for in the backwards pass of the output layer.
	for i, pred := range outputs {
		for j, actual := range y[i] {
			m.outputLayer().dCdA[i][j] = pred[j] - float64(actual)
			// neuron := m.outputNeurons()[j]
			// neuron.dCdA[i] = neuron.activations[i] - float64(actual)
		}
	}

	// Regularize the loss with L2 regularization
	// 0.5 * lambda * sum(W^2)
	regLoss := 0.5 * m.lambda * m.sumSquaredWeights()

	return outputs, Accuracy(outputs, y), loss + regLoss, nil
}

func (m *MLP) Backward(lr float64) error {
	// For now only batch size of 1 supported
	// if len(outputs) != 1 || len(y) != 1 {
	// 	return errors.New("only batch size of 1 is currently supported for training")
	// }

	// Exclude input layer from backwards pass
	for i := len(m.Layers) - 1; i >= 1; i-- {
		m.Layers[i].Backward()
	}

	for i := len(m.Layers) - 1; i >= 1; i-- {
		m.Layers[i].Update(lr)
	}

	return nil
}

func Accuracy(predictions [][]float64, y [][]int) float64 {
	var numCorrect int
	for i, pred := range predictions {
		idx := slices.Index(pred, slices.Max(pred))
		if y[i][idx] == 1 {
			numCorrect++
		}
	}
	return float64(numCorrect) / float64(len(predictions)) * 100
}

func (m *MLP) LogWeights() {
	for _, l := range m.Layers[1:] {
		l.logWweights()
	}
}

func (m *MLP) LogBiases() {
	for _, l := range m.Layers[1:] {
		l.logBiases()
	}
}

func (m *MLP) LogGrads() {
	for _, l := range m.Layers[1:] {
		l.logGrads()
	}
}

// func (m *MLP) LogActivations() {
// 	for _, l := range m.Layers[3:] {
// 		l.logActivations()
// 	}
// }

func (m *MLP) LogLossDeriv() {
	for _, l := range m.Layers[1:] {
		l.logLossDeriv()
	}
}

// func (m *MLP) LogLogits() {
// 	for _, l := range m.Layers[3:] {
// 		l.logLogits()
// 	}
// }

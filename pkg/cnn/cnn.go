package cnn

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"strings"

	"github.com/edatts/ml/pkg/model"
	"github.com/edatts/ml/pkg/shape"
)

var _ model.Model = &CNN{}

type CNN struct {
	layers       []Layer
	layersByName map[string]Layer
	layerCounts  map[layerType]int
	identity     string
}

type Option func(*CNN)

func WithConv2D(kernelSize, numKernels int) Option {
	return func(c *CNN) {
		prev := c.layers[len(c.layers)-1]
		c.layers = append(c.layers, c.newConv2D(prev, kernelSize, numKernels))
		fmt.Printf("After Conv: %+v\n", c.layers[len(c.layers)-1].Shape())
	}
}

func WithPool2D() Option {
	return func(c *CNN) {
		prev := c.layers[len(c.layers)-1]
		c.layers = append(c.layers, c.newPool2D(prev))
		fmt.Printf("After Pool: %+v\n", c.layers[len(c.layers)-1].Shape())
	}
}

func WithDropout(dropoutRatio float64) Option {
	return func(c *CNN) {
		prev := c.layers[len(c.layers)-1]
		c.layers = append(c.layers, c.newDropout(prev, dropoutRatio))
		fmt.Printf("After Dropout: %+v\n", c.layers[len(c.layers)-1].Shape())
	}
}

func WithFullyConnected(width int) Option {
	return func(c *CNN) {
		prev := c.layers[len(c.layers)-1]
		c.layers = append(c.layers, c.newFullyConnected(prev, width))
		fmt.Printf("After Fully Connected: %+v\n", c.layers[len(c.layers)-1].Shape())
	}
}

func New(inputShape shape.Shape, outputSize int, optFns ...Option) (*CNN, error) {
	if inputShape.Channels() != 1 {
		return nil, fmt.Errorf("only 1 input channel is currently supported")
	}

	c := &CNN{
		layers:       make([]Layer, 0),
		layersByName: make(map[string]Layer),
		layerCounts:  make(map[layerType]int),
	}

	var x Layer
	x = c.newInput(inputShape)
	fmt.Printf("Input Shape: %+v\n", x.Shape())
	c.layers = append(c.layers, x)

	if len(optFns) == 0 {
		return nil, fmt.Errorf("no layers provided")
	}

	for _, fn := range optFns {
		fn(c)
	}

	out := c.newOutput(outputSize, c.layers[len(c.layers)-1])
	fmt.Printf("After Out: %+v\n", out.Shape())
	c.layers = append(c.layers, out)

	for _, l := range c.layers {
		c.layersByName[l.Name()] = l
	}

	return c, nil
}

func (c *CNN) inputLayer() Layer {
	if len(c.layers) == 0 {
		panic("no layers in model")
	}

	return c.layers[0]
}

func (c *CNN) outputLayer() Layer {
	if len(c.layers) == 0 {
		panic("no layers in model")
	}

	return c.layers[len(c.layers)-1]
}

func (c *CNN) Forward(inputs [][]float32, isTest bool) ([][]float32, error) {
	if len(c.layers) <= 2 {
		return nil, fmt.Errorf("not enough layers")
	}

	if len(inputs) == 0 {
		return nil, fmt.Errorf("empty input data")
	}

	batch, err := NewBatchFromSlices(c.inputLayer().Shape(), inputs)
	if err != nil {
		return nil, fmt.Errorf("failed instantiating batch: %w", err)
	}

	if !bytes.Equal(batch[0].Shape.Bytes(), c.inputLayer().Shape().Bytes()) {
		return nil, fmt.Errorf("invalid sample shape, sample=%v input=%v", batch[0].Shape, c.layers[0].Shape())
	}

	// Set activations of input layer
	c.inputLayer().init(len(batch))
	for i := range c.inputLayer().batch() {
		c.inputLayer().batch()[i] = batch[i]
	}

	for i, l := range c.layers[1:] {
		l.init(len(batch))
		if err := l.Forward(isTest); err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
	}

	// Extract activations from output layer
	return AsSlices(c.outputLayer().batch()), nil
}

func (c *CNN) Backward(grads [][]float32, lr, lambda, beta float64) error {
	// To run the backwards pass we have to first calculate the loss
	// and then compute the derivatives of weights and biases throughout
	// the network with respect to the loss.

	// Set grads in output layer
	for i, sample := range grads {
		c.outputLayer().activationDeltas()[i] = Sample{data: sample, Shape: c.outputLayer().Shape()}
	}

	// Run backwards pass
	for i := len(c.layers) - 1; i > 0; i-- {
		if err := c.layers[i].Backward(); err != nil {
			return fmt.Errorf("layer %d failed running backwards pass: %w", i, err)
		}
	}

	// Update layers
	for _, l := range c.layers {
		l.update(lr, lambda, beta)
	}

	return nil
}

// func (c *CNN) Classify(batch []Sample, y [][]int) ([][]float32, float32, float32, error) {
// 	output, err := c.Forward(batch)
// 	if err != nil {
// 		return nil, 0, 0, err
// 	}

// 	preds := ToData(output)

// 	loss, err := c.CategoricalCrossEntropy(preds, y)
// 	if err != nil {
// 		return nil, 0, 0, err
// 	}

// 	// Regularize the loss with L2 regularization
// 	// 0.5 * lambda * sum(W^2)
// 	regLoss := 0.5 * c.lambda * c.sumSquaredWeights()

// 	return preds, c.Accuracy(preds, y), loss + regLoss, nil
// }

// func (c *CNN) Accuracy(preds [][]float32, y [][]int) float32 {
// 	var numCorrect float32
// 	for i, pred := range preds {
// 		predIdx := slices.Index(pred, slices.Max(pred))
// 		if y[i][predIdx] == 1 {
// 			numCorrect++
// 		}
// 	}
// 	return (numCorrect / float32(len(preds))) * float32(100)
// }

// // Formule for CCE: - sum(y * ln(ŷ))
// func (c *CNN) CategoricalCrossEntropy(preds [][]float32, y [][]int) (float32, error) {
// 	if len(preds) != len(y) {
// 		return 0, fmt.Errorf("mismatches batch size between preds and targets, len(preds)=%d, len(targets)=%d", len(preds), len(y))
// 	}

// 	if len(preds) == 0 {
// 		return 0, fmt.Errorf("predictions length is zero")
// 	}

// 	// The categorical cross entrppy is calculated as the negative of the sum over
// 	// all output classes of the target multiplied by natural log of the prediciton
// 	// for each class. This simplies to -ln(ŷ), the negative of the natural log of
// 	// the prediction for the correct class. This is due to the targets being one-hot
// 	// encoded so we can ignore all the other classes.
// 	var correctIndices = make([]int, len(preds))
// 	for i, sample := range y {
// 		correctIndices[i] = slices.Index(sample, 1)
// 	}

// 	// Simplifies to -ln(ŷ). We take the average over the batch.
// 	var crossEntSum float32
// 	for i, pred := range preds {
// 		// We prevent zeros because ln(0) is undefined.
// 		if pred[correctIndices[i]] == 0 {
// 			pred[correctIndices[i]] = 1e-10
// 		}
// 		crossEntSum += -float32(math.Log(float64(pred[correctIndices[i]])))
// 	}

// 	return crossEntSum / float32(len(preds)), nil
// }

func (c *CNN) SumSquaredWeights() float64 {
	// Take the sum of the square of all weights in the network
	var sum float64
	for _, l := range c.layers {
		sum += float64(l.sumSquaredWeights())
	}
	return sum
}

func (c *CNN) Weights() []model.Tensor {
	var out = []model.Tensor{}

	for _, l := range c.layers {
		for _, tensor := range l.parameters() {
			out = append(out, tensor)
		}
	}

	return out
}

func (c *CNN) LoadWeights(parameters []model.Tensor) error {
	for _, tensor := range parameters {
		parts := strings.Split(tensor.Name, ".")
		layerName := strings.Join(parts[:len(parts)-1], ".")
		l, ok := c.getLayer(layerName)
		if !ok {
			return fmt.Errorf("no layer found with name '%s'", layerName)
		}

		slog.Info("loading parameters", "layer", layerName, "tensor", tensor.Name)
		if err := l.loadParameters(tensor); err != nil {
			return fmt.Errorf("failed loading parameters '%s': %w", tensor.Name, err)
		}
	}

	return nil
}

func (c *CNN) getLayer(name string) (Layer, bool) {
	l, ok := c.layersByName[name]
	return l, ok
}

func (c *CNN) assignName(lType layerType) string {
	idx := c.layerCounts[lType]
	c.layerCounts[lType] += 1
	return fmt.Sprintf("%s_%d", lType, idx)
}

func (c *CNN) Identity() string {
	if c.identity != "" {
		return c.identity
	}

	var b []byte
	for _, l := range c.layers {
		switch l.(type) {
		case *dropout, *flatten:
			continue
		}
		b = append(b, []byte(l.Name())...)
		b = append(b, l.Shape().Bytes()...)
	}

	h := sha256.New()
	_, _ = h.Write(b)
	c.identity = hex.EncodeToString(h.Sum(nil))
	return c.identity
}

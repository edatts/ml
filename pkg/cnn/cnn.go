package cnn

import (
	"fmt"
	"math"
	"slices"
)

// Here, in this package, we shall create functions for the instantiation and training
// of convolutional neural networks... Innit blud.

type CNN struct {
	lambda float32

	layers []Layer
}

// The shape takes into account height, width, and channels.
// We ignore batch size for now but might include it later.
type Shape [3]int

func (s Shape) Height() int {
	return s[0]
}

func (s Shape) Width() int {
	return s[1]
}

func (s Shape) Channels() int {
	return s[2]
}

func New(inputShape, outputShape Shape) (*CNN, error) {
	if inputShape.Channels() != 1 {
		return nil, fmt.Errorf("only 1 input channel is currently supported")
	}

	c := &CNN{
		lambda: 1e-4,
		layers: nil,
	}

	var x Layer
	x = newInput(inputShape)
	fmt.Printf("Input Shape: %+v\n", x.Shape())
	x = newConv2D(x, 32, c.lambda)
	fmt.Printf("After Conv: %+v\n", x.Shape())
	x = newPool2D(x)
	fmt.Printf("After Pool: %+v\n", x.Shape())
	x = newConv2D(x, 64, c.lambda)
	fmt.Printf("After Conv: %+v\n", x.Shape())
	x = newConv2D(x, 128, c.lambda)
	fmt.Printf("After Conv: %+v\n", x.Shape())
	x = newPool2D(x)
	fmt.Printf("After Pool: %+v\n", x.Shape())
	x = newConv2D(x, 128, c.lambda)
	fmt.Printf("After Conv: %+v\n", x.Shape())
	x = newConv2D(x, 256, c.lambda)
	fmt.Printf("After Conv: %+v\n", x.Shape())
	x = newFullyConnected(256, x, c.lambda)
	fmt.Printf("After Full: %+v\n", x.Shape())
	x = newFullyConnected(256, x, c.lambda)
	fmt.Printf("After Full: %+v\n", x.Shape())
	out := newOutput(outputShape, x, c.lambda)
	fmt.Printf("After Out: %+v\n", out.Shape())

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

func (c *CNN) Forward(batch []Sample) ([]Sample, error) {
	if len(c.layers) <= 2 {
		return nil, fmt.Errorf("not enough layers")
	}

	if len(batch) == 0 {
		return nil, fmt.Errorf("empty input data")
	}

	if batch[0].Shape != c.inputLayer().Shape() {
		return nil, fmt.Errorf("invalid sample shape, sample=%v input=%v", batch[0].Shape, c.layers[0].Shape())
	}

	// Set activations of input layer
	c.inputLayer().init(len(batch))
	for i := range c.inputLayer().batch() {
		c.inputLayer().batch()[i] = batch[i]
	}

	for i, l := range c.layers[1:] {
		l.init(len(batch))
		if err := l.Forward(); err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
	}

	// Extract activations from output layer
	return c.outputLayer().batch(), nil
}

func (c *CNN) Backward(lr float32) error {
	// To run the backwards pass we have to first calculate the loss
	// and then compute the derivatives of weights and biases throughout
	// the network with respect to the loss.

	// Run backwards pass
	for i := len(c.layers) - 1; i > 0; i-- {
		if err := c.layers[i].Backward(); err != nil {
			return fmt.Errorf("layer %d failed running backwards pass: %w", i, err)
		}
	}

	// Update layers
	for _, l := range c.layers {
		l.Update(lr)
	}

	return nil
}

func (c *CNN) Classify(batch []Sample, y [][]int) ([][]float32, float32, float32, error) {
	output, err := c.Forward(batch)
	if err != nil {
		return nil, 0, 0, err
	}

	preds := ToData(output)

	loss, err := c.CategoricalCrossEntropy(preds, y)
	if err != nil {
		return nil, 0, 0, err
	}

	// Regularize the loss with L2 regularization
	// 0.5 * lambda * sum(W^2)
	regLoss := 0.5 * c.lambda * c.sumSquaredWeights()

	return preds, c.Accuracy(preds, y), loss + regLoss, nil
}

func (c *CNN) Accuracy(preds [][]float32, y [][]int) float32 {
	var numCorrect float32
	for i, pred := range preds {
		predIdx := slices.Index(pred, slices.Max(pred))
		if y[i][predIdx] == 1 {
			numCorrect++
		}
	}
	return (numCorrect / float32(len(preds))) * float32(100)
}

// Formule for CCE: - sum(y * ln(ŷ))
func (c *CNN) CategoricalCrossEntropy(preds [][]float32, y [][]int) (float32, error) {
	if len(preds) != len(y) {
		return 0, fmt.Errorf("mismatches batch size between preds and targets, len(preds)=%d, len(targets)=%d", len(preds), len(y))
	}

	if len(preds) == 0 {
		return 0, fmt.Errorf("predictions length is zero")
	}

	// The categorical cross entrppy is calculated as the negative of the sum over
	// all output classes of the target multiplied by natural log of the prediciton
	// for each class. This simplies to -ln(ŷ), the negative of the natural log of
	// the prediction for the correct class. This is due to the targets being one-hot
	// encoded so we can ignore all the other classes.
	var correctIndices = make([]int, len(preds))
	for i, sample := range y {
		correctIndices[i] = slices.Index(sample, 1)
	}

	// Simplifies to -ln(ŷ). We take the average over the batch.
	var crossEntSum float32
	for i, pred := range preds {
		// We prevent zeros because ln(0) is undefined.
		if pred[correctIndices[i]] == 0 {
			pred[correctIndices[i]] = 1e-10
		}
		crossEntSum += -float32(math.Log(float64(pred[correctIndices[i]])))
	}

	return crossEntSum / float32(len(preds)), nil
}

func (c *CNN) sumSquaredWeights() float32 {
	// Take the sum of the square of all weights in the network
	var sum float32
	for _, l := range c.layers {
		sum += l.sumSquaredWeights()
	}
	return sum
}

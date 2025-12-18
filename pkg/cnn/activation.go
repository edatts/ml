package cnn

import (
	"math"

	"slices"
)

type ActivationFunc interface {
	Forward(float32) float32
	Backward(float32) float32
}

type NoOp struct{}

func (n NoOp) Forward(in float32) float32 {
	return in
}

func (n NoOp) Backward(in float32) float32 {
	return 1
}

// y = 1/(1+e^-x)
type Sigmoid struct{}

func (s Sigmoid) Forward(val float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-val))))
}

func (s Sigmoid) Backward(val float32) float32 {
	return val * (1 - val)
}

type ReLU struct{}

func (r ReLU) Forward(val float32) float32 {
	if val > 0 {
		return val
	}
	return 0
}

func (r ReLU) Backward(val float32) float32 {
	if val > 0 {
		return 1
	}
	return 0
}

func LeakyReLU(in float32) float32 {
	if in > 0 {
		return in
	}
	return in * 0.01
}

type SoftMax struct{}

func (s SoftMax) Forward(in []float32) []float32 {
	if len(in) == 0 {
		return in
	}

	var (
		out = make([]float32, len(in))
		max = slices.Max(in)
		sum float32
	)

	for i, val := range in {
		expVal := float32(math.Exp(float64(val - max)))
		out[i] = expVal
		sum += expVal
	}

	// Nomalize
	for i, val := range out {
		out[i] = val / sum
	}

	return out
}

// This simply returns slice of 1s because we have already incorporated
// the derivative of softmax when we set dCdA in the Classify receiver.
func (s SoftMax) Backward(in []float32) []float32 {
	return slices.Repeat([]float32{1}, len(in))
}

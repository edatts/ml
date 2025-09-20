package mlp

import (
	"math"

	"slices"
)

type ActivationFunc interface {
	Forward([]float64) []float64
	Backward([]float64) []float64
}

type NoOp struct{}

func (n NoOp) Forward(in []float64) []float64 {
	return in
}

func (n NoOp) Backward(in []float64) []float64 {
	return slices.Repeat([]float64{1}, len(in))
}

// y = 1/(1+e^-x)
type Sigmoid struct{}

func (s Sigmoid) Forward(in []float64) []float64 {
	var out = make([]float64, len(in))
	for i, val := range in {
		out[i] = 1 / (1 + math.Exp(-val))
	}

	return out
}

func (s Sigmoid) Backward(in []float64) []float64 {
	var out = make([]float64, len(in))
	for i, val := range in {
		out[i] = val * (1 - val)
	}

	return out
}

type ReLU struct{}

func (r ReLU) Forward(in []float64) []float64 {
	var out = make([]float64, len(in))
	for i, val := range in {
		if val > 0 {
			out[i] = val
		}
	}
	return out
}

func (r ReLU) Backward(in []float64) []float64 {
	var out = make([]float64, len(in))
	for i, val := range in {
		if val > 0 {
			out[i] = 1
		}
	}
	return out
}

func LeakyReLU(in float64) float64 {
	if in > 0 {
		return in
	}
	return in * 0.01
}

type SoftMax struct{}

func (s SoftMax) Forward(in []float64) []float64 {
	if len(in) == 0 {
		return in
	}

	var (
		out = make([]float64, len(in))
		max = slices.Max(in)
		sum float64
	)

	for i, val := range in {
		expVal := math.Exp(val - max)
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
func (s SoftMax) Backward(in []float64) []float64 {
	return slices.Repeat([]float64{1}, len(in))
}

package mlp

import (
	"math"

	"golang.org/x/exp/slices"
)

type ActivationFunc interface {
	Forward(float64) float64
	Backward(float64) float64
}

type NoOp struct{}

func (n NoOp) Forward(in float64) float64 {
	return in
}

func (n NoOp) Backward(_ float64) float64 {
	return 1
}

// y = 1/(1+e^-x)
type Sigmoid struct{}

func (s Sigmoid) Forward(in float64) float64 {
	return 1 / (1 + math.Exp(-in))
}

func (s Sigmoid) Backward(in float64) float64 {
	return in * (1 - in)
}

type ReLU struct{}

func (r ReLU) Forward(in float64) float64 {
	if in > 0 {
		return in
	}
	return 0
}

func (r ReLU) Backward(in float64) float64 {
	if in > 0 {
		return 1
	}
	return 0
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

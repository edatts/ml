package types

import (
	"github.com/edatts/ml/pkg/shape"
	"github.com/x448/float16"
)

type Float16 = float16.Float16

type Model interface {
	Forward(batch [][]Float16, isTest bool) ([][]Float16, error)
	Backward(dCdA [][]Float16, lr, lambda, beta float64) error
	SumSquaredWeights() float64
	Weights() []Tensor
	LoadWeights([]Tensor) error
	Identity() string
}

type Tensor struct {
	Name  string
	Shape shape.Shape
	Data  []Float16
}

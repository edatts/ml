package model

import (
	"github.com/edatts/ml/pkg/shape"
	"github.com/edatts/ml/pkg/types"
)

type Model interface {
	Forward(batch [][]float32, isTest bool) ([][]float32, error)
	Backward(dCdA [][]float32, lr, lambda, beta float64) error
	SumSquaredWeights() float64
	Weights() []types.Tensor
	LoadWeights([]types.Tensor) error
	Identity() string
}

package model

type Model interface {
	Forward([][]float32) ([][]float32, error)
	Backward(dCdA [][]float32, lr float64) error
	SumSquaredWeights() float64
	Weights() [][]float32
}

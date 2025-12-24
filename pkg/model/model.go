package model

type Model interface {
	Forward([][]float32) [][]float32
	Backward(lr float64) error
	Weights() [][]float32
}

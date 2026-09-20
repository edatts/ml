package test

import (
	"crypto/sha256"
	"encoding/hex"

	"github.com/edatts/ml/pkg/model"
)

type option func(*mockModel)

func WithIdentity(ident string) option {
	return func(m *mockModel) {
		m.identity = ident
	}
}

func WithWeights(weights []model.Tensor) option {
	return func(m *mockModel) {
		m.weights = weights
	}
}

var _ model.Model = &mockModel{}

func NewMockModel(optFns ...option) *mockModel {
	m := &mockModel{}
	for _, fn := range optFns {
		fn(m)
	}
	return m
}

// Mock model will multiply all inputs by 2
type mockModel struct {
	identity string
	weights  []model.Tensor
}

func (m *mockModel) Forward(inputs [][]float32, _ bool) ([][]float32, error) {
	var out = make([][]float32, len(inputs))
	for i, in := range inputs {
		out[i] = make([]float32, len(in))
		for j, val := range in {
			out[i][j] = val * 2
		}
	}
	return out, nil
}

func (m *mockModel) Backward(_ [][]float32, _, _, _ float64) error {
	return nil
}

func (m *mockModel) Weights() []model.Tensor {
	return m.weights
}

func (m *mockModel) SumSquaredWeights() float64 {
	return 0
}

func (m *mockModel) LoadWeights(parameters []model.Tensor) error {
	m.weights = parameters
	return nil
}

func (m *mockModel) Identity() string {
	if m.identity != "" {
		return m.identity
	}

	sum := sha256.Sum256([]byte(`I am a mock of a model`))
	return hex.EncodeToString(sum[:])
}

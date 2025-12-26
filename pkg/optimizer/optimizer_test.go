package optimizer_test

import (
	"testing"

	"github.com/edatts/ml/pkg/model"
	"github.com/edatts/ml/pkg/optimizer"
	"github.com/stretchr/testify/require"
)

var _ model.Model = &mockModel{}

// Mock modal will multiply all inputs by 2
type mockModel struct{}

func (m *mockModel) Forward(inputs [][]float32) ([][]float32, error) {
	var out = make([][]float32, len(inputs))
	for i, in := range inputs {
		out[i] = make([]float32, len(in))
		for j, val := range in {
			out[i][j] = val * 2
		}
	}
	return out, nil
}

func (m *mockModel) Backward(ouputs [][]float32, lr float64) error {
	return nil
}

func (m *mockModel) Weights() [][]float32 {
	return nil
}

func (m *mockModel) SumSquaredWeights() float64 {
	return 0
}

func TestOptimizer(t *testing.T) {

	t.Run("Instantiate and Run", func(t *testing.T) {

	})

	trainingDataProvider := func() ([][]float32, any, error) {
		return [][]float32{{1}, {2}, {3}}, [][]float32{{2}, {4}, {6}}, nil
	}

	testDataProvider := func() ([][]float32, any, error) {
		return [][]float32{{4}, {5}, {6}}, [][]float32{{8}, {10}, {12}}, nil
	}

	o := optimizer.New(
		optimizer.WithNumEpochs(3),
		optimizer.WithBatchSize(1),
		optimizer.WithLoggingInterval(1),
		optimizer.WithTrainDataProvider(trainingDataProvider),
		optimizer.WithTestDataProvider(testDataProvider),
		optimizer.WithSampler(&optimizer.DefaultSampler[float32]{}),
	)

	require.NoError(t, o.Run(&mockModel{}))
}

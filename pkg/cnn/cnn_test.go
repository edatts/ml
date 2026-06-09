package cnn_test

import (
	"testing"

	"github.com/edatts/ml/pkg/cnn"
	"github.com/edatts/ml/pkg/mnist"
	"github.com/edatts/ml/pkg/optimizer"
	"github.com/stretchr/testify/require"
)

func TestCNN(t *testing.T) {
	network, err := cnn.New(cnn.Shape{28, 28, 1}, 10)
	require.NoError(t, err)

	t.Run("mnist handwritten digits", func(t *testing.T) {

		X_train, Y_train, X_test, Y_test, err := mnist.LoadData()
		require.NoError(t, err)

		trainDataProvider := func() ([][]float32, any, error) {
			return X_train, Y_train, nil
		}

		testDataProvider := func() ([][]float32, any, error) {
			return X_test, Y_test, nil
		}

		o := optimizer.New(
			optimizer.WithNumEpochs(20),
			optimizer.WithBatchSize(32),
			optimizer.WithClassification(),
			optimizer.WithTrainDataProvider(trainDataProvider),
			optimizer.WithTestDataProvider(testDataProvider),
			optimizer.WithSampler(optimizer.NewRS2Sampler[int](0.20)),
		)

		require.NoError(t, o.Run(network))
	})

}

// func TestPadding(t *testing.T) {
// 	var data = []float32{
// 		1, 1, 1,
// 		1, 1, 1,
// 		1, 1, 1,
// 	}

// 	var expected = []float32{
// 		1, 1, 1, 0,
// 		1, 1, 1, 0,
// 		1, 1, 1, 0,
// 		0, 0, 0, 0,
// 	}

// }

func TestIm2Col(t *testing.T) {
	var data = []float32{
		1, 1, 1, 1,
		1, 2, 2, 1,
		1, 2, 2, 1,
		1, 1, 1, 1,

		1, 2, 3, 4,
		2, 1, 2, 3,
		3, 2, 1, 2,
		4, 3, 2, 1,

		3, 3, 3, 3,
		1, 1, 1, 1,
		1, 1, 1, 1,
		3, 3, 3, 3,
	}

	var expected = []float32{
		1, 1, 1, 2,
		1, 1, 2, 2,
		1, 1, 2, 1,
		1, 2, 1, 2,
		2, 2, 2, 2,
		2, 1, 2, 1,
		1, 2, 1, 1,
		2, 2, 1, 1,
		2, 1, 1, 1,

		1, 2, 2, 1,
		2, 3, 1, 2,
		3, 4, 2, 3,
		2, 1, 3, 2,
		1, 2, 2, 1,
		2, 3, 1, 2,
		3, 2, 4, 3,
		2, 1, 3, 2,
		1, 2, 2, 1,

		3, 3, 1, 1,
		3, 3, 1, 1,
		3, 3, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 3, 3,
		1, 1, 3, 3,
		1, 1, 3, 3,
	}

	s, err := cnn.NewSampleFromData(cnn.Shape{4, 4, 3}, data)
	require.NoError(t, err)

	sampleMatrix, err := s.Im2Col(1, 3, 3)
	require.NoError(t, err)

	// for i := range sampleMatrix.NumRows() {
	// 	slog.Info("row", "row", sampleMatrix.Row(i))
	// }

	for i, elem := range sampleMatrix.Data() {
		require.Equal(t, expected[i], elem)
	}
}

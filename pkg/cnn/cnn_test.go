package cnn_test

import (
	"testing"

	"github.com/edatts/ml/pkg/cnn"
	"github.com/edatts/ml/pkg/mnist"
	"github.com/edatts/ml/pkg/optimizer"
	"github.com/stretchr/testify/require"
)

func TestCNN(t *testing.T) {
	network, err := cnn.New(cnn.Shape{1, 28, 28}, 10)
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

func TestIm2Col(t *testing.T) {

	t.Run("valid padding 3D kernel", func(t *testing.T) {
		s, err := cnn.NewSampleFromData(cnn.Shape{3, 4, 4}, data3x4x4)
		require.NoError(t, err)

		sampleMatrix, err := s.Im2Col(1, 3, 3, cnn.Valid, false)
		require.NoError(t, err)

		require.Equal(t, 27, sampleMatrix.NumRows())
		require.Equal(t, 4, sampleMatrix.NumCols())

		for i, elem := range sampleMatrix.Data() {
			require.Equal(t, expectedValidPadding3DKernel[i], elem, "unexpected element at index '%d'", i)
		}
	})

	t.Run("valid padding 2D kernel", func(t *testing.T) {
		s, err := cnn.NewSampleFromData(cnn.Shape{3, 4, 4}, data3x4x4)
		require.NoError(t, err)

		sampleMatrix, err := s.Im2Col(1, 3, 3, cnn.Valid, true)
		require.NoError(t, err)

		require.Equal(t, 9, sampleMatrix.NumRows())
		require.Equal(t, 12, sampleMatrix.NumCols())

		for i, elem := range sampleMatrix.Data() {
			require.Equal(t, expectedValidPadding2DKernel[i], elem, "unexpected element at index '%d'", i)
		}
	})

	t.Run("same padding", func(t *testing.T) {
		s, err := cnn.NewSampleFromData(cnn.Shape{3, 4, 4}, data3x4x4)
		require.NoError(t, err)

		sampleMatrix, err := s.Im2Col(1, 3, 3, cnn.Same, false)
		require.NoError(t, err)

		require.Equal(t, 27, sampleMatrix.NumRows())
		require.Equal(t, 16, sampleMatrix.NumCols())

		for i, elem := range sampleMatrix.Data() {
			require.Equal(t, expectedSamePadding[i], elem, "unexpected element at index '%d'", i)
		}
	})

	t.Run("full padding", func(t *testing.T) {
		s, err := cnn.NewSampleFromData(cnn.Shape{3, 2, 2}, data3x2x2)
		require.NoError(t, err)

		sampleMatrix, err := s.Im2Col(1, 3, 3, cnn.Full, false)
		require.NoError(t, err)

		require.Equal(t, 27, sampleMatrix.NumRows())
		require.Equal(t, 16, sampleMatrix.NumCols())

		for i, elem := range sampleMatrix.Data() {
			require.Equal(t, expectedFullPadding[i], elem, "unexpected element at index '%d'", i)
		}
	})

}

var data3x4x4 = []float32{
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

var expectedValidPadding3DKernel = []float32{
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

var expectedValidPadding2DKernel = []float32{
	1, 1, 1, 2, 1, 2, 2, 1, 3, 3, 1, 1,
	1, 1, 2, 2, 2, 3, 1, 2, 3, 3, 1, 1,
	1, 1, 2, 1, 3, 4, 2, 3, 3, 3, 1, 1,
	1, 2, 1, 2, 2, 1, 3, 2, 1, 1, 1, 1,
	2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1,
	2, 1, 2, 1, 2, 3, 1, 2, 1, 1, 1, 1,
	1, 2, 1, 1, 3, 2, 4, 3, 1, 1, 3, 3,
	2, 2, 1, 1, 2, 1, 3, 2, 1, 1, 3, 3,
	2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 3, 3,
}

var expectedSamePadding = []float32{
	// 0, 0, 0, 0, 0, 0,
	// 0, 1, 1, 1, 1, 0,
	// 0, 1, 2, 2, 1, 0,
	// 0, 1, 2, 2, 1, 0,
	// 0, 1, 1, 1, 1, 0,
	// 0, 0, 0, 0, 0, 0,

	0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2,
	0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1,
	0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0,
	0, 1, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 1, 1,
	1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1,
	1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 1, 1, 1, 0,
	0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 1, 1, 0, 0, 0, 0,
	1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0,
	2, 2, 1, 0, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0,

	// 0, 0, 0, 0, 0, 0,
	// 0, 1, 2, 3, 4, 0,
	// 0, 2, 1, 2, 3, 0,
	// 0, 3, 2, 1, 2, 0,
	// 0, 4, 3, 2, 1, 0,
	// 0, 0, 0, 0, 0, 0,

	0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 1, 2, 0, 3, 2, 1,
	0, 0, 0, 0, 1, 2, 3, 4, 2, 1, 2, 3, 3, 2, 1, 2,
	0, 0, 0, 0, 2, 3, 4, 0, 1, 2, 3, 0, 2, 1, 2, 0,
	0, 1, 2, 3, 0, 2, 1, 2, 0, 3, 2, 1, 0, 4, 3, 2,
	1, 2, 3, 4, 2, 1, 2, 3, 3, 2, 1, 2, 4, 3, 2, 1,
	2, 3, 4, 0, 1, 2, 3, 0, 2, 1, 2, 0, 3, 2, 1, 0,
	0, 2, 1, 2, 0, 3, 2, 1, 0, 4, 3, 2, 0, 0, 0, 0,
	2, 1, 2, 3, 3, 2, 1, 2, 4, 3, 2, 1, 0, 0, 0, 0,
	1, 2, 3, 0, 2, 1, 2, 0, 3, 2, 1, 0, 0, 0, 0, 0,

	// 0, 0, 0, 0, 0, 0,
	// 0, 3, 3, 3, 3, 0,
	// 0, 1, 1, 1, 1, 0,
	// 0, 1, 1, 1, 1, 0,
	// 0, 3, 3, 3, 3, 0,
	// 0, 0, 0, 0, 0, 0,

	0, 0, 0, 0, 0, 3, 3, 3, 0, 1, 1, 1, 0, 1, 1, 1,
	0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,
	0, 0, 0, 0, 3, 3, 3, 0, 1, 1, 1, 0, 1, 1, 1, 0,
	0, 3, 3, 3, 0, 1, 1, 1, 0, 1, 1, 1, 0, 3, 3, 3,
	3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3,
	3, 3, 3, 0, 1, 1, 1, 0, 1, 1, 1, 0, 3, 3, 3, 0,
	0, 1, 1, 1, 0, 1, 1, 1, 0, 3, 3, 3, 0, 0, 0, 0,
	1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0,
	1, 1, 1, 0, 1, 1, 1, 0, 3, 3, 3, 0, 0, 0, 0, 0,
}

var data3x2x2 = []float32{
	1, 1,
	1, 2,

	1, 2,
	2, 1,

	3, 3,
	1, 1,
}

var expectedFullPadding = []float32{

	// 0, 0, 0, 0, 0, 0,
	// 0, 0, 0, 0, 0, 0,
	// 0, 0, 1, 1, 0, 0,
	// 0, 0, 1, 2, 0, 0,
	// 0, 0, 0, 0, 0, 0,
	// 0, 0, 0, 0, 0, 0,

	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0,
	0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0,
	0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

	// 0, 0, 0, 0, 0, 0,
	// 0, 0, 0, 0, 0, 0,
	// 0, 0, 1, 2, 0, 0,
	// 0, 0, 2, 1, 0, 0,
	// 0, 0, 0, 0, 0, 0,
	// 0, 0, 0, 0, 0, 0,

	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 2, 1,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 2, 1, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 2, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0,
	0, 0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

	// 0, 0, 0, 0, 0, 0,
	// 0, 0, 0, 0, 0, 0,
	// 0, 0, 3, 3, 0, 0,
	// 0, 0, 1, 1, 0, 0,
	// 0, 0, 0, 0, 0, 0,
	// 0, 0, 0, 0, 0, 0,

	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 1, 1,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 1, 1, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 1, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 1, 1, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 3, 3, 0, 0, 1, 1, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 3, 3, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
	0, 0, 3, 3, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 3, 3, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	3, 3, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
}

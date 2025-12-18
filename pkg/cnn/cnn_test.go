package cnn_test

import (
	"log/slog"
	"testing"

	"github.com/edatts/ml/pkg/cnn"
	"github.com/stretchr/testify/require"
)

func TestCNN(t *testing.T) {
	c, err := cnn.New(cnn.Shape{28, 28, 1}, cnn.Shape{1, 1, 10})
	require.NoError(t, err)

	_ = c
}

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

	for i := range sampleMatrix.NumRows() {
		slog.Info("row", "row", sampleMatrix.Row(i))
	}

	for i, elem := range sampleMatrix.Data() {
		require.Equal(t, expected[i], elem)
	}
}

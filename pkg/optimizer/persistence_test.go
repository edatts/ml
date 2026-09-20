package optimizer

import (
	"os"
	"testing"

	"github.com/edatts/ml/pkg/model"
	"github.com/edatts/ml/pkg/optimizer/test"
	"github.com/stretchr/testify/require"
)

var parameters = []model.Tensor{
	{
		Name:  "full.0.weights",
		Shape: model.Shape{0, 0, 0, 9},
		Data:  []float32{1, 2, 3, 4, 5, 4, 3, 2, 1},
	},
	{
		Name:  "full.0.biases",
		Shape: model.Shape{0, 0, 0, 3},
		Data:  []float32{1, 2, 1},
	},
}

var mostRecentParams = []model.Tensor{
	{
		Name:  "conv2d.0.weights",
		Shape: model.Shape{1, 1, 2, 2},
		Data:  []float32{2, 2, 2, 2},
	},
	{
		Name:  "conv2d.0.biases",
		Shape: model.Shape{0, 0, 0, 1},
		Data:  []float32{6},
	},
}

func TestPersistence(t *testing.T) {
	o, cleanupFn := setup(t)

	t.Run("save and load weights", func(t *testing.T) {
		t.Cleanup(cleanupFn)
		require.NoError(t, o.saveWeights())

		entries, err := os.ReadDir(o.cfg.weightsDir)
		require.NoError(t, err)
		require.Len(t, entries, 1)
		require.Contains(t, entries[0].Name(), o.model.Identity())

		require.NoError(t, o.model.LoadWeights(nil))
		require.NoError(t, o.loadWeights())
		require.ElementsMatch(t, parameters, o.model.Weights())
	})

	t.Run("loadWeights ignores files for a different identity", func(t *testing.T) {
		t.Cleanup(cleanupFn)
		require.NoError(t, o.saveWeights())

		o.model = test.NewMockModel(test.WithIdentity("anotherIdentity"))

		require.NoError(t, o.loadWeights())
		require.Nil(t, o.model.Weights())

	})

	t.Run("loadWeights only loads the most recent weights", func(t *testing.T) {
		t.Cleanup(cleanupFn)
		require.NoError(t, o.saveWeights())
		require.NoError(t, o.saveWeights())

		require.NoError(t, o.model.LoadWeights(mostRecentParams))
		require.NoError(t, o.saveWeights())

		require.NoError(t, o.model.LoadWeights(nil))
		require.NoError(t, o.loadWeights())
		require.ElementsMatch(t, mostRecentParams, o.model.Weights())
	})
}

func setup(t *testing.T) (*optimizer, func()) {
	weightsDir := t.TempDir()

	o := &optimizer{
		cfg: Config{
			weightsDir: weightsDir,
		},
		model: test.NewMockModel(test.WithWeights(parameters)),
	}

	cleanupFn := func() {
		o = &optimizer{
			cfg: Config{
				weightsDir: weightsDir,
			},
			model: test.NewMockModel(test.WithWeights(parameters)),
		}

		entries, err := os.ReadDir(t.TempDir())
		require.NoError(t, err)

		for _, entry := range entries {
			require.NoError(t, os.Remove(entry.Name()))
		}

		// Reset the weights in the mock.
		require.NoError(t, o.model.LoadWeights(parameters))
	}

	return o, cleanupFn
}

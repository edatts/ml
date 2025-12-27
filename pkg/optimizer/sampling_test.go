package optimizer_test

import (
	"testing"

	"github.com/edatts/ml/pkg/optimizer"
	"github.com/stretchr/testify/require"
)

func TestSampling(t *testing.T) {

	t.Run("Configuration Errors", func(t *testing.T) {
		s := optimizer.NewConvenienceSampler[int]()
		require.False(t, s.Next())
		require.ErrorIs(t, s.Err(), optimizer.ErrSamplerNotInitialized)

		s = optimizer.NewRS2Sampler[int](1)
		err := s.Init(0, func() ([][]float32, any, error) { return [][]float32{}, [][]int{}, nil })
		require.ErrorIs(t, err, optimizer.ErrBatchSize)

		s = optimizer.NewRS2Sampler[int](0.001)
		err = s.Init(1, func() ([][]float32, any, error) { return [][]float32{}, [][]int{}, nil })
		require.NoError(t, err)
		require.False(t, s.Next())
		require.ErrorIs(t, s.Err(), optimizer.ErrInvalidSubsetRatio)

		s = optimizer.NewRS2Sampler[int](1.001)
		err = s.Init(1, func() ([][]float32, any, error) { return [][]float32{}, [][]int{}, nil })
		require.NoError(t, err)
		require.False(t, s.Next())
		require.ErrorIs(t, s.Err(), optimizer.ErrInvalidSubsetRatio)
	})

	t.Run("Bad Data Provider Errors", func(t *testing.T) {
		mismatchedSampleAndTargetLengths := func() ([][]float32, any, error) {
			return [][]float32{{1}, {2}}, [][]int{{1}}, nil
		}

		s := optimizer.NewConvenienceSampler[int]()
		err := s.Init(5, mismatchedSampleAndTargetLengths)
		require.ErrorIs(t, err, optimizer.ErrDataLengths)

		wrongTargetDataType := func() ([][]float32, any, error) {
			return [][]float32{{1}}, [][]float32{{1}}, nil
		}

		s = optimizer.NewConvenienceSampler[int]()
		err = s.Init(5, wrongTargetDataType)
		require.ErrorIs(t, err, optimizer.ErrTargetsDataType)
	})

	t.Run("RS2 Sampler", func(t *testing.T) {
		var (
			numBatches        int
			totalSamples      int
			trainDataProvider = func() ([][]float32, any, error) {
				X := [][]float32{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}
				Y := [][]int{{2}, {4}, {6}, {8}, {10}, {12}, {14}, {16}}
				return X, Y, nil
			}
		)

		s := optimizer.NewRS2Sampler[int](0.2)
		require.NoError(t, s.Init(5, trainDataProvider))

		cSampler, ok := s.(optimizer.ClassificationSampler)
		require.True(t, ok)

		for epoch := cSampler.NewEpoch(); epoch <= 5; epoch = cSampler.NewEpoch() {
			for cSampler.Next() {
				batch, err := cSampler.Batch()
				require.NoError(t, err)

				numBatches++
				totalSamples += len(batch.Inputs)
				require.NotEqual(t, [][]float32{{1}, {2}, {3}, {4}, {5}}, batch)
			}

			require.NoError(t, cSampler.Err())
		}

		require.Equal(t, 5, numBatches)
		require.Equal(t, 5*5, totalSamples)

	})
}

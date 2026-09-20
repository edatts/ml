package optimizer_test

import (
	"testing"

	"github.com/edatts/ml/pkg/optimizer"
	"github.com/edatts/ml/pkg/optimizer/test"
	"github.com/stretchr/testify/require"
)

func TestOptimizer(t *testing.T) {
	trainDataProvider := func() ([][]float32, any, error) {
		return [][]float32{{1}, {2}, {3}}, [][]float32{{2}, {4}, {6}}, nil
	}

	testDataProvider := func() ([][]float32, any, error) {
		return [][]float32{{4}, {5}, {6}}, [][]float32{{8}, {10}, {12}}, nil
	}

	t.Run("Instantiate and Run", func(t *testing.T) {
		o := optimizer.New(
			optimizer.WithModel(test.NewMockModel()),
			optimizer.WithNumEpochs(3),
			optimizer.WithBatchSize(1),
			optimizer.WithLoggingInterval(1),
			optimizer.WithTrainDataProvider(trainDataProvider),
			optimizer.WithTestDataProvider(testDataProvider),
			optimizer.WithSampler(&optimizer.DefaultSampler[float32]{}),
		)

		require.NoError(t, o.Run())
	})

	t.Run("Errors on missing components", func(t *testing.T) {
		o := optimizer.New()

		err := o.Run()
		require.Error(t, err)
		require.ErrorIs(t, err, optimizer.ErrNoModel)

		o = optimizer.New(optimizer.WithModel(test.NewMockModel()))

		err = o.Run()
		require.Error(t, err)
		require.ErrorIs(t, err, optimizer.ErrNoSampler)

		o = optimizer.New(
			optimizer.WithModel(test.NewMockModel()),
			optimizer.WithSampler(optimizer.NewConvenienceSampler[float32]()),
		)

		err = o.Run()
		require.Error(t, err)
		require.ErrorIs(t, err, optimizer.ErrNoDataProvider)

		o = optimizer.New(
			optimizer.WithModel(test.NewMockModel()),
			optimizer.WithSampler(optimizer.NewConvenienceSampler[float32]()),
			optimizer.WithTrainDataProvider(trainDataProvider),
		)

		err = o.Run()
		require.Error(t, err)
		require.ErrorIs(t, err, optimizer.ErrNoDataProvider)

		o = optimizer.New(
			optimizer.WithModel(test.NewMockModel()),
			optimizer.WithSampler(optimizer.NewConvenienceSampler[float32]()),
			optimizer.WithTrainDataProvider(trainDataProvider),
			optimizer.WithTestDataProvider(testDataProvider),
		)

		require.NoError(t, o.Run())
	})

}

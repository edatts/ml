package mlp_test

import (
	"testing"

	"github.com/edatts/ml/pkg/mlp"
	"github.com/edatts/ml/pkg/mnist"
	"github.com/edatts/ml/pkg/model"
	"github.com/edatts/ml/pkg/optimizer"
	"github.com/stretchr/testify/require"
)

func TestF16MLP(t *testing.T) {

	X_train, Y_train, X_test, Y_test, err := mnist.LoadData()
	require.NoError(t, err)

	model, err := mlp.NewF16(
		784, 10,
		mlp.WithClassifcation(),
		mlp.WithHiddenLayers(512, 384, 256),
	)
	require.NoError(t, err)

	o := optimizer.New(
		optimizer.WithModel(model),
		optimizer.WithClassification(),
		optimizer.WithNumEpochs(10),
		optimizer.WithBatchSize(128),
		optimizer.WithLearningRate(0.075),
		optimizer.WithLearningRateDecay(0.0025),
		optimizer.WithSampler(optimizer.NewRS2Sampler[int](0.10)),
		optimizer.WithTrainDataProvider(func() ([][]float32, any, error) { return X_train, Y_train, nil }),
		optimizer.WithTestDataProvider(func() ([][]float32, any, error) { return X_test, Y_test, nil }),
	)

	require.NoError(t, o.Run())
}

func BenchmarkMLP(b *testing.B) {
	X_train, Y_train, X_test, Y_test, err := mnist.LoadData()
	require.NoError(b, err)

	trainDataFn := func() ([][]float32, any, error) {
		return X_train, Y_train, nil
	}

	testDataFn := func() ([][]float32, any, error) {
		return X_test, Y_test, nil
	}

	f32Model, f16Model := buildModels(b)

	b.Run("FP32", func(b *testing.B) {
		o := getOptimizer(trainDataFn, testDataFn, f32Model)

		for b.Loop() {
			require.NoError(b, o.Run())
		}
	})

	b.Run("FP16", func(b *testing.B) {
		o := getOptimizer(trainDataFn, testDataFn, f16Model)

		for b.Loop() {
			require.NoError(b, o.Run())
		}
	})
}

func buildModels(b *testing.B) (*mlp.MLP, *mlp.F16MLP) {
	model, err := mlp.New(
		784, 10,
		mlp.WithClassifcation(),
		mlp.WithHiddenLayers(512, 384, 256),
	)
	require.NoError(b, err)

	f16Model, err := mlp.NewF16(
		784, 10,
		mlp.WithClassifcation(),
		mlp.WithHiddenLayers(512, 384, 256),
	)
	require.NoError(b, err)

	return model, f16Model
}

func getOptimizer(trainDataFn, testDataFn func() ([][]float32, any, error), model model.Model) optimizer.Optimizer {
	return optimizer.New(
		optimizer.WithModel(model),
		optimizer.WithClassification(),
		optimizer.WithNumEpochs(1),
		optimizer.WithBatchSize(128),
		optimizer.WithLearningRate(0.075),
		optimizer.WithLearningRateDecay(0.0025),
		optimizer.WithSampler(optimizer.NewRS2Sampler[int](0.02)),
		optimizer.WithTrainDataProvider(trainDataFn),
		optimizer.WithTestDataProvider(testDataFn),
	)
}

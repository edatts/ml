package mlp_test

import (
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"net/http"
	"sync"
	"testing"

	"github.com/edatts/ml/pkg/mlp"
	"github.com/edatts/ml/pkg/mnist"
	"github.com/edatts/ml/pkg/optimizer"
	"github.com/go-echarts/go-echarts/v2/components"
	"github.com/stretchr/testify/require"
)

var mu = sync.Mutex{}
var allCarts []components.Charter
var page *components.Page

func httpHandler(rw http.ResponseWriter, _ *http.Request) {
	mu.Lock()
	defer mu.Unlock()
	page := components.NewPage().AddCharts(allCarts...)
	page.SetLayout(components.PageFullLayout)
	slog.Info("rendering page...")
	if err := page.Render(rw); err != nil {
		panic("failed rendering page")
	}
}

func TestActivation(t *testing.T) {
	t.Run("softmax", func(t *testing.T) {
		values := []float32{4.8, 1.21, 2.385}
		expected := []float32{0.8952827, 0.024708305, 0.08000901}
		require.Equal(t, expected, mlp.SoftMax{}.Forward(values))
	})

	t.Run("relu", func(t *testing.T) {
		require.Equal(t, [][]float32{{4.8}, {0}}, [][]float32{[]float32{mlp.ReLU{}.Forward(4.8)}, []float32{mlp.ReLU{}.Forward(-1.21)}})
	})
}

func TestMLP(t *testing.T) {
	// ctx, cancel := signal.NotifyContext(t.Context(), os.Interrupt, os.Kill)
	// defer cancel()

	// // Server for serving charts...
	// go func() {
	// 	svr := &http.Server{
	// 		Addr: ":8000",
	// 	}

	// 	http.Handle("GET /", http.HandlerFunc(httpHandler))
	// 	if err := svr.ListenAndServe(); err != http.ErrServerClosed {
	// 		panic(fmt.Sprintf("server err: %s", err.Error()))
	// 	}
	// }()

	t.Run("spiral dataset", func(t *testing.T) {
		var (
			N = 100
			K = 3
			X = make([][]float32, N*K)
			Y = make([][]int, N*K)
		)

		for i := range K {
			r := linspace(0, 1, N)
			t := linspaceRand(float64(4*i), float64(4*(i+1)), N, 0.2)
			for j := range N {
				dataPoint := []float32{float32(r[j] * math.Sin(t[j])), float32(r[j] * math.Cos(t[j]))}
				X[i*100+j] = dataPoint
				y := make([]int, K)
				y[i] = 1
				Y[i*100+j] = y
			}
		}

		X_train, X_test, Y_train, Y_test := trainTestSplit(X, Y, 0.8)

		trainDataProvider := func() ([][]float32, any, error) {
			return X_train, Y_train, nil
		}

		testDataProvider := func() ([][]float32, any, error) {
			return X_test, Y_test, nil
		}

		model, err := mlp.New(
			2, 3, mlp.WithClassifcation(),
			mlp.WithHiddenLayers(32, 64, 64),
		)
		require.NoError(t, err)

		o := optimizer.New(
			optimizer.WithClassification(),
			optimizer.WithNumEpochs(50),
			optimizer.WithBatchSize(16),
			optimizer.WithLearningRate(0.1),
			optimizer.WithLearningRateDecay(0.0025),
			optimizer.WithLoggingInterval(100),
			optimizer.WithTrainDataProvider(trainDataProvider),
			optimizer.WithTestDataProvider(testDataProvider),
			optimizer.WithSampler(optimizer.NewRS2Sampler[int](1)),
		)

		require.NoError(t, o.Run(model))

		_, _, accuracy, regLoss, err := o.Classify(X_test, Y_test)
		require.NoError(t, err)

		slog.Info("loss", "loss", regLoss)
		slog.Info("percentage accuracy", "accuracy", accuracy)

		require.Greater(t, accuracy, float64(90))
	})

	t.Run("sine wave", func(t *testing.T) {
		var (
			numDatapoints         = 5000
			X, Y                  = make([][]float32, numDatapoints), make([][]float32, numDatapoints)
			trainRatio    float64 = 0.80
		)

		for i := range numDatapoints {
			x := (rand.Float64() * 3.14159 * 2) - 3.14159
			y := float32(math.Sin(x))
			X[i], Y[i] = []float32{float32(x / 3.14159)}, []float32{y / 3.14159}
		}

		X_train, X_test, Y_train, Y_test := trainTestSplit(X, Y, trainRatio)

		trainDataProvider := func() ([][]float32, any, error) {
			return X_train, Y_train, nil
		}

		testDataProvider := func() ([][]float32, any, error) {
			return X_test, Y_test, nil
		}

		model, err := mlp.New(
			1, 1,
			mlp.WithHiddenLayers(64, 128, 64),
		)
		require.NoError(t, err)

		o := optimizer.New(
			optimizer.WithNumEpochs(50),
			optimizer.WithBatchSize(32),
			optimizer.WithLearningRate(0.2),
			optimizer.WithLearningRateDecay(0.0002),
			optimizer.WithLoggingInterval(200),
			optimizer.WithTrainDataProvider(trainDataProvider),
			optimizer.WithTestDataProvider(testDataProvider),
			optimizer.WithSampler(optimizer.NewConvenienceSampler[float32]()),
		)

		require.NoError(t, o.Run(model))

		outputs, _, mse, regLoss, err := o.Regress(X_test, Y_test)
		require.NoError(t, err)

		slog.Info("loss", "loss", regLoss)
		slog.Info("Mean Squared Error", "MSE", mse)

		var numCloseEnough int
		for i, target := range Y_test {
			//  require.True(t, withinExpectedRange(outputs[i][0], target[0], 0.05), "actual not within range, actual=%f, expected=%f, tolerancePercentage=%d", outputs[i][0], target[0], 5)
			// slog.Info("results", "prediction", outputs[i][0], "target", target[0])
			if withinExpectedRange(outputs[i][0], target[0], 0.01) {
				numCloseEnough++
			}
		}

		percentageInRange := (float64(numCloseEnough) / float64(len(Y_test))) * 100

		slog.Info("len(Y_test)", "len", len(Y_test))
		slog.Info("numCloseEnough", "num", numCloseEnough)
		slog.Info("percentage of close enough answers", "percentage", percentageInRange)
		require.Greater(t, percentageInRange, float64(90))

		// sinePlot := charts.NewScatter()
		// sinePlot.SetGlobalOptions(
		// 	charts.WithTitleOpts(opts.Title{Title: "Sine actual"}),
		// 	charts.WithGridOpts(opts.Grid{
		// 		Show: opts.Bool(true),
		// 	}),
		// )

		// trainData := []opts.ScatterData{}
		// // for i, x := range X_train {
		// for i, x := range X_test {
		// 	trainData = append(trainData, opts.ScatterData{
		// 		// Value: []float32{x[0], Y_train[i][0]},
		// 		Value:      []float32{x[0], outputs[i][0]},
		// 		SymbolSize: 7,
		// 	})
		// }

		// sinePlot.AddSeries("Predicted", trainData)

		// mu.Lock()
		// allCarts = append(allCarts, sinePlot)
		// mu.Unlock()

	})

	t.Run("mnist handwritten digits", func(t *testing.T) {
		slog.Info("preparing mnist data")

		// The input data is scaled and the targets are one-hot encoded.
		X_train, Y_train, X_test, Y_test, err := mnist.LoadData()
		require.NoError(t, err)

		slog.Info("loaded mnist data")

		model, err := mlp.New(
			784, 10,
			mlp.WithClassifcation(),
			mlp.WithHiddenLayers(512, 384, 256),
		)
		require.NoError(t, err)

		o := optimizer.New(
			optimizer.WithClassification(),
			optimizer.WithNumEpochs(10),
			optimizer.WithBatchSize(128),
			optimizer.WithLearningRate(0.075),
			optimizer.WithLearningRateDecay(0.0025),
			optimizer.WithSampler(optimizer.NewRS2Sampler[int](0.10)),
			optimizer.WithTrainDataProvider(func() ([][]float32, any, error) { return X_train, Y_train, nil }),
			optimizer.WithTestDataProvider(func() ([][]float32, any, error) { return X_test, Y_test, nil }),
		)

		slog.Info("starting model training...")

		require.NoError(t, o.Run(model))

	})

	// <-ctx.Done()
}

type Batch[XT, YT any] struct {
	Inputs  []XT
	Targets []YT
}

func NewBatch[XT, YT any](size int) *Batch[XT, YT] {
	return &Batch[XT, YT]{
		Inputs:  make([]XT, size),
		Targets: make([]YT, size),
	}
}

// Here we're just going to use repeated sampling of random subsets.
func selectBatches[XT, YT any](batchSize int, X []XT, Y []YT, subsetRatio float64) []*Batch[XT, YT] {
	if len(X) != len(Y) {
		panic(fmt.Sprintf("selecting batches: X and Y are different lengths len(X)=%d, len(Y)=%d", len(X), len(Y)))
	}

	if subsetRatio < 0.01 || subsetRatio > 1 {
		panic("batch sampling subset ratio must be between 0.01 and 1 inclusive")
	}

	shuffle(3, X, Y)

	var (
		numSamples     = int(float64(len(X)) * subsetRatio)
		out            = make([]*Batch[XT, YT], int(math.Ceil(float64(numSamples)/float64(batchSize))))
		batchNum   int = -1
	)
	for i := 0; i < numSamples; i++ {
		if i%batchSize == 0 {
			batchNum++
			out[batchNum] = NewBatch[XT, YT](min(batchSize, numSamples-i))
		}
		out[batchNum].Inputs[i%batchSize] = X[i]
		out[batchNum].Targets[i%batchSize] = Y[i]
	}

	return out
}

func TestSelectBatches(t *testing.T) {
	var (
		batchSize = 5
		X         = []int{1, 2, 3, 4, 5, 6, 7, 8}
		Y         = []int{2, 4, 6, 8, 10, 12, 14, 16}
	)

	batches := selectBatches(batchSize, X, Y, 1)
	require.Len(t, batches, 2)
	require.Len(t, batches[0].Inputs, 5)
	require.Len(t, batches[0].Targets, 5)
	require.Len(t, batches[1].Inputs, 3)
	require.Len(t, batches[1].Targets, 3)
}

func linspace(start, end float64, n int) []float64 {
	var out = make([]float64, n)
	unit := (end - start) / float64(n)
	for i := range n {
		out[i] = start + (unit * float64(i+1))
	}
	return out
}

func linspaceRand(start, end float64, n int, randFactor float64) []float64 {
	var (
		out  = make([]float64, n)
		unit = (end - start) / float64(n)
	)

	for i := range n {
		out[i] = (start + (unit * float64(i+1))) + ((rand.Float64()*2 - 1) * randFactor)
	}

	return out
}

func trainTestSplit[XT, YT any](X []XT, Y []YT, trainRatio float64) ([]XT, []XT, []YT, []YT) {
	if len(X) != len(Y) {
		panic("X and Y are different lengths")
	}

	shuffle(5, X, Y)

	var (
		trainSize = int(float64(len(X)) * trainRatio)
		X_train   = make([]XT, trainSize)
		X_test    = make([]XT, len(X)-trainSize)
		Y_train   = make([]YT, trainSize)
		Y_test    = make([]YT, len(Y)-trainSize)
	)

	for i := range len(X) {
		if i >= trainSize {
			X_test[i-trainSize] = X[i]
			Y_test[i-trainSize] = Y[i]
			continue
		}
		X_train[i] = X[i]
		Y_train[i] = Y[i]
	}

	return X_train, X_test, Y_train, Y_test
}

func withinExpectedRange(value, expected, absRange float32) bool {
	return value >= expected-absRange && value <= expected+absRange
}

func shuffle[XT, YT any](numRounds int, X []XT, Y []YT) {
	for range numRounds {
		rand.Shuffle(len(X), func(i, j int) {
			X[i], X[j] = X[j], X[i]
			Y[i], Y[j] = Y[j], Y[i]
		})
	}
}

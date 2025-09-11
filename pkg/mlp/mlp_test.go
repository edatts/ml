package mlp_test

import (
	"encoding/csv"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"testing"

	"github.com/edatts/ml/pkg/mlp"
	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/components"
	"github.com/go-echarts/go-echarts/v2/opts"
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
		values := []float64{4.8, 1.21, 2.385}
		expected := []float64{0.8952826639572619, 0.02470830678209937, 0.0800090292606387}
		require.Equal(t, expected, mlp.SoftMax{}.Forward(values))
	})

	t.Run("relu", func(t *testing.T) {
		require.Equal(t, []float64{4.8, 0}, []float64{mlp.ReLU{}.Forward(4.8), mlp.ReLU{}.Forward(-1.21)})
	})
}

func TestMLP(t *testing.T) {
	ctx, cancel := signal.NotifyContext(t.Context(), os.Interrupt, os.Kill)
	defer cancel()

	go func() {
		svr := &http.Server{
			Addr: ":8000",
		}

		http.Handle("GET /", http.HandlerFunc(httpHandler))
		if err := svr.ListenAndServe(); err != http.ErrServerClosed {
			panic(fmt.Sprintf("server err: %s", err.Error()))
		}
	}()

	// t.Run("spiral dataset", func(t *testing.T) {
	// 	var (
	// 		N = 100
	// 		K = 3
	// 		X = make([][]float64, N*K)
	// 		Y = make([][]int, N*K)
	// 	)

	// 	for i := range K {
	// 		r := linspace(0, 1, N)
	// 		t := linspaceRand(float64(4*i), float64(4*(i+1)), N, 0.2)
	// 		for j := range N {
	// 			dataPoint := []float64{r[j] * math.Sin(t[j]), r[j] * math.Cos(t[j])}
	// 			X[i*100+j] = dataPoint
	// 			y := make([]int, K)
	// 			y[i] = 1
	// 			Y[i*100+j] = y
	// 		}
	// 	}

	// 	scatter := charts.NewScatter()
	// 	scatter.SetGlobalOptions(
	// 		charts.WithTitleOpts(opts.Title{Title: "Spiral scatter"}),
	// 		charts.WithGridOpts(opts.Grid{
	// 			Show: opts.Bool(true),
	// 		}),
	// 		charts.WithXAxisOpts(opts.XAxis{
	// 			Min: -1.1, Max: 1.1,
	// 		}),
	// 		charts.WithYAxisOpts(opts.YAxis{
	// 			Min: -1.1, Max: 1.1,
	// 		}),
	// 	)

	// 	data := []opts.ScatterData{}
	// 	for i, x := range X {
	// 		data = append(data, opts.ScatterData{
	// 			Value:      x,
	// 			SymbolSize: 7,
	// 		})
	// 		if (i+1)%100 == 0 {
	// 			scatter.AddSeries(strconv.Itoa((i+1)/100), data)
	// 			data = nil
	// 		}
	// 	}

	// 	mu.Lock()
	// 	allCarts = append(allCarts, scatter)
	// 	mu.Unlock()

	// 	slog.Info("X", "X", X[:5])
	// 	slog.Info("X", "X", X[100:105])
	// 	slog.Info("X", "X", X[200:205])

	// 	slog.Info("Y", "Y", Y[:5])
	// 	slog.Info("Y", "Y", Y[100:105])
	// 	slog.Info("Y", "Y", Y[200:205])

	// 	model, err := mlp.New(2, 3, 2, mlp.WithClassifcation())
	// 	require.NoError(t, err)

	// 	_, _, _, err = model.Classify(X, Y)
	// 	require.NoError(t, err)

	// 	var (
	// 		numIterations         = 20000
	// 		batchSize             = 1
	// 		learningRate  float64 = 0.001
	// 		lossSum       float64 = 0
	// 		accSum        float64 = 0
	// 	)

	// 	X_train, X_test, Y_train, Y_test := trainTestSplit(X, Y, 0.8)

	// 	for i := range numIterations {
	// 		// slog.Info("new iteration")
	// 		// Pick random data point
	// 		// idx := rand.Intn(len(X))
	// 		// input, target := X[idx], Y[idx]

	// 		var batchInput = make([][]float64, batchSize)
	// 		var batchTarget = make([][]int, batchSize)
	// 		for j := range batchSize {
	// 			batchInput[j] = X_train[(j+(i*batchSize))%len(X_train)]
	// 			batchTarget[j] = Y_train[(j+(i*batchSize))%len(Y_train)]
	// 		}

	// 		// slog.Info("data", "inputs", batchInput, "targets", batchTarget)

	// 		// out, acc, loss, err := model.Classify([][]float64{input}, [][]int{target})
	// 		_, acc, loss, err := model.Classify(batchInput, batchTarget)
	// 		require.NoError(t, err)

	// 		lossSum += loss
	// 		accSum += acc
	// 		if (i+1)%500 == 0 {
	// 			learningRate *= 0.99
	// 			slog.Info("training info", "iteration", i+1, "lr", learningRate, "loss", loss, "avgLoss", lossSum/float64(500), "avgAccuracy", accSum/float64(500))

	// 			// model.LogLossDeriv()
	// 			require.NoError(t, model.Backward(learningRate))
	// 			// model.LogWeights()
	// 			// model.LogLogits()
	// 			// model.LogGrads()
	// 			// model.LogActivations()
	// 			// slog.Info("targets", "targets", batchTarget)
	// 			// slog.Info("outputs", "outputs", outputs)
	// 			lossSum, accSum = 0, 0
	// 			continue
	// 		}

	// 		require.NoError(t, model.Backward(learningRate))
	// 	}

	// 	_, accuracy, loss, err := model.Classify(X_test, Y_test)
	// 	require.NoError(t, err)

	// 	slog.Info("loss", "loss", loss)
	// 	slog.Info("percentage accuracy", "accuracy", accuracy)

	// 	require.Greater(t, accuracy, float64(90))

	// })

	t.Run("sine wave", func(t *testing.T) {
		var (
			numDatapoints         = 1000
			numIterations         = 25000
			batchSize             = 4
			learningRate  float64 = 0.025
			X, Y                  = make([][]float64, numDatapoints), make([][]float64, numDatapoints)
			trainRatio    float64 = 0.85
			sumLoss       float64 = 0
			sumMSE        float64 = 0
		)

		for i := range numDatapoints {
			x := rand.Float64() * 3.14159 * 2
			y := math.Sin(x)
			X[i], Y[i] = []float64{x}, []float64{y}
		}

		X_train, X_test, Y_train, Y_test := trainTestSplit(X, Y, trainRatio)

		// sinePlot := charts.NewScatter()
		// sinePlot.SetGlobalOptions(
		// 	charts.WithTitleOpts(opts.Title{Title: "Sine actual"}),
		// 	charts.WithGridOpts(opts.Grid{
		// 		Show: opts.Bool(true),
		// 	}),
		// )

		// trainData := []opts.ScatterData{}
		// for i, x := range X_train {
		// 	trainData = append(trainData, opts.ScatterData{
		// 		Value:      []float64{x[0], Y_train[i][0]},
		// 		SymbolSize: 7,
		// 	})
		// }

		// sinePlot.AddSeries("Predicted", trainData)

		// mu.Lock()
		// allCarts = append(allCarts, sinePlot)
		// mu.Unlock()

		// slog.Info("ready to render page.. ")

		model, err := mlp.New(1, 1, 2)
		require.NoError(t, err)

		for i := range numIterations {
			var batchInput = make([][]float64, batchSize)
			var batchTarget = make([][]float64, batchSize)
			for j := range batchSize {
				batchInput[j] = X_train[(j+(i*batchSize))%len(X_train)]
				batchTarget[j] = Y_train[(j+(i*batchSize))%len(Y_train)]
			}

			outputs, mse, loss, err := model.Regress(batchInput, batchTarget)
			require.NoError(t, err)
			sumLoss += loss
			sumMSE += mse

			if (i+1)%1000 == 0 {
				slog.Info("training info", "iteration", i+1, "lr", learningRate, "avgLoss", sumLoss/float64(1000), "avgMSE", sumMSE/float64(1000), "pred_0", outputs[0], "target_0", batchTarget[0])
				sumLoss, sumMSE = 0, 0
				learningRate *= 0.98
			}

			require.NoError(t, model.Backward(learningRate))
		}

		outputs, mse, loss, err := model.Regress(X_test, Y_test)
		require.NoError(t, err)

		slog.Info("loss", "loss", loss)
		slog.Info("Mean Squared Error", "MSE", mse)

		var numCloseEnough int
		for i, target := range Y_test {
			//  require.True(t, withinExpectedRange(outputs[i][0], target[0], 0.05), "actual not within range, actual=%f, expected=%f, tolerancePercentage=%d", outputs[i][0], target[0], 5)
			if withinExpectedRange(outputs[i][0], target[0], 0.01) {
				numCloseEnough++
			}
		}

		slog.Info("len(Y_test)", "len", len(Y_test))
		slog.Info("numCloseEnough", "num", numCloseEnough)
		slog.Info("percentage of close enough answers", "percentage", (float64(numCloseEnough)/float64(len(Y_test)))*100)

		scatter := charts.NewScatter()
		scatter.SetGlobalOptions(
			charts.WithTitleOpts(opts.Title{Title: "Sine predictions"}),
			charts.WithGridOpts(opts.Grid{
				Show: opts.Bool(true),
			}),
		)

		data := []opts.ScatterData{}
		for i, x := range X_test {
			data = append(data, opts.ScatterData{
				Value:      []float64{x[0], outputs[i][0]},
				SymbolSize: 7,
			})
		}

		scatter.AddSeries("Predicted", data)

		mu.Lock()
		allCarts = append(allCarts, scatter)
		mu.Unlock()

	})

	t.Run("mnist handwritten digits", func(t *testing.T) {
		var (
			X_train = make([][]float64, 60000)
			Y_train = make([][]int, 60000)
			X_test  = make([][]float64, 10000)
			Y_test  = make([][]int, 10000)
		)

		train_samples, train_labels, test_samples, test_labels := loadMNISTData(t)
		formatMNISTSamples(t, train_samples, test_samples, X_train, X_test)
		formatMNISTLabels(t, train_labels, test_labels, Y_train, Y_test)

	})

	<-ctx.Done()
}

func loadMNISTData(t *testing.T) ([][]string, [][]string, [][]string, [][]string) {
	files := []string{
		"test/datasets/mnist/train-images.csv",
		"test/datasets/mnist/train-labels.csv",
		"test/datasets/mnist/t10k-images.csv",
		"test/datasets/mnist/t10k-labels.csv",
	}

	fileContents := make([][][]string, 4)
	for i, file := range files {
		f, err := os.Open(file)
		require.NoError(t, err)

		fileContents[i], err = csv.NewReader(f).ReadAll()
		require.NoError(t, err)
	}

	return fileContents[0], fileContents[1], fileContents[2], fileContents[3]
}

func formatMNISTSamples(t *testing.T, train, test [][]string, X_train, X_test [][]float64) {
	// Parse them into ints then scale them into float64s
	for i, sample := range train {
		var row = make([]float64, 784)
		for j, elem := range sample {
			num, err := strconv.ParseUint(elem, 10, 8)
			require.NoError(t, err)
			row[j] = float64(num) / 255
		}
		X_train[i] = row
	}

	for i, sample := range test {
		var row = make([]float64, 784)
		for j, elem := range sample {
			num, err := strconv.ParseUint(elem, 10, 8)
			require.NoError(t, err)
			row[j] = float64(num) / 255
		}
		X_test[i] = row
	}
}

func formatMNISTLabels(t *testing.T, train, test [][]string, Y_train, Y_test [][]int) {
	// Parse them into ints then 1 hot encode them
	for i, sample := range train {
		var row = make([]int, 10)
		num, err := strconv.ParseUint(sample[0], 10, 8)
		require.NoError(t, err)
		row[num] = 1
		Y_train[i] = row
	}

	for i, sample := range test {
		var row = make([]int, 10)
		num, err := strconv.ParseUint(sample[0], 10, 8)
		require.NoError(t, err)
		row[num] = 1
		Y_test[i] = row
	}
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

	// Shuffle
	for range 5 {
		for i := range len(X) {
			swapIdx := rand.Intn(len(X))
			X[i], X[swapIdx] = X[swapIdx], X[i]
			Y[i], Y[swapIdx] = Y[swapIdx], Y[i]
		}
	}

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

func withinExpectedRange(value, expected, absRange float64) bool {
	return value >= expected-absRange && value <= expected+absRange
}

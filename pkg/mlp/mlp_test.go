package mlp_test

import (
	"archive/tar"
	"compress/gzip"
	"encoding/csv"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/edatts/ml/pkg/idx"
	"github.com/edatts/ml/pkg/mlp"
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
		values := []float64{4.8, 1.21, 2.385}
		expected := []float64{0.8952826639572619, 0.02470830678209937, 0.0800090292606387}
		require.Equal(t, expected, mlp.SoftMax{}.Forward(values))
	})

	t.Run("relu", func(t *testing.T) {
		require.Equal(t, [][]float64{{4.8}, {0}}, [][]float64{mlp.ReLU{}.Forward([]float64{4.8}), mlp.ReLU{}.Forward([]float64{-1.21})})
	})
}

func TestMLP(t *testing.T) {
	// ctx, cancel := signal.NotifyContext(t.Context(), os.Interrupt, os.Kill)
	// defer cancel()

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

	// t.Run("sine wave", func(t *testing.T) {
	// 	var (
	// 		numDatapoints         = 1000
	// 		numIterations         = 10000
	// 		batchSize             = 64
	// 		learningRate  float64 = 0.015
	// 		X, Y                  = make([][]float64, numDatapoints), make([][]float64, numDatapoints)
	// 		trainRatio    float64 = 0.85
	// 		sumLoss       float64 = 0
	// 		sumMSE        float64 = 0
	// 	)

	// 	for i := range numDatapoints {
	// 		x := rand.Float64() * 3.14159 * 2
	// 		y := math.Sin(x)
	// 		X[i], Y[i] = []float64{x}, []float64{y}
	// 	}

	// 	X_train, X_test, Y_train, Y_test := trainTestSplit(X, Y, trainRatio)

	// 	// sinePlot := charts.NewScatter()
	// 	// sinePlot.SetGlobalOptions(
	// 	// 	charts.WithTitleOpts(opts.Title{Title: "Sine actual"}),
	// 	// 	charts.WithGridOpts(opts.Grid{
	// 	// 		Show: opts.Bool(true),
	// 	// 	}),
	// 	// )

	// 	// trainData := []opts.ScatterData{}
	// 	// for i, x := range X_train {
	// 	// 	trainData = append(trainData, opts.ScatterData{
	// 	// 		Value:      []float64{x[0], Y_train[i][0]},
	// 	// 		SymbolSize: 7,
	// 	// 	})
	// 	// }

	// 	// sinePlot.AddSeries("Predicted", trainData)

	// 	// mu.Lock()
	// 	// allCarts = append(allCarts, sinePlot)
	// 	// mu.Unlock()

	// 	// slog.Info("ready to render page.. ")

	// 	model, err := mlp.New(1, 1, 2)
	// 	require.NoError(t, err)

	// 	for i := range numIterations {
	// 		var batchInput = make([][]float64, batchSize)
	// 		var batchTarget = make([][]float64, batchSize)
	// 		for j := range batchSize {
	// 			batchInput[j] = X_train[(j+(i*batchSize))%len(X_train)]
	// 			batchTarget[j] = Y_train[(j+(i*batchSize))%len(Y_train)]
	// 		}

	// 		outputs, mse, loss, err := model.Regress(batchInput, batchTarget)
	// 		require.NoError(t, err)
	// 		sumLoss += loss
	// 		sumMSE += mse

	// 		if (i+1)%500 == 0 {
	// 			slog.Info("training info", "iteration", i+1, "lr", learningRate, "avgLoss", sumLoss/float64(1000), "avgMSE", sumMSE/float64(1000), "pred_0", outputs[0], "target_0", batchTarget[0])
	// 			sumLoss, sumMSE = 0, 0
	// 			learningRate *= 0.98
	// 			require.NoError(t, model.Backward(learningRate))
	// 			// model.LogGrads()
	// 			// model.LogLossDeriv()
	// 			// model.LogWeights()
	// 			// model.LogBiases()
	// 			continue
	// 		}

	// 		require.NoError(t, model.Backward(learningRate))
	// 	}

	// 	outputs, mse, loss, err := model.Regress(X_test, Y_test)
	// 	require.NoError(t, err)

	// 	slog.Info("loss", "loss", loss)
	// 	slog.Info("Mean Squared Error", "MSE", mse)

	// 	var numCloseEnough int
	// 	for i, target := range Y_test {
	// 		//  require.True(t, withinExpectedRange(outputs[i][0], target[0], 0.05), "actual not within range, actual=%f, expected=%f, tolerancePercentage=%d", outputs[i][0], target[0], 5)
	// 		if withinExpectedRange(outputs[i][0], target[0], 0.01) {
	// 			numCloseEnough++
	// 		}
	// 	}

	// 	slog.Info("len(Y_test)", "len", len(Y_test))
	// 	slog.Info("numCloseEnough", "num", numCloseEnough)
	// 	slog.Info("percentage of close enough answers", "percentage", (float64(numCloseEnough)/float64(len(Y_test)))*100)

	// 	scatter := charts.NewScatter()
	// 	scatter.SetGlobalOptions(
	// 		charts.WithTitleOpts(opts.Title{Title: "Sine predictions"}),
	// 		charts.WithGridOpts(opts.Grid{
	// 			Show: opts.Bool(true),
	// 		}),
	// 	)

	// 	data := []opts.ScatterData{}
	// 	for i, x := range X_test {
	// 		data = append(data, opts.ScatterData{
	// 			Value:      []float64{x[0], outputs[i][0]},
	// 			SymbolSize: 7,
	// 		})
	// 	}

	// 	scatter.AddSeries("Predicted", data)

	// 	mu.Lock()
	// 	allCarts = append(allCarts, scatter)
	// 	mu.Unlock()

	// })

	t.Run("mnist handwritten digits", func(t *testing.T) {
		slog.Info("preparing mnist data")

		train, Y_train, test, Y_test := loadMNISTData(t)

		slog.Info("loaded mnist data")

		// This func scales the inputs data between 0 and 1 and
		// one-hot encodes the targets.
		X_train, X_test := formatMNISTData(t, train, test, Y_train, Y_test)

		slog.Info("finished formatting data")

		model, err := mlp.New(784, 10, 3, mlp.WithClassifcation())
		require.NoError(t, err)

		var (
			numEpochs    = 16
			batchSize    = 64
			learningRate = 0.05
		)

		slog.Info("starting model training...")
		for epoch := 1; epoch <= numEpochs; epoch++ {
			var accSum, lossSum float64
			var numBatches int
			for _, batch := range selectBatches(batchSize, X_train, Y_train, 0.20) {
				_, acc, loss, err := model.Classify(batch.Inputs, batch.Targets)
				require.NoError(t, err)
				accSum += acc
				lossSum += loss
				numBatches++

				if numBatches%20 == 0 {
					slog.Info("training info", "epoch", epoch, "lr", learningRate, "loss", lossSum/float64(20), "accuracy", accSum/float64(20))
					accSum, lossSum = 0, 0
				}

				require.NoError(t, model.Backward(learningRate))
				learningRate *= 0.99925
			}
		}

		_, acc, loss, err := model.Classify(X_test, Y_test)
		require.NoError(t, err)

		slog.Info("testing info", "loss", loss, "accuracy", acc)

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

func loadMNISTData(t *testing.T) ([][]int, [][]int, [][]int, [][]int) {
	var (
		tarFile = "test/datasets/mnist/mnist.tar.gz"
		out     = map[string]idx.Data{}
	)

	f, err := os.Open(tarFile)
	require.NoError(t, err)
	defer f.Close()

	gr, err := gzip.NewReader(f)
	require.NoError(t, err)

	tr := tar.NewReader(gr)

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)

		slog.Info("reading file", "name", hdr.FileInfo().Name())
		// slog.Info("file size", "size", hdr.FileInfo().Size())
		slog.Info("file size", "size", hdr.Size)

		// var b = []byte{0}
		// for i := range 47040016 {
		// 	if n, err := tr.Read(b); err != nil {
		// 		slog.Error("failed reading", "error", err, "i", i, "n", n)
		// 		panic(err)
		// 	}
		// }

		name := strings.Split(hdr.Name, ".")[0]
		out[name], err = idx.ParseIdxFile(tr)
		require.NoError(t, err)
	}

	return out["train-images"].GetInt(), out["train-labels"].GetInt(), out["t10k-images"].GetInt(), out["t10k-labels"].GetInt()
}

func loadMNISTDataFromCsv(t *testing.T) ([][]string, [][]string, [][]string, [][]string) {
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
		defer f.Close()

		// Discard header
		cr := csv.NewReader(f)
		_, err = cr.Read()
		require.NoError(t, err)

		fileContents[i], err = cr.ReadAll()
		require.NoError(t, err)
	}

	return fileContents[0], fileContents[1], fileContents[2], fileContents[3]
}

func formatMNISTData(t *testing.T, train, test [][]int, Y_train, Y_test [][]int) ([][]float64, [][]float64) {
	// Scale image data
	var X_train = make([][]float64, len(train))
	for i, sample := range train {
		var scaled = make([]float64, len(sample))
		for j, num := range sample {
			scaled[j] = float64(num) / 255
		}
		X_train[i] = scaled
	}

	var X_test = make([][]float64, len(test))
	for i, sample := range test {
		var scaled = make([]float64, len(sample))
		for j, num := range sample {
			scaled[j] = float64(num) / 255
		}
		X_test[i] = scaled
	}

	// Convert labels to 1 hot encodings
	for i, sample := range Y_train {
		var row = make([]int, 10)
		row[sample[0]] = 1
		Y_train[i] = row
	}

	for i, sample := range Y_test {
		var row = make([]int, 10)
		row[sample[0]] = 1
		Y_test[i] = row
	}

	return X_train, X_test
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

func withinExpectedRange(value, expected, absRange float64) bool {
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

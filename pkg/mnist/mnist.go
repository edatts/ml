package mnist

import (
	"archive/tar"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/edatts/ml/pkg/idx"
)

func LoadData() (X_train [][]float32, Y_train [][]int, X_test [][]float32, Y_test [][]int, err error) {
	var (
		tarFile = "../../data/mnist/mnist.tar.gz"
		out     = map[string]idx.Data{}
	)

	f, err := os.Open(tarFile)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed opening file: %w", err)
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed instantiating gzip reader: %w", err)
	}

	tr := tar.NewReader(gr)

	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("tar reader error: %w", err)
		}

		// slog.Info("reading file", "name", hdr.FileInfo().Name())
		// slog.Info("file size", "size", hdr.Size)

		// var b = []byte{0}
		// for i := range 47040016 {
		// 	if n, err := tr.Read(b); err != nil {
		// 		slog.Error("failed reading", "error", err, "i", i, "n", n)
		// 		panic(err)
		// 	}
		// }

		name := strings.Split(hdr.Name, ".")[0]
		out[name], err = idx.ParseIdxFile(tr)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("failed parsing idx file: %w", err)
		}
	}

	Y_train, Y_test = out["train-labels"].GetInt(), out["t10k-labels"].GetInt()
	X_train, X_test = formatData(out["train-images"].GetInt(), out["t10k-images"].GetInt(), Y_train, Y_test)

	return X_train, Y_train, X_test, Y_test, nil
}

func formatData(train, test [][]int, Y_train, Y_test [][]int) ([][]float32, [][]float32) {
	// Scale image data
	var X_train = make([][]float32, len(train))
	for i, sample := range train {
		var scaled = make([]float32, len(sample))
		for j, num := range sample {
			scaled[j] = float32(num) / 255
		}
		X_train[i] = scaled
	}

	var X_test = make([][]float32, len(test))
	for i, sample := range test {
		var scaled = make([]float32, len(sample))
		for j, num := range sample {
			scaled[j] = float32(num) / 255
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

package optimizer

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/edatts/ml/pkg/safetensors"
)

func (o *optimizer) loadWeights() error {
	// Load most recent weights for this run.
	entries, err := os.ReadDir(o.cfg.weightsDir)
	if err != nil {
		return fmt.Errorf("failed reading directory '%s': %w", o.cfg.weightsDir, err)
	}

	var latestFileName string
	var latestTime time.Time
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		if !isValidFormat(entry.Name()) {
			continue
		}

		if !hasIdentity(entry.Name(), o.model.Identity()) {
			continue
		}

		if isMoreRecent(entry.Name(), latestTime) {
			latestFileName = entry.Name()
		}
	}

	// If we can't find weights for the model then just warn and return nil
	if latestFileName == "" {
		slog.Warn("no weights found for model", "identity", o.model.Identity(), "dir", o.cfg.weightsDir)
		return nil
	}

	f, err := os.Open(filepath.Join(o.cfg.weightsDir, latestFileName))
	if err != nil {
		return fmt.Errorf("failed opening file '%s': %w", filepath.Join(o.cfg.weightsDir, latestFileName), err)
	}
	defer f.Close()

	b, err := io.ReadAll(f)
	if err != nil {
		return fmt.Errorf("failed reading weights file '%s': %w", latestFileName, err)
	}

	parameters, err := safetensors.Parse(b)
	if err != nil {
		return fmt.Errorf("failed parsing file content as SafeTensors format: %w", err)
	}

	if err := o.model.LoadWeights(parameters); err != nil {
		return fmt.Errorf("failed loading parameters into model: %w", err)
	}

	return nil
}

// File name format: weights_<modelIdentity>_<unixNano>
func isValidFormat(name string) bool {
	parts := strings.Split(name, "_")
	_, err := strconv.ParseInt(parts[2], 10, 64)
	return len(parts) == 3 && parts[0] == "weights" && err == nil
}

func hasIdentity(name, identity string) bool {
	return strings.Split(name, "_")[1] == identity
}

func isMoreRecent(name string, latestTime time.Time) bool {
	unixNanos := strings.Split(name, "_")[2]
	nanos, err := strconv.ParseInt(unixNanos, 10, 64)
	if err != nil {
		slog.Warn("unparsable unix nanos in file name", "error", err)
		return false
	}

	return time.Unix(0, nanos).After(latestTime)
}

func (o *optimizer) saveWeights() error {
	if o.model == nil {
		return ErrNoModel
	}

	b, err := safetensors.Format(o.model.Weights())
	if err != nil {
		return fmt.Errorf("failed formatting model weights: %w", err)
	}

	var weightsDir = "weights"
	if o.cfg.weightsDir != "" {
		weightsDir = o.cfg.weightsDir
	}

	// File name format: weights_<modelIdentity>_<unixNano>
	var fileName = fmt.Sprintf("weights_%s_%d", o.model.Identity(), time.Now().UnixNano())
	f, err := os.Create(filepath.Join(weightsDir, fileName))
	if err != nil {
		return fmt.Errorf("failed creating file: %w", err)
	}
	defer f.Close()

	if _, err := f.Write(b); err != nil {
		return fmt.Errorf("failed writing model parameters to file: %w", err)
	}

	slog.Info("wrote weights to file", "filePath", filepath.Join(weightsDir, fileName))

	return nil
}

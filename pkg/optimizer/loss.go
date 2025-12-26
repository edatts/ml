package optimizer

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// 1/N * sum((y-ŷ)^2)
func MeanSquaredError(predictions [][]float32, actual [][]float32) (float64, error) {
	if len(actual) != len(predictions) {
		return 0, fmt.Errorf("mismatched lengths, len(predictions)='%d', len(actual)='%d'", len(predictions), len(actual))
	}

	if len(predictions) == 0 {
		return 0, errors.New("predictions length is zero")
	}

	var sumSquaredErrs float64
	for i, pred := range predictions {
		var sumSquaredErr float64
		for j, value := range pred {
			sumSquaredErr += math.Pow(float64(actual[i][j]-value), 2)
		}
		sumSquaredErrs += sumSquaredErr / float64(len(pred))
	}

	return sumSquaredErrs / float64(len(predictions)), nil
}

// - sum(y * ln(ŷ))
func CategoricalCrossEntropy(predictions [][]float32, targets [][]int) (float64, error) {
	if len(targets) != len(predictions) {
		return 0, fmt.Errorf("mismatched lengths, len(predictions)='%d', len(actual)='%d'", len(predictions), len(targets))
	}

	if len(predictions) == 0 {
		return 0, errors.New("predictions length is zero")
	}

	var correctIndices = make([]int, len(targets))
	for i, correct := range targets {
		correctIndices[i] = slices.Index(correct, 1)
	}

	var crossEntSum float64
	for i, prediction := range predictions {
		if prediction[correctIndices[i]] == 0 {
			prediction[correctIndices[i]] = 1e-10
		}
		// Since we are doing classification and the correct answer is a one-hot encoding
		// of the output vector, the formula simplifies to -ln(ŷ).
		crossEntSum += -math.Log(float64(prediction[correctIndices[i]]))
	}

	return crossEntSum / float64(len(predictions)), nil
}

func Accuracy(predictions [][]float32, y [][]int) float64 {
	var numCorrect int
	for i, pred := range predictions {
		idx := slices.Index(pred, slices.Max(pred))
		if y[i][idx] == 1 {
			numCorrect++
		}
	}
	return float64(numCorrect) / float64(len(predictions)) * 100
}

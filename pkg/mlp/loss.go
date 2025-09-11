package mlp

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// - sum(y * ln(ŷ))
func CategoricalCrossEntropy(predictions [][]float64, actual [][]int) (float64, error) {
	if len(actual) != len(predictions) {
		return 0, fmt.Errorf("mismatched lengths, len(predictions)='%d', len(actual)='%d'", len(predictions), len(actual))
	}

	if len(predictions) == 0 {
		return 0, errors.New("predictions length is zero")
	}

	var correctIndices = make([]int, len(actual))
	for i, correct := range actual {
		correctIndices[i] = slices.Index(correct, 1)
	}

	var crossEntSum float64
	for i, prediction := range predictions {
		if prediction[correctIndices[i]] == 0 {
			prediction[correctIndices[i]] = 1e-10
		}
		// Since we are doing classification and the correct answer is a one-hot encoding
		// of the output vector, the formula simplifies to -ln(ŷ).
		crossEntSum += -math.Log(prediction[correctIndices[i]])
	}

	return crossEntSum / float64(len(predictions)), nil
}

// 1/N * sum((y-ŷ)^2)
func MeanSquaredError(predictions [][]float64, actual [][]float64) (float64, error) {
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
			sumSquaredErr += math.Pow(actual[i][j]-value, 2)
		}
		sumSquaredErrs += sumSquaredErr / float64(len(pred))
	}

	return sumSquaredErrs / float64(len(predictions)), nil
}

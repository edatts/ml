package tabular

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAgentEncodeDecode(t *testing.T) {
	var (
		state1 = &SnakeState{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}
		state2 = &SnakeState{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
		state3 = &SnakeState{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
	)

	var initial = &agent[SnakeState, *SnakeState]{
		numActions:   4,
		stateSize:    16,
		learningRate: 0.1,
		lrDecay:      0.001,
		minLr:        0.001,
		gamma:        0.90,
		epsilon:      0.5,
		epsilonDecay: 0.005,
		minEpsilon:   0.0001,
		qValues: map[string][]float32{
			state1.String(): {1, 2, 3, 4},
			state2.String(): {4, 3, 2, 1},
			state3.String(): {1, 2, 2, 1},
		},
	}

	b, err := initial.Encode()
	require.NoError(t, err)

	decoded, err := DecodeAgent[SnakeState](b)
	require.NoError(t, err)

	require.Equal(t, initial, decoded)
}

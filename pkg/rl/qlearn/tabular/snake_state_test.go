package tabular_test

import (
	"encoding/json"
	"testing"

	qlearn "github.com/edatts/ml/pkg/rl/qlearn/tabular"
	"github.com/stretchr/testify/require"
)

func TestSnakeStateEncDec(t *testing.T) {
	var (
		raw     = qlearn.SnakeState{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}
		encoded = []byte{0b10101010, 0b10101010}
	)

	require.Equal(t, encoded, raw.Encode())

	var s qlearn.SnakeState
	require.NoError(t, s.Decode(encoded))
	require.Equal(t, raw, s)

}

func BenchmarkSnakeStateEncDec(b *testing.B) {
	var raw = qlearn.SnakeState{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}

	b.Run("json", func(b *testing.B) {
		// Validate the outputs before benching
		var s qlearn.SnakeState
		encoded, err := json.Marshal(raw)
		require.NoError(b, err)
		require.NoError(b, json.Unmarshal(encoded, &s))
		require.Equal(b, raw, s)

		for b.Loop() {
			encoded, _ := json.Marshal(raw)
			_ = json.Unmarshal(encoded, &s)
		}
	})

	b.Run("binary", func(b *testing.B) {
		// Validate the outputs before benching
		var s qlearn.SnakeState
		encoded := raw.Encode()
		require.NoError(b, s.Decode(encoded))
		require.Equal(b, raw, s)

		for b.Loop() {
			encoded := raw.Encode()
			_ = s.Decode(encoded)
		}
	})
}

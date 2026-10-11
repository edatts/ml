package qlearn_test

import (
	"encoding/json"
	"testing"

	"github.com/edatts/ml/pkg/rl/qlearn"
	"github.com/stretchr/testify/require"
)

func TestSnakeStateEncDec(t *testing.T) {
	var (
		raw     = qlearn.SnakeState{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}
		encoded = []byte{0b10101010, 0b10101010}
	)

	require.Equal(t, encoded, raw.Encode())

	var s qlearn.SnakeState
	decoded, err := s.Decode(encoded)
	require.NoError(t, err)
	require.Equal(t, raw, decoded)

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
		decoded, err := s.Decode(encoded)
		require.NoError(b, err)
		require.Equal(b, raw, decoded)

		for b.Loop() {
			encoded := raw.Encode()
			_, _ = s.Decode(encoded)
		}
	})
}

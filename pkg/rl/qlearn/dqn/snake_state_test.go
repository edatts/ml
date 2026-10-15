package dqn_test

import (
	"testing"

	"github.com/edatts/ml/pkg/games/snake/core"
	"github.com/edatts/ml/pkg/rl/qlearn/dqn"
	"github.com/stretchr/testify/require"
)

func TestSnakeState(t *testing.T) {
	var tilesData = []core.Tile{
		0, 0, 1, 0, 0, 0, 0, 0,
		0, 2, 0, 0, 0, 0, 0, 0,
		0, 3, 0, 0, 0, 0, 0, 0,
		0, 3, 3, 3, 0, 0, 0, 0,
		0, 0, 0, 3, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	var expected = []float32{
		1, 1, 1, 1, 1,
		1, 0, 0, 0.33333334, 0,
		1, 0, 0.6666667, 0, 0,
		1, 0, 1, 0, 0,
		1, 0, 1, 1, 1,

		1, 0, 0, 0,
		1, 0, 0, 1,
	}

	tiles, err := core.NewTiles(8, 8, tilesData)
	require.NoError(t, err)

	state := dqn.NewSnakeState(tiles, core.Up)

	require.Equal(t, expected, state.Data())
}

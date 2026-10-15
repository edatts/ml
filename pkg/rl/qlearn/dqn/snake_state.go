package dqn

import (
	"encoding/binary"
	"fmt"
	"math"

	snake "github.com/edatts/ml/pkg/games/snake/core"
	"github.com/edatts/ml/pkg/rl/qlearn"
)

// There are a number of ways we could represent the game state for use with
// our DQN. The main objective is to use a state representation that gives
// us much more information that in the tabular Q-learning agent.
//
// One such representation is to provide a constrained view of the board
// centred arond the head of the snake. This could be 5x5 or 7x7 and the
// walls, body, and food would simply move in and out of frame and move
// relative to the snakes head. This representation should also be much
// easier for a simple MLP to work with and learn from.
//
// Alternatively we could represent each different tile type on a separate
// input channel where the channel is the size of the entire board. This
// would keep spatial information for the head, food, and obstacles separated
// so that each tile type gets managed by a different sub-set of neurons.
//
// We could also exclusively track the coordinates of the active tiles and
// flatten that such that each coordinate is a single input element.
//

const dims = 9

var _ qlearn.State = &SnakeState{}

// Initially we're going to implement the head-centred sub-frame of the
// game board as the game state then we can experiment with others later.
type SnakeState []float32

func NewSnakeState(tiles snake.Tiles, direction snake.Direction) qlearn.State {
	var (
		area  = dims * dims
		state = make(SnakeState, area+8)
		diff  = dims / 2

		// We want to build the constrained view around this centre index.
		headIdx  = tiles.HeadIndex()
		foodIdx  = tiles.FoodIndex()
		x_h, y_h = tiles.X(headIdx), tiles.Y(headIdx)
		x_f, y_f = tiles.X(foodIdx), tiles.Y(foodIdx)
	)

	// To get the bounding box we want to take the coordinates that are
	// +\- (dims/2) from the head index. Any element with indices that
	// are out of range of the game board should have values set to the
	// same as the snake body so that the model views them as obstacles.
	x1, y1 := x_h-diff, y_h-diff
	// x2, y2 := x_h+diff, y_h+diff

	for i := range dims {
		for j := range dims {
			x, y := x1+i, y1+j

			if outOfBounds(x, y) {
				state[i+j*dims] = float32(snake.Body) / 3
			} else {
				state[i+j*dims] = float32(tiles.Get(tiles.Index(x1+i, y1+j))) / 3
			}
		}
	}

	// Append direction and food direction to state
	putDirections(state[area:], direction)
	putFoodDirections(state[area+4:], x_h, y_h, x_f, y_f)
	return &state
}

func outOfBounds(x, y int) bool {
	h, w := snake.Height, snake.Width
	return x >= w || x < 0 || y >= h || y < 0
}

func putDirections(dst []float32, d snake.Direction) {
	switch d {
	case snake.Up:
		dst[0] = 1
	case snake.Down:
		dst[1] = 1
	case snake.Left:
		dst[2] = 1
	case snake.Right:
		dst[3] = 1
	}
}

func putFoodDirections(dst []float32, x_h, y_h, x_f, y_f int) {
	if y_h > y_f {
		dst[0] = 1
	}

	if y_h < y_f {
		dst[1] = 1
	}

	if x_h > x_f {
		dst[2] = 1
	}

	if x_h < x_f {
		dst[3] = 1
	}
}

func (s SnakeState) Data() []float32 {
	return s
}

func (s SnakeState) Encode() []byte {
	var out = make([]byte, len(s)*4)
	for i, num := range s {
		binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(num))
	}
	return out
}

func (s *SnakeState) Decode(b []byte) error {
	if len(b)%4 != 0 {
		return fmt.Errorf("invalid unput length, bytes should be a multiple of 4")
	}

	var data = make(SnakeState, len(b)/4)
	for i := 0; i < len(b); i += 4 {
		data[i/4] = math.Float32frombits(binary.LittleEndian.Uint32(b[i : i+4]))
	}

	*s = data
	return nil
}

// This receiver is not really useful for this datatype...
func (s SnakeState) String() string {
	return ""
}

func (s *SnakeState) FromString(str string) error {
	return fmt.Errorf("unsupported for dqn snake state")
}

func (s SnakeState) IsTerminalAction(action int) bool {
	switch qlearn.SnakeActions[action] {
	case snake.Up:
		x := dims / 2
		y := dims/2 - 1
		return s[x+y*dims] == float32(snake.Body)/3
	case snake.Down:
		x := dims / 2
		y := dims/2 + 1
		return s[x+y*dims] == float32(snake.Body)/3
	case snake.Left:
		x := dims/2 - 1
		y := dims / 2
		return s[x+y*dims] == float32(snake.Body)/3
	case snake.Right:
		x := dims/2 + 1
		y := dims / 2
		return s[x+y*dims] == float32(snake.Body)/3
	default:
		panic(fmt.Sprintf("unknown action '%d'", action))
	}
}

package tabular

import (
	"fmt"
	"strconv"

	snake "github.com/edatts/ml/pkg/games/snake/core"
	"github.com/edatts/ml/pkg/rl/qlearn"
)

var _ qlearn.State = &SnakeState{}

// foodUp         bool
// foodDown       bool
// foodLeft       bool
// foodRight      bool
// obstacleUp     bool
// obstacleDown   bool
// obstacleLeft   bool
// obstacleRight  bool
// obstacleUp2    bool
// obstacleDown2  bool
// obstacleLeft2  bool
// obstacleRight2 bool
// directionUp    bool
// directionDown  bool
// directionLeft  bool
// directionRight bool
type SnakeState [16]uint8

func NewSnakeState(tiles snake.Tiles, d snake.Direction) qlearn.State {
	var out SnakeState
	out.Build(tiles, d)
	return &out
}

func (s SnakeState) IsTerminalAction(action int) bool {
	switch qlearn.SnakeActions[action] {
	case snake.Up:
		return s[4] == 1
	case snake.Down:
		return s[5] == 1
	case snake.Left:
		return s[6] == 1
	case snake.Right:
		return s[7] == 1
	default:
		panic(fmt.Sprintf("unknown action '%d'", action))
	}
}

func (s *SnakeState) FromString(str string) error {
	if len(str) != 16 {
		return fmt.Errorf("unsupported length '%d' for snake state, expected 16", len(str))
	}

	for i, elem := range str {
		val, err := strconv.ParseUint(string(elem), 10, 1)
		if err != nil {
			return fmt.Errorf("failed parsing element: %w", err)
		}

		s[i] = uint8(val)
	}

	return nil
}

func (s *SnakeState) Build(tiles snake.Tiles, d snake.Direction) {
	var (
		headIdx  = tiles.HeadIndex()
		foodIdx  = tiles.FoodIndex()
		x_h, y_h = tiles.X(headIdx), tiles.Y(headIdx)
		x_f, y_f = tiles.X(foodIdx), tiles.Y(foodIdx)
	)

	s.setFoodDirections(x_h, y_h, x_f, y_f)
	s.setImmediateObstacles(x_h, y_h, tiles)
	s.set2TileObstacles(x_h, y_h, tiles)
	s.setDirection(d)
}

func (s *SnakeState) setFoodDirections(x, y, x_food, y_food int) {
	s[0], s[1], s[2], s[3] = 0, 0, 0, 0

	if y > y_food {
		s[0] = 1
	}

	if y < y_food {
		s[1] = 1
	}

	if x > x_food {
		s[2] = 1
	}

	if x < x_food {
		s[3] = 1
	}
}

func (s *SnakeState) setImmediateObstacles(x, y int, tiles snake.Tiles) {
	var (
		upIdx    = tiles.Index(x, y-1)
		downIdx  = tiles.Index(x, y+1)
		leftIdx  = tiles.Index(x-1, y)
		rightIdx = tiles.Index(x+1, y)
	)

	s[4], s[5], s[6], s[7] = 0, 0, 0, 0

	if y == 0 { // Top wall
		s[4] = 1
	} else if tiles.Get(upIdx) == snake.Body {
		s[4] = 1
	}

	if y == 29 { // Bottom wall
		s[5] = 1
	} else if tiles.Get(downIdx) == snake.Body {
		s[5] = 1
	}

	if x == 0 { // Left Wall
		s[6] = 1
	} else if tiles.Get(leftIdx) == snake.Body {
		s[6] = 1
	}

	if x == 29 { // Right wall
		s[7] = 1
	} else if tiles.Get(rightIdx) == snake.Body {
		s[7] = 1
	}
}

func (s *SnakeState) set2TileObstacles(x, y int, tiles snake.Tiles) {
	var (
		upIdx    = tiles.Index(x, y-2)
		downIdx  = tiles.Index(x, y+2)
		leftIdx  = tiles.Index(x-2, y)
		rightIdx = tiles.Index(x+2, y)
	)

	s[8], s[9], s[10], s[11] = 0, 0, 0, 0

	if y-1 == 0 || y == 0 { // Top wall
		s[8] = 1
	} else if tiles.Get(upIdx) == snake.Body {
		s[8] = 1
	}

	if y+1 == 29 || y == 29 { // Bottom wall
		s[9] = 1
	} else if tiles.Get(downIdx) == snake.Body {
		s[9] = 1
	}

	if x-1 == 0 || x == 0 { // Left Wall
		s[10] = 1
	} else if tiles.Get(leftIdx) == snake.Body {
		s[10] = 1
	}

	if x+1 == 29 || x == 29 { // Right wall
		s[11] = 1
	} else if tiles.Get(rightIdx) == snake.Body {
		s[11] = 1
	}
}

func (s *SnakeState) setDirection(d snake.Direction) {
	switch d {
	case snake.Up:
		s[12], s[13], s[14], s[15] = 1, 0, 0, 0
	case snake.Down:
		s[12], s[13], s[14], s[15] = 0, 1, 0, 0
	case snake.Left:
		s[12], s[13], s[14], s[15] = 0, 0, 1, 0
	case snake.Right:
		s[12], s[13], s[14], s[15] = 0, 0, 0, 1
	}
}

func (s SnakeState) Data() []float32 {
	var out = make([]float32, len(s))
	for i, elem := range s {
		out[i] = float32(elem)
	}
	return out
}

func (s SnakeState) String() string {
	return fmt.Sprintf("%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d", s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15])
}

func (s SnakeState) Encode() []byte {
	var out = [2]byte{}
	for i := range out {
		var n uint8
		for j := i * 8; j < (i+1)*8; j++ {
			if s[j] == 1 {
				n = n | 1
			}

			if (j+1)%8 != 0 {
				n = n << 1
			}
		}

		out[i] = n
	}

	return out[:]
}

func (s *SnakeState) Decode(b []byte) error {
	if len(b) != 2 {
		return fmt.Errorf("invalid input length '%d'", len(b))
	}

	for i := range 2 {
		elem := b[i]
		for j := 7 + (i * 8); j >= i*8; j-- {
			s[j] = elem & 1
			elem = elem >> 1
		}
	}

	return nil
}

package qlearn

import (
	"math"

	snake "github.com/edatts/ml/pkg/games/snake/core"
)

var SnakeActions = [4]snake.Direction{snake.Up, snake.Down, snake.Left, snake.Right}

var _ Env[SnakeState] = &snakeEnv{}

func NewSnakeEnv() *snakeEnv {
	return &snakeEnv{
		core: snake.New(),
	}
}

type snakeEnv struct {
	core      *snake.Core
	state     SnakeState
	prevScore int
}

func (e *snakeEnv) Init() {
	e.core.Init()
}

func (e *snakeEnv) Reset() {
	e.core.Reset()
	e.prevScore = 0
}

func (e *snakeEnv) Step(action int) (float64, bool) {
	// Convert action index into a direction
	e.core.SetDirection(SnakeActions[action])

	// If game over issue a negative reward (we currently have
	// no "win" condition anyway so all terminal states should
	// issue a negative reward).
	if e.core.Step() {
		// tiles, d := e.core.State()
		// slog.Info("final direction", "direction", d, "action", action)
		// slog.Info("final state", "state", e.state.String())
		// tiles.Log()
		return -10, true
	}

	// If we got the food issue a positive reward
	if score := e.core.Score(); score > e.prevScore {
		e.prevScore = score
		return 20, false
	}

	tiles, _ := e.core.State()
	idx_h := tiles.HeadIndex()
	idx_f := tiles.FoodIndex()
	xFoodDistance := math.Abs(float64(tiles.X(idx_h) - tiles.X(idx_f)))
	yFoodDistance := math.Abs(float64(tiles.Y(idx_h) - tiles.Y(idx_f)))

	x_avg, y_avg := tiles.CentreOfMass()
	xBodyDistance := math.Abs(float64(tiles.X(idx_h)-x_avg)) + 1
	yBodyDistance := math.Abs(float64(tiles.Y(idx_h)-y_avg)) + 1

	var reward float64

	// We want to encourage the snake not to box itself in, overly
	// complicated reward functions can have unintended effects and
	// limit the agent's strategy formation. So we should stick to
	// simple rules.
	//
	// Here we add a positive reward proportional to the distance from
	// the body centre of mass.
	reward += ((0.0003 * xBodyDistance) + (0.0003 * yBodyDistance))

	// Let's apply a negative reward for every step but the size
	// of the reward will be proportional to the distance from
	// the food, this should encourage the agent to seek out food.
	reward -= ((0.0002 * xFoodDistance) + (0.0002 * yFoodDistance))

	// What if we add a massive negative reward when the snake has
	// boxed itself in? We could detect this by checking if there
	// is a valid path to the food. That could be checked here or
	// on termination of an episode when Step() returns true.

	return reward, false
}

// Here we want to convert the core state (the game tiles) into
// a state suitable for use with a Q-table. The number of discrete
// states for a 30x30 snake game is enormous so we simplify to the
// following state:
//
//	foodUp -> 0 or 1
//	foodDown -> 0 or 1
//	foodLeft -> 0 or 1
//	foodRight -> 0 or 1
//	obstacleUp -> 0 or 1
//	obstacleDown -> 0 or 1
//	obstacleLeft -> 0 or 1
//	obstacleRight -> 0 or 1
//	directionUp -> 0 or 1
//	directionDown -> 0 or 1
//	directionLeft -> 0 or 1
//	directionRight -> 0 or 1
//
// The maximum number of states for 12 bits of 2^12 (or 4096), but
// in our snake game this will be less since direction bits and
// certain food direction bits are mutually exclusive. This means
// that the observation space is a managable size for Q-learning.
func (e *snakeEnv) State() SnakeState {
	e.state.Build(e.core.State())
	return e.state
}

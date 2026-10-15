package qlearn

import (
	"fmt"
	"log/slog"

	snake "github.com/edatts/ml/pkg/games/snake/core"
)

var SnakeActions = [4]snake.Direction{snake.Up, snake.Down, snake.Left, snake.Right}

type SnakeStateFn = func(tiles snake.Tiles, direction snake.Direction) State
type SnakeRewardFn = func(gotFood, hitWall, hitSelf bool, tiles snake.Tiles)

type EnvOption func(*snakeEnv)

func WithStateConstructor(fn func(snake.Tiles, snake.Direction) State) EnvOption {
	return func(e *snakeEnv) {
		e.stateFn = fn
	}
}

func WithRewardFunc(fn SnakeRewardFn) EnvOption {
	return func(e *snakeEnv) {
		e.rewardFn = fn
	}
}

var _ Env = &snakeEnv{}

func NewSnakeEnv(optFns ...EnvOption) (*snakeEnv, error) {
	e := &snakeEnv{
		core: snake.New(),
	}

	for _, fn := range optFns {
		fn(e)
	}

	if e.stateFn == nil {
		return nil, fmt.Errorf("state constructor func is required")
	}

	return e, nil
}

type snakeEnv struct {
	core      *snake.Core
	state     State
	prevScore int

	stateFn  SnakeStateFn
	rewardFn SnakeRewardFn
}

func (e *snakeEnv) Init() {
	e.core.Init()
	e.state = e.stateFn(e.core.State())
}

func (e *snakeEnv) Reset() {
	e.core.Reset()
	e.prevScore = 0
}

// TODO: Implement reward shapig using state potentials and make
//
//	the reward function configurable.
func (e *snakeEnv) Step(action int) (float64, float64, bool) {
	defer func() {
		e.state = e.stateFn(e.core.State())
	}()

	// Convert action index into a direction
	e.core.SetDirection(SnakeActions[action])

	// Get state before we step the core sim.
	tiles, direction := e.core.State()

	// If game over issue a negative reward (we currently have
	// no "win" condition anyway so all terminal states should
	// issue a negative reward).
	if e.core.Step() {
		// tiles, d := e.core.State()
		// slog.Info("final direction", "direction", d, "action", action)
		// slog.Info("final state", "state", e.state.String())
		// tiles.Log()
		return -10, float64(e.core.Score()), true
	}

	// If we got the food issue a positive reward
	if score := e.core.Score(); score > e.prevScore {
		e.prevScore = score
		return 10, float64(e.core.Score()), false
	}

	// If we didn't die we apply a reward based on where we moved.
	// We use the state from before we stepped the simulation.
	headIdx := tiles.HeadIndex()
	foodIdx := tiles.FoodIndex()
	x_h, y_h := tiles.X(headIdx), tiles.Y(headIdx)
	x_f, y_f := tiles.X(foodIdx), tiles.Y(foodIdx)

	// xFoodDistance := math.Abs(float64(x_h - x_f))
	// yFoodDistance := math.Abs(float64(y_h - y_f))

	// x_avg, y_avg := tiles.CentreOfMass()
	// xBodyDistance := math.Abs(float64(tiles.X(idx_h)-x_avg)) + 1
	// yBodyDistance := math.Abs(float64(tiles.Y(idx_h)-y_avg)) + 1

	var reward float64

	// Here we apply a negative reward for every step but the size
	// of the reward will be proportional to the distance from
	// the food, this should encourage the agent to seek out food
	// while penalizing it for surviving without seeking out food.
	// reward -= ((0.005 * xFoodDistance) + (0.005 * yFoodDistance))
	//
	// Just try a flat negative reward
	reward -= 0.05

	// Here we add directional rewards to further encourage the
	// snake to move towards food but matched with a negative reward
	// for moving away from food so that it does not farm rewards by
	// moving close to food without eating it.
	switch direction {
	case snake.Up:
		if y_h > y_f {
			reward += 0.1
		} else if y_h < y_f {
			reward -= 0.1
		}
	case snake.Down:
		if y_h < y_f {
			reward += 0.1
		} else if y_h > y_f {
			reward -= 0.1
		}
	case snake.Left:
		if x_h > x_f {
			reward += 0.1
		} else if x_h < x_f {
			reward -= 0.1
		}
	case snake.Right:
		if x_h < x_f {
			reward += 0.1
		} else if x_h > x_f {
			reward -= 0.1
		}
	}

	// We will add a small reward for moving close to walls and
	// body segments. This encourages the snake to utilize it's
	// available space efficiently.
	// if y_h == 0 || tiles.Get(tiles.Index(x_h, y_h-1)) == 3 {
	// 	reward += 0.02
	// }

	// if y_h == 29 || tiles.Get(tiles.Index(x_h, y_h+1)) == 3 {
	// 	reward += 0.02
	// }

	// if x_h == 0 || tiles.Get(tiles.Index(x_h-1, y_h)) == 3 {
	// 	reward += 0.02
	// }

	// if x_h == 29 || tiles.Get(tiles.Index(x_h+1, y_h)) == 3 {
	// 	reward += 0.02
	// }

	return reward, float64(e.core.Score()), false
}

func (e *snakeEnv) State() State {
	return e.state
}

func (e *snakeEnv) Test(agent Agent) error {
	agent.SetTraining(false)
	e.Reset()

	var (
		bestScore    float64
		avgScore100  float64
		recentScores []float64
	)

	for i := range 1000 {
		var nSteps int
		var done bool
		var episodeScore float64
		for !done {
			action, err := agent.NextAction(e.State())
			if err != nil {
				return err
			}

			_, episodeScore, done = e.Step(action)
			nSteps++
			if nSteps >= 10000 {
				done = true
			}
		}

		e.Reset()
		bestScore = max(bestScore, episodeScore)
		recentScores = append(recentScores, episodeScore)
		if len(recentScores) == 101 {
			recentScores = recentScores[1:]
		}

		var sum float64
		for _, elem := range recentScores {
			sum += elem
		}
		avgScore100 = sum / 100

		if i%50 == 0 {
			slog.Info("testing agent", "bestScore", bestScore, "averageSore", avgScore100)
		}
	}

	return nil
}

package snake

import "github.com/edatts/ml/pkg/rl/qlearn"

type Speed string

const (
	Slow     Speed = "slow"
	Medium   Speed = "medium"
	Fast     Speed = "fast"
	VeryFast Speed = "very_fast"
	Unreal   Speed = "unreal"
)

type GameOption func(*Game)

func WithGameSpeed(s string) GameOption {
	return func(g *Game) {
		g.speed = Speed(s)
	}
}

func WithQLearningAgent(a qlearn.Agent) GameOption {
	return func(g *Game) {
		g.agent = a
	}
}

func WithStateFunc(fn qlearn.SnakeStateFn) GameOption {
	return func(g *Game) {
		g.stateFn = fn
	}
}

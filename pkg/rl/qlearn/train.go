package qlearn

import (
	"fmt"
	"log/slog"
	"os"
	"time"
)

type State interface {
	Data() []float32
	Encode() []byte
	Decode([]byte) error
	String() string
	FromString(string) error
	IsTerminalAction(int) bool
}

type Env interface {
	Init()
	Step(action int) (float64, float64, bool)
	State() State
	Reset()
	Test(agent Agent) error
}

type Params struct {
	ActN         uint16
	StateSize    uint16
	LearningRate float64
	LrDecay      float64
	MinLr        float64
	Gamma        float64
	Epsilon      float64
	EpsilonDecay float64
	MinEpsilon   float64
}

type opts struct {
	numEpisodes     int
	maxEpisodeSteps int
	saveAgent       bool
	modelType       string
}

type Option func(*opts)

func WithNumEpisodes(n int) Option {
	return func(o *opts) {
		o.numEpisodes = n
	}
}

func WithMaxEpisodeSteps(n int) Option {
	return func(o *opts) {
		o.maxEpisodeSteps = n
	}
}

func WithSaveAgent(saveAgent bool) Option {
	return func(o *opts) {
		o.saveAgent = saveAgent
	}
}

func WithModelType(mType string) Option {
	return func(o *opts) {
		o.modelType = mType
	}
}

func Train(agent Agent, env Env, optFns ...Option) error {
	opts := opts{
		numEpisodes:     100000,
		maxEpisodeSteps: 10000,
	}

	for _, fn := range optFns {
		fn(&opts)
	}

	var (
		numSteps     int
		numEpisodes  int
		totalRewards float64
		bestScore    float64
		avgScore100  float64
		recentScores []float64
	)

	env.Init()
	for range opts.numEpisodes {
		numEpisodes++
		numSteps = 0

		var done bool
		var episodeScore float64
		for !done {
			numSteps++

			var (
				reward float64
				state  = env.State()
			)

			action, err := agent.NextAction(state)
			if err != nil {
				return err
			}

			reward, episodeScore, done = env.Step(action)
			totalRewards += reward

			if err := agent.UpdateQValues(state, action, reward, env.State(), done); err != nil {
				return err
			}

			// if numSteps%5000 == 0 {
			// 	slog.Info("training info", "step", numSteps, "score", episodeScore, "bestScore", bestScore, "totalRewards", totalRewards)
			// }

			if numSteps == opts.maxEpisodeSteps {
				done = true
			}
		}

		env.Reset()
		agent.UpdateParameters()

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

		if numEpisodes%50 == 0 {
			slog.Info("episode complete", "episode", numEpisodes, "score", episodeScore, "avgScore", avgScore100, "stepsTaken", numSteps, "totalRewards", totalRewards, "epsilon", agent.Epsilon(), "lr", agent.LearningRate())
		}
	}

	slog.Info("training complete", "bestScore", bestScore, "avgScore", avgScore100, "finalEpsilon", agent.Epsilon())

	// At the end of training we should probably run some test simulations
	// with epsilon set to 0 so we can estimate the actual performance of
	// the agent.

	if err := env.Test(agent); err != nil {
		return err
	}

	if opts.saveAgent {
		fileName := fmt.Sprintf("qlearn_%s_snake_%d", opts.modelType, time.Now().Unix())
		f, err := os.Create(fileName)
		if err != nil {
			return fmt.Errorf("failed creating agent file: %w", err)
		}
		defer f.Close()

		b, err := agent.Encode()
		if err != nil {
			return fmt.Errorf("failed encoding agent: %w", err)
		}

		if _, err := f.Write(b); err != nil {
			return fmt.Errorf("failed writing agent bytes to file: %w", err)
		}
	}

	return nil
}

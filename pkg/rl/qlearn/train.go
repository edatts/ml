package qlearn

import (
	"fmt"
	"log/slog"
	"os"
	"time"
)

type State interface {
	Encode() []byte
	Decode([]byte) (State, error)
}

type ComparableState interface {
	State
	comparable
}

type Env[T ComparableState] interface {
	Init()
	Step(action int) (float64, bool)
	State() T
	Reset()
}

type opts struct {
	numEpisodes     int
	maxEpisodeSteps int
	saveAgent       bool
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

func Train[T ComparableState](agent *agent[T], env Env[T], optFns ...Option) error {
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
		bestScore    int
		avgScore100  int
		recentScores []int
	)

	env.Init()
	for range opts.numEpisodes {
		numEpisodes++
		numSteps = 0

		var done bool
		var episodeScore int
		for !done {
			numSteps++

			var (
				reward float64
				state  = env.State()
				action = agent.NextAction(state)
			)

			reward, done = env.Step(action)
			totalRewards += reward

			agent.UpdateQTable(state, action, reward, env.State())

			if reward >= 1 {
				episodeScore++
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

		var sum int
		for _, elem := range recentScores {
			sum += elem
		}
		avgScore100 = sum / 100

		if numEpisodes%500 == 0 {
			slog.Info("episode complete", "episode", numEpisodes, "score", episodeScore, "avgScore", avgScore100, "stepsTaken", numSteps, "totalRewards", totalRewards, "epsilon", agent.epsilon, "lr", agent.learningRate)
		}
	}

	slog.Info("training complete", "bestScore", bestScore, "avgScore", avgScore100, "finalEpsilon", agent.epsilon)

	if opts.saveAgent {
		fileName := fmt.Sprintf("qlearn_snake_%d", time.Now().Unix())
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

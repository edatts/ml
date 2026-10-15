package snakecmd

import (
	"fmt"

	qlearnagents "github.com/edatts/ml/agents/rl/qlearn"
	"github.com/edatts/ml/pkg/games/snake"
	"github.com/edatts/ml/pkg/rl/qlearn/dqn"
	"github.com/edatts/ml/pkg/rl/qlearn/tabular"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/spf13/cobra"
)

var Cmd = &cobra.Command{
	Use:   "snake",
	Short: "Runs the snake game",
	RunE: func(cmd *cobra.Command, _ []string) error {
		var opts = []snake.GameOption{}

		switch agent {
		case "qtable":
			a, err := tabular.DecodeAgent[tabular.SnakeState](qlearnagents.SnakeTabular)
			if err != nil {
				return fmt.Errorf("failed decoding Q-learning agent: %w", err)
			}

			opts = append(opts, snake.WithQLearningAgent(a))
			opts = append(opts, snake.WithStateFunc(tabular.NewSnakeState))
		case "dqn":
			a, err := dqn.DecodeAgent(qlearnagents.SnakeDQN)
			if err != nil {
				return fmt.Errorf("failed decoding DQN agent: %w", err)
			}

			opts = append(opts, snake.WithQLearningAgent(a))
			opts = append(opts, snake.WithStateFunc(dqn.NewSnakeState))
		}

		if speed != "" {
			opts = append(opts, snake.WithGameSpeed(speed))
		}

		return ebiten.RunGame(snake.NewGame(opts...))
	},
}

var (
	agent string
	speed string
)

func init() {
	Cmd.Flags().StringVar(&agent, "agent", "qtable", "Loads an agent trained with the specified algorithm to play the game")
	Cmd.Flags().StringVar(&speed, "speed", "fast", "Sets the game speed")
}

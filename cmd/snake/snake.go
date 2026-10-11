package snakecmd

import (
	"fmt"

	qlearnagents "github.com/edatts/ml/agents/rl/qlearn"
	"github.com/edatts/ml/pkg/games/snake"
	"github.com/edatts/ml/pkg/rl/qlearn"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/spf13/cobra"
)

var Cmd = &cobra.Command{
	Use:   "snake",
	Short: "Runs the snake game",
	RunE: func(cmd *cobra.Command, _ []string) error {
		var opts = []snake.GameOption{}

		switch agent {
		case "qlearn":
			a, err := qlearn.DecodeAgent[qlearn.SnakeState](qlearnagents.Snake)
			if err != nil {
				return fmt.Errorf("failed decoding Q-learning agent: %w", err)
			}

			opts = append(opts, snake.WithQLearningAgent(a))
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
	Cmd.Flags().StringVar(&agent, "agent", "", "Loads an agent trained with the specified algorithm to play the game")
	Cmd.Flags().StringVar(&speed, "speed", "fast", "Sets the game speed")
}

package snakecmd

import (
	"github.com/edatts/ml/pkg/rl/qlearn"
	"github.com/spf13/cobra"
)

var Cmd = &cobra.Command{
	Use:   "snake",
	Short: "train a Q-learning agent to play snake",
	RunE:  trainAgent,
}

var (
	saveAgent bool
)

func init() {
	Cmd.Flags().BoolVar(&saveAgent, "save-agent", false, "When set the trained agent is saved to the current working directory")
}

func trainAgent(cmd *cobra.Command, _ []string) error {
	agent := qlearn.NewAgent[qlearn.SnakeState](16, 4)
	env := qlearn.NewSnakeEnv()
	qlearn.Train(agent, env, qlearn.WithSaveAgent(saveAgent))
	return nil
}

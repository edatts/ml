package snakecmd

import (
	"fmt"

	"github.com/edatts/ml/pkg/mlp"
	"github.com/edatts/ml/pkg/rl/qlearn"
	"github.com/edatts/ml/pkg/rl/qlearn/dqn"
	"github.com/edatts/ml/pkg/rl/qlearn/tabular"
	"github.com/edatts/ml/pkg/types"
	"github.com/spf13/cobra"
)

var Cmd = &cobra.Command{
	Use:   "snake",
	Short: "train a Q-learning agent to play snake",
	RunE:  trainAgent,
}

var (
	saveAgent bool
	modelType string
)

func init() {
	Cmd.Flags().BoolVar(&saveAgent, "save-agent", false, "When set the trained agent is saved to the current working directory")
	Cmd.Flags().StringVar(&modelType, "model-type", "tabular", "The type of Q-learning model to train")
}

func trainAgent(cmd *cobra.Command, _ []string) error {
	var (
		agent   qlearn.Agent
		stateFn qlearn.SnakeStateFn
		err     error
	)

	switch modelType {
	case "tabular":
		agent = tabular.NewAgent[tabular.SnakeState](16, 4)
		agent.SetTraining(false)
		stateFn = tabular.NewSnakeState
	case "dqn":
		agent, err = buildDQNAgent()
		if err != nil {
			return fmt.Errorf("failed instantiating agent: %w", err)
		}
		agent.SetTraining(false)
		stateFn = dqn.NewSnakeState
	default:
		return fmt.Errorf("unknown model type '%s'", modelType)
	}

	env, err := qlearn.NewSnakeEnv(qlearn.WithStateConstructor(stateFn))
	if err != nil {
		return err
	}

	agent.SetTraining(true)

	return qlearn.Train(
		agent, env,
		qlearn.WithSaveAgent(saveAgent),
		qlearn.WithModelType(modelType),
		qlearn.WithNumEpisodes(15_000),
		qlearn.WithMaxEpisodeSteps(3000),
	)
}

func buildDQNAgent() (qlearn.Agent, error) {
	onlineModel, err := mlp.New(89, 4, mlp.WithHiddenLayers(256, 256, 128))
	if err != nil {
		return nil, fmt.Errorf("failed instantiating online model: %w", err)
	}

	targetModel, err := mlp.New(89, 4, mlp.WithHiddenLayers(256, 256, 128))
	if err != nil {
		return nil, fmt.Errorf("failed instantiating online model: %w", err)
	}

	copiedWeights, err := copyWeights(onlineModel.Weights())
	if err != nil {
		return nil, fmt.Errorf("failed copying weights: %w", err)
	}

	if err := targetModel.LoadWeights(copiedWeights); err != nil {
		return nil, fmt.Errorf("failed loading weights into target model")
	}

	return dqn.NewAgent(onlineModel, targetModel, 4)
}

func copyWeights(tensors []types.Tensor) ([]types.Tensor, error) {
	var out = make([]types.Tensor, len(tensors))
	for i, t := range tensors {
		tensor := types.Tensor{
			Name:     t.Name,
			Shape:    t.Shape, // Don't think we need to copy the slice data in shape...
			DataType: t.DataType,
		}

		switch t.DataType {
		case types.F32:
			orig, ok := t.FP32Data()
			if !ok {
				return nil, fmt.Errorf("unexpected data type in original tensor '%s'", t.Name)
			}

			d := &types.F32Data{
				Data: make([]float32, len(orig)),
			}

			copy(d.Data, orig)

			tensor.Data = d
		default:
			return nil, fmt.Errorf("unsupported data type '%s'", t.DataType)
		}

		out[i] = tensor
	}

	return out, nil
}

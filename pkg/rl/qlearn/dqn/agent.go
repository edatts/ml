package dqn

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/edatts/ml/pkg/mlp"
	"github.com/edatts/ml/pkg/model"
	"github.com/edatts/ml/pkg/optimizer"
	"github.com/edatts/ml/pkg/rl/qlearn"
	"github.com/edatts/ml/pkg/safetensors"
	"github.com/edatts/ml/pkg/types"
)

// Now we are going to implement a Deep Q-Leaning agent. There are a few
// differences between this agent and a tabular Q-learning agent.
//
// Firstly, instead of storing Q values for every discrete state in a Q
// table we use a Neural Network to estimate the Q values for a particular
// state. This neural network is optimized to minimize the loss between
// the predicted Q values and the Q values resulting from the reward + the
// predicted Q values for the next state. The prediction for Q values of
// the next state are made with the same network but using a frozen set
// of weights from a previous training step, the weights of this target
// network are updated infrequently to assist with numerical stability.
//
// As an alternative to these "hard" updates we can use Polyak averaging
// to perform continuous "soft" updates of the target network weights
// using the formula: θ− = τθ+(1−τ)θ− Where tau is a very small number
// like 0.001 or 0.005.
//
// Additionally, we store past exteriences in a replay buffer so that
// we can train the model on mini-batches of past experiences. This
// allows us to train the network on individual experiences multiple
// times instead of only training on the current step. Thie also breaks
// temporal correlation between steps and allows the network to learn
// long-term strategies and avoid "forgetting" ones learned earlier in
// training. We will need to seed the replay buffer before we start
// sampling from it so that we have some initial data to work with.
//

const warmupPeriod = 10000

var _ qlearn.Agent = &agent{}

type agent struct {
	actN         uint16
	learningRate float64
	lrDecay      float64
	minLr        float64
	gamma        float64
	epsilon      float64
	epsilonDecay float64
	minEpsilon   float64
	batchSize    int
	regFactor    float64
	momentum     float64
	tau          float64
	isTraining   bool

	optimizeInterval int
	nextOptimization int

	replay *replayBuffer

	onlineModel model.Model
	targetModel model.Model
}

func NewAgent(model, targetModel model.Model, actN uint16) (*agent, error) {
	if model.Identity() != targetModel.Identity() {
		return nil, fmt.Errorf("online and target networks are incompatible")
	}

	return &agent{
		actN:         actN,
		learningRate: 0.0002,
		lrDecay:      0,
		minLr:        0.0001,
		gamma:        0.99,
		epsilon:      1,
		epsilonDecay: 0.00035,
		minEpsilon:   0.10,
		batchSize:    32,
		regFactor:    1e-4,
		momentum:     0.95,
		tau:          0.005,

		optimizeInterval: 8,

		replay:      NewReplayBuffer(500_000),
		onlineModel: model,
		targetModel: targetModel,
	}, nil
}

func (a *agent) SetTraining(isTraining bool) {
	a.isTraining = isTraining
}

func (a *agent) NextAction(state qlearn.State) (int, error) {
	// We're going to apply some action masking to the epsilon
	// case of our action selecting, to do this we need to check
	// the state and strip out actions from random selection that
	// would result in episodes ending early. If all actions
	// result in an end of the episode then we don't apply the
	// mask.
	//
	// Instead of getting the index of a random action we will
	// store the raw values of valid actions and select one at
	// random.

	if rand.Float64() < a.epsilon && a.isTraining {
		// Pick random action with mask
		if a.isWarmingUp() {
			return rand.IntN(int(a.actN)), nil
		}

		var validActions []int
		for i := range int(a.actN) {
			if !state.IsTerminalAction(int(i)) {
				validActions = append(validActions, i)
			} else if rand.IntN(100) < 3 { // Allow 3% of terminal actions.
				validActions = append(validActions, i)
			}
		}

		if len(validActions) == 0 { // All actions terminal, skip mask
			return rand.IntN(int(a.actN)), nil
		}

		return validActions[rand.IntN(len(validActions))], nil
	} else {
		// Return max predicted Q value
		batch := [][]float32{state.Data()}
		output, err := a.onlineModel.Forward(batch, true)
		if err != nil {
			return 0, err
		}

		return slices.Index(output[0], slices.Max(output[0])), nil
	}
}

func (a *agent) UpdateQValues(state qlearn.State, action int, reward float64, nextState qlearn.State, isTerminal bool) error {
	// This is where we will want to calculate gradients to train the model.
	// We want to use the following loss function (with MSE):
	//
	//	Loss =  ((R + gamma * Q(S',A')) - Q(S,A))^2

	if !a.isTraining {
		return fmt.Errorf("agent is not in training mode")
	}

	// Add the experience to the replay buffer and get batch
	a.replay.Push(state, nextState, action, reward, isTerminal)

	if a.isWarmingUp() {
		return nil
	}

	if a.nextOptimization != 0 {
		a.nextOptimization--
		return nil
	}
	a.nextOptimization = a.optimizeInterval

	experiences, err := a.replay.Batch(a.batchSize)
	if err != nil {
		return err
	}

	var (
		states     = experiences.states
		actions    = experiences.actions
		rewards    = experiences.rewards
		nextStates = experiences.nextStates
		terminal   = experiences.isTerminal
	)

	// Get predictions
	outputs, err := a.onlineModel.Forward(states, false)
	if err != nil {
		return fmt.Errorf("failed forward: %w", err)
	}

	// Get target prediction
	targetOutputs, err := a.targetModel.Forward(nextStates, false)
	if err != nil {
		return fmt.Errorf("failed next states forward: %w", err)
	}

	// Now we need to use the respective actions for each experience
	// to select the relevant Q values for the loss calculation. We
	// only select one Q value per sample to use for backprop.
	var preds = make([][]float32, a.batchSize)
	var targets = make([][]float32, a.batchSize)
	for i := range a.batchSize {
		maxTargetIdx := slices.Index(targetOutputs[i], slices.Max(targetOutputs[i]))

		var terminalFactor float32 = 1
		if terminal[i] {
			terminalFactor = 0
		}

		preds[i] = []float32{outputs[i][actions[i]]}
		targets[i] = []float32{float32(rewards[i]) + (float32(a.gamma) * targetOutputs[i][maxTargetIdx] * terminalFactor)}
	}

	// Calculate error
	loss, err := optimizer.MeanSquaredError(preds, targets)
	if err != nil {
		return err
	}

	// We don't really need to calc the loss because we manually
	// calculate the derivative of the loss for MSE below...
	_ = loss

	// Derivative of the MSE loss simplifies to 2(ŷ - y)
	var dCdA = make([][]float32, len(preds))
	for i, pred := range preds {
		dCdA[i] = make([]float32, len(outputs[i]))

		// Only the gradient corresponding to the action Q-value
		// will be non zero at the output layer. All other grads
		// are zero because their corresponding actions are not
		// evaluated by the loss function.
		dCdA[i][actions[i]] = 2 * (pred[0] - targets[i][0])
	}

	// TODO: Update MLP to add momentum to SGD (we should probably also
	// 		 move the gradient calculations and weight updates out of
	// 		 the model implementations and into the optimizer...)
	if err := a.onlineModel.Backward(dCdA, a.learningRate, a.regFactor, a.momentum); err != nil {
		return fmt.Errorf("failed backward: %w", err)
	}

	// Let's skip hard updates of the target network and go straight to
	// soft updates using Polyak averaging.
	//	θ− = τθ+(1−τ)θ−

	// These methods don't copy the data so modifying the slices in the
	// tensors modifies the underling arrays.
	weights := a.onlineModel.Weights()
	targetWeights := a.targetModel.Weights()

	for i, tensor := range weights {
		if err := polyakAverage(tensor, targetWeights[i], a.tau); err != nil {
			return fmt.Errorf("failed averaging weights for target network: %w", err)
		}
	}

	// TODO: Add some proper logging for the NN training for loss etc...

	return nil
}

func polyakAverage(onlineTensor, targetTensor types.Tensor, tau float64) error {
	// Polyak averaging uses the following formula:
	//	θ− = τθ+(1−τ)θ−
	if onlineTensor.DataType != targetTensor.DataType {
		return fmt.Errorf("mismatched data type for tensors: online='%s'='%s', target='%s'='%s'", onlineTensor.Name, onlineTensor.DataType, targetTensor.Name, targetTensor.DataType)
	}

	switch onlineTensor.DataType {
	case types.F32:
		weights, ok := onlineTensor.FP32Data()
		if !ok {
			return fmt.Errorf("data type is not FP32 for tensor '%s'", onlineTensor.Name)
		}

		targetWeights, _ := targetTensor.FP32Data()
		if !ok {
			return fmt.Errorf("data type is not FP32 for tensor '%s'", targetTensor.Name)
		}

		for i, w := range weights {
			targetWeights[i] = float32(tau)*w + float32(1-tau)*targetWeights[i]
		}

	// TODO: Add FP16 support
	case types.F16:
		fallthrough
	default:
		return fmt.Errorf("unsupported data type '%s'", onlineTensor.DataType)
	}

	return nil
}

func (a *agent) UpdateParameters() {
	if a.isWarmingUp() {
		return
	}

	a.epsilon = max(a.minEpsilon, a.epsilon-(a.epsilon*a.epsilonDecay))
	a.learningRate = max(a.minLr, a.learningRate-(a.learningRate*a.lrDecay))
}

func (a *agent) Encode() ([]byte, error) {
	var out []byte
	out = binary.LittleEndian.AppendUint16(out, a.actN)
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.learningRate))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.lrDecay))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.minLr))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.gamma))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.epsilon))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.epsilonDecay))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.minEpsilon))
	out = binary.LittleEndian.AppendUint64(out, uint64(a.batchSize))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.regFactor))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.momentum))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.tau))
	out = binary.LittleEndian.AppendUint64(out, uint64(a.optimizeInterval))

	// Do we need to serialize the replay buffer? If we want to continue
	// training then we should probably persist it... For now let's skip
	// the replay buffer.

	// Serialize the neural networks. The target network is only for training
	// so we can probably omitt it for now.
	modelBytes, err := safetensors.Format(a.onlineModel.Weights())
	if err != nil {
		return nil, fmt.Errorf("failed formatting network as safetensors: %w", err)
	}

	out = append(out, modelBytes...)

	return out, nil
}

func DecodeAgent(b []byte) (*agent, error) {
	var a = &agent{}
	a.actN = binary.LittleEndian.Uint16(b[0:2])
	a.learningRate = math.Float64frombits(binary.LittleEndian.Uint64(b[2:10]))
	a.lrDecay = math.Float64frombits(binary.LittleEndian.Uint64(b[10:18]))
	a.minLr = math.Float64frombits(binary.LittleEndian.Uint64(b[18:26]))
	a.gamma = math.Float64frombits(binary.LittleEndian.Uint64(b[26:34]))
	a.epsilon = math.Float64frombits(binary.LittleEndian.Uint64(b[34:42]))
	a.epsilonDecay = math.Float64frombits(binary.LittleEndian.Uint64(b[42:50]))
	a.minEpsilon = math.Float64frombits(binary.LittleEndian.Uint64(b[50:58]))
	a.batchSize = int(binary.LittleEndian.Uint64(b[58:66]))
	a.regFactor = math.Float64frombits(binary.LittleEndian.Uint64(b[66:74]))
	a.momentum = math.Float64frombits(binary.LittleEndian.Uint64(b[74:82]))
	a.tau = math.Float64frombits(binary.LittleEndian.Uint64(b[82:90]))
	a.optimizeInterval = int(binary.LittleEndian.Uint64(b[90:98]))

	// Decode model
	tensors, err := safetensors.Parse(b[98:])
	if err != nil {
		return nil, fmt.Errorf("failed parsing safetensors model: %w", err)
	}

	// For now load into the hardcoded model but shortly after testing we
	// need to add metadata to the safetensors format so we can properly
	// decode and instantiate the model...
	m, err := mlp.New(89, 4, mlp.WithHiddenLayers(256, 256, 128))
	if err != nil {
		return nil, fmt.Errorf("failed instantiating new MLP: %w", err)
	}

	if err := m.LoadWeights(tensors); err != nil {
		return nil, fmt.Errorf("failed loading weights: %w", err)
	}

	a.onlineModel = m
	return a, nil
}

func (a *agent) Epsilon() float64 {
	return a.epsilon
}

func (a *agent) LearningRate() float64 {
	return a.learningRate
}

func (a *agent) isWarmingUp() bool {
	return a.replay.size < warmupPeriod
}

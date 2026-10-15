package tabular

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/edatts/ml/pkg/rl/qlearn"
)

var _ qlearn.Agent = &agent[SnakeState, *SnakeState]{}

type agent[T any, PT qlearn.AgentState[T]] struct {
	numActions   uint16
	stateSize    uint16
	learningRate float64
	lrDecay      float64
	minLr        float64
	gamma        float64
	epsilon      float64
	epsilonDecay float64
	minEpsilon   float64
	isTraining   bool
	qValues      map[string][]float32
}

func NewAgent[T any, PT qlearn.AgentState[T]](stateSize, actN uint16) *agent[T, PT] {
	return &agent[T, PT]{
		numActions:   actN,
		stateSize:    stateSize,
		learningRate: 0.60,
		lrDecay:      0.00004,
		minLr:        0.001,
		gamma:        0.95,
		epsilon:      0.75,
		epsilonDecay: 0.000085,
		minEpsilon:   0.00001,
		qValues:      make(map[string][]float32),
	}
}

func (a *agent[T, PT]) SetTraining(isTraining bool) {
	a.isTraining = isTraining
}

func (a *agent[T, PT]) NextAction(state qlearn.State) (int, error) {
	// Based on epsilon value we either pick a random action
	// or select the highest Q value for the current state.

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
		var validActions []int
		for i := range a.numActions {
			if !state.IsTerminalAction(int(i)) {
				validActions = append(validActions, int(i))
			}
		}

		if len(validActions) == 0 { // Al actions terminal, skip mask
			return rand.IntN(int(a.numActions)), nil
		}

		return validActions[rand.IntN(len(validActions))], nil
	} else {
		// Pick action with largest Q value
		values, ok := a.qValues[state.String()]
		if !ok {
			values = make([]float32, a.numActions)
			a.qValues[state.String()] = values
		}
		return slices.Index(values, slices.Max(values)), nil
	}
}

func (a *agent[T, PT]) UpdateQValues(state qlearn.State, action int, reward float64, nextState qlearn.State, isTerminal bool) error {
	// Here we want to update our Q values using the temporal difference
	// equaiton which approximates the bellman equation:
	//
	// Q(S,A) = Q(S,A) + lr * (R + gamma * Q(S',A') - Q(S,A))
	//

	values, ok := a.qValues[state.String()]
	if !ok {
		values = make([]float32, a.numActions)
		a.qValues[state.String()] = values
	}

	currentQValue := float64(values[action])

	nextValues, ok := a.qValues[nextState.String()]
	if !ok {
		nextValues = make([]float32, a.numActions)
		a.qValues[nextState.String()] = nextValues
	}

	var terminalFactor float64 = 1
	if isTerminal {
		terminalFactor = 0
	}

	nextQValue := float64(nextValues[slices.Index(nextValues, slices.Max(nextValues))])
	values[action] = float32(currentQValue + a.learningRate*(reward+a.gamma*nextQValue*terminalFactor-currentQValue))
	return nil
}

func (a *agent[T, PT]) UpdateParameters() {
	a.epsilon = max(a.minEpsilon, a.epsilon-(a.epsilon*a.epsilonDecay))
	a.learningRate = max(a.minLr, a.learningRate-(a.learningRate*a.lrDecay))
}

func (a *agent[T, PT]) Epsilon() float64 {
	return a.epsilon
}

func (a *agent[T, PT]) LearningRate() float64 {
	return a.learningRate
}

func (a *agent[T, PT]) MarshalJSON() ([]byte, error) {
	return nil, errors.New("MarshalJSON not supported for agent[T State]")

	// var intermediate = jsonAgent{
	// 	NumActions:   a.numActions,
	// 	StateSize:    a.stateSize,
	// 	LearningRate: a.learningRate,
	// 	LrDecay:      a.lrDecay,
	// 	MinLr:        a.minLr,
	// 	Gamma:        a.gamma,
	// 	Epsilon:      a.epsilon,
	// 	EpsilonDecay: a.epsilonDecay,
	// 	MinEpsilon:   a.minEpsilon,
	// 	QValues:      a.qValues,
	// }

	// return json.Marshal(intermediate)
}

func (a *agent[T, PT]) UnmarshalJSON(b []byte) error {
	return errors.New("UnmarshalJSON not supported for agent[T State]")
	// var intermediate jsonAgent
	// if err := json.Unmarshal(b, &intermediate); err != nil {
	// 	return err
	// }

	// a.numActions = intermediate.NumActions
	// a.stateSize = intermediate.StateSize
	// a.learningRate = intermediate.LearningRate
	// a.lrDecay = intermediate.LrDecay
	// a.minLr = intermediate.MinLr
	// a.gamma = intermediate.Gamma
	// a.epsilon = intermediate.Epsilon
	// a.epsilonDecay = intermediate.EpsilonDecay
	// a.minEpsilon = intermediate.MinEpsilon
	// a.qValues = intermediate.QValues
	// return nil
}

// type jsonAgent struct {
// 	NumActions   uint16          `json:"num_actions"`
// 	StateSize    uint16          `json:"state_size"`
// 	LearningRate float64         `json:"learning_rate"`
// 	LrDecay      float64         `json:"lr_decay"`
// 	MinLr        float64         `json:"min_lr"`
// 	Gamma        float64         `json:"gamma"`
// 	Epsilon      float64         `json:"epsilon"`
// 	EpsilonDecay float64         `json:"epsilon_decay"`
// 	MinEpsilon   float64         `json:"min_epsilon"`
// 	QValues      map[State][]float32 `json:"q_values"`
// }

// Encode encodes the agent into a binary representation. Originally JSON
// was used to persist the agent but binary marshalling was chosen as it
// results in a smaller storage profile. This is desirable since we keep
// copies of the agents in the repo and embed them in our go binaries.
func (a *agent[T, PT]) Encode() ([]byte, error) {
	// We should really have a metadata header that allows us to determine
	// if a serialized agent is compatible when decoding. Then we can fail
	// the decode instead of loading a corrupted model...

	var out []byte
	out = binary.LittleEndian.AppendUint16(out, a.numActions)
	out = binary.LittleEndian.AppendUint16(out, a.stateSize)
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.learningRate))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.lrDecay))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.minLr))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.gamma))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.epsilon))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.epsilonDecay))
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(a.minEpsilon))

	// If we change the size of the state or action space then we won't be able
	// to load old agents unless we track version and keep old implementations...
	var s = PT(new(T))
	for k, v := range a.qValues {
		if err := s.FromString(k); err != nil {
			return nil, fmt.Errorf("failed parsing state string '%s': %w", k, err)
		}

		out = append(out, s.Encode()...)
		for _, fp := range v {
			out = binary.LittleEndian.AppendUint32(out, math.Float32bits(fp))
		}
	}

	return out, nil
}

func DecodeAgent[T any, PT qlearn.AgentState[T]](b []byte) (*agent[T, PT], error) {
	var a = &agent[T, PT]{}
	a.numActions = binary.LittleEndian.Uint16(b[0:2])
	a.stateSize = binary.LittleEndian.Uint16(b[2:4])
	a.learningRate = math.Float64frombits(binary.LittleEndian.Uint64(b[4:12]))
	a.lrDecay = math.Float64frombits(binary.LittleEndian.Uint64(b[12:20]))
	a.minLr = math.Float64frombits(binary.LittleEndian.Uint64(b[20:28]))
	a.gamma = math.Float64frombits(binary.LittleEndian.Uint64(b[28:36]))
	a.epsilon = math.Float64frombits(binary.LittleEndian.Uint64(b[36:44]))
	a.epsilonDecay = math.Float64frombits(binary.LittleEndian.Uint64(b[44:52]))
	a.minEpsilon = math.Float64frombits(binary.LittleEndian.Uint64(b[52:60]))
	a.qValues = make(map[string][]float32)

	var (
		// Ceiling division: ceil(a/b) == (a+b-1) / b
		stateByteLen   = int((a.stateSize + 7) / 8)
		qValuesByteLen = int(a.numActions) * 4
	)

	var state = PT(new(T))
	var entryByteLen = stateByteLen + qValuesByteLen // 4 bytes per f32
	for i := 60; i < len(b); i += entryByteLen {
		var (
			qValuesStartIdx = i + stateByteLen
			stateBytes      = b[i:qValuesStartIdx]
			qValuesBytes    = b[qValuesStartIdx : qValuesStartIdx+qValuesByteLen]
		)

		if err := state.Decode(stateBytes); err != nil {
			return nil, fmt.Errorf("failed decoding state: %w", err)
		}

		qValues := make([]float32, a.numActions)
		for j := range a.numActions {
			qValues[j] = math.Float32frombits(binary.LittleEndian.Uint32(qValuesBytes[j*4:]))
		}

		a.qValues[state.String()] = qValues
	}

	return a, nil
}

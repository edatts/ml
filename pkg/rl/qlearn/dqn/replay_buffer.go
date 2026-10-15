package dqn

import (
	"fmt"
	"math/rand/v2"

	"github.com/edatts/ml/pkg/rl/qlearn"
)

type experience struct {
	state      qlearn.State
	action     int
	reward     float64
	nextState  qlearn.State
	isTerminal bool
}

type experiences struct {
	states     [][]float32
	actions    []int
	rewards    []float64
	nextStates [][]float32
	isTerminal []bool
}

type replayBuffer struct {
	data []experience
	size int
	cap  int
	head int
}

func NewReplayBuffer(cap int) *replayBuffer {
	return &replayBuffer{
		data: make([]experience, cap),
		cap:  cap,
	}
}

func (b *replayBuffer) Push(state, nextState qlearn.State, action int, reward float64, isTerminal bool) {
	b.data[b.head] = experience{state: state, action: action, reward: reward, nextState: nextState, isTerminal: isTerminal}
	b.head = (b.head + 1) % b.cap

	if b.size < b.cap {
		b.size++
	}
}

func (b *replayBuffer) Batch(n int) (experiences, error) {
	if n > b.size {
		return experiences{}, fmt.Errorf("batch size '%d' is larger than availble samples '%d'", n, b.size)
	}

	var batch = experiences{
		states:     make([][]float32, n),
		actions:    make([]int, n),
		rewards:    make([]float64, n),
		nextStates: make([][]float32, n),
		isTerminal: make([]bool, n),
	}

	for i := range n {
		idx := rand.IntN(b.size)
		batch.states[i] = b.data[idx].state.Data()
		batch.actions[i] = b.data[idx].action
		batch.rewards[i] = b.data[idx].reward
		batch.nextStates[i] = b.data[idx].nextState.Data()
		batch.isTerminal[i] = b.data[idx].isTerminal
	}

	return batch, nil
}

package qlearn

type AgentState[B any] interface {
	State
	*B
}

type Agent interface {
	NextAction(state State) (int, error)
	UpdateQValues(state State, action int, reward float64, nextState State, isTerminal bool) error
	UpdateParameters()
	Encode() ([]byte, error)

	// Params
	Epsilon() float64
	LearningRate() float64

	SetTraining(bool)
}

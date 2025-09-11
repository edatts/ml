package mlp

import (
	"log/slog"
)

type LayerType int

const (
	Input LayerType = iota
	Hidden
	Output
)

type Layer struct {
	prev    *Layer
	lType   LayerType
	Neurons []*Neuron
}

func newLayer(lType LayerType, n int16, prev *Layer, lambda float64) *Layer {
	return &Layer{
		prev:    prev,
		lType:   lType,
		Neurons: newNeurons(n, prev, lType, lambda),
	}
}

func (l *Layer) Forward(newBatch bool, batchLen int) {
	for _, neuron := range l.Neurons {
		neuron.Forward(newBatch, batchLen)
	}
}

func (l *Layer) Backward() {
	for _, neuron := range l.Neurons {
		neuron.Backward()
	}
}

func (l *Layer) Update(lr float64) {
	for _, neuron := range l.Neurons {
		neuron.Update(lr)
	}
}

func (l *Layer) logWweights() {
	var w []float64
	for _, neuron := range l.Neurons {
		for _, conn := range neuron.Inputs {
			w = append(w, conn.Weight)
		}
	}
	slog.Info("weights", "weights", w[:5])
}

func (l *Layer) logGrads() {
	var g []float64
	for _, neuron := range l.Neurons {
		for _, grad := range neuron.dCdz {
			g = append(g, grad)
		}
	}
	slog.Info("grads", "grads", g)
}

func (l *Layer) logActivations() {
	var a []float64
	for _, neuron := range l.Neurons {
		for _, act := range neuron.activations {
			a = append(a, act)
		}
	}
	slog.Info("activations", "acts", a)
}

func (l *Layer) logLossDeriv() {
	var dCdA []float64
	for _, neuron := range l.Neurons {
		for _, val := range neuron.dCdA {
			dCdA = append(dCdA, val)
		}
	}
	slog.Info("Loss deriv", "dCdA", dCdA)
}

func (l *Layer) logLogits() {
	var logits []float64
	for _, neuron := range l.Neurons {
		for _, val := range neuron.logits {
			logits = append(logits, val)
		}
	}
	slog.Info("Logits", "logits", logits)
}

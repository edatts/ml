package optimizer

import "github.com/edatts/ml/pkg/model"

type Config struct {
	numEpochs    int
	batchSize    int
	learningRate float64
	saveWeights  bool
}

func NewConfig() Config {
	return Config{
		numEpochs:    10,
		batchSize:    32,
		learningRate: 0.01,
	}
}

func WithNumEpochs(epochs int) option {
	return func(o *optimizer) {
		o.cfg.numEpochs = epochs
	}
}

func WithBatchSize(size int) option {
	return func(o *optimizer) {
		o.cfg.batchSize = size
	}
}

func WithLearningRate(lr float64) option {
	return func(o *optimizer) {
		o.cfg.learningRate = lr
	}
}

func WithSaveWeights() option {
	return func(o *optimizer) {
		o.cfg.saveWeights = true
	}
}

func WithModel(model model.Model) option {
	return func(o *optimizer) {
		o.model = model
	}
}

func WithDataProvider(prov func() ([]float32, any, error)) option {
	return func(o *optimizer) {
		o.dataProvider = prov
	}
}

func WithSampler(sampler Sampler) option {
	return func(o *optimizer) {
		o.sampler = sampler
	}
}

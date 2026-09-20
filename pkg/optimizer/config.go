package optimizer

import "github.com/edatts/ml/pkg/model"

type Config struct {
	numEpochs           int
	batchSize           int
	learningRate        float64
	lrDecay             float64
	lambda              float64
	trainLogWindow      int
	saveFinalWeights    bool
	loadModelWeights    bool
	classification      bool
	checkpointInterval  int
	weightsDir          string
	momentumCoefficient float64
}

func NewConfig() Config {
	return Config{
		numEpochs:           10,
		batchSize:           32,
		learningRate:        0.01,
		lambda:              2.5e-5,
		trainLogWindow:      20,
		momentumCoefficient: 0, // default to 0 for now
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

func WithLearningRateDecay(decay float64) option {
	return func(o *optimizer) {
		o.cfg.lrDecay = decay
	}
}

func WithRegularizationFactor(lambda float64) option {
	return func(o *optimizer) {
		o.cfg.lambda = lambda
	}
}

func WithLoggingInterval(numBatches int) option {
	return func(o *optimizer) {
		o.cfg.trainLogWindow = numBatches
	}
}

func WithSaveFinalWeights() option {
	return func(o *optimizer) {
		o.cfg.saveFinalWeights = true
	}
}

func WithClassification() option {
	return func(o *optimizer) {
		o.cfg.classification = true
	}
}

func WithTrainDataProvider(prov func() ([][]float32, any, error)) option {
	return func(o *optimizer) {
		o.trainDataProvider = prov
	}
}

func WithTestDataProvider(prov func() ([][]float32, any, error)) option {
	return func(o *optimizer) {
		o.testDataProvider = prov
	}
}

func WithSampler(sampler Sampler) option {
	return func(o *optimizer) {
		o.sampler = sampler
	}
}

func WithCheckpoints(epochs int) option {
	return func(o *optimizer) {
		o.cfg.checkpointInterval = epochs
	}
}

func WithWeightsDir(dir string) option {
	return func(o *optimizer) {
		o.cfg.weightsDir = dir
	}
}

func WithModel(model model.Model) option {
	return func(o *optimizer) {
		o.model = model
	}
}

func WithLoadModelWeights() option {
	return func(o *optimizer) {
		o.cfg.loadModelWeights = true
	}
}

func WithMomentumCoefficient(beta float64) option {
	return func(o *optimizer) {
		o.cfg.momentumCoefficient = beta
	}
}

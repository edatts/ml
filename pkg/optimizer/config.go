package optimizer

type Config struct {
	numEpochs      int
	batchSize      int
	learningRate   float64
	lrDecay        float64
	lambda         float64
	trainLogWindow int
	saveWeights    bool
	classification bool
}

func NewConfig() Config {
	return Config{
		numEpochs:      10,
		batchSize:      32,
		learningRate:   0.01,
		trainLogWindow: 20,
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

func WithSaveWeights() option {
	return func(o *optimizer) {
		o.cfg.saveWeights = true
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

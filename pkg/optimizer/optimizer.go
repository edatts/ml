package optimizer

import (
	"fmt"
	"log/slog"

	"github.com/edatts/ml/pkg/model"
)

type Optimizer interface {
	Run() error
}

type dataProvider func() ([][]float32, any, error)

type optimizer struct {
	cfg               Config
	model             model.Model
	trainDataProvider dataProvider
	testDataProvider  dataProvider
	sampler           Sampler
	currentRun        TrainingRun
}

type option func(o *optimizer)

func New(optFns ...option) *optimizer {
	o := &optimizer{
		cfg: NewConfig(),
	}

	for _, fn := range optFns {
		fn(o)
	}

	return o
}

func (o *optimizer) Run(model model.Model) error {
	if o.currentRun != nil {
		return ErrRunInProgress
	}

	if model == nil {
		return ErrNoModel
	}

	if o.sampler == nil {
		return ErrNoSampler
	}

	if o.trainDataProvider == nil {
		return fmt.Errorf("train data: %w", ErrNoDataProvider)
	}

	if o.testDataProvider == nil {
		return fmt.Errorf("test data: %w", ErrNoDataProvider)
	}

	o.model = model

	if err := o.sampler.Init(o.cfg.batchSize, o.trainDataProvider); err != nil {
		return fmt.Errorf("failed initializing sampler: %w", err)
	}

	// The optimizer will run until completion. The stopping condition will be
	// a certain number of epochs, we could implement early stopping later on.
	if err := o.train(); err != nil {
		return fmt.Errorf("failed training: %w", err)
	}

	if err := o.test(); err != nil {
		return fmt.Errorf("failed testing: %w", err)
	}

	if err := o.saveWeights(); err != nil {
		return fmt.Errorf("failed saving weights: %w", err)
	}

	return nil
}

func (o *optimizer) train() error {
	if err := o.initTrainingRun(); err != nil {
		return fmt.Errorf("failed initializing training run: %w", err)
	}

	for epoch := o.sampler.NewEpoch(); epoch < o.cfg.numEpochs; epoch = o.sampler.NewEpoch() {
		for o.sampler.Next() { // Returns false at end of epoch
			if err := o.currentRun.Step(); err != nil {
				return fmt.Errorf("failed step: %w", err)
			}
		}

		if err := o.sampler.Err(); err != nil {
			return fmt.Errorf("failed sampling: %w", err)
		}
	}

	slog.Info("maximum number of epochs reached, finished training")

	// TODO: Write the model to disk. Add configuration for storage directory.
	// if err := o.persistModel(); err != nil {
	// 	return fmt.Errorf("failed persisting model: %w", err)
	// }

	return nil
}

func (o *optimizer) initTrainingRun() error {
	if o.cfg.classification {
		sampler, ok := o.sampler.(ClassificationSampler)
		if !ok {
			return fmt.Errorf("failed asserting sampler to ClassificationSampler, type of sampler is %T", o.sampler)
		}

		o.currentRun = &ClassificationRun{
			optimizer:           o,
			sampler:             sampler,
			currentLearningRate: o.cfg.learningRate,
			lrDecay:             o.cfg.lrDecay,
		}
		return nil
	}

	slog.Info("classification option not set, defaulting to regression")

	sampler, ok := o.sampler.(RegressionSampler)
	if !ok {
		return fmt.Errorf("failed asserting sampler to RegressionSampler, type of sampler is %T", o.sampler)
	}

	o.currentRun = &RegressionRun{
		optimizer:           o,
		sampler:             sampler,
		currentLearningRate: o.cfg.learningRate,
		lrDecay:             o.cfg.lrDecay,
	}

	return nil
}

func (o *optimizer) Classify(inputs [][]float32, targets [][]int) ([][]float32, [][]float32, float64, float64, error) {
	outputs, err := o.model.Forward(inputs)
	if err != nil {
		return nil, nil, 0, 0, fmt.Errorf("failed forward pass: %w", err)
	}

	// Could make this configurable? Are there any other common calssification loss algorithms?
	loss, err := CategoricalCrossEntropy(outputs, targets)
	if err != nil {
		return nil, nil, 0, 0, fmt.Errorf("failed calculating categorical cross entropy: %w", err)
	}

	// Derivative of the loss simplifies to ŷ - y, including the SoftMax activaiton.
	var dCdA = make([][]float32, len(outputs))
	for i, pred := range outputs {
		dCdA[i] = make([]float32, len(outputs[i]))
		for j, actual := range targets[i] {
			dCdA[i][j] = pred[j] - float32(actual)
		}
	}

	regLoss := 0.5 * o.cfg.lambda * o.model.SumSquaredWeights()

	return outputs, dCdA, Accuracy(outputs, targets), loss + regLoss, nil
}

func (o *optimizer) Regress(inputs [][]float32, targets [][]float32) ([][]float32, [][]float32, float64, float64, error) {
	outputs, err := o.model.Forward(inputs)
	if err != nil {
		return nil, nil, 0, 0, fmt.Errorf("failed forward pass: %w", err)
	}

	// TODO: Should make this configurable.
	loss, err := MeanSquaredError(outputs, targets)
	if err != nil {
		return nil, nil, 0, 0, fmt.Errorf("failed calculating mean squared error: %w", err)
	}

	// Derivative of the loss simplifies to 2(ŷ - y)
	var dCdA = make([][]float32, len(outputs))
	for i, pred := range outputs {
		dCdA[i] = make([]float32, len(outputs[i]))
		for j, actual := range targets[i] {
			dCdA[i][j] = 2 * (pred[j] - actual)
		}
	}

	regLoss := 0.5 * o.cfg.lambda * o.model.SumSquaredWeights()

	return outputs, dCdA, loss, loss + regLoss, nil
}

func (o *optimizer) test() error {
	// panic("optimizer.test() is unimplemented")

	X, Y, err := o.testDataProvider()
	if err != nil {
		return fmt.Errorf("failed getting test data: %w", err)
	}

	if o.cfg.classification {
		targets, ok := Y.([][]int)
		if !ok {
			return fmt.Errorf("failed asserting test targets to [][]int, type of data is %T", Y)
		}

		outputs, grads, acc, loss, err := o.Classify(X, targets)
		if err != nil {
			return fmt.Errorf("failed classifying: %w", err)
		}

	}

	return nil
}

func (o *optimizer) saveWeights() error {
	if !o.cfg.saveWeights {
		return nil
	}

	// Write weights to disk. What format? Where on disk?

	return nil
}

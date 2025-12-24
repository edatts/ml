package optimizer

import (
	"fmt"

	"github.com/edatts/ml/pkg/model"
)

type Optimizer interface {
	Run() error
}

type optimizer struct {
	cfg          Config
	model        model.Model
	dataProvider func() ([]float32, any, error)
	sampler      Sampler
}

type option func(o *optimizer)

func New(model model.Model, optFns ...option) *optimizer {
	o := &optimizer{
		cfg:   NewConfig(),
		model: model,
	}

	for _, fn := range optFns {
		fn(o)
	}

	return o
}

func (o *optimizer) Run() error {
	if o.model == nil {
		return ErrNoModel
	}

	if o.sampler == nil {
		return ErrNoSampler
	}

	if o.dataProvider == nil {
		return ErrNoDataProvider
	}

	o.sampler.init(o.dataProvider)

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
	if o.cfg.classification {
		return o.trainClassifier()
	}

	return o.trainRegressor()
}

func (o *optimizer) trainClassifier() error {
	sampler, ok := o.sampler.(*Sampler[int])
	if !ok {
		return fmt.Errorf("failed asserting sampler to *Sampler[float32, int], type of sampler is %T", o.sampler)
	}

	for epoch := 1; epoch < o.cfg.numEpochs; {
		sample, epoch := sampler.Next()

	}

	return nil
}

func (o *optimizer) trainRegressor() error {

	return nil
}

func (o *optimizer) test() error {
	panic("optimizer test is unimplemented")
	return nil
}

func (o *optimizer) saveWeights() error {
	if !o.cfg.saveWeights {
		return nil
	}

	// Write weights to disk. What format? Where on disk?

	return nil
}

package optimizer

import (
	"fmt"
	"log/slog"
)

type TrainingRun interface {
	Step() error
	isTrainingRun()
}

func NewTrainingRun(isClassification bool) TrainingRun {
	if isClassification {
		return &ClassificationRun{}
	}
	return &RegressionRun{}
}

var _ TrainingRun = &ClassificationRun{}

type ClassificationRun struct {
	optimizer           *optimizer
	sampler             ClassificationSampler
	epoch               int
	numBatches          int
	currentLearningRate float64
	lrDecay             float64
	accuracy            float64
	regLoss             float64
}

func (c *ClassificationRun) Step() error {
	if c.optimizer == nil {
		return fmt.Errorf("classification run has no parent optimizer")
	}

	if c.sampler == nil {
		return fmt.Errorf("classificaiton run: %w", ErrNoSampler)
	}

	batch, err := c.sampler.Batch()
	if err != nil {
		return err
	}

	_, dCdA, acc, regLoss, err := c.optimizer.Classify(batch.Inputs, batch.Targets)
	if err != nil {
		return fmt.Errorf("failed classifying: %w", err)
	}

	c.numBatches++
	c.accuracy += acc
	c.regLoss += regLoss
	if c.numBatches%c.optimizer.cfg.trainLogWindow == 0 {
		slog.Info("training info", "run", c)
	}

	// Backpropagate
	if err := c.optimizer.model.Backward(dCdA, c.currentLearningRate); err != nil {
		return fmt.Errorf("failed backpropagating: %w", err)
	}

	c.currentLearningRate *= (1 - c.lrDecay)
	return nil
}

func (c *ClassificationRun) isTrainingRun() {}

var _ slog.LogValuer = &ClassificationRun{}

func (c *ClassificationRun) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Any("epoch", c.epoch),
		slog.Any("batchNum", c.numBatches),
		slog.Any("lr", c.currentLearningRate),
		slog.Any("accuracy", c.accuracy),
		slog.Any("regLoss", c.regLoss),
	)
}

type RegressionRun struct {
	optimizer           *optimizer
	sampler             RegressionSampler
	epoch               int
	numBatches          int
	currentLearningRate float64
	lrDecay             float64
	loss                float64
	regLoss             float64
}

func (r *RegressionRun) Step() error {
	if r.optimizer == nil {
		return fmt.Errorf("regression run has no parent optimizer")
	}

	if r.sampler == nil {
		return fmt.Errorf("regression run: %w", ErrNoSampler)
	}

	batch, err := r.sampler.Batch()
	if err != nil {
		return err
	}

	_, dCdA, loss, regLoss, err := r.optimizer.Regress(batch.Inputs, batch.Targets)
	if err != nil {
		return fmt.Errorf("failed classifying: %w", err)
	}

	r.numBatches++
	r.loss += loss
	r.regLoss += regLoss
	if r.numBatches%r.optimizer.cfg.trainLogWindow == 0 {
		slog.Info("training info", "run", r)
	}

	// Backpropagate
	if err := r.optimizer.model.Backward(dCdA, r.currentLearningRate); err != nil {
		return fmt.Errorf("failed backpropagating: %w", err)
	}

	r.currentLearningRate *= (1 - r.lrDecay)
	return nil
}

func (r *RegressionRun) isTrainingRun() {}

func (r *RegressionRun) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Any("epoch", r.epoch),
		slog.Any("batchNum", r.numBatches),
		slog.Any("lr", r.currentLearningRate),
		slog.Any("loss", r.loss),
		slog.Any("regLoss", r.regLoss),
	)
}

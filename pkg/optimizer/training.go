package optimizer

import (
	"fmt"
	"log/slog"
)

type TrainingRun interface {
	NewEpoch()
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
	numBatches          int
	epoch               int
	epochBatches        int
	currentLearningRate float64
	lrDecay             float64

	lastAccuracy float64
	lastRegLoss  float64
	accuracySum  float64
	regLossSum   float64
}

func (c *ClassificationRun) NewEpoch() {
	c.epoch++
	c.epochBatches = 0
	c.accuracySum = 0
	c.regLossSum = 0
}

func (c *ClassificationRun) Step() error {
	if c.optimizer == nil {
		return fmt.Errorf("classification run has no parent optimizer")
	}

	if c.sampler == nil {
		return fmt.Errorf("classification run: %w", ErrNoSampler)
	}

	batch, err := c.sampler.Batch()
	if err != nil {
		return err
	}

	// t := time.Now()

	_, dCdA, acc, regLoss, err := c.optimizer.Classify(batch.Inputs, batch.Targets, false)
	if err != nil {
		return fmt.Errorf("failed classifying: %w", err)
	}

	c.numBatches++
	c.epochBatches++
	c.lastAccuracy = acc
	c.lastRegLoss = regLoss
	c.accuracySum += acc
	c.regLossSum += regLoss
	if c.numBatches%c.optimizer.cfg.trainLogWindow == 0 {
		slog.Info("training info", "run", c)
	}

	// Backpropagate
	if err := c.optimizer.model.Backward(
		dCdA,
		c.currentLearningRate,
		c.optimizer.cfg.lambda,
		c.optimizer.cfg.momentumCoefficient,
	); err != nil {
		return fmt.Errorf("failed backpropagating: %w", err)
	}

	// Recent values pre-optimization:
	//	- Batch Size 64: 0.4304s
	// 	- Batch Size 128: 0.8735s
	//
	// After applying Im2Col to kernels once in backprop:
	//	- Batch Size 64: 0.1103s
	// 	- Batch Size 128: 0.2250s
	//
	// After adding some parallelization:
	//	- Batch Size 64: 0.0426s
	// 	- Batch Size 128: 0.0833s
	//
	// if c.numBatches%c.optimizer.cfg.trainLogWindow == 0 {
	// 	slog.Info("time to process batch", "batchSize", len(batch.Inputs), "timeInSeconds", time.Since(t).Seconds())
	// }

	c.currentLearningRate *= (1 - c.lrDecay)
	return nil
}

func (c *ClassificationRun) isTrainingRun() {}

var _ slog.LogValuer = &ClassificationRun{}

func (c *ClassificationRun) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Any("epoch", c.epoch),
		slog.Any("batchNum", c.numBatches),
		slog.Any("lr", fmt.Sprintf("%.6f", c.currentLearningRate)),
		// slog.Any("lastAccuracy", c.lastAccuracy),
		slog.Any("epochAccuracy", fmt.Sprintf("%.6f", c.accuracySum/float64(c.epochBatches))),
		// slog.Any("lastRegLoss", c.lastRegLoss),
		slog.Any("epochRegLoss", fmt.Sprintf("%.6f", c.regLossSum/float64(c.epochBatches))),
	)
}

type RegressionRun struct {
	optimizer           *optimizer
	sampler             RegressionSampler
	numBatches          int
	epoch               int
	epochBatches        int
	currentLearningRate float64
	lrDecay             float64

	lastLoss    float64
	lossSum     float64
	lastRegLoss float64
	regLossSum  float64
}

func (r *RegressionRun) NewEpoch() {
	r.epoch++
	r.epochBatches = 0
	r.lossSum = 0
	r.regLossSum = 0
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

	_, dCdA, loss, regLoss, err := r.optimizer.Regress(batch.Inputs, batch.Targets, false)
	if err != nil {
		return fmt.Errorf("failed regressing: %w", err)
	}

	r.numBatches++
	r.epochBatches++
	r.lastLoss = loss
	r.lastRegLoss = regLoss
	r.lossSum += loss
	r.regLossSum += regLoss
	if r.numBatches%r.optimizer.cfg.trainLogWindow == 0 {
		slog.Info("training info", "run", r)
	}

	// Backpropagate
	if err := r.optimizer.model.Backward(
		dCdA,
		r.currentLearningRate,
		r.optimizer.cfg.lambda,
		r.optimizer.cfg.momentumCoefficient,
	); err != nil {
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
		// slog.Any("lastLoss", r.lastLoss),
		slog.Any("epochLoss", r.lossSum/float64(r.epochBatches)),
		// slog.Any("lastRegLoss", r.lastRegLoss),
		slog.Any("epochRegLoss", r.regLossSum/float64(r.epochBatches)),
	)
}

package optimizer

import (
	"fmt"
	"math/rand/v2"
)

// TODO: Implement different sampling methods:
//	- Convenience
//	- Stratified
//	- Repeated Sampling of Random Subsets

// TODO: Implement a streaming sampler:
// 	- Should buffer batches in memory
//	- Sampling methods should be applied before streaming
//	- Should maintain data order when batching
// 	- Add Close() receiver to gracefully stop streaming

type Sampler interface {
	NewEpoch() int
	Init(batchSize int, dataProvider func() ([][]float32, any, error)) error
	Next() bool
	Err() error
	isSampler()
}

type ClassificationSampler interface {
	Sampler
	Batch() (Batch[int], error)
}

type RegressionSampler interface {
	Sampler
	Batch() (Batch[float32], error)
}

type baseSampler[YT Numeric] struct {
	initialized bool
	data        Data[YT]
	epoch       int
	batchSize   int
	batchOffset int
	batch       *Batch[YT]
	err         error
}

func (s *baseSampler[YT]) Init(batchSize int, provider func() ([][]float32, any, error)) error {
	if batchSize == 0 {
		return ErrBatchSize
	}
	s.batchSize = batchSize
	s.data = Data[YT]{}

	samples, targets, err := provider()
	if err != nil {
		return fmt.Errorf("failed getting data: %w", err)
	}

	s.data.X = samples

	var ok bool
	s.data.Y, ok = targets.([][]YT)
	if !ok {
		return fmt.Errorf("dataset targets are wrong type, got %T, expected %T", targets, *new(YT))
	}

	if len(s.data.X) != len(s.data.Y) {
		return fmt.Errorf("samples and targets are different lengths, len(samples)=%d, len(targets)=%d", len(s.data.X), len(s.data.Y))
	}

	if len(s.data.X) < s.batchSize {
		return fmt.Errorf("batch size to large, must be less than number of samples")
	}

	s.initialized = true
	return nil
}

func (s *baseSampler[YT]) Batch() (Batch[YT], error) {
	if s.batch == nil {
		return Batch[YT]{}, ErrNoBatch
	}

	return *s.batch, nil
}

func (s *baseSampler[YT]) Err() error {
	return s.err
}

var _ Sampler = &DefaultSampler[int]{}
var _ ClassificationSampler = &DefaultSampler[int]{}
var _ RegressionSampler = &DefaultSampler[float32]{}

// The default sampler will simply run through the entire dataset sequentially
// for each epoch. Often referred to as convenience sampling.
type DefaultSampler[YT Numeric] struct {
	baseSampler[YT]
}

func NewConvenienceSampler[YT Numeric]() Sampler {
	return &DefaultSampler[YT]{}
}

func (s *DefaultSampler[YT]) NewEpoch() int {
	s.batchOffset = 0
	s.batch = nil
	s.epoch++
	return s.epoch
}

func (s *DefaultSampler[YT]) Next() bool {
	if !s.initialized {
		s.err = ErrSamplerNotInitialized
		return false
	}

	var (
		start = s.batchOffset
		end   = s.batchOffset + s.batchSize
	)

	if start >= len(s.data.X) {
		// Start new epoch when Epoch called.
		return false
	}

	// Prepare next batch
	s.batch = &Batch[YT]{
		Inputs:  s.data.X[start:min(end, len(s.data.X))],
		Targets: s.data.Y[start:min(end, len(s.data.X))],
	}

	s.batchOffset += s.batchSize
	return true
}

func (s *DefaultSampler[YT]) isSampler() {}

var _ Sampler = &RS2Sampler[int]{}
var _ ClassificationSampler = &RS2Sampler[int]{}
var _ RegressionSampler = &RS2Sampler[float32]{}

type RS2Sampler[YT Numeric] struct {
	baseSampler[YT]
	subsetRatio float64
}

func NewRS2Sampler[YT Numeric](subsetRatio float64) Sampler {
	return &RS2Sampler[YT]{
		subsetRatio: subsetRatio,
	}
}

// In this implementation we shuffle the dataset
func (s *RS2Sampler[YT]) NewEpoch() int {
	s.batchOffset = 0
	s.batch = nil
	rand.Shuffle(len(s.data.X), func(i, j int) {
		s.data.X[i], s.data.X[j] = s.data.X[j], s.data.X[i]
		s.data.Y[i], s.data.Y[j] = s.data.Y[j], s.data.Y[i]
	})
	s.epoch++
	return s.epoch
}

func (s RS2Sampler[YT]) Next() bool {
	if !s.initialized {
		s.err = ErrSamplerNotInitialized
		return false
	}

	if s.subsetRatio < 0.01 || s.subsetRatio > 1 {
		s.err = ErrInvalidSubsetRatio
		return false
	}

	var (
		start = s.batchOffset
		end   = s.batchOffset + s.batchSize
	)

	if start >= len(s.data.X)*int(s.subsetRatio) {
		// Subset fully consumed, end epoch.
		return false
	}

	s.batch = &Batch[YT]{
		Inputs:  s.data.X[start:min(end, len(s.data.X))],
		Targets: s.data.Y[start:min(end, len(s.data.X))],
	}

	s.batchOffset += s.batchSize
	return true
}

func (s *RS2Sampler[YT]) isSampler() {}

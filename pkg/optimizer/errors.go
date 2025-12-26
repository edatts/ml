package optimizer

import "errors"

var (
	ErrRunInProgress         = errors.New("a training run is already in progress")
	ErrNoModel               = errors.New("no model provided, model is nil")
	ErrNoSampler             = errors.New("no sampler has been loaded")
	ErrNoDataProvider        = errors.New("data provider is nil")
	ErrSamplerNotInitialized = errors.New("sampler used before initialization")
	ErrBatchSize             = errors.New("batch size must be positive and non-zero")
	ErrNoBatch               = errors.New("the sampler does not have a batch available")
	ErrInvalidSubsetRatio    = errors.New("sampling subset ratio for RS2 must be between 0.01 and 1 inclusive")
)

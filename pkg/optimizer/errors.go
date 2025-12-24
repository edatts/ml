package optimizer

import "errors"

var (
	ErrNoModel        = errors.New("no model has been loaded")
	ErrNoSampler      = errors.New("no sampler has been loaded")
	ErrNoDataProvider = errors.New("data provider is nil")
)

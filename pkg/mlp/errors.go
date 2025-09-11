package mlp

import "errors"

var (
	ErrNotEnoughLayers  = errors.New("the model does not have enough layers")
	ErrEmptyInputData   = errors.New("no input data provided in batch")
	ErrInvalidInputSize = errors.New("input data is not the correct size")
)

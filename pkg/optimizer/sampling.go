package optimizer

import "fmt"

type Epocher interface {
	Epoch() int
}

type Sampler interface {
	Epocher

	Init(dataProvider func() ([]float32, any, error)) error

	isSampler()
}

type CalssificationSampler interface {
	Epocher

	Next() []Batch[int]
}

type RegressionSampler interface {
	Epocher

	Next() []Batch[float32]
}

// // We are going to delegate handling of the epochs to the sampler
// type Sampler[XT, YT Numeric] interface {
// 	Next() Batch[XT, YT]
// 	Epoch() int
// }

var _ Sampler = &DefaultSampler[int]{}

// The default sampler will simply run through the entire dataset sequentially
// for each epoch.
type DefaultSampler[YT Numeric] struct {
	data  Data[YT]
	epoch int
}

// func NewDefault() *DefaultSampler[float64, int] {
// 	return &DefaultSampler[float64, int]{}
// }

func (s *DefaultSampler[YT]) init(provider func() ([]float32, any, error)) error {
	data := Data[YT]{}

	samples, targets, err := provider()
	if err != nil {
		return fmt.Errorf("failed getting data: %w", err)
	}

	data.X = samples

	var ok bool
	data.Y, ok = targets.([]YT)
	if !ok {
		return fmt.Errorf("dataset targets are wrong type, got %T, expected %T", targets, *new(YT))
	}

	s.data = data
	return nil
}

func (s *DefaultSampler[YT]) Next() Batch[YT] {
	panic("default sampler next is unimplemented")
}

func (s *DefaultSampler[YT]) Epoch() int {
	return s.epoch
}

func (s *DefaultSampler[YT]) isSampler() {}

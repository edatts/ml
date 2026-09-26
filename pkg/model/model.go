package model

import (
	"github.com/edatts/ml/pkg/shape"
	"github.com/x448/float16"
)

type Model interface {
	Forward(batch [][]float32, isTest bool) ([][]float32, error)
	Backward(dCdA [][]float32, lr, lambda, beta float64) error
	SumSquaredWeights() float64
	Weights() []Tensor
	LoadWeights([]Tensor) error
	Identity() string
}

// TODO: Pull these out into separate pkg and use them everywhere...
// type Shape [4]int // NCHW

// func (s Shape) Samples() int {
// 	return s[0]
// }

// func (s Shape) Channels() int {
// 	return s[1]
// }

// func (s Shape) Height() int {
// 	return s[2]
// }

// func (s Shape) Width() int {
// 	return s[3]
// }

// func (s Shape) Bytes() []byte {
// 	var b []byte
// 	for _, n := range s {
// 		if n != 0 {
// 			b = binary.LittleEndian.AppendUint64(b, uint64(n))
// 		}
// 	}
// 	return b
// }

// func (s Shape) MarshalJSON() ([]byte, error) {
// 	var intermediate = []int{}
// 	for _, x := range s {
// 		if x != 0 {
// 			intermediate = append(intermediate, x)
// 		}
// 	}
// 	return json.Marshal(intermediate)
// }

// func (s *Shape) UnmarshalJSON(b []byte) error {
// 	var intermediate = []int{}
// 	if err := json.Unmarshal(b, &intermediate); err != nil {
// 		return err
// 	}

// 	if len(intermediate) > 4 {
// 		return fmt.Errorf("cannot unmarshal, maximum rank is 4, input rank is %d", len(intermediate))
// 	}

// 	diff := 4 - len(intermediate)
// 	for i := diff; i < 4; i++ {
// 		s[i] = intermediate[i-diff]
// 	}

// 	return nil
// }

type Float16 = float16.Float16

type Numeric interface {
	~int | ~float32 | Float16
}

// type Tensor[T Numeric] interface {
// 	Name() string
// 	Data() []T
// 	Shape() Shape
// }

// var _ Tensor[float32] = &tensor{}

type Tensor struct {
	Name  string
	Shape shape.Shape
	Data  []float32
}

// func (t *Tensor) Name() string {
// 	return t.name
// }

// func (t *Tensor) Data() []float32 {
// 	return t.data
// }

// func (t *Tensor) Shape() Shape {
// 	return t.shape
// }

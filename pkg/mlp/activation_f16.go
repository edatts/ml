package mlp

import (
	"math"

	"slices"

	"github.com/edatts/ml/pkg/types"
	"github.com/x448/float16"
)

type F16ActivationFunc interface {
	Forward(types.Float16) types.Float16
	Backward(types.Float16) types.Float16
}

type F16NoOp struct{}

func (n F16NoOp) Forward(in types.Float16) types.Float16 {
	return in
}

func (n F16NoOp) Backward(in types.Float16) types.Float16 {
	return float16.Fromfloat32(1)
}

// y = 1/(1+e^-x)
type F16Sigmoid struct{}

func (s F16Sigmoid) Forward(in types.Float16) types.Float16 {
	return float16.Fromfloat32(1 / (1 + float32(math.Exp(-float64(in.Float32())))))
}

func (s F16Sigmoid) Backward(in types.Float16) types.Float16 {
	return float16.Fromfloat32(in.Float32() * (1 - in.Float32()))
}

type F16ReLU struct{}

func (r F16ReLU) Forward(in types.Float16) types.Float16 {
	if in.Float32() > 0 {
		return in
	}
	return 0
}

func (r F16ReLU) Backward(in types.Float16) types.Float16 {
	if in.Float32() > 0 {
		return 1
	}
	return 0
}

type F16SoftMax struct{}

func (s F16SoftMax) Forward(in []types.Float16) []types.Float16 {
	if len(in) == 0 {
		return in
	}

	var f32In = make([]float32, len(in))
	for i, elem := range in {
		f32In[i] = elem.Float32()
	}

	var (
		outF32 = make([]float32, len(in))
		out    = make([]types.Float16, len(in))
		max    = slices.Max(f32In)
		sum    float32
	)

	for i, val := range f32In {
		expVal := float32(math.Exp(float64(val - max)))
		outF32[i] = expVal
		sum += expVal
	}

	// fmt.Printf("outF32: %v\n\n", outF32)

	// Nomalize
	var outF32Norm = make([]float32, len(outF32))
	for i, val := range outF32 {
		out[i] = float16.Fromfloat32(val / sum)
		outF32Norm[i] = val / sum
	}

	// fmt.Printf("outF32 norm: %v\n\n", outF32Norm)

	return out
}

// This simply returns slice of 1s because we have already incorporated
// the derivative of softmax when we set dCdA in the Classify receiver.
func (s F16SoftMax) Backward(in []types.Float16) []types.Float16 {
	return slices.Repeat([]types.Float16{float16.Fromfloat32(1)}, len(in))
}

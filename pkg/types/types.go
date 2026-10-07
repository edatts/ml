package types

import (
	"encoding/binary"
	"math"

	"github.com/edatts/ml/pkg/shape"
	"github.com/x448/float16"
)

type Float16 = float16.Float16

type Numeric interface {
	~float32 | Float16
}

type DataType = string

const (
	F16 DataType = "F16"
	F32 DataType = "F32"
)

type Data interface {
	Len() int
	Bytes() []byte
	ByteLen() int
	isData()
}

type Tensor struct {
	Name     string
	Shape    shape.Shape
	DataType DataType
	Data     Data
}

func (t Tensor) FP32Data() ([]float32, bool) {
	if x, ok := t.Data.(*F32Data); ok {
		return x.Data, true
	}
	return nil, false
}

func (t Tensor) FP16Data() ([]Float16, bool) {
	if x, ok := t.Data.(*F16Data); ok {
		return x.Data, true
	}
	return nil, false
}

type F32Data struct {
	Data []float32
}

func (d *F32Data) Len() int {
	return len(d.Data)
}

func (d *F32Data) Bytes() []byte {
	var out = make([]byte, len(d.Data)*4)
	for i, elem := range d.Data {
		binary.LittleEndian.PutUint32(out[i:i+4], math.Float32bits(elem))
	}
	return out
}

func (d *F32Data) ByteLen() int {
	return len(d.Data) * 4
}

type F16Data struct {
	Data []Float16
}

func (d *F16Data) Len() int {
	return len(d.Data)
}

func (d *F16Data) Bytes() []byte {
	var out = make([]byte, len(d.Data)*2)
	for i, elem := range d.Data {
		binary.LittleEndian.PutUint16(out[i:i+2], uint16(elem))
	}
	return out
}

func (d *F16Data) ByteLen() int {
	return len(d.Data) * 2
}

func (*F32Data) isData() {}
func (*F16Data) isData() {}

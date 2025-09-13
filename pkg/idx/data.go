package idx

import (
	"fmt"
	"go/types"
	"io"
	"log/slog"
)

type Data interface {
	Type() types.BasicKind
	GetInt() [][]int
	GetFloat32() [][]float32
	GetFloat64() [][]float64
}

type data struct {
	content  any
	dataType types.BasicKind
}

func NewData(numRows, numCols int, dataType types.BasicKind, r io.Reader) (*data, error) {
	var data = &data{
		dataType: dataType,
	}

	slog.Info("new data", "numRows", numRows, "numCols", numCols, "dataType", types.Typ[dataType])

	var err error
	switch dataType {
	case types.Uint8:
		data.content, err = readUint8Content(numRows, numCols, r)
	case types.Int8:
		data.content, err = readInt8Content(numRows, numCols, r)
	case types.Int16:
		data.content, err = readInt16Content(numRows, numCols, r)
	case types.Int32:
		data.content, err = readInt32Content(numRows, numCols, r)
	case types.Float32:
		data.content, err = readFloat32Content(numRows, numCols, r)
	case types.Float64:
		data.content, err = readFloat64Content(numRows, numCols, r)
	default:
		return nil, fmt.Errorf("unsupported data type '%s'", types.Typ[dataType].Name())
	}

	if err != nil {
		return nil, fmt.Errorf("failed reading data: %w", err)
	}

	return data, nil
}

func (d *data) Type() types.BasicKind {
	if d != nil {
		return d.dataType
	}
	return types.Invalid
}

func (d *data) get() any {
	if d != nil {
		return d.content
	}
	return nil
}

func (d *data) GetInt() [][]int {
	if content, ok := d.get().([][]int); ok {
		return content
	}
	return nil
}

func (d *data) GetFloat32() [][]float32 {
	if content, ok := d.get().([][]float32); ok {
		return content
	}
	return nil
}

func (d *data) GetFloat64() [][]float64 {
	if content, ok := d.get().([][]float64); ok {
		return content
	}
	return nil
}

package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"

	"github.com/edatts/ml/pkg/model"
)

// File format is SafeTensors:
//	- First a littleEndian uint64 with the header length
//	- Then the JSON header
//	- Then the data
//
// The JSON header has the following format:
//	{
//		"TENSOR_NAME_1": {
//			"dtype": "DATA_TYPE",
//			"shape": [...],
//			"data_offsets": [START, END]
//		},
//		"TENSOR_NAME_2": {...}
//		...,
//		"__metadata__": {...}
//	}
//
// The offsets are from the start of the data section not the
// start of the file.
//
// The following are valid for DATA_TYPE: [ "F64", "F32", "F16",
// "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"]

type Header map[string]json.RawMessage

type SafeTensorMetadata struct {
	DataType string      `json:"dtype"`
	Shape    model.Shape `json:"shape"`
	Offsets  [2]int      `json:"data_offsets"`
}

// Format accepts a slice of tensors as input and returns the full
// SafeTensors encoding of the data or an error. Currently only
// float32 elements are supported.
func Format(parameters []model.Tensor) ([]byte, error) {
	var (
		header = Header{}
		n      int
		data   []byte
	)

	// Create and marshal metadata
	for _, tensor := range parameters {
		b, err := json.Marshal(SafeTensorMetadata{
			DataType: "F32",
			Shape:    tensor.Shape,
			Offsets:  [2]int{n, n + len(tensor.Data)*4},
		})
		if err != nil {
			return nil, fmt.Errorf("failed marshalling metadata for tensor '%s': %w", tensor.Name, err)
		}

		header[tensor.Name] = b
		n += len(tensor.Data) * 4
	}

	headerBytes, err := json.Marshal(header)
	if err != nil {
		return nil, fmt.Errorf("failed marshalling JSON header: %w", err)
	}

	data = binary.LittleEndian.AppendUint64(data, uint64(len(headerBytes)))
	data = append(data, headerBytes...)
	for _, tensor := range parameters {
		for _, elem := range tensor.Data {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(elem))
		}
	}

	return data, nil
}

// Parse accepts a slice of bytes in SafeTensors format and decodes the
// data, returning a slice of tensors or an error. Currently the only
// supported data type is float32.
func Parse(b []byte) ([]model.Tensor, error) {
	// Read header len
	metadataLen := binary.LittleEndian.Uint64(b)

	// Read and parse metadata
	header := Header{}
	if err := json.Unmarshal(b[8:8+metadataLen], &header); err != nil {
		return nil, fmt.Errorf("failed unmarshalling metadata: %w", err)
	}

	var (
		allData = b[8+metadataLen:]
		out     = []model.Tensor{}
	)

	// For each tensor, read raw data
	for name, meta := range header {
		var m SafeTensorMetadata
		if err := json.Unmarshal(meta, &m); err != nil {
			return nil, fmt.Errorf("failed unmarshalling metadata for layer '%s': %w", name, err)
		}

		if m.DataType != "F32" {
			return nil, fmt.Errorf("unsupported data type '%s', only F32 currently supported", m.DataType)
		}

		var (
			startOffset = m.Offsets[0]
			endOffset   = m.Offsets[1]
			data        = make([]float32, (endOffset-startOffset)/4)
		)

		for i := startOffset; i < endOffset; i += 4 {
			data[(i-startOffset)/4] = math.Float32frombits(binary.LittleEndian.Uint32(allData[i : i+4]))
		}

		out = append(out, model.Tensor{
			Name:  name,
			Shape: m.Shape,
			Data:  data,
		})
	}

	return out, nil
}

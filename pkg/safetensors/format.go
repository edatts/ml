package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"

	"github.com/edatts/ml/pkg/shape"
	"github.com/edatts/ml/pkg/types"
	"github.com/x448/float16"
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
	Shape    shape.Shape `json:"shape"`
	Offsets  [2]int      `json:"data_offsets"`
}

// Format accepts a slice of tensors as input and returns the full
// SafeTensors encoding of the data or an error. Currently only
// float32 elements are supported.
func Format(parameters []types.Tensor) ([]byte, error) {
	var (
		header = Header{}
		n      int
		data   []byte
	)

	// Create and marshal metadata
	for _, tensor := range parameters {
		b, err := json.Marshal(SafeTensorMetadata{
			DataType: string(tensor.DataType),
			Shape:    tensor.Shape,
			Offsets:  [2]int{n, n + tensor.Data.ByteLen()},
		})
		if err != nil {
			return nil, fmt.Errorf("failed marshalling metadata for tensor '%s': %w", tensor.Name, err)
		}

		header[tensor.Name] = b
		n += tensor.Data.ByteLen()
	}

	headerBytes, err := json.Marshal(header)
	if err != nil {
		return nil, fmt.Errorf("failed marshalling JSON header: %w", err)
	}

	data = binary.LittleEndian.AppendUint64(data, uint64(len(headerBytes)))
	data = append(data, headerBytes...)
	for _, tensor := range parameters {
		data = append(data, tensor.Data.Bytes()...)
	}

	return data, nil
}

// Parse accepts a slice of bytes in SafeTensors format and decodes the
// data, returning a slice of tensors or an error. Currently the only
// supported data type is float32.
func Parse(b []byte) ([]types.Tensor, error) {
	// Read header len
	metadataLen := binary.LittleEndian.Uint64(b)

	// Read and parse metadata
	header := Header{}
	if err := json.Unmarshal(b[8:8+metadataLen], &header); err != nil {
		return nil, fmt.Errorf("failed unmarshalling metadata: %w", err)
	}

	var (
		allData = b[8+metadataLen:]
		out     = []types.Tensor{}
	)

	// For each tensor, read raw data
	for name, meta := range header {
		var m SafeTensorMetadata
		if err := json.Unmarshal(meta, &m); err != nil {
			return nil, fmt.Errorf("failed unmarshalling metadata for layer '%s': %w", name, err)
		}

		if m.DataType != "F32" && m.DataType != "F16" {
			return nil, fmt.Errorf("unsupported data type '%s', only F32 and F16 currently supported", m.DataType)
		}

		var (
			startOffset = m.Offsets[0]
			endOffset   = m.Offsets[1]
			numBytes    = byteLength(m.DataType)
			data        types.Data
		)

		switch m.DataType {
		case "F32":
			var nums = make([]float32, (endOffset-startOffset)/4)
			for i := startOffset; i < endOffset; i += numBytes {
				nums[(i-startOffset)/4] = math.Float32frombits(binary.LittleEndian.Uint32(allData[i : i+4]))
			}
			data = &types.F32Data{Data: nums}
		case "F16":
			var nums = make([]types.Float16, (endOffset-startOffset)/2)
			for i := startOffset; i < endOffset; i += numBytes {
				nums[(i-startOffset)/2] = float16.Frombits(binary.LittleEndian.Uint16(allData[i : i+2]))
			}
			data = &types.F16Data{Data: nums}
		default:
			return nil, fmt.Errorf("unsupported data type '%s'", m.DataType)
		}

		out = append(out, types.Tensor{
			Name:     name,
			DataType: m.DataType,
			Shape:    m.Shape,
			Data:     data,
		})
	}

	return out, nil
}

func byteLength(dType string) int {
	switch dType {
	case "F32":
		return 4
	case "F16":
		return 2
	}
	panic(fmt.Sprintf("unsupported data type '%s'", dType))
}

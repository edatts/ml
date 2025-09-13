package idx

import (
	"encoding/binary"
	"fmt"
	"go/types"
	"io"
	"log/slog"
)

// Package for parsing idx3-ubyte and idx1-ubyte extension files.
func ParseIdxFile(r io.Reader) (Data, error) {
	slog.Info("parsing idx file...")
	// First 4 bytes are the "magic number", first 2 are always 0
	var buf = make([]byte, 4)
	if _, err := r.Read(buf); err != nil {
		return nil, fmt.Errorf("failed reading magic number: %w", err)
	}

	// Third byte encodes the data type:
	// 0x08: unsigned byte
	// 0x09: signed byte
	// 0x0B: short (2 bytes)
	// 0x0C: int (4 bytes)
	// 0x0D: float (4 bytes)
	// 0x0E: double (8 bytes)
	dataType, err := getDataType(buf[2])
	if err != nil {
		return nil, err
	}

	// Fourth byte encodes the dimensionality of the data
	nDims := int(buf[3])
	slog.Info("dimensionality of data", "numDims", nDims)

	// The sizes in each dimension are big endian 4 byte intgers.
	var dimSizes = make([]uint32, nDims)
	for i := range nDims {
		buf = make([]byte, 4)
		if _, err := r.Read(buf); err != nil {
			return nil, fmt.Errorf("failed reading dimension sizes: %w", err)
		}
		dimSizes[i] = binary.BigEndian.Uint32(buf)
	}

	// So now we have the data type, the number of dims, and the dim sizes.
	// We can parse the rest of the bytes to build the output.
	numCols := 1
	for _, size := range dimSizes[1:] {
		numCols *= int(size)
	}

	return NewData(int(dimSizes[0]), numCols, dataType, r)
}

func getDataType(b byte) (types.BasicKind, error) {
	switch b {
	case 0x08:
		return types.Uint8, nil
	case 0x09:
		return types.Int8, nil
	case 0x0B:
		return types.Int16, nil
	case 0x0C:
		return types.Int32, nil
	case 0x0D:
		return types.Float32, nil
	case 0x0E:
		return types.Float64, nil
	default:
		return 0, fmt.Errorf("unknown data type for IDX format '%x'", b)
	}
}

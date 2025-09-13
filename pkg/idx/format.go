package idx

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

func readInt8Content(numRows, numCols int, r io.Reader) ([][]int, error) {
	var content = make([][]int, numRows)
	var sample = make([]byte, numCols)
	for i := range numRows {
		if _, err := io.ReadFull(r, sample); err != nil {
			return nil, fmt.Errorf("failed reading sample: %w", err)
		}
		content[i] = formatInt8Sample(sample)
	}
	return content, nil
}

func formatInt8Sample(sample []byte) []int {
	var out = make([]int, len(sample))
	for i, b := range sample {
		out[i] = int(b)
	}
	return out
}

func readUint8Content(numRows, numCols int, r io.Reader) ([][]int, error) {
	var content = make([][]int, numRows)
	var sample = make([]byte, numCols)
	for i := range numRows {
		if _, err := io.ReadFull(r, sample); err != nil {
			return nil, fmt.Errorf("failed reading sample: %w", err)
		}
		content[i] = formatUint8Sample(sample)
	}
	return content, nil
}

func formatUint8Sample(sample []byte) []int {
	var out = make([]int, len(sample))
	for i, b := range sample {
		out[i] = int(b)
	}
	return out
}

func readInt16Content(numRows, numCols int, r io.Reader) ([][]int, error) {
	var content = make([][]int, numRows)
	var sample = make([]byte, numCols*2)
	for i := range numRows {
		if _, err := io.ReadFull(r, sample); err != nil {
			return nil, fmt.Errorf("failed reading sample: %w", err)
		}
		content[i] = formatInt16Sample(sample)
	}
	return content, nil
}

func formatInt16Sample(sample []byte) []int {
	var out = make([]int, len(sample)/2)
	for i := range len(sample) / 2 {
		out[i] = int(binary.BigEndian.Uint16(sample[i*2 : i*2+1]))
	}
	return out
}

func readInt32Content(numRows, numCols int, r io.Reader) ([][]int, error) {
	var content = make([][]int, numRows)
	var sample = make([]byte, numCols*4)
	for i := range numRows {
		if _, err := io.ReadFull(r, sample); err != nil {
			return nil, fmt.Errorf("failed reading sample: %w", err)
		}
		content[i] = formatInt32Sample(sample)
	}
	return content, nil
}

func formatInt32Sample(sample []byte) []int {
	var out = make([]int, len(sample)/2)
	for i := range len(sample) / 2 {
		out[i] = int(binary.BigEndian.Uint32(sample[i*4 : i*4+1]))
	}
	return out
}

func readFloat32Content(numRows, numCols int, r io.Reader) ([][]float32, error) {
	var content = make([][]float32, numRows)
	var sample = make([]byte, numCols*4)
	for i := range numRows {
		if _, err := io.ReadFull(r, sample); err != nil {
			return nil, fmt.Errorf("failed reading sample: %w", err)
		}
		content[i] = formatFloat32Sample(sample)
	}
	return content, nil
}

func formatFloat32Sample(sample []byte) []float32 {
	var out = make([]float32, len(sample)/2)
	for i := range len(sample) / 2 {
		out[i] = math.Float32frombits(binary.BigEndian.Uint32(sample[i*4 : i*4+1]))
	}
	return out
}

func readFloat64Content(numRows, numCols int, r io.Reader) ([][]float64, error) {
	var content = make([][]float64, numRows)
	var sample = make([]byte, numCols*4)
	for i := range numRows {
		if _, err := io.ReadFull(r, sample); err != nil {
			return nil, fmt.Errorf("failed reading sample: %w", err)
		}
		content[i] = formatFloat64Sample(sample)
	}
	return content, nil
}

func formatFloat64Sample(sample []byte) []float64 {
	var out = make([]float64, len(sample)/2)
	for i := range len(sample) / 2 {
		out[i] = math.Float64frombits(binary.BigEndian.Uint64(sample[i*4 : i*4+1]))
	}
	return out
}

package shape

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
)

var (
	ErrNoDims          = errors.New("no dimensions provided")
	ErrMaxRank         = errors.New("maximum rank is 4")
	ErrZeroDim         = errors.New("input has invalid dimesion with length zero")
	ErrNegativeDim     = errors.New("input has invalid dimesion with negative length")
	ErrIncompatiblDims = errors.New("index dimensions are not compatible with strides")
)

func maxRankErr(r int) error {
	return fmt.Errorf("%w, input rank is %d", ErrMaxRank, r)
}

// New creates a new Shape from the specified dimensions. The maximum
// rank is 4 and the default format is channels first (NCHW).
func New(dims ...int) Shape {
	if err := validate(dims); err != nil {
		panic(err)
	}

	return Shape{
		dims: dims,
	}
}

func validate(input []int) error {
	if len(input) == 0 {
		return ErrNoDims
	}

	if len(input) > 4 {
		return maxRankErr(len(input))
	}

	if slices.Contains(input, 0) {
		return fmt.Errorf("invalid input '%v': %w", input, ErrZeroDim)
	}

	if slices.ContainsFunc(input, func(i int) bool { return i < 0 }) {
		return ErrNegativeDim
	}

	return nil
}

// Shape is a representation of a tensor shape, it uses channels
// first formats by default (NCHW).
type Shape struct {
	dims    []int
	strides []int
}

func (s Shape) Samples() int {
	if len(s.dims) < 4 {
		return 0
	}

	return s.dims[len(s.dims)-4]
}

func (s Shape) Channels() int {
	if len(s.dims) < 3 {
		return 0
	}

	return s.dims[len(s.dims)-3]
}

func (s Shape) Height() int {
	if len(s.dims) < 2 {
		return 0
	}

	return s.dims[len(s.dims)-2]
}

func (s Shape) Width() int {
	if len(s.dims) < 1 {
		return 0
	}

	return s.dims[len(s.dims)-1]
}

func (s Shape) Volume() int {
	if len(s.dims) == 0 {
		return 0
	}

	var out int = 1
	for _, dim := range s.dims {
		out *= int(dim)
	}
	return out
}

func (s Shape) Bytes() []byte {
	var b []byte
	for _, n := range s.dims {
		if n != 0 {
			b = binary.LittleEndian.AppendUint64(b, uint64(n))
		}
	}
	return b
}

func (s Shape) MarshalJSON() ([]byte, error) {
	var intermediate = []int{}
	for _, x := range s.dims {
		if x != 0 {
			intermediate = append(intermediate, x)
		}
	}
	return json.Marshal(intermediate)
}

func (s *Shape) UnmarshalJSON(b []byte) error {
	var intermediate = []int{}
	err := json.Unmarshal(b, &intermediate)
	if err != nil {
		return err
	}

	if err := validate(intermediate); err != nil {
		return fmt.Errorf("failed validating: %w", err)
	}

	s.dims = intermediate
	return nil
}

func (s Shape) Strides() Strides {
	if len(s.strides) != 0 {
		return Strides{elems: s.strides, maxIdx: s.Volume()}
	}

	s.strides = make([]int, len(s.dims))
	s.strides[len(s.dims)-1] = 1
	for i := len(s.dims) - 2; i >= 0; i-- {
		s.strides[i] = s.strides[i+1] * int(s.dims[i+1])
	}

	return Strides{elems: s.strides, maxIdx: s.Volume()}
}

// Strides is a convenience type for converting flat indices into
// tensor indices and vice-versa.
//
// If the tensor dimensions are powers of 2 (or can be padded to powers
// of 2) then we can replace usage of pre-computed strides with usage
// of bithsift and logical & operators.
type Strides struct {
	elems  []int
	maxIdx int
}

func (s Strides) FlatIndex(indices ...int) int {
	if len(indices) != len(s.elems) {
		panic(fmt.Sprintf("indices and strides are different lengths, indices=%d, striudes=%d", len(indices), len(s.elems)))
	}

	var out int
	for i, idx := range indices {
		out += int(s.elems[i] * idx)
	}

	return out
}

func (s Strides) Indices(idx int) []int {
	if idx >= s.maxIdx {
		panic(fmt.Sprintf("provided index '%d' exceeds maximum for shape '%d'", idx, s.maxIdx))
	}

	var indices = make([]int, len(s.elems))
	indices[0] = idx / s.elems[0]
	remainder := idx - (indices[0] * s.elems[0])
	for i := 1; i < len(s.elems); i++ {
		indices[i] = remainder / s.elems[i]
		remainder = remainder - (indices[i] * s.elems[i])
	}
	return indices
}

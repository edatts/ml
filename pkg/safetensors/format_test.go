package safetensors_test

import (
	"encoding/binary"
	"testing"

	"github.com/edatts/ml/pkg/model"
	"github.com/edatts/ml/pkg/safetensors"
	"github.com/edatts/ml/pkg/shape"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

var (
	parameters = []model.Tensor{
		{
			Name:  "layers.conv2d_1.weights",
			Shape: shape.New(2, 1, 3, 3),
			Data: []float32{
				0, 1, 2, 3, 4, 5, 6, 7, 8,
				8, 7, 6, 5, 4, 3, 2, 1, 0,
			},
		},
		{
			Name:  "layers.conv2d_1.biases",
			Shape: shape.New(2),
			Data:  []float32{3, 5},
		},
		{
			Name:  "layers.full_1.weights",
			Shape: shape.New(4, 4),
			Data: []float32{
				1, 1, 1, 1, 1, 1, 1, 1,
				2, 2, 2, 2, 2, 2, 2, 2,
			},
		},
		{
			Name:  "layers.full_1.biases",
			Shape: shape.New(4),
			Data:  []float32{3, 3, 3, 3},
		},
	}

	expectedConv1Weights = `{"dtype":"F32","shape":[2,1,3,3],"data_offsets":[0,72]}`
	expectedConv1Biases  = `{"dtype":"F32","shape":[2],"data_offsets":[72,80]}`
	expectedFull1Weights = `{"dtype":"F32","shape":[4,4],"data_offsets":[80,144]}`
	expectedFull1Biases  = `{"dtype":"F32","shape":[4],"data_offsets":[144,160]}`

	expectedData = []byte{
		0x00, 0x00, 0x00, 0x00, 0x3f, 0x80, 0x00, 0x00,
		0x40, 0x00, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00,
		0x40, 0x80, 0x00, 0x00, 0x40, 0xa0, 0x00, 0x00,
		0x40, 0xc0, 0x00, 0x00, 0x40, 0xe0, 0x00, 0x00,
		0x41, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00,
		0x40, 0xe0, 0x00, 0x00, 0x40, 0xc0, 0x00, 0x00,
		0x40, 0xa0, 0x00, 0x00, 0x40, 0x80, 0x00, 0x00,
		0x40, 0x40, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
		0x3f, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,

		0x40, 0x40, 0x00, 0x00, 0x40, 0xa0, 0x00, 0x00,

		0x3f, 0x80, 0x00, 0x00, 0x3f, 0x80, 0x00, 0x00,
		0x3f, 0x80, 0x00, 0x00, 0x3f, 0x80, 0x00, 0x00,
		0x3f, 0x80, 0x00, 0x00, 0x3f, 0x80, 0x00, 0x00,
		0x3f, 0x80, 0x00, 0x00, 0x3f, 0x80, 0x00, 0x00,
		0x40, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
		0x40, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
		0x40, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
		0x40, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,

		0x40, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00,
		0x40, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00,
	}
)

func TestFormat(t *testing.T) {
	b, err := safetensors.Format(parameters)
	require.NoError(t, err)

	metadataLen := binary.LittleEndian.Uint64(b)
	require.Equal(t, uint64(313), metadataLen)

	headerBytes := b[8 : 8+metadataLen]

	res := gjson.GetBytes(headerBytes, `layers\.conv2d_1\.weights`)
	require.True(t, res.Exists())
	require.Equal(t, expectedConv1Weights, res.String())

	res = gjson.GetBytes(headerBytes, `layers\.conv2d_1\.biases`)
	require.True(t, res.Exists())
	require.Equal(t, expectedConv1Biases, res.String())

	res = gjson.GetBytes(headerBytes, `layers\.full_1\.weights`)
	require.True(t, res.Exists())
	require.Equal(t, expectedFull1Weights, res.String())

	res = gjson.GetBytes(headerBytes, `layers\.full_1\.biases`)
	require.True(t, res.Exists())
	require.Equal(t, expectedFull1Biases, res.String())

	require.ElementsMatch(t, expectedData, b[8+metadataLen:])
}

func TestParse(t *testing.T) {
	b, err := safetensors.Format(parameters)
	require.NoError(t, err)

	tensors, err := safetensors.Parse(b)
	require.NoError(t, err)

	for _, tensor := range parameters {
		require.Contains(t, tensors, tensor)
	}
}

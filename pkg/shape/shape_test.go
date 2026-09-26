package shape_test

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"testing"

	"github.com/edatts/ml/pkg/shape"
	"github.com/stretchr/testify/require"
)

func TestShape(t *testing.T) {

	t.Run("constructor", func(t *testing.T) {
		fn := func() { _ = shape.New() }
		require.PanicsWithError(t, shape.ErrNoDims.Error(), fn)

		fn = func() { _ = shape.New(-5) }
		require.PanicsWithError(t, shape.ErrNegativeDim.Error(), fn)

		fn = func() { _ = shape.New(1, 1, 1, 1, 1) }
		errString := fmt.Sprintf("%s, input rank is %d", shape.ErrMaxRank, 5)
		require.PanicsWithError(t, errString, fn)

		fn = func() { _ = shape.New(0, 1, 1, 1) }
		errString = fmt.Sprintf("invalid input '%v': %s", []uint16{0, 1, 1, 1}, shape.ErrZeroDim)
		require.PanicsWithError(t, errString, fn)
	})

	t.Run("elements", func(t *testing.T) {
		var s shape.Shape
		fn := func() { s = shape.New(3, 2, 6, 5) }
		require.NotPanics(t, fn)

		require.Equal(t, 3, s.Samples())
		require.Equal(t, 2, s.Channels())
		require.Equal(t, 6, s.Height())
		require.Equal(t, 5, s.Width())

	})

	t.Run("JSON", func(t *testing.T) {
		var s shape.Shape
		fn := func() { s = shape.New(1, 2, 3, 1) }
		require.NotPanics(t, fn)

		b, err := json.Marshal(s)
		require.NoError(t, err)

		var parsed shape.Shape
		require.NoError(t, json.Unmarshal(b, &parsed))

		require.Equal(t, s, parsed)
	})

	t.Run("strides", func(t *testing.T) {
		var s shape.Shape
		fn := func() { s = shape.New(1, 2, 3, 2) }
		require.NotPanics(t, fn)

		marshalled, err := json.Marshal(s)
		require.NoError(t, err)

		slog.Info("shape", "shape", marshalled)

		require.Equal(t, []int{0, 0, 0, 0}, s.Strides().Indices(0))
		require.Equal(t, []int{0, 0, 0, 1}, s.Strides().Indices(1))
		require.Equal(t, []int{0, 0, 1, 0}, s.Strides().Indices(2))
		require.Equal(t, []int{0, 0, 2, 1}, s.Strides().Indices(5))
		require.Equal(t, []int{0, 1, 2, 0}, s.Strides().Indices(10))
		require.Equal(t, []int{0, 1, 2, 1}, s.Strides().Indices(11))
		require.Panics(t, func() { s.Strides().Indices(13) })
	})
}

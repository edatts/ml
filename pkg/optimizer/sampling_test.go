package optimizer_test

import (
	"testing"

	"github.com/edatts/ml/pkg/optimizer"
	"github.com/stretchr/testify/require"
)

func TestSampling(t *testing.T) {
	t.Run("RS2 Sampler", func(t *testing.T) {
		s := optimizer.NewRS2Sampler[int](0.2)

		cSampler, ok := s.(optimizer.ClassificationSampler)
		require.True(t, ok)

		_ = cSampler
	})
}

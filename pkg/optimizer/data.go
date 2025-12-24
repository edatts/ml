// Will probably move this to another package later
package optimizer

// TODO: Find out where the stdlib numeric interface lives...
type Numeric interface {
	~int | ~float32 | ~float64
}

type Data[YT Numeric] struct {
	X []float32
	Y []YT
}

type Batch[YT Numeric] struct {
	Inputs  [][]float32
	Targets [][]YT
}

func NewBatch[XT, YT Numeric](size int) Batch[YT] {
	return Batch[YT]{
		Inputs:  make([][]float32, size),
		Targets: make([][]YT, size),
	}
}

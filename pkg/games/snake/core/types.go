package core

import (
	"fmt"
	"slices"
)

type Direction string

const (
	Up    Direction = "up"
	Down  Direction = "down"
	Left  Direction = "left"
	Right Direction = "right"
)

type Tile int

const (
	Empty Tile = 0
	Food  Tile = 1
	Head  Tile = 2
	Body  Tile = 3
)

// node represents a node in a doubly-linked list that
// we use to track the indices of the snake body.
type node struct {
	index int
	next  *node
	prev  *node
}

func NewTiles(height, width int, data []Tile) (Tiles, error) {
	if len(data) != height*width {
		return Tiles{}, fmt.Errorf("data is incompatible with height and width")
	}

	return Tiles{
		data:   data,
		height: height,
		width:  width,
	}, nil
}

type Tiles struct {
	data          []Tile
	height, width int
}

// func (t Tiles) ScaledData() []float32 {
// 	// We want to scale the data between 0 and 1 and since the
// 	// data is alread in range 0-3 we can just divide by 3.
// 	var out = make([]float32, len(t.data))
// 	for i, datum := range t.data {
// 		out[i] = float32(datum) / 3
// 	}
// 	return out
// }

func (t Tiles) X(idx int) int {
	return idx % t.width
}

func (t Tiles) Y(idx int) int {
	return idx / t.height
}

func (t Tiles) Index(x, y int) int {
	return y*t.width + x
}

func (t Tiles) Len() int {
	return len(t.data)
}

func (t Tiles) Get(idx int) Tile {
	return Tile(t.data[idx])
}

func (t Tiles) FoodIndex() int {
	if idx := slices.Index(t.data, Food); idx != -1 {
		return idx
	}
	panic("food not found in tiles")
}

func (t Tiles) HeadIndex() int {
	if idx := slices.Index(t.data, Head); idx != -1 {
		return idx
	}
	panic("head not found in tiles")
}

func (t Tiles) CentreOfMass() (int, int) {
	var (
		total        int
		x_avg, y_avg int
	)

	for i, tile := range t.data {
		if tile == Body {
			total++
			x_avg += t.X(i)
			y_avg += t.Y(i)
		}
	}

	x_avg /= total
	y_avg /= total

	return x_avg, y_avg
}

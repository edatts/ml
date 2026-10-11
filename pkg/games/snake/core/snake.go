package core

import (
	"fmt"
	"math/rand/v2"
)

const height, width = 30, 30

func New() *Core {
	return &Core{
		tiles: Tiles{
			height: height,
			width:  width,
			data:   make([]Tile, height*width),
		},
	}
}

// Core is a standalone component that contains the core game logic
// and state for snake. Core is not aware of ebitengine so we can
// run it completely separately for training agents.
//
// The game state is represented by an array of integers called tiles,
// this state is also used by the Game object to render the UI when
// running with ebitengine.
//
// Empty tiles are represented with a 0, food tiles are represente
// with a 1, the head of the snake is 2, the body of the snake is 3.
type Core struct {
	// height    int
	// width     int
	// boardSize int
	// tiles     []int
	tiles Tiles

	currentDirection Direction

	head *node
	tail *node

	score int
}

func (c *Core) Init() {
	// Snake starts in the middle row on the left
	snakeIdx := (c.tiles.width * c.tiles.height / 2) + 3
	c.head = &node{index: snakeIdx}
	c.head.prev = &node{next: c.head, index: snakeIdx - 1}
	c.tail = &node{next: c.head.prev, index: snakeIdx - 2}
	c.tail.next = c.head.prev
	c.head.prev.prev = c.tail
	c.currentDirection = Right
	c.tiles.data[snakeIdx] = Head
	c.tiles.data[snakeIdx-1] = Body
	c.tiles.data[snakeIdx-2] = Body

	// Add random food
	c.newFood()
}

func (c *Core) Reset() {
	c.score = 0
	c.tiles.data = make([]Tile, len(c.tiles.data))
	c.currentDirection = ""
	c.Init()
}

// setDirection is used to indicate the direction that the snake
// should move in during the next step. It is used by the Game
// component to translate user input into directions and is also
// used to input actions from agents when they are playing.
func (c *Core) SetDirection(d Direction) {
	if d != "" && !isOpposite(c.currentDirection, d) {
		c.currentDirection = d
	}
}

func (c *Core) State() (Tiles, Direction) {
	return c.tiles, c.currentDirection
}

func (c *Core) Score() int {
	return c.score
}

func (c *Core) RawTiles() []Tile {
	return c.tiles.data
}

// step carries out the next step of the snake simulation. It returns a
// boolean indicating whether the episode is complete or not.
func (c *Core) Step() bool {
	// Here we will update our state. There are a few things we
	// want to do here.
	//	- Advance the snake by one tile.
	//	- If hitting a wall, game over.
	// 	- If hitting the snakes body, game over.
	//	- If hitting food, increase score, grow snake.

	// start := time.Now()
	// defer func() {
	// 	elapsed := time.Since(start).Microseconds()
	// 	sloc.Info("elapsed time in micros", "elapsed", elapsed, "percentage", (float64(elapsed)/float64(1000000/60))*100)
	// }()

	// Move snake in current direction
	newHeadIdx := c.nextHeadIndex()

	if c.hitWall(newHeadIdx) {
		// Game over
		return true
	}

	if c.hitBody(newHeadIdx) {
		// Game over
		return true
	}

	if c.hitFood(newHeadIdx) {
		// Grow snake (tail remains where it is)
		c.tiles.data[c.head.index], c.tiles.data[newHeadIdx] = Body, Head
		c.head.next = &node{prev: c.head, index: newHeadIdx}
		c.head = c.head.next
		c.score++

		// New food
		c.newFood()
		return false
	}

	// New head position
	c.tiles.data[c.head.index], c.tiles.data[newHeadIdx] = Body, Head
	c.head.next = &node{prev: c.head, index: newHeadIdx}
	c.head = c.head.next

	// Delete tail
	c.tiles.data[c.tail.index] = Empty
	c.tail = &node{next: c.tail.next.next, index: c.tail.next.index}

	return false
}

func (c *Core) newFood() {
	foodIdx := rand.IntN(c.tiles.Len())
	for c.tiles.data[foodIdx] != 0 {
		foodIdx = rand.IntN(c.tiles.Len())
	}
	c.tiles.data[foodIdx] = Food
}

func (c *Core) nextHeadIndex() int {
	switch c.currentDirection {
	case Up:
		return c.head.index - c.tiles.width
	case Down:
		return c.head.index + c.tiles.width
	case Left:
		return c.head.index - 1
	case Right:
		return c.head.index + 1
	default:
		panic(fmt.Sprintf("unknown direction '%s'", c.currentDirection))
	}
}

func (c *Core) hitWall(newIdx int) bool {
	switch c.currentDirection {
	case Up:
		return newIdx < 0
	case Down:
		return newIdx >= c.tiles.Len()
	case Left:
		return (newIdx+1)%c.tiles.width == 0
	case Right:
		return newIdx%c.tiles.width == 0
	default:
		panic(fmt.Sprintf("unknown direction '%s'", c.currentDirection))
	}
}

func (c *Core) hitBody(newIdx int) bool {
	return c.tiles.data[newIdx] == 3
}

func (c *Core) hitFood(newIdx int) bool {
	return c.tiles.data[newIdx] == 1
}

func isOpposite(d1, d2 Direction) bool {
	switch d1 {
	case Up:
		return d2 == Down
	case Down:
		return d2 == Up
	case Left:
		return d2 == Right
	case Right:
		return d2 == Left
	default:
		panic(fmt.Sprintf("unknown direction '%s'", d1))
	}
}

package snake

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	_ "image/png"
	"os"

	"github.com/edatts/ml/pkg/games/snake/core"
	"github.com/edatts/ml/pkg/games/snake/images"
	"github.com/edatts/ml/pkg/rl/qlearn"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

const tileSize = 12

var (
	blackImage      = ebiten.NewImage(tileSize, tileSize)
	lightGreenImage = ebiten.NewImage(tileSize, tileSize)
	greenImage      = ebiten.NewImage(tileSize, tileSize)
	redImage        = ebiten.NewImage(tileSize, tileSize)
	gameOverImage   *ebiten.Image
)

func init() {
	decodedGameOver, _, err := image.Decode(bytes.NewReader(images.GameOver_PNG))
	if err != nil {
		panic(fmt.Sprintf("failed decoding embedded image 'game-over.png': %s", err))
	}
	gameOverImage = ebiten.NewImageFromImage(decodedGameOver)
	blackImage.Fill(color.Black)
	lightGreenImage.Fill(color.RGBA{94, 255, 61, 255})
	greenImage.Fill(color.RGBA{24, 177, 24, 255})
	redImage.Fill(color.RGBA{222, 0, 0, 255})
}

var _ ebiten.Game = &Game{}

// Game is the core type that implements the game logic and rendering
// logic required by the ebitengine Game interface. Currently it runs
// the snake game using the ebitengine UI.
//
// We want to be able to run the core game logic as a simulation,
// decoupled from ebitengine so that we can run it very quickly and
// in parallel for training agents.
//
// Alternatively we could modify the tick and draw rate so that an
// agent can learn from the game while it is running on ebitengine.
// In this case it might be a bit more complicated to train agents
// in parallel.
//
// At most our update loop takes a few microseconds to execute, while
// a single the Draw() call takes a few hundred microseconds. Overall
// we have about 16.6ms per frame so we can probably run about 20 of
// these in parallel without impacting framerate (assuming we render
// all instances).
//
// I think as a starting point we should pull the core game logic out
// into a separate component that is not aware of ebitengine. This way
// we can wrap it in the Game type for use with ebitengine or we can
// run it standalone to train our agents.
type Game struct {
	core        *core.Core
	speed       Speed
	inputBuffer []ebiten.Key
	agent       qlearn.Agent
	stateFn     qlearn.SnakeStateFn

	gameOver bool
}

func NewGame(optFns ...GameOption) *Game {
	g := &Game{
		core:  core.New(),
		speed: Fast,
	}

	for _, fn := range optFns {
		fn(g)
	}

	return g
}

func (g *Game) Update() error {
	// So we don't want the snake to move every tick, we want it
	// to move every n ticks where n is larger for a slower moving
	// snake and smaller for a faster moving snake.
	//
	// How we will relate this to our states for Reinforcement
	// Learning is interesting because we want each time step from
	// the perspective of our algorithm to be each time the snake
	// moves not each logical game update...
	//

	if g.gameOver {
		// Check for key press to start new game or exit.
		for _, key := range inpututil.AppendJustPressedKeys(nil) {
			if key == ebiten.KeyEnter { // Restart
				g.core.Reset()
				g.gameOver = false
				return nil
			}

			if key == ebiten.KeyEscape {
				os.Exit(1)
			}
		}
		return nil
	}

	// We queue up the ket presses every tick.
	g.queueKeyPresses()

	if ebiten.Tick() == 0 {
		g.core.Init()
		return nil
	}

	if ebiten.Tick()%g.actionInterval() == 0 {
		// Here we de-queue one key from the input buffer and use it
		// to set the direction for the next step of the game.
		var d core.Direction
		if len(g.inputBuffer) != 0 {
			d = g.directionFromKey(g.inputBuffer[0])
			g.inputBuffer = g.inputBuffer[1:]
		}

		if g.agent != nil {
			// Build the agent state from core
			state := g.stateFn(g.core.State())
			action, err := g.agent.NextAction(state)
			if err != nil {
				return err
			}

			d = qlearn.SnakeActions[action]
		}

		g.core.SetDirection(d)
		if done := g.core.Step(); done {
			g.gameOver = done
		}

		return nil
	}

	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(color.White)
	var numTiles = 360 / tileSize
	for i, tileType := range g.core.RawTiles() {
		tileImg := g.getTileImage(tileType)

		x := 1 + (i%numTiles)*tileSize
		y := 1 + (i/numTiles)*tileSize

		opt := &ebiten.DrawImageOptions{}
		opt.GeoM.Translate(float64(x), float64(y))
		screen.DrawImage(tileImg, opt)
	}

	if g.gameOver {
		dx := int(float64(gameOverImage.Bounds().Dx()) / 2)
		dy := int(float64(gameOverImage.Bounds().Dy()) / 2)
		opt := &ebiten.DrawImageOptions{}
		opt.GeoM.Translate(float64(362-dx), float64(362-dy))
		opt.GeoM.Scale(0.5, 0.5)
		screen.DrawImage(gameOverImage, opt)

	}
}

func (g *Game) Layout(_, _ int) (width int, height int) {
	return 362, 362
}

func (g *Game) actionInterval() int64 {
	switch g.speed {
	case Slow:
		return 8
	case Medium:
		return 6
	case Fast:
		return 4
	case VeryFast:
		return 2
	case Unreal:
		return 1
	default:
		panic(fmt.Sprintf("unknown game speed: %s", g.speed))
	}
}

func (g *Game) queueKeyPresses() {
	// Currently we just throw all key presses into the input buffer
	// but we should probably filter out unmapped keys early and
	// intercept keys for pausing and ending the game...
	g.inputBuffer = inpututil.AppendJustPressedKeys(g.inputBuffer)
}

func (g *Game) getTileImage(tileType core.Tile) *ebiten.Image {
	switch tileType {
	case 0: // Empty
		return blackImage
	case 1: // Food
		return redImage
	case 2: // Head
		return lightGreenImage
	case 3: // Body
		return greenImage
	default:
		panic(fmt.Sprintf("unknown tile type '%d'", tileType))
	}
}

func (g *Game) directionFromKey(key ebiten.Key) core.Direction {
	switch key {
	case ebiten.KeyArrowUp:
		return core.Up
	case ebiten.KeyArrowDown:
		return core.Down
	case ebiten.KeyArrowLeft:
		return core.Left
	case ebiten.KeyArrowRight:
		return core.Right
	default:
		return ""
	}
}

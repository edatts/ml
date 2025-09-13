package parse

import (
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
)

var ParseIDXCmd = &cobra.Command{
	Use:   "parse-idx",
	Short: "Parses IDX files and outputs the results as CSV",
	RunE: func(cmd *cobra.Command, _ []string) error {
		// Get files from flags
		var outDir = "."
		if dir != "" {
			outDir = dir
			if err := fs.WalkDir(os.DirFS(dir), ".", func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return err
				}

				if d.Type().IsRegular() && hasIDXExtension(d.Name()) {
					fileNames = append(fileNames, filepath.Join(dir, d.Name()))
				}

				return nil
			}); err != nil {
				return fmt.Errorf("failed walking dir '%s': %w", dir, err)
			}
		}

		if len(fileNames) == 0 {
			return fmt.Errorf("no supported files found in dir '%s'", dir)
		}

		for _, path := range fileNames {
			slog.Info("file name", "path", path)

			f, err := os.Open(path)
			if err != nil {
				return fmt.Errorf("failed opening file '%s': %w", path, err)
			}
			defer f.Close()

			// First 4 bytes are the "magic number", first 2 are always 0
			var buf = make([]byte, 4)
			if _, err = f.Read(buf); err != nil {
				return fmt.Errorf("failed reading magic number: %w", err)
			}

			slog.Info("read magic number", "buf", fmt.Sprintf("%+v", buf))

			// Third byte encodes the data type:
			// 0x08: unsigned byte
			// 0x09: signed byte
			// 0x0B: short (2 bytes)
			// 0x0C: int (4 bytes)
			// 0x0D: float (4 bytes)
			// 0x0E: double (8 bytes)
			slog.Info("data type", "val", buf[2])
			// dataType := buf[2]

			// Fourth byte encodes the dimensionality of the data
			nDims := buf[3]
			slog.Info("dimensionality of data", "numDims", nDims)

			// The sizes in each dimension are big endian 4 byte intgers.
			var dimSizes = make([]uint32, nDims)
			for i := range nDims {
				buf = make([]byte, 4)
				if _, err = f.Read(buf); err != nil {
					return fmt.Errorf("failed reading next 20 bytes: %w", err)
				}
				slog.Info("dimension size", "dim", i, "bytes", fmt.Sprintf("%+v", buf), "val", binary.BigEndian.Uint32(buf))
				dimSizes[i] = binary.BigEndian.Uint32(buf[:4])
			}

			newfileName := fmt.Sprintf("%s.csv", strings.Split(filepath.Base(path), ".")[0])
			csvFile, err := os.Create(filepath.Join(outDir, newfileName))
			if err != nil {
				return fmt.Errorf("failed creating file '%s': %w", filepath.Join(outDir, newfileName), err)
			}
			defer csvFile.Close()

			numCols := 1
			for _, size := range dimSizes[1:] {
				numCols *= int(size)
			}

			slog.Info("numCols", "numCols", numCols)

			var csvHeader = make([]string, numCols)
			for i := range numCols {
				csvHeader[i] = strconv.Itoa(i)
			}

			csvWriter := csv.NewWriter(csvFile)
			if err := csvWriter.Write(csvHeader); err != nil {
				return fmt.Errorf("failed writing csv header: %w", err)
			}

			var datum = make([]uint8, numCols)
			var numBytes = 16
			for range dimSizes[0] {
				// I know the type is uint8 so won't handle other types yet
				n, err := f.Read(datum)
				if err != nil {
					return fmt.Errorf("failed reading datum: %w", err)
				}

				numBytes += n

				if err := csvWriter.Write(toString(datum)); err != nil {
					return fmt.Errorf("failed writing datum to csv: %w", err)
				}

				csvWriter.Flush()
				if err := csvWriter.Error(); err != nil {
					return fmt.Errorf("csv writer flush err: %w", err)
				}
			}
			slog.Info("num bytes read", "numBytes", numBytes)

		}

		return nil
	},
}

func toString(in []uint8) []string {
	var out = make([]string, len(in))
	for i, num := range in {
		out[i] = strconv.Itoa(int(num))
	}
	return out
}

func hasIDXExtension(name string) bool {
	switch {
	case strings.Contains(name, "idx3-ubyte"):
		return true
	case strings.Contains(name, "idx1-ubyte"):
		return true
	default:
		return false
	}
}

var (
	fileNames []string
	dir       string
)

func init() {
	ParseIDXCmd.Flags().StringArrayVarP(&fileNames, "file-name", "f", nil, "Used to specify the relatve path to a file to pass. Can be used more than once to parse multiple files")
	ParseIDXCmd.Flags().StringVar(&dir, "dir", "", "Used to specify a directory in which all IDX files will be parsed.")

	ParseIDXCmd.MarkFlagsOneRequired("file-name", "dir")
	ParseIDXCmd.MarkFlagsMutuallyExclusive("file-name", "dir")
}

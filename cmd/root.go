package cmd

import (
	"log/slog"
	"os"

	"github.com/edatts/ml/cmd/parse"
	qlearncmd "github.com/edatts/ml/cmd/qlearn"
	snakecmd "github.com/edatts/ml/cmd/snake"
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:          "main",
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		slog.Info("Hello")
		return cmd.Usage()
	},
}

func init() {
	rootCmd.AddCommand(parse.ParseIDXCmd)
	rootCmd.AddCommand(snakecmd.Cmd)
	rootCmd.AddCommand(qlearncmd.Cmd)
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		slog.Error("failed executing root command", "error", err)
		os.Exit(1)
	}
}

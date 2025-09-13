package cmd

import (
	"log/slog"
	"os"

	"github.com/edatts/ml/cmd/parse"
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
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		slog.Error("failed executing root command", "error", err)
		os.Exit(1)
	}
}

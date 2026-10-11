package qlearncmd

import (
	snakecmd "github.com/edatts/ml/cmd/qlearn/snake"
	"github.com/spf13/cobra"
)

var Cmd = &cobra.Command{
	Use:   "qlearn",
	Short: "Q learning sub-command",
	RunE: func(cmd *cobra.Command, _ []string) error {
		return cmd.Usage()
	},
}

func init() {
	Cmd.AddCommand(snakecmd.Cmd)
}

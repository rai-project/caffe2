package main

import (
	"fmt"
	"os"

	"github.com/rai-project/caffe2"
	_ "github.com/rai-project/caffe2/predict"
	cmd "github.com/rai-project/dlframework/framework/cmd/server"
)

func main() {

	rootCmd, err := cmd.NewRootCommand(caffe2.FrameworkManifest)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}

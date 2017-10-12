package main

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	"github.com/rai-project/caffe2"
)

// example Usage:
// go run caffe_reader.go ~/data/carml/dlframework/caffe2_0.8.1/squeezenet_1.0/predict_net.pb
func main() {
	modelFile := os.Args[1]
	if !com.IsFile(modelFile) {
		fmt.Println("unable to find", modelFile)
		os.Exit(-1)
	}
	buf, err := ioutil.ReadFile(modelFile)
	if err != nil {
		fmt.Println("unable to read file", err)
		os.Exit(-1)
	}
	def := &caffe2.NetDef{}
	err = def.Unmarshal(buf)
	if err != nil {
		fmt.Println("unable to unmarshal file", modelFile)
		os.Exit(-1)
	}
	// pp.Println(def.DeviceOption)
	for _, op := range def.Op {
		pp.Println(op.GetType())
	}
}

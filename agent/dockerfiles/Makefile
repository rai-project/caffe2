buildlocalppc:
	docker build . -f Dockerfile.ppc64le -t carml/caffe2:base-ppc64le-gpu
buildlocalamd64gpu:
	docker build . -f Dockerfile.amd64_gpu -t carml/caffe2:base-amd64-gpu
buildlocalamd64cpu:
	docker build . -f Dockerfile.amd64_cpu -t carml/caffe2:base-amd64-cpu

buildlocalamd64: buildlocalamd64cpu buildlocalamd64gpu

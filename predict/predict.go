package predict

import (
	"bufio"
	"os"
	"strings"

	context "golang.org/x/net/context"

	opentracing "github.com/opentracing/opentracing-go"
	"github.com/pkg/errors"
	"github.com/rai-project/caffe2"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	common "github.com/rai-project/dlframework/framework/predict"
	"github.com/rai-project/downloadmanager"
	gocaffe2 "github.com/rai-project/go-caffe2"
	"github.com/rai-project/image/types"
)

type ImagePredictor struct {
	common.ImagePredictor
	features  []string
	predictor *gocaffe2.Predictor
	inputDims []uint32
}

func New(model dlframework.ModelManifest, opts dlframework.PredictionOptions) (common.Predictor, error) {
	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}

	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(ImagePredictor)

	return predictor.Load(context.Background(), model, opts)
}

func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts dlframework.PredictionOptions) (common.Predictor, error) {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "Load"); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &ImagePredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework:         framework,
				Model:             model,
				PredictionOptions: opts,
			},
			WorkDir: workDir,
		},
	}

	if ip.download(ctx) != nil {
		return nil, err
	}

	if ip.loadPredictor(ctx) != nil {
		return nil, err
	}

	return ip, nil
}

func (p *ImagePredictor) GetPreprocessOptions(ctx context.Context) (common.PreprocessOptions, error) {
	mean, err := p.GetMeanImage()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	scale, err := p.GetScale()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	imageDims, err := p.GetImageDimensions()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	return common.PreprocessOptions{
		MeanImage: mean,
		Scale:     scale,
		Size:      []int{int(imageDims[1]), int(imageDims[2])},
		ColorMode: types.BGRMode,
	}, nil
}

func (p *ImagePredictor) download(ctx context.Context) error {
	if span, newCtx := opentracing.StartSpanFromContext(
		ctx,
		"Download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"traget_graph_file":   p.GetGraphPath(),
			"weights_url":         p.GetWeightsUrl(),
			"traget_weights_file": p.GetWeightsPath(),
			"feature_url":         p.GetFeaturesUrl(),
			"traget_feature_file": p.GetFeaturesPath(),
		},
	); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

	model := p.Model
	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		_, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx))
		if err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
		return nil
	}

	checksum := p.GetGraphChecksum()
	if checksum == "" {
		return errors.New("Need graph file checksum in the model manifest")
	}

	if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	checksum = p.GetWeightsChecksum()
	if checksum == "" {
		return errors.New("Need weights file checksum in the model manifest")
	}

	if _, err := downloadmanager.DownloadFile(p.GetWeightsUrl(), p.GetWeightsPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	checksum = p.GetFeaturesChecksum()
	if checksum == "" {
		return errors.New("Need features file checksum in the model manifest")
	}

	if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	return nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "LoadPredictor"); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

	var features []string
	f, err := os.Open(p.GetFeaturesPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetFeaturesPath())
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		features = append(features, line)
	}
	p.features = features

	p.inputDims, err = p.GetImageDimensions()
	if err != nil {
		return err
	}

	pred, err := gocaffe2.New(p.GetGraphPath(), p.GetWeightsPath())
	if err != nil {
		return err
	}
	p.predictor = pred

	return nil
}

func (p *ImagePredictor) Predict(ctx context.Context, data []float32, opts dlframework.PredictionOptions) (dlframework.Features, error) {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "Predict", opentracing.Tags{
		"model_name":        p.Model.GetName(),
		"model_version":     p.Model.GetVersion(),
		"framework_name":    p.Model.GetFramework().GetName(),
		"framework_version": p.Model.GetFramework().GetVersion(),
	}); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

	predictions, err := p.predictor.Predict(data, int(p.BatchSize()), int(p.inputDims[0]), int(p.inputDims[1]), int(p.inputDims[2]))
	if err != nil {
		return nil, err
	}

	rprobs := make([]*dlframework.Feature, len(predictions))
	for ii, pred := range predictions {
		rprobs[ii] = &dlframework.Feature{
			Index:       int64(pred.Index),
			Name:        p.features[pred.Index],
			Probability: pred.Probability,
		}
	}
	res := dlframework.Features(rprobs)

	return res, nil
}

func (p *ImagePredictor) Reset(ctx context.Context) error {

	return nil
}

func (p *ImagePredictor) Close() error {
	if p.predictor != nil {
		p.predictor.Close()
	}

	return nil
}

func init() {
	config.AfterInit(func() {
		framework := caffe2.FrameworkManifest
		agent.AddPredictor(framework, &ImagePredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}

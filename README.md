# MobileNet

A tensorflow implementation of Google's [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

## Base Module

<div align="center">
<img src="https://github.com/Zehaos/MobileNet/blob/master/figures/dwl_pwl.png"><br><br>
</div>

## Accuracy on ImageNet-2012 Validation Set

| Model | Width Multiplier |Preprocessing  | Accuracy-Top1|Accuracy-Top5 |
|--------|:---------:|:------:|:------:|:------:|
| MobileNet |1.0| Same as Inception | 66.51% | 87.09% |

[Click](https://pan.baidu.com/s/1i5xFjal) to download the pretrained weight.

**Loss**
<div align="center">
<img src="https://github.com/Zehaos/MobileNet/blob/master/figures/epoch90_full_preprocess.png"><br><br>
</div>

## Time Benchmark
Environment: Ubuntu 16.04 LTS, Xeon E3-1231 v3, 4 Cores @ 3.40GHz, GTX1060.

TF 1.0.1(native pip install), TF 1.1.0(build from source, optimization flag '-mavx2')

Input image size: (224, 224, 3)

Batch size: 1

| Device | Forward| Forward-Backward |Instruction set|Quantized|Fused-BN|Remark|
|--------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|CPU|52ms|503ms|-|-|-|TF 1.0.1|
|CPU|44ms|177ms|-|-|On|TF 1.0.1|
|CPU|31ms| - |-|8-bits|-|TF 1.0.1|
|CPU|26ms| 75ms|AVX2|-|-|TF 1.1.0|
|CPU|128ms| - |AVX2|8-bits|-|TF 1.1.0|
|CPU|**19ms**| 89ms|AVX2|-|On|TF 1.1.0|
|GPU|3ms|16ms|-|-|-|TF 1.0.1, CUDA8.0, CUDNN5.1|
|GPU|**3ms**|15ms|-|-|On|TF 1.0.1, CUDA8.0, CUDNN5.1|

## Usage

### Train on Imagenet

1. Prepare imagenet data. Please refer to Google's tutorial for [training inception](https://github.com/tensorflow/models/tree/master/inception#getting-started).

2. Modify './script/train_mobilenet_on_imagenet.sh' according to your environment.

```
bash ./script/train_mobilenet_on_imagenet.sh
```

### Benchmark speed
```
python time_benchmark.py
```

## Trouble Shooting

1. About the MobileNet model size

According to the paper, MobileNet has 3.3 Million Parameters, which does not vary based on the input resolution. It means that the number of final model parameters should be larger than 3.3 Million, because of the fc layer.

When using RMSprop training strategy, the checkpoint file size should be almost 3 times as large as the model size, because of some auxiliary parameters used in RMSprop. You can use the inspect_checkpoint.py to figure it out.

2. Slim multi-gpu performance problems

[#1390](https://github.com/tensorflow/models/issues/1390)

## TODO
- [x] Train on Imagenet
- [x] Add Width Multiplier Hyperparameters
- [x] Report training result
- [ ] Intergrate into object detection task

## Reference
[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

[SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)

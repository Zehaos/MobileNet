# MobileNet

A tensorflow implementation of Google's [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

## Base Module

<div align="center">
<img src="https://github.com/Zehaos/MobileNet/blob/master/figures/dwl_pwl.png"><br><br>
</div>

## Known bug

[#issues1](https://github.com/Zehaos/MobileNet/issues/1)

[Is it OK to mix tf layer with slim layer?](https://stackoverflow.com/questions/43538420/can-not-mix-slim-layer-with-tensorflow)

## Usage

### First

Prepare imagenet data.

Please refer to Google's tutorial for [training inception](https://github.com/tensorflow/models/tree/master/inception#getting-started).

### Second

Modify './script/train_mobilenet_on_imagenet.sh' according to your environment.

```
bash ./script/train_mobilenet_on_imagenet.sh
```


## TODO
- [x] Train on Imagenet
- [x] Add Width Multiplier Hyperparameters
- [ ] Report training result
- [ ] Intergrate into object detection task

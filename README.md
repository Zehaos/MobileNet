# MobileNet

A tensorflow implementation of Google's [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

## Usage

### First

Preparing imagenet data

Please refer to Google's tutorial for [training inception](https://github.com/tensorflow/models/tree/master/inception#getting-started)

### Second

Modify './script/train_mobilenet_on_imagenet.sh' according to your environment.

```
bash ./script/train_mobilenet_on_imagenet.sh
```

## TODO
- [x] Train on Imagenet
- [ ] Add Hyperparameters
- [ ] Report training result
- [ ] Intergrate into object detection task

# Xception
A PyTorch implementation of Xception

### Overview

This repository is a PyTorch reimplementation of [Xception](https://arxiv.org/abs/1610.02357), and almost is an op-to-op translation from the [official implementation](https://github.com/tensorflow/models/blob/master/research/deeplab/core/xception.py). Moreover, we provide a function to convert the official TensorFlow pretrained weights(which can be download in [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md)) to PyTorch weights, hence it is very convenient to infer or finetune your own datasets.

As mentioned in the official version, the Xception implemented here made a few more changes:
1. Fully convolutional: All the max-pooling layers are replaced with separable
  conv2d with stride = 2. This allows us to use atrous convolution to extract
  feature maps at any resolution.

2. We support adding ReLU and BatchNorm after depthwise convolution, motivated
  by the design of MobileNetv1.
 
At the moment, you can easily:
+ Load pretrained Xception models
+ Use Xception models for classification or feature extraction.
 
 
### Usage

##### Convert pretrained weights

First, you need to download the **official pretrained weights** at the bottom of the [page](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md). There are three pretrained models:  **xception_xx_imagenet**, where xx is one of [**41**, **65**, **71**]. Then, run the following command:
```
python3 xception_test.py --tf_checkpoint_path "xxxx.....xxx/model.ckpt" --model_name "xception_xx"
```
You will find a new created folder 'pretrained_models' where the output pytorch model file is stored, and print a few lines in console like this (if model_name is not specified or model_name == xception_65):
```
TensorFlow predicion:
[286]
[[279 288 282 283 286]]
PyTorch prediction:
[286]
[[279 288 282 283 286]]
Save model to:  ./pretrained_models/xception_65.pth
Load pretrained weights successfully.
PyTorch prediction:
[286]
[[279 288 282 283 286]]
```

##### Load pretrained models

Load a Xception:
```
import xception
xception_65 = xception.xception_65(pretrained=False)
```
Load a pretrained Xception:
```
import xception
xception_65 = xception.xception_65(pretrained=True)
```

##### Example: Classification

In this case, `num_classes` must be specified, like this:
```
import xception
model = xception.xception_65(num_classes=8, pretrained=True)
```

##### Example: Feature extration

In this case, please set the keywords `num_classes=None, global_pool=False`:
```
import xception
model = xception.xception_65(num_classes=None, global_pool=False, pretrained=True)
```

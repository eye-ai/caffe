# caffe-augmentation

Caffe with real-time data augmentation


## Introduction
Data augmentation is a simple yet effective way to enrich training data. However, we don't want to re-create a dataset (such as ImageNet) with more than millions of images every time when we change our augmentation strategy. To address this problem, this project provides real-time training data augmentation. During training, caffe will augment training data with random combination of different geometric transformations (scaling, rotation, cropping), image variations (blur, sharping, JPEG compression), and lighting adjustments.

<img src="https://www.csie.ntu.edu.tw/~r01944012/bb.gif" width="500">


## Realtime data augmentation
Realtime data augmentation is implemented within the `ImageData` layer. We provide several augmentations as below:
- Geometric transform: random flipping, cropping, resizing, rotation
- Smooth filtering
- JPEG compression
- Contrast & brightness adjustment


## How to use
You could specify your network prototxt as:

    layer {
    name: "data"
    type: "ImageData"
    top: "data"
    top: "label"
    include {
      phase: TRAIN
    }
    transform_param {
      mirror: true
      crop_size: 227
      mean_file: "/home/your/imagenet_mean.binaryproto"
      contrast_adjustment: true
      smooth_filtering: true
      jpeg_compression: true
      rotation_angle_interval: 30
      display: true
    }
    image_data_param {
      source: "/home/your/image/list.txt"
      batch_size: 32
      shuffle: true
      new_height: 256
      new_width: 256
    }
    }

You could also find a toy example at `/examples/SSDH/train_val.prototxt`

Note: ImageData Layer is currently not supported in TEST mode


## Setup caffe-augmentation
Adjust Makefile.config and simply run the following commands:

    $ make all -j8

For a faster build, compile in parallel by doing `make all -j8` where 8 is the number of parallel threads for compilation (a good choice for the number of threads is the number of cores in your machine).


## Acknowledgment
This project is based upon [@ChenlongChen](https://github.com/ChenglongChen)'s [caffe-windows](https://github.com/ChenglongChen/caffe-windows), [@ShaharKatz](https://github.com/ShaharKatz)'s [Caffe-Data-Augmentation](https://github.com/ShaharKatz/Caffe-Data-Augmentation), and [@senecaur](https://github.com/senecaur)'s [caffe-rta](https://github.com/senecaur/caffe-rta). Thank you for your inspiration!






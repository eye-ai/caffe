# Caffe For Image Search
Customized `caffe` for Image Search. Installation is the same as `BVLC/caffe` [installation](http://caffe.berkeleyvision.org/installation.html)

## Base caffe version: 
`BVLC/caffe v1.0` https://github.com/BVLC/caffe/tree/1.0

## Additional Components
Check on each public repository for documentation on additional layers and fearures.

### 1. Caffe layers from `Lifted Structured Feature Embedding`
From: https://github.com/rksltnl/Deep-Metric-Learning-CVPR16.git

Layers added:
* `lifted_struct_similarity_softmax_layer`

File change: 
* `src/caffe/proto/caffe.proto`

### 2. Caffe layers from `Hard-Aware-Deeply-Cascaded-Embedding`
From: https://github.com/PkuRainBow/Hard-Aware-Deeply-Cascaded-Embedding_release.git

Layers added:
* `normalization_layer`
* `pair_fast_loss_layer`

File change: 
* `src/caffe/proto/caffe.proto`

### 3. Caffe with real-time data augmentation
From: https://github.com/kevinlin311tw/caffe-augmentation.git

Main changes: 
* `src/caffe/data_transformer.cpp`
* `src/caffe/proto/caffe.proto`

### 4. Caffe layers from `Center Loss`
From: https://github.com/ydwen/caffe-face

Layers added:
* `center_loss_layer`

File change:
* `include/caffe/layers/center_loss_layer.hpp`
* `src/caffe/layers/center_loss_layer.cpp`
* `src/caffe/layers/center_loss_layer.cu`
* `src/caffe/proto/caffe.proto`

#include <algorithm>
#include <vector>
#include <stdio.h>
#include "caffe/layers/contrastive_batch_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ContrastiveBatchLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int npairs = (num*(num-1))/2;
  diff_.Reshape(npairs, channels, 1, 1);
  dist_sq_.Reshape(npairs, 1, 1, 1);
}

template <typename Dtype>
void ContrastiveBatchLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->height(), 1);
    CHECK_EQ(bottom[0]->width(), 1);
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int npairs = (num*(num-1))/2;
    diff_.Reshape(npairs, channels, 1, 1);
    dist_sq_.Reshape(npairs, 1, 1, 1);

    vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
    top[0]->Reshape(loss_shape);
}


template <typename Dtype>
void ContrastiveBatchLossLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int npairs = (num*(num-1))/2;
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  const int image_batch_copies = this->layer_param_.contrastive_loss_param().image_batch_copies();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  Dtype loss(0.0);
  for (int i = 0, c = 0; i < bottom[0]->num()-1; ++i) {
      for (int j = i+1; j < bottom[0]->num(); ++j, ++c) {
          caffe_sub(
              channels,
              bottom[0]->cpu_data()+(i*channels),  // a
              bottom[0]->cpu_data()+(j*channels),  // b
              diff_.mutable_cpu_data()+(c*channels));  // a_i-b_i
          dist_sq_.mutable_cpu_data()[c] = caffe_cpu_dot(channels,
            diff_.cpu_data() + (c*channels), diff_.cpu_data() + (c*channels));
          if ((i / image_batch_copies) == (j / image_batch_copies)) { // similar pairs
            loss += dist_sq_.cpu_data()[c];
          } else {  // dissimilar pairs
            if (legacy_version) {
              loss += std::max(margin - dist_sq_.cpu_data()[c], Dtype(0.0));
            } else {
              Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq_.cpu_data()[c]),
                Dtype(0.0));
              loss += dist*dist;
            }
          }
      }
  }
  loss = loss / static_cast<Dtype>(npairs) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ContrastiveBatchLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    Dtype margin = this->layer_param_.contrastive_loss_param().margin();
    bool legacy_version =
        this->layer_param_.contrastive_loss_param().legacy_version();
    if (!propagate_down[0])
        return;
    Dtype* bout = bottom[0]->mutable_cpu_diff();
    const int num = bottom[0]->num();
    const int npairs = (num*(num-1))/2;
    const Dtype alpha_p = top[0]->cpu_diff()[0] /
        static_cast<Dtype>(npairs);
    const Dtype alpha_n = -alpha_p;
    int channels = bottom[0]->channels();
    int image_batch_copies = this->layer_param_.contrastive_loss_param().image_batch_copies();

    caffe_set(channels*num, Dtype(0), bout);
    for (int i = 0, c = 0; i < bottom[0]->num()-1; ++i) {
        for (int j = i+1; j < bottom[0]->num(); ++j, ++c) {
            if (i / image_batch_copies == j / image_batch_copies) { // similar pairs
                caffe_cpu_axpby(
                        channels,
                        alpha_p,
                        diff_.cpu_data() + (c*channels),
                        Dtype(1.0),
                        bout + (i*channels));
                caffe_cpu_axpby(
                        channels,
                        alpha_n,
                        diff_.cpu_data() + (c*channels),
                        Dtype(1.0),
                        bout + (j*channels));
            } else {  // dissimilar pairs
                Dtype mdist(0.0);
                Dtype beta_p(0.0);
                Dtype beta_n(0.0);
                if (legacy_version) {
                    mdist = margin - dist_sq_.cpu_data()[c];
                    beta_p = -alpha_p;
                    beta_n = -alpha_n;
                } else {
                    Dtype dist = sqrt(dist_sq_.cpu_data()[c]);
                    mdist = margin - dist;
                    beta_p = -alpha_p * mdist / (dist + Dtype(1e-4));
                    beta_n = -alpha_n * mdist / (dist + Dtype(1e-4));
                }
                if (mdist > Dtype(0.0)) {
                    caffe_cpu_axpby(
                            channels,
                            beta_p,
                            diff_.cpu_data() + (c*channels),
                            Dtype(1.0),
                            bout + (i*channels));
                    caffe_cpu_axpby(
                            channels,
                            beta_n,
                            diff_.cpu_data() + (c*channels),
                            Dtype(1.0),
                            bout + (j*channels));
                }

            }
        }
    }
}

INSTANTIATE_CLASS(ContrastiveBatchLossLayer);
REGISTER_LAYER_CLASS(ContrastiveBatchLoss);

}  // namespace caffe

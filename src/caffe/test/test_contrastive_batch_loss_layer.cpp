#include <algorithm>
#include <cmath>
#include <vector>
#include <stdio.h>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/contrastive_batch_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ContrastiveBatchLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;


 protected:
  ContrastiveBatchLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(4, 2, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~ContrastiveBatchLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_loss_;
  }

  Dtype dist_sq_calc(const Dtype *a, const Dtype *b, int n)
  {
      Dtype dist_sq(0);
      for (int j = 0; j < n; ++j) {
          Dtype diff = a[j] - b[j];
          dist_sq += diff*diff;
      }
      return dist_sq;
  }

  Dtype loss_single(bool same, bool legacy, Dtype margin, const Dtype *a, const Dtype *b, int n)
  { 
      Dtype dist_sq = dist_sq_calc(a, b, n);
      if (same)
          return dist_sq;
      if (legacy)
          return std::max(margin - dist_sq, Dtype(0.0));
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      return dist*dist;
  }


  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ContrastiveBatchLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(ContrastiveBatchLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_contrastive_loss_param()->set_image_batch_copies(2);
  ContrastiveBatchLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const Dtype margin = layer_param.contrastive_loss_param().margin();
  const int num = this->blob_bottom_data_->num();
  Dtype loss(0);
  const Dtype *bottom = this->blob_bottom_data_->cpu_data();
  loss = this->loss_single(true, false, margin, bottom, bottom+1*2, 2)+
      this->loss_single(true, false, margin, bottom+2*2, bottom+3*2, 2)+
      this->loss_single(false, false, margin, bottom, bottom+2*2, 2)+
      this->loss_single(false, false, margin, bottom, bottom+3*2, 2)+
      this->loss_single(false, false, margin, bottom+1*2, bottom+2*2, 2)+
      this->loss_single(false, false, margin, bottom+1*2, bottom+3*2, 2);
  loss /= static_cast<Dtype>(6) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
}

TYPED_TEST(ContrastiveBatchLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_contrastive_loss_param()->set_image_batch_copies(2);
  ContrastiveBatchLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(ContrastiveBatchLossLayerTest, TestLegacyForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_contrastive_loss_param()->set_image_batch_copies(2);
  layer_param.mutable_contrastive_loss_param()->set_legacy_version(true);
  ContrastiveBatchLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const Dtype margin = layer_param.contrastive_loss_param().margin();
  const int num = this->blob_bottom_data_->num();
  Dtype loss(0);
  const Dtype *bottom = this->blob_bottom_data_->cpu_data();
  loss = this->loss_single(true, true, margin, bottom, bottom+1*2, 2)+
      this->loss_single(true, true, margin, bottom+2*2, bottom+3*2, 2)+
      this->loss_single(false, true, margin, bottom, bottom+2*2, 2)+
      this->loss_single(false, true, margin, bottom, bottom+3*2, 2)+
      this->loss_single(false, true, margin, bottom+1*2, bottom+2*2, 2)+
      this->loss_single(false, true, margin, bottom+1*2, bottom+3*2, 2);
  loss /= static_cast<Dtype>(6) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
}

TYPED_TEST(ContrastiveBatchLossLayerTest, TestGradientLegacy) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_contrastive_loss_param()->set_legacy_version(true);
  layer_param.mutable_contrastive_loss_param()->set_image_batch_copies(2);
  ContrastiveBatchLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
}  // namespace caffe

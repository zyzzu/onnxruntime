/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#include "roialign_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T, bool is_mode_avg>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  if (is_mode_avg) {
    return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  } else {
    return max(max(max(w1 * v1, w2 * v2), w3 * v3), w4 * v4);  // mode Max
  }
}

static const int ROI_COLS = 4;

template <typename T, bool is_mode_avg>
__global__ void RoIAlignForward(
    const int64_t nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t stride_HxW,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const fast_divmod div_pooled_W,
    const fast_divmod div_pooled_HxW,
    const fast_divmod div_pooled_CxHxW,
    const int64_t sampling_ratio,
    const T* bottom_rois,
    T* top_data,
    const int64_t* batch_indices_ptr,
    const T min_T_value) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(index, nthreads);
  {
    // (n, c, ph, pw) is an element in the pooled output
    int n, c, ph, pw;
    div_pooled_CxHxW.divmod(index, n, pw);
    div_pooled_HxW.divmod(pw, c, pw);
    div_pooled_W.divmod(pw, ph, pw);

    // RoI must have 4 columns
    const T* offset_bottom_rois = bottom_rois + n * ROI_COLS;

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;

    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);

    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    const int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(bin_size_h);
    const int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(bin_size_w);

    T output_val = min_T_value;
    const T* offset_bottom_data =
        bottom_data + static_cast<int64_t>((batch_indices_ptr[n] * channels + c) * stride_HxW);
    for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate<T, is_mode_avg>(
            offset_bottom_data, height, width, y, x, index);

        if (is_mode_avg) {
          output_val += val;
        } else {
          output_val = max(output_val, val);
        }
      }
    }
    if (is_mode_avg) {
      output_val /= (roi_bin_grid_h * roi_bin_grid_w);
    }

    top_data[index] = output_val;
  }
}

template <typename T>
T MinValueOf() {
  return -std::numeric_limits<T>::infinity();
}

template <typename T>
void RoiAlignImpl(
    const int64_t nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t sampling_ratio,
    const T* bottom_rois,
    T* top_data,
    const bool is_mode_avg,
    const int64_t* batch_indices_ptr) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(nthreads) / GridDim::maxThreadsPerBlock));
  T min_T_value = is_mode_avg ? T(0.) : MinValueOf<T>();
  const fast_divmod div_pooled_W(pooled_width);
  const fast_divmod div_pooled_HxW(pooled_width * pooled_height);
  const fast_divmod div_pooled_CxHxW(pooled_height * pooled_width * channels);

  if (is_mode_avg) {
    RoIAlignForward<T, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        nthreads,
        bottom_data,
        spatial_scale,
        channels,
        height,
        width,
        height * width,
        pooled_height,
        pooled_width,
        div_pooled_W,
        div_pooled_HxW,
        div_pooled_CxHxW,
        sampling_ratio,
        bottom_rois,
        top_data,
        batch_indices_ptr,
        min_T_value);
  } else {
    RoIAlignForward<T, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        nthreads,
        bottom_data,
        spatial_scale,
        channels,
        height,
        width,
        height * width,
        pooled_height,
        pooled_width,
        div_pooled_W,
        div_pooled_HxW,
        div_pooled_CxHxW,
        sampling_ratio,
        bottom_rois,
        top_data,
        batch_indices_ptr,
        min_T_value);
  }
}

// template __device__ T bilinear_interpolate<T, true>(  \
//   const T* bottom_data,                             \
//   const int height,                                 \
//   const int width,                                  \
//   T y,                                              \
//   T x,                                              \
//   const int index);                                 \
// template __device__ T bilinear_interpolate<T, false>( \
//   const T* bottom_data,                             \
//   const int height,                                 \
//   const int width,                                  \
//   T y,                                              \
//   T x,                                              \
//   const int index);                                 \

#define SPECIALIZED_IMPL(T)         \
  template void RoiAlignImpl<T>(    \
      const int64_t nthreads,       \
      const T* bottom_data,         \
      const T spatial_scale,        \
      const int64_t channels,       \
      const int64_t height,         \
      const int64_t width,          \
      const int64_t pooled_height,  \
      const int64_t pooled_width,   \
      const int64_t sampling_ratio, \
      const T* bottom_rois,         \
      T* top_data,                  \
      const bool is_mode_avg,       \
      const int64_t* batch_indices_ptr);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime

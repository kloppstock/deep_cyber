#include "../include/conv2d.h"
#include <assert.h>

Tensor conv2d(Tensor X, Tensor w, Tensor b, uint16_t stride_rows,
              uint16_t stride_cols, uint16_t padding, uint16_t groups) {
  // extract parameters
  uint16_t kernel_rows = w.a;
  uint16_t kernel_cols = w.b;
  uint16_t channels_in = w.c;
  uint16_t channels_out = w.d;

  uint16_t batch = X.a;
  uint16_t rows_in = X.b;
  uint16_t cols_in = X.c;
  uint16_t channels_in_ = X.d;

  uint grouped_channels_out = channels_out / groups;

  assert((channels_in * groups == channels_in_) &&
         "Error: Invalid number of groups or input channels in Conv2D!\n");
  assert((channels_out % groups == 0) &&
         "Error: Invalid number of groups in Conv2D!\n");

  // same padding
  if (padding) {
    uint16_t rows_offset = kernel_rows / 2;
    uint16_t cols_offset = kernel_cols / 2;

    // calculate output dimensions and create output tensor
    uint16_t rows_out = (rows_in + stride_rows - 1) / stride_rows;
    uint16_t cols_out = (cols_in + stride_cols - 1) / stride_cols;
    Tensor out =
        create_tensor(batch, rows_out, cols_out, groups * channels_out);

    // prefill output with bias
    uint16_t i, y, x, g, co, ci;
    for (i = 0; i < batch; ++i)
      for (y = 0; y < rows_out; ++y)
        for (x = 0; x < cols_out; ++x)
          for (g = 0; g < groups; ++g)
            for (co = 0; co < grouped_channels_out; ++co)
              *at4(&out, i, y, x, g * grouped_channels_out + co) = *at1(&b, co);

    // convolute
    for (i = 0; i < batch; ++i) {
      for (y = 0; y < rows_out; ++y) {
        for (x = 0; x < cols_out; ++x) {
          int16_t sy = y * stride_rows - rows_offset;
          int16_t sx = x * stride_cols - cols_offset;
          for (g = 0; g < groups; ++g) {
            uint16_t gc = g * grouped_channels_out;
            for (co = 0; co < grouped_channels_out; ++co) {
              for (ci = 0; ci < channels_in; ++ci) {
                int16_t ky, kx;
                for (ky = sy; ky < sy + kernel_rows; ++ky) {
                  for (kx = sx; kx < sx + kernel_cols; ++kx) {
                    if (ky >= 0 && ky < rows_in && kx >= 0 && kx < cols_in) {
                      *at4(&out, i, y, x, gc + co) +=
                          *at4(&X, i, ky, kx, gc + ci) *
                          *at4(&w, ky - sy, kx - sx, ci, co);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    return out;

    // zero padding
  } else {
    // calculate output dimensions and create output tensor
    uint16_t rows_out = (rows_in - (kernel_rows - stride_rows)) / stride_rows;
    uint16_t cols_out = (cols_in - (kernel_cols - stride_cols)) / stride_cols;
    Tensor out =
        create_tensor(batch, rows_out, cols_out, groups * channels_out);

    // prefill output with bias
    uint16_t i, y, x, g, co, ci;
    for (i = 0; i < batch; ++i)
      for (y = 0; y < rows_out; ++y)
        for (x = 0; x < cols_out; ++x)
          for (g = 0; g < groups; ++g)
            for (co = 0; co < grouped_channels_out; ++co)
              *at4(&out, i, y, x, g * grouped_channels_out + co) = *at1(&b, co);

    // convolute
    for (i = 0; i < batch; ++i) {
      for (y = 0; y < rows_out; ++y) {
        for (x = 0; x < cols_out; ++x) {
          int16_t sy = y * stride_rows;
          int16_t sx = x * stride_cols;
          for (g = 0; g < groups; ++g) {
            uint16_t gc = g * grouped_channels_out;
            for (co = 0; co < grouped_channels_out; ++co) {
              for (ci = 0; ci < channels_in; ++ci) {
                int16_t ky, kx;
                for (ky = sy; ky < sy + kernel_rows; ++ky) {
                  for (kx = sx; kx < sx + kernel_cols; ++kx) {
                    *at4(&out, i, y, x, gc + co) +=
                        *at4(&X, i, ky, kx, gc + ci) *
                        *at4(&w, ky - sy, kx - sx, ci, co);
                  }
                }
              }
            }
          }
        }
      }
    }

    return out;
  }
}

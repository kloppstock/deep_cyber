#include "../include/deep_cyber.h"
#include <assert.h>

/* general 2D convolution */
Tensor conv2d(Tensor X, Tensor w, Tensor b, uint32_t stride_rows,
              uint32_t stride_cols, char padding, uint32_t groups) {
  /* extract parameters */
  uint32_t kernel_rows = w.a;
  uint32_t kernel_cols = w.b;
  uint32_t channels_in = w.c;
  uint32_t channels_out = w.d;

  uint32_t batch = X.a;
  int32_t rows_in = X.b;
  int32_t cols_in = X.c;
  uint32_t channels_in_ = X.d;

  uint32_t grouped_channels_out = channels_out / groups;

  assert((channels_in * groups == channels_in_) &&
         "Error: Invalid number of groups or input channels in Conv2D!\n");
  assert((channels_out % groups == 0) &&
         "Error: Invalid number of groups in Conv2D!\n");

  /* same padding */
  if (padding) {
    uint32_t rows_offset = kernel_rows / 2;
    uint32_t cols_offset = kernel_cols / 2;

    /* calculate output dimensions and create output tensor */
    uint32_t rows_out = (rows_in + stride_rows - 1) / stride_rows;
    uint32_t cols_out = (cols_in + stride_cols - 1) / stride_cols;
    Tensor out =
        create_tensor(batch, rows_out, cols_out, groups * channels_out);

    /* prefill output with bias */
    uint32_t i, y, x, g, co, ci;
    for (i = 0; i < batch; ++i)
      for (y = 0; y < rows_out; ++y)
        for (x = 0; x < cols_out; ++x)
          for (g = 0; g < groups; ++g)
            for (co = 0; co < grouped_channels_out; ++co)
              AT4(out, i, y, x, g * grouped_channels_out + co) = AT1(b, co);

    /* convolute */
    for (i = 0; i < batch; ++i) {
      for (y = 0; y < rows_out; ++y) {
        for (x = 0; x < cols_out; ++x) {
          int32_t sy = y * stride_rows - rows_offset;
          int32_t sx = x * stride_cols - cols_offset;
          for (g = 0; g < groups; ++g) {
            uint32_t gc = g * grouped_channels_out;
            for (co = 0; co < grouped_channels_out; ++co) {
              for (ci = 0; ci < channels_in; ++ci) {
                int32_t ky, kx;
                for (ky = sy; ky < sy + (int32_t)kernel_rows; ++ky) {
                  for (kx = sx; kx < sx + (int32_t)kernel_cols; ++kx) {
                    if (ky >= 0 && ky < rows_in && kx >= 0 && kx < cols_in) {
                      AT4(out, i, y, x, gc + co) +=
                          AT4(X, i, ky, kx, gc + ci) *
                          AT4(w, ky - sy, kx - sx, ci, co);
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

    /* zero padding */
  } else {
    /* calculate output dimensions and create output tensor */
    uint32_t rows_out = (rows_in - (kernel_rows - stride_rows)) / stride_rows;
    uint32_t cols_out = (cols_in - (kernel_cols - stride_cols)) / stride_cols;
    Tensor out =
        create_tensor(batch, rows_out, cols_out, groups * channels_out);

    /* prefill output with bias */
    uint32_t i, y, x, g, co, ci;
    for (i = 0; i < batch; ++i)
      for (y = 0; y < rows_out; ++y)
        for (x = 0; x < cols_out; ++x)
          for (g = 0; g < groups; ++g)
            for (co = 0; co < grouped_channels_out; ++co)
              AT4(out, i, y, x, g * grouped_channels_out + co) = AT1(b, co);

    /* convolute */
    for (i = 0; i < batch; ++i) {
      for (y = 0; y < rows_out; ++y) {
        for (x = 0; x < cols_out; ++x) {
          uint32_t sy = y * stride_rows;
          uint32_t sx = x * stride_cols;
          for (g = 0; g < groups; ++g) {
            uint32_t gc = g * grouped_channels_out;
            for (co = 0; co < grouped_channels_out; ++co) {
              for (ci = 0; ci < channels_in; ++ci) {
                uint32_t ky, kx;
                for (ky = sy; ky < sy + kernel_rows; ++ky) {
                  for (kx = sx; kx < sx + kernel_cols; ++kx) {
                    AT4(out, i, y, x, gc + co) +=
                        AT4(X, i, ky, kx, gc + ci) *
                        AT4(w, ky - sy, kx - sx, ci, co);
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

#include "../include/deep_cyber.h"
#include <assert.h>

/* general 2D convolution */
Tensor conv2d(Tensor X, Tensor w, Tensor b, unsigned int stride_rows,
              unsigned int stride_cols, char padding, unsigned int groups) {
  /* extract parameters */
  int kernel_rows = w.a;
  int kernel_cols = w.b;
  unsigned int channels_in = w.c;
  unsigned int channels_out = w.d;

  unsigned int batch = X.a;
  int rows_in = X.b;
  int cols_in = X.c;
  unsigned int channels_in_ = X.d;

  unsigned int grouped_channels_out = channels_out / groups;

  assert((channels_in * groups == channels_in_) &&
         "Error: Invalid number of groups or input channels in Conv2D!\n");
  assert((channels_out % groups == 0) &&
         "Error: Invalid number of groups in Conv2D!\n");

  /* same padding */
  if (padding) {
    unsigned int rows_offset = 2; /*kernel_rows / 4; */
    unsigned int cols_offset = 2; /*kernel_cols / 4; */

    /* calculate output dimensions and create output tensor */
    unsigned int rows_out = (rows_in + stride_rows - 1) / stride_rows;
    unsigned int cols_out = (cols_in + stride_cols - 1) / stride_cols;
    Tensor out =
        create_tensor(batch, rows_out, cols_out, groups * channels_out);

    /* prefill output with bias */
    unsigned int i, y, x, g, co, ci;
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
          int sy = y * stride_rows - rows_offset;
          int sx = x * stride_cols - cols_offset;
          for (g = 0; g < groups; ++g) {
            unsigned int gc = g * grouped_channels_out;
            for (co = 0; co < grouped_channels_out; ++co) {
              for (ci = 0; ci < channels_in; ++ci) {
                int ky, kx;
                for (ky = sy; ky < sy + kernel_rows; ++ky) {
                  for (kx = sx; kx < sx + kernel_cols; ++kx) {
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
    unsigned int rows_out =
        (rows_in - (kernel_rows - stride_rows)) / stride_rows;
    unsigned int cols_out =
        (cols_in - (kernel_cols - stride_cols)) / stride_cols;
    Tensor out =
        create_tensor(batch, rows_out, cols_out, groups * channels_out);

    /* prefill output with bias */
    unsigned int i, y, x, g, co, ci;
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
          int sy = y * stride_rows;
          int sx = x * stride_cols;
          for (g = 0; g < groups; ++g) {
            unsigned int gc = g * grouped_channels_out;
            for (co = 0; co < grouped_channels_out; ++co) {
              for (ci = 0; ci < channels_in; ++ci) {
                int ky, kx;
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

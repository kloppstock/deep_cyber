#include "../include/convolutional_layer.h"
#include <stdlib.h>

convolutional_layer alloc_convolutional_layer(uint8_t size, uint8_t stride,
                                              uint8_t filters, uint8_t padding,
                                              float (*activate)(float)) {
  return {size,
          stride,
          filters,
          padding,
          (float *)calloc(size * size * stride, sizeof(float)),
          (float *)calloc(filters, sizeof(float)),
          activate};
}

void free_convolutional_layer(convolutional_layer l) {
  free(l.weights);
  free(l.biases);
}

void run_convolution(matrix in, convolutional_layer layer, matrix out) {
  // prefill the output with the bias
  int i;
  for (i = 0; i < out.channels * out.height * out.width; ++i) {
    // calculate filter bias position
    uint16_t bias_position = i / out.height / out.width;

    // copy the bias value
    out.data[i] = layer.biases[bias_position];
  }

  // run horizontal part of the convolution
  for (i = 0; i < in.channels * in.height * in.width; ++i) {

    // calculate index in image
    uint16_t c = i / in.height * in.width;
    uint16_t y = (i % in.height * in.width) / in.width;
    uint16_t x = i % in.width;

    // iterate over all possible kernel positions
    int half_size = layer.size / 2;
    int kx, ky;
    for (ky = -half_size; ky < layer.size - half_size; ++ky) {
      for (kx = -half_size; kx < layer.size - half_size; ++kx) {

        // check if kernel position is potential center
        if ((kx + x) % layer.stride == 0 && kx + x > 0 && kx + x < in.width &&
            (ky + y) % layer.stride == 0 && ky + y > 0 && ky + y < in.height) {

          // calculate buffer output, input and kernel position
          uint16_t in_position = c * in.channels * in.height * in.width +
                                 (y + ky) * in.width + (x + kx);
          uint16_t out_position = c * in.channels * in.height * in.width +
                                  (y + ky) * in.width / layer.stride +
                                  (x + kx) / layer.stride;
          uint16_t kernel_position =
              (ky + half_size) * layer.size + kx + half_size;

          // iterate over all filters
          int f;
          for (f = 0; f < layer.filters; ++f) {

            // calculate filter position
            uint16_t filter_position =
                f * layer.size * layer.size + kernel_position;

            // calculate intermediate output
            out.data[out_position] +=
                in.data[in_position] * layer.weights[filter_position];
          }
        }
      }
    }

    // activate the layer
    for (i = 0; i < out.channels * out.height * out.width; ++i) {
      out.data[i] = layer.activate(out.data[i]);
    }
  }
}

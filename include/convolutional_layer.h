#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "../include/matrix.h"
#include <stdint.h>

typedef struct convolutional_layer {
  uint8_t size, stride, filters;
  uint8_t padding;

  float *weights;
  float *biases;

  float (*activate)(float);
} convolutional_layer;

convolutional_layer alloc_convolutional_layer(uint16_t width, uint16_t height,
                                              uint16_t channels, uint8_t size,
                                              uint8_t stride, uint8_t filters,
                                              bool padding,
                                              void (*activate)(matrix));
void free_convolutional_layer(convolutional_layer);
void run_convolution(matrix, convolutional_layer, matrix);

#endif // CONVOLUTIONAL_LAYER_H

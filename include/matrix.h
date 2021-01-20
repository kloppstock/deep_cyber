#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

typedef struct matrix {
  uint16_t width, height, channels;
  float *data;
} matrix;

matrix alloc_matrix(uint16_t width, uint16_t height, uint16_t channels);
void free_matrix(matrix);

#endif // MATRIX_H

#include "../include/matrix.h"

#include <stdlib.h>

matrix alloc_matrix(uint16_t width, uint16_t height, uint16_t channels) {
  return {width, height, channels,
          (float *)calloc(width * height * channels, sizeof(float))};
}

void free_matrix(matrix m) {
    free(m.data);
}

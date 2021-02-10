#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Tensor {
  uint8_t a, b, c, d;
  float *data;
} Tensor;

Tensor create_tensor(uint8_t a, uint8_t b, uint8_t c, uint8_t d);
Tensor create_tensor4(uint8_t a, uint8_t b, uint8_t c, uint8_t d);
Tensor create_tensor3(uint8_t a, uint8_t b, uint8_t c);
Tensor create_tensor2(uint8_t a, uint8_t b);
Tensor create_tensor1(uint8_t a);

void free_tensor(Tensor t);

void reshape(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d);
void reshape4(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d);
void reshape3(Tensor *t, uint8_t a, uint8_t b, uint8_t c);
void reshape2(Tensor *t, uint8_t a, uint8_t b);
void reshape1(Tensor *t, uint8_t a);

float *at(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d);
float *at4(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d);
float *at3(Tensor *t, uint8_t a, uint8_t b, uint8_t c);
float *at2(Tensor *t, uint8_t a, uint8_t b);
float *at1(Tensor *t, uint8_t a);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H

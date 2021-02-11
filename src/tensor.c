#include "../include/tensor.h"

Tensor create_tensor(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    Tensor t;
    t.a = a;
    t.b = b;
    t.c = c;
    t.d = d;
    t.data = (float *)malloc(a * b * c * d * sizeof(float));
    return t;
}

Tensor create_tensor4(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  return create_tensor(a, b, c, d);
}

Tensor create_tensor3(uint8_t a, uint8_t b, uint8_t c) {
  return create_tensor(1, a, b, c);
}

Tensor create_tensor2(uint8_t a, uint8_t b) {
  return create_tensor(1, 1, a, b);
}

Tensor create_tensor1(uint8_t a) { return create_tensor(1, 1, 1, a); }

void free_tensor(Tensor t) { free(t.data); }

void reshape(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  assert((t->a * t->b * t->c * t->d != a * b * c * d) &&
         "Error: Incompatible tensor shapes while reshaping!\n");

  t->a = a;
  t->b = b;
  t->c = c;
  t->d = d;
}

void reshape1(Tensor *t, uint8_t a) { reshape(t, 1, 1, 1, a); }

void reshape2(Tensor *t, uint8_t a, uint8_t b) { reshape(t, 1, 1, a, b); }

void reshape3(Tensor *t, uint8_t a, uint8_t b, uint8_t c) {
  reshape(t, 1, a, b, c);
}

void reshape4(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  reshape(t, a, b, c, d);
}

float *at(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  return &t->data[a * t->b * t->c * t->d + b * t->c * t->d + c * t->d + d];
}

float *at1(Tensor *t, uint8_t a) { return at(t, 1, 1, 1, a); }

float *at2(Tensor *t, uint8_t a, uint8_t b) { return at(t, 1, 1, a, b); }

float *at3(Tensor *t, uint8_t a, uint8_t b, uint8_t c) {
  return at(t, 1, a, b, c);
}

float *at4(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  return at(t, a, b, c, d);
}

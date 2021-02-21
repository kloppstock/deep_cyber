#include "../include/tensor.h"

/* general tensor creation function */
Tensor create_tensor(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  Tensor t;
  t.a = a;
  t.b = b;
  t.c = c;
  t.d = d;
  t.data = (float *)malloc((size_t)a * b * c * d * sizeof(float));
  return t;
}

// 4D creation wrapper */
Tensor create_tensor4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  return create_tensor(a, b, c, d);
}

/* 3D creation wrapper */
Tensor create_tensor3(uint32_t a, uint32_t b, uint32_t c) {
  return create_tensor(1, a, b, c);
}

/* 2D creation wrapper */
Tensor create_tensor2(uint32_t a, uint32_t b) {
  return create_tensor(1, 1, a, b);
}

/* 1D creation wrapper */
Tensor create_tensor1(uint32_t a) { return create_tensor(1, 1, 1, a); }

/* frees tensor data */
void free_tensor(Tensor t) {
  if (t.data)
    free(t.data);
}

/* general reshape function */
void reshape(Tensor *t, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  assert((t->a * t->b * t->c * t->d == a * b * c * d) &&
         "Error: Incompatible tensor shapes while reshaping!\n");

  t->a = a;
  t->b = b;
  t->c = c;
  t->d = d;
}

/* 4D reshape wrapper */
void reshape4(Tensor *t, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  reshape(t, a, b, c, d);
}

/* 3D reshape wrapper */
void reshape3(Tensor *t, uint32_t a, uint32_t b, uint32_t c) {
  reshape(t, 1, a, b, c);
}

/* 2D reshape wrapper */
void reshape2(Tensor *t, uint32_t a, uint32_t b) { reshape(t, 1, 1, a, b); }

/* 1D reshape wrapper */
void reshape1(Tensor *t, uint32_t a) { reshape(t, 1, 1, 1, a); }

/* general indexing function */
float *at(Tensor *t, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  return &t->data[(size_t)a * t->b * t->c * t->d + (size_t)b * t->c * t->d +
                  (size_t)c * t->d + (size_t)d];
}

/* 4D indexing wrapper */
float *at4(Tensor *t, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  return at(t, a, b, c, d);
}

/* 3D indexing wrapper */
float *at3(Tensor *t, uint32_t a, uint32_t b, uint32_t c) {
  return at(t, 0, a, b, c);
}

/* 2D indexing wrapper */
float *at2(Tensor *t, uint32_t a, uint32_t b) { return at(t, 0, 0, a, b); }

/* 1D indexing wrapper */
float *at1(Tensor *t, uint32_t a) { return at(t, 0, 0, 0, a); }

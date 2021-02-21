#include "../include/tensor.h"

/* general tensor creation function */
Tensor create_tensor(unsigned int a, unsigned int b, unsigned int c,
                     unsigned int d) {
  Tensor t;
  t.a = a;
  t.b = b;
  t.c = c;
  t.d = d;
  t.data = (float *)malloc((size_t)a * b * c * d * sizeof(float));
  return t;
}

/* 4D creation wrapper */
Tensor create_tensor4(unsigned int a, unsigned int b, unsigned int c,
                      unsigned int d) {
  return create_tensor(a, b, c, d);
}

/* 3D creation wrapper */
Tensor create_tensor3(unsigned int a, unsigned int b, unsigned int c) {
  return create_tensor(1, a, b, c);
}

/* 2D creation wrapper */
Tensor create_tensor2(unsigned int a, unsigned int b) {
  return create_tensor(1, 1, a, b);
}

/* 1D creation wrapper */
Tensor create_tensor1(unsigned int a) { return create_tensor(1, 1, 1, a); }

/* frees tensor data */
void free_tensor(Tensor t) {
  if (t.data)
    free(t.data);
}

/* general reshape function */
void reshape(Tensor *t, unsigned int a, unsigned int b, unsigned int c,
             unsigned int d) {
  assert((t->a * t->b * t->c * t->d == a * b * c * d) &&
         "Error: Incompatible tensor shapes while reshaping!\n");

  t->a = a;
  t->b = b;
  t->c = c;
  t->d = d;
}

/* 4D reshape wrapper */
void reshape4(Tensor *t, unsigned int a, unsigned int b, unsigned int c,
              unsigned int d) {
  reshape(t, a, b, c, d);
}

/* 3D reshape wrapper */
void reshape3(Tensor *t, unsigned int a, unsigned int b, unsigned int c) {
  reshape(t, 1, a, b, c);
}

/* 2D reshape wrapper */
void reshape2(Tensor *t, unsigned int a, unsigned int b) {
  reshape(t, 1, 1, a, b);
}

/* 1D reshape wrapper */
void reshape1(Tensor *t, unsigned int a) { reshape(t, 1, 1, 1, a); }

/* general indexing function */
float *at(Tensor *t, unsigned int a, unsigned int b, unsigned int c,
          unsigned int d) {
  return &t->data[(size_t)a * t->b * t->c * t->d + (size_t)b * t->c * t->d +
                  (size_t)c * t->d + (size_t)d];
}

/* 4D indexing wrapper */
float *at4(Tensor *t, unsigned int a, unsigned int b, unsigned int c,
           unsigned int d) {
  return at(t, a, b, c, d);
}

/* 3D indexing wrapper */
float *at3(Tensor *t, unsigned int a, unsigned int b, unsigned int c) {
  return at(t, 0, a, b, c);
}

/* 2D indexing wrapper */
float *at2(Tensor *t, unsigned int a, unsigned int b) {
  return at(t, 0, 0, a, b);
}

/* 1D indexing wrapper */
float *at1(Tensor *t, unsigned int a) { return at(t, 0, 0, 0, a); }

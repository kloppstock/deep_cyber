#include "../include/deep_cyber.h"
#include <assert.h>

/* general dense layer implementation */
Tensor dense(Tensor X, Tensor w, Tensor b) {
  /* extract parameters */
  unsigned int batches = X.c;
  unsigned int size = X.d;
  unsigned int w_s = w.c;
  unsigned int units = w.d;

  Tensor out = create_tensor2(batches, units);

  unsigned int i, j, k;

  assert((size == w_s) && "Error: Input and weight size doesn't match!\n");

  /* prefill tensor with bias */
  for (i = 0; i < batches; ++i)
    for (j = 0; j < units; ++j)
      AT2(out, i, j) = AT1(b, j);

  /* matrix multiplication */
  for (i = 0; i < batches; ++i)
    for (k = 0; k < size; ++k)
      for (j = 0; j < units; ++j)
        AT2(out, i, j) += AT2(X, i, k) * AT2(w, k, j);

  return out;
}

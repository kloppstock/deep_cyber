#include "../include/deep_cyber.h"
#include <assert.h>

/* general dense layer implementation */
Tensor dense(Tensor X, Tensor w, Tensor b) {
  /* extract parameters */
  uint32_t batches = X.c;
  uint32_t size = X.d;
  uint32_t w_s = w.c;
  uint32_t units = w.d;

  Tensor out = create_tensor2(batches, units);

  uint32_t i, j, k;

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

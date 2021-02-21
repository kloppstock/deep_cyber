#include "../include/deep_cyber.h"
#include <assert.h>

/*!
 * \brief Inline exponential function approximation.
 * \param y The exponent.
 * \return e^y
 */
#define EXP(y)                                                                 \
  (((union {                                                                   \
     float f;                                                                  \
     int32_t i;                                                                \
   })((int32_t)(12102203 * y + 1064986816)))                                   \
       .f)

/* ReLU */
Tensor relu(Tensor X) {
  /* create output */
  Tensor out = create_tensor(X.a, X.b, X.c, X.d);

  uint32_t ai, bi, ci, di;

  /* activate */
  for (ai = 0; ai < X.a; ++ai)
    for (bi = 0; bi < X.b; ++bi)
      for (ci = 0; ci < X.c; ++ci)
        for (di = 0; di < X.d; ++di)
          if (AT(X, ai, bi, ci, di) <= 0.f)
            AT(out, ai, bi, ci, di) = 0.f;
          else
            AT(out, ai, bi, ci, di) = AT(X, ai, bi, ci, di);

  return out;
}

/* sigmoid */
Tensor sigmoid(Tensor X) {
  /* create output */
  Tensor out = create_tensor(X.a, X.b, X.c, X.d);

  uint32_t ai, bi, ci, di;

  /* activate */
  for (ai = 0; ai < X.a; ++ai)
    for (bi = 0; bi < X.b; ++bi)
      for (ci = 0; ci < X.c; ++ci)
        for (di = 0; di < X.d; ++di)
          AT(out, ai, bi, ci, di) = 1.f / (1 + EXP(-AT(X, ai, bi, ci, di)));

  return out;
}

/* softmax */
Tensor softmax(Tensor X) {
  /* extract inputs */
  uint32_t batches = X.c;
  uint32_t cells = X.d;

  Tensor out = create_tensor2(batches, cells);

  uint32_t i, j;

  /* calculate sum of exponentials and store intermediate results for later use
   */
  for (i = 0; i < batches; ++i) {
    for (j = 0; j < cells; ++j) {
      if (j != cells - 1)
        AT2(out, i, j) = EXP(AT2(X, i, j));
      if (j == 0)
        AT2(out, i, cells - 1) = EXP(AT2(X, i, j));
      else
        AT2(out, i, cells - 1) += EXP(AT2(X, i, j));
    }
  }

  /* activate */
  for (i = 0; i < batches; ++i)
    for (j = 0; j < cells; ++j)
      if (j < cells - 1)
        AT2(out, i, j) /= AT2(out, i, cells - 1);
      else
        AT2(out, i, j) = EXP(AT2(X, i, j)) / AT2(out, i, cells - 1);

  return out;
}

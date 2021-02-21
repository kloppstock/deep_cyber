#include "../include/deep_cyber.h"
#include <assert.h>
#include <math.h>

/* ReLU */
Tensor relu(Tensor X) {
  /* create output */
  Tensor out = create_tensor(X.a, X.b, X.c, X.d);

  unsigned int ai, bi, ci, di;

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

  unsigned int ai, bi, ci, di;

  /* activate */
  for (ai = 0; ai < X.a; ++ai)
    for (bi = 0; bi < X.b; ++bi)
      for (ci = 0; ci < X.c; ++ci)
        for (di = 0; di < X.d; ++di)
          AT(out, ai, bi, ci, di) = 1.f / (1 + powf(E, -AT(X, ai, bi, ci, di)));

  return out;
}

/* softmax */
Tensor softmax(Tensor X) {
  /* extract inputs */
  unsigned int batches = X.c;
  unsigned int cells = X.d;

  Tensor out = create_tensor2(batches, cells);

  unsigned int i, j;

  /* calculate sum of exponentials */
  for (i = 0; i < batches; ++i)
    for (j = 0; j < cells; ++j)
      if (j == 0)
        AT2(out, i, cells - 1) = powf(E, AT2(X, i, j));
      else
        AT2(out, i, cells - 1) += powf(E, AT2(X, i, j));

  /* activate */
  for (i = 0; i < batches; ++i)
    for (j = 0; j < cells; ++j)
      AT2(out, i, j) = AT2(out, i, cells - 1);//pow((double)E, (double)AT2(X, i, j));// / AT2(out, i, cells - 1);

  return out;
}

/*
Tensor softmax(Tensor X) {
  /* extract inputs * /
  unsigned int batches = X.c;
  unsigned int cells = X.d;

  Tensor out = create_tensor2(batches, cells);

  unsigned int i, j;

  /* calculate sum of exponentials * /
  for (i = 0; i < batches; ++i) {
    for (j = 0; j < cells; ++j) {
      if (j != cells - 1)
        AT2(out, i, j) = pow(E, AT2(X, i, j));
      if (j == 0)
        AT2(out, i, cells - 1) = pow(E, AT2(X, i, j));
      else
        AT2(out, i, cells - 1) += pow(E, AT2(X, i, j));
    }
  }

  /* activate * /
  for (i = 0; i < batches; ++i)
    for (j = 0; j < cells; ++j)
      if (j < cells - 1)
        AT2(out, i, j) /= AT2(out, i, cells - 1);
      else
        AT2(out, i, j) = pow(E, AT2(X, i, j)) / AT2(out, i, cells - 1);

  return out;
}*/

#include "../include/deep_cyber.h"

/*!
 * \brief Minimal floating point value;
 */
#define FLT_MIN 1.175494e-38

/*!
 * \brief Returns the maximum of two values.
 * \param a The first value.
 * \param b The second value.
 * \return The maximum.
 */
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/* specialized max pooling */
Tensor maxpool2d_2x2(Tensor X, char padding) {
  /* extract parameters */
  uint32_t batches = X.a;
  int32_t rows_in = X.b;
  int32_t cols_in = X.c;
  uint32_t channels = X.d;

  /* same padding */
  if (padding) {
    /* allocate output */
    uint32_t out_rows = (rows_in + 1) / 2;
    uint32_t out_cols = (cols_in + 1) / 2;
    Tensor out = create_tensor(batches, out_rows, out_cols, channels);
    uint32_t ai, bi, ci, di;
    uint32_t i, y, x, c;
    int32_t ky, kx;

    /* prefill with smallest possible value */
    for (ai = 0; ai < out.a; ++ai)
      for (bi = 0; bi < out.b; ++bi)
        for (ci = 0; ci < out.c; ++ci)
          for (di = 0; di < out.d; ++di)
            AT(out, ai, bi, ci, di) = FLT_MIN;

    /* pool */
    for (i = 0; i < batches; ++i) {
      for (y = 0; y < out_rows; ++y) {
        for (x = 0; x < out_cols; ++x) {
          int32_t sy = y * 2;
          int32_t sx = x * 2;
          for (c = 0; c < channels; ++c) {
            for (ky = sy; ky < sy + 2; ++ky) {
              for (kx = sx; kx < sx + 2; ++kx) {
                if (ky >= 0 && ky < rows_in && kx >= 0 && kx < cols_in)
                  AT(out, i, y, x, c) =
                      MAX(AT(out, i, y, x, c), AT(X, i, ky, kx, c));
                else
                  AT(out, i, y, x, c) = MAX(AT(out, i, y, x, c), FLT_MIN);
              }
            }
          }
        }
      }
    }

    return out;

    /* zero padding*/
  } else {
    /* allocate output */
    uint32_t out_rows = rows_in / 2;
    uint32_t out_cols = cols_in / 2;
    Tensor out = create_tensor(batches, out_rows, out_cols, channels);
    uint32_t i, y, x, c, ky, kx;

    /* pool */
    for (i = 0; i < batches; ++i) {
      for (y = 0; y < out_rows; ++y) {
        for (x = 0; x < out_cols; ++x) {
          uint32_t sy = y * 2;
          uint32_t sx = x * 2;
          for (c = 0; c < channels; ++c) {
            for (ky = sy; ky < sy + 2; ++ky) {
              for (kx = sx; kx < sx + 2; ++kx) {
                if (ky == sy && kx == sx)
                  AT(out, i, y, x, c) = AT(X, i, ky, kx, c);
                else
                  AT(out, i, y, x, c) =
                      MAX(AT(out, i, y, x, c), AT(X, i, ky, kx, c));
              }
            }
          }
        }
      }
    }

    return out;
  }
}

/* max pooling */
Tensor maxpool2d(Tensor X, uint32_t pool_rows, uint32_t pool_cols,
                 uint32_t stride_rows, uint32_t stride_cols, char padding) {
  /* extract parameters */
  uint32_t batches = X.a;
  int32_t rows_in = X.b;
  int32_t cols_in = X.c;
  uint32_t channels = X.d;

  /* check if specialized version is applicable */
  if (pool_rows == 2 && pool_cols == 2 && stride_rows == 2 && stride_cols == 2)
    return maxpool2d_2x2(X, padding);

  /* same padding */
  if (padding) {
    /* allocate output */
    uint32_t out_rows = (rows_in + stride_rows - 1) / stride_rows;
    uint32_t out_cols = (cols_in + stride_cols - 1) / stride_cols;
    Tensor out = create_tensor(batches, out_rows, out_cols, channels);
    uint32_t ai, bi, ci, di;
    uint32_t i, y, x, c;
    int32_t ky, kx;

    /* prefill with smallest possible value */
    for (ai = 0; ai < out.a; ++ai)
      for (bi = 0; bi < out.b; ++bi)
        for (ci = 0; ci < out.c; ++ci)
          for (di = 0; di < out.d; ++di)
            AT(out, ai, bi, ci, di) = FLT_MIN;

    /* pool */
    for (i = 0; i < batches; ++i) {
      for (y = 0; y < out_rows; ++y) {
        for (x = 0; x < out_cols; ++x) {
          int32_t sy = y * stride_rows;
          int32_t sx = x * stride_cols;
          for (c = 0; c < channels; ++c) {
            for (ky = sy; ky < sy + (int32_t)pool_rows; ++ky) {
              for (kx = sx; kx < sx + (int32_t)pool_cols; ++kx) {
                if (ky >= 0 && ky < rows_in && kx >= 0 && kx < cols_in)
                  AT(out, i, y, x, c) =
                      MAX(AT(out, i, y, x, c), AT(X, i, ky, kx, c));
                else
                  AT(out, i, y, x, c) = MAX(AT(out, i, y, x, c), FLT_MIN);
              }
            }
          }
        }
      }
    }

    return out;

    /* zero padding*/
  } else {
    /* allocate output */
    uint32_t out_rows = (rows_in - (pool_rows - stride_rows)) / stride_rows;
    uint32_t out_cols = (cols_in - (pool_cols - stride_cols)) / stride_cols;
    Tensor out = create_tensor(batches, out_rows, out_cols, channels);
    uint32_t i, y, x, c, ky, kx;

    /* pool */
    for (i = 0; i < batches; ++i) {
      for (y = 0; y < out_rows; ++y) {
        for (x = 0; x < out_cols; ++x) {
          uint32_t sy = y * stride_rows;
          uint32_t sx = x * stride_cols;
          for (c = 0; c < channels; ++c) {
            for (ky = sy; ky < sy + pool_rows; ++ky) {
              for (kx = sx; kx < sx + pool_cols; ++kx) {
                if (ky == sy && kx == sx)
                  AT(out, i, y, x, c) = AT(X, i, ky, kx, c);
                else
                  AT(out, i, y, x, c) =
                      MAX(AT(out, i, y, x, c), AT(X, i, ky, kx, c));
              }
            }
          }
        }
      }
    }

    return out;
  }
}

/* specialized average pooling */
Tensor avgpool2d_2x2(Tensor X, char padding) {
  /* extract parameters */
  uint32_t batches = X.a;
  int32_t rows_in = X.b;
  int32_t cols_in = X.c;
  uint32_t channels = X.d;

  /* same padding */
  if (padding) {
    /* allocate output */
    uint32_t out_rows = (rows_in + 1) / 2;
    uint32_t out_cols = (cols_in + 1) / 2;
    Tensor out = create_tensor(batches, out_rows, out_cols, channels);
    uint32_t ai, bi, ci, di;
    uint32_t i, y, x, c;
    int32_t ky, kx;

    /* prefill zero */
    for (ai = 0; ai < out.a; ++ai)
      for (bi = 0; bi < out.b; ++bi)
        for (ci = 0; ci < out.c; ++ci)
          for (di = 0; di < out.d; ++di)
            AT(out, ai, bi, ci, di) = 0.f;

    /* pool */
    for (i = 0; i < batches; ++i) {
      for (y = 0; y < out_rows; ++y) {
        for (x = 0; x < out_cols; ++x) {
          int32_t sy = y * 2;
          int32_t sx = x * 2;
          for (c = 0; c < channels; ++c) {
            uint32_t area = 0.f;
            for (ky = sy; ky < sy + 2; ++ky) {
              for (kx = sx; kx < sx + 2; ++kx) {
                if (ky >= 0 && ky < rows_in && kx >= 0 && kx < cols_in) {
                  AT(out, i, y, x, c) += AT(X, i, ky, kx, c);
                  area += 1;
                }
              }
            }
            AT(out, i, y, x, c) /= area;
          }
        }
      }
    }

    return out;

    /* zero padding*/
  } else {
    /* allocate output */
    uint32_t out_rows = rows_in / 2;
    uint32_t out_cols = cols_in / 2;
    Tensor out = create_tensor(batches, out_rows, out_cols, channels);
    uint32_t i, y, x, c, ky, kx;

    /* pool */
    for (i = 0; i < batches; ++i) {
      for (y = 0; y < out_rows; ++y) {
        for (x = 0; x < out_cols; ++x) {
          uint32_t sy = y * 2;
          uint32_t sx = x * 2;
          for (c = 0; c < channels; ++c) {
            uint32_t area = 0.f;
            for (ky = sy; ky < sy + 2; ++ky) {
              for (kx = sx; kx < sx + 2; ++kx) {
                if (ky == sy && kx == sx)
                  AT(out, i, y, x, c) = AT(X, i, ky, kx, c);
                else
                  AT(out, i, y, x, c) += AT(X, i, ky, kx, c);
                area += 1;
              }
            }
            AT(out, i, y, x, c) /= area;
          }
        }
      }
    }

    return out;
  }
}

/* average pooling */
Tensor avgpool2d(Tensor X, uint32_t pool_rows, uint32_t pool_cols,
                 uint32_t stride_rows, uint32_t stride_cols, char padding) {
  /* extract parameters */
  uint32_t batches = X.a;
  int32_t rows_in = X.b;
  int32_t cols_in = X.c;
  uint32_t channels = X.d;

  /* check if specialized version is applicable */
  if (pool_rows == 2 && pool_cols == 2 && stride_rows == 2 && stride_cols == 2)
    return avgpool2d_2x2(X, padding);

  /* same padding */
  if (padding) {
    /* allocate output */
    uint32_t out_rows = (rows_in + stride_rows - 1) / stride_rows;
    uint32_t out_cols = (cols_in + stride_cols - 1) / stride_cols;
    Tensor out = create_tensor(batches, out_rows, out_cols, channels);
    uint32_t ai, bi, ci, di;
    uint32_t i, y, x, c;
    int32_t ky, kx;

    /* prefill zero */
    for (ai = 0; ai < out.a; ++ai)
      for (bi = 0; bi < out.b; ++bi)
        for (ci = 0; ci < out.c; ++ci)
          for (di = 0; di < out.d; ++di)
            AT(out, ai, bi, ci, di) = 0.f;

    /* pool */
    for (i = 0; i < batches; ++i) {
      for (y = 0; y < out_rows; ++y) {
        for (x = 0; x < out_cols; ++x) {
          int32_t sy = y * stride_rows;
          int32_t sx = x * stride_cols;
          for (c = 0; c < channels; ++c) {
            uint32_t area = 0.f;
            for (ky = sy; ky < sy + (int32_t)pool_rows; ++ky) {
              for (kx = sx; kx < sx + (int32_t)pool_cols; ++kx) {
                if (ky >= 0 && ky < rows_in && kx >= 0 && kx < cols_in) {
                  AT(out, i, y, x, c) += AT(X, i, ky, kx, c);
                  area += 1;
                }
              }
            }
            AT(out, i, y, x, c) /= area;
          }
        }
      }
    }

    return out;

    /* zero padding*/
  } else {
    /* allocate output */
    uint32_t out_rows = (rows_in - (pool_rows - stride_rows)) / stride_rows;
    uint32_t out_cols = (cols_in - (pool_cols - stride_cols)) / stride_cols;
    Tensor out = create_tensor(batches, out_rows, out_cols, channels);
    uint32_t i, y, x, c, ky, kx;

    /* pool */
    for (i = 0; i < batches; ++i) {
      for (y = 0; y < out_rows; ++y) {
        for (x = 0; x < out_cols; ++x) {
          uint32_t sy = y * stride_rows;
          uint32_t sx = x * stride_cols;
          for (c = 0; c < channels; ++c) {
            uint32_t area = 0.f;
            for (ky = sy; ky < sy + pool_rows; ++ky) {
              for (kx = sx; kx < sx + pool_cols; ++kx) {
                if (ky == sy && kx == sx)
                  AT(out, i, y, x, c) = AT(X, i, ky, kx, c);
                else
                  AT(out, i, y, x, c) += AT(X, i, ky, kx, c);
                area += 1;
              }
            }
            AT(out, i, y, x, c) /= area;
          }
        }
      }
    }

    return out;
  }
}

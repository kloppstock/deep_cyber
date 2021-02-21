#ifndef CONV2D_H
#define CONV2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include "tensor.h"

/*!
 * \brief A 2D convolution.
 * \param X The input uint32.
 * \param w The weight uint32.
 * \param b The bias uint32.
 * \param stride_rows The number of rows to stride.
 * \param stride_cols The number of cols to stride.
 * \param padding The padding (0 = zero padding, same otherwise).
 * \param groups The number of groups.
 * \return The output uint32.
 */
Tensor conv2d(Tensor X, Tensor w, Tensor b, unsigned int stride_rows,
              unsigned int stride_cols, char padding, unsigned int groups);

#ifdef __cplusplus
}
#endif

#endif // CONV2D_H

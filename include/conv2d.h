#ifndef CONV2D_H
#define CONV2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include "tensor.h"

/*!
 * \brief A 2D convolution.
 * \param X The input tensor.
 * \param w The weight tensor.
 * \param b The bias tensor.
 * \param stride_rows The number of rows to stride.
 * \param stride_cols The number of cols to stride.
 * \param padding The padding (0 = zero padding, same otherwise).
 * \param groups The number of groups.
 * \return The output tensor.
 */
Tensor conv2d(Tensor X, Tensor w, Tensor b, uint16_t stride_rows,
              uint16_t stride_cols, uint8_t padding, uint16_t groups);

#ifdef __cplusplus
}
#endif

#endif // CONV2D_H

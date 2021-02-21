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
 * \param stride_cols The number of columns to stride.
 * \param padding The padding (0 = zero padding, same otherwise).
 * \param groups The number of groups.
 * \return The output tensor.
 */
Tensor conv2d(Tensor X, Tensor w, Tensor b, unsigned int stride_rows,
              unsigned int stride_cols, char padding, unsigned int groups);

/*!
 * \brief The dense layer implementation
 * \param X The input tensor.
 * \param w The weight tensor.
 * \param b The bias tensor.
 * \return The output tensor.
 */
Tensor dense(Tensor X, Tensor w, Tensor b);

/*!
 * \brief The ReLU implementation.
 * \param X The input tensor.
 * \return The output tensor.
 */
Tensor relu(Tensor X);

/*!
 * \brief The sigmoid implementation.
 * \param X The input tensor.
 * \return The output tensor.
 */
Tensor sigmoid(Tensor X);

/*!
 * \brief The softmax implementation.
 * \param X The input tensor.
 * \return The output tensor.
 */
Tensor softmax(Tensor X);

/*!
 * \brief The 2D max pooling layer implementation.
 * \param X The input tensor.
 * \param pool_rows The number of rows to pool.
 * \param pool_cols The number of columns to pool.
 * \param stride_rows The number of rows to stride.
 * \param stride_cols The number of columns to stride.
 * \param padding The padding (0 = zero padding, same otherwise).
 * \return The output tensor.
 */
Tensor maxpool2d(Tensor X, unsigned int pool_rows, unsigned int pool_cols,
                 unsigned int stride_rows, unsigned int stride_cols,
                 char padding);

/*!
 * \brief The 2D average pooling layer implementation.
 * \param X The input tensor.
 * \param pool_rows The number of rows to pool.
 * \param pool_cols The number of columns to pool.
 * \param stride_rows The number of rows to stride.
 * \param stride_cols The number of columns to stride.
 * \param padding The padding (0 = zero padding, same otherwise).
 * \return The output tensor.
 */
Tensor avgpool2d(Tensor X, unsigned int pool_rows, unsigned int pool_cols,
                 unsigned int stride_rows, unsigned int stride_cols,
                 char padding);

#ifdef __cplusplus
}
#endif

#endif // CONV2D_H

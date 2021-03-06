#ifndef CIFAR10_WEIGHTS_H
#define CIFAR10_WEIGHTS_H

#include "../include/deep_cyber.h"

/*!
 * \brief The weights for the first convolutional layer.
 */
extern Tensor C1W;

/*!
 * \brief The biases fo the first convolutional layer.
 */
extern Tensor C1B;

/*!
 * \brief The weights for the second convolutional layer.
 */
extern Tensor C2W;

/*!
 * \brief The biases fo the second convolutional layer.
 */
extern Tensor C2B;

/*!
 * \brief The weights for the third convolutional layer.
 */
extern Tensor C3W;

/*!
 * \brief The biases fo the third convolutional layer.
 */
extern Tensor C3B;

/*!
 * \brief The weights for the fourth convolutional layer.
 */
extern Tensor C4W;

/*!
 * \brief The biases fo the fourth convolutional layer.
 */
extern Tensor C4B;

/*!
 * \brief The weights for the first fully connected layer.
 */
extern Tensor D1W;

/*!
 * \brief The biases fo the first fully connected layer.
 */
extern Tensor D1B;

/*!
 * \brief The weights for the second fully connected layer.
 */
extern Tensor D2W;

/*!
 * \brief The biases fo the second fully connected layer.
 */
extern Tensor D2B;

#endif // CIFAR10_WEIGHTS_H

#ifndef CIFAR10_H
#define CIFAR10_H

#include "../include/deep_cyber.h"

/*!
 * \brief Run the CIFAR10 model.
 * \param X The input tensor.
 * \return The index of the predicted class.
 */
uint8_t cifar10(Tensor X);

/*!
 * \brief Returns the class name for the corresponding class index.
 * \param class The class index.
 * \return The class name.
 */
const char *get_class_name(uint8_t class);

#endif // CIFAR10_H

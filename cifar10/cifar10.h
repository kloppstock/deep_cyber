#ifndef CIFAR10_H
#define CIFAR10_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../include/deep_cyber.h"

/*!
 * \brief Run the CIFAR10 model.
 * \param X The input tensor.
 * \return The index tensor of the predicted classes.
 */
Tensor cifar10(Tensor X);

/*!
 * \brief Returns the class name for the corresponding class index.
 * \param class The class index.
 * \return The class name.
 */
const char *get_class_name(uint8_t class_id);

#ifdef __cplusplus
}
#endif

#endif // CIFAR10_H

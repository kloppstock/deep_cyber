#ifndef CIFAR10_DATA_H
#define CIFAR10_DATA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "include/deep_cyber.h"

/*!
 * \brief A subset of the CIFAR10 data set.
 */
extern Tensor CIFAR10;

/*!
 * \brief The reference result for the selected CIFAR10 data.
 */
extern const uint8_t CIFAR10_REFERENCE[];

#ifdef __cplusplus
}
#endif

#endif // CIFAR10_DATA_H

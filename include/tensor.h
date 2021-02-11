#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

/*!
 * \brief Tensor struct. Holds a Tensor with (a, b, c, d) dimensions.
 */
typedef struct {
  uint8_t a, b, c, d;
  float *data;
} Tensor;

/*!
 * \brief Creates a 4D tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \param c Third dimension.
 * \param d Fourth dimension.
 * \return The tensor.
 */
Tensor create_tensor(uint8_t a, uint8_t b, uint8_t c, uint8_t d);

/*!
 * \brief Creates a 4D tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \param c Third dimension.
 * \param d Fourth dimension.
 * \return The tensor.
 */
Tensor create_tensor4(uint8_t a, uint8_t b, uint8_t c, uint8_t d);

/*!
 * \brief Creates a 3D tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \param c Third dimension.
 * \return The tensor.
 */
Tensor create_tensor3(uint8_t a, uint8_t b, uint8_t c);

/*!
 * \brief Creates a 2D tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \return The tensor.
 */
Tensor create_tensor2(uint8_t a, uint8_t b);

/*!
 * \brief Creates a 1D tensor.
 * \param a First dimension.
 * \return The tensor.
 */
Tensor create_tensor1(uint8_t a);

/*!
 * \brief Frees a tensor.
 * \param t The tensor to free.
 */
void free_tensor(Tensor t);

/*!
 * \brief Reshapes a 4D tensor.
 * \param t The input tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \param c Third dimension.
 * \param d Fourth dimension.
 */
void reshape(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d);

/*!
 * \brief Reshapes a 4D tensor.
 * \param t The input tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \param c Third dimension.
 * \param d Fourth dimension.
 */
void reshape4(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d);

/*!
 * \brief Reshapes a 3D tensor.
 * \param t The input tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \param c Third dimension.
 */
void reshape3(Tensor *t, uint8_t a, uint8_t b, uint8_t c);

/*!
 * \brief Reshapes a 2D tensor.
 * \param t The input tensor.
 * \param a First dimension.
 * \param b Second dimension.
 */
void reshape2(Tensor *t, uint8_t a, uint8_t b);

/*!
 * \brief Reshapes a 1D tensor.
 * \param t The input tensor.
 * \param a First dimension.
 */
void reshape1(Tensor *t, uint8_t a);

/*!
 * \brief 4D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \param c The index in the third dimension.
 * \param d The index in the fourth dimension.
 * \return A pointer to the requested Element.
 */
float *at(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d);

/*!
 * \brief 4D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \param c The index in the third dimension.
 * \param d The index in the fourth dimension.
 * \return A pointer to the requested Element.
 */
float *at4(Tensor *t, uint8_t a, uint8_t b, uint8_t c, uint8_t d);

/*!
 * \brief 3D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \param c The index in the third dimension.
 * \return A pointer to the requested Element.
 */
float *at3(Tensor *t, uint8_t a, uint8_t b, uint8_t c);

/*!
 * \brief 2D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \return A pointer to the requested Element.
 */
float *at2(Tensor *t, uint8_t a, uint8_t b);

/*!
 * \brief 1D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \return A pointer to the requested Element.
 */
float *at1(Tensor *t, uint8_t a);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H

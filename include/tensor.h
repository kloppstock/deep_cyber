#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stdlib.h>

/*!
 * \brief Tensor struct. Holds a Tensor with (a, b, c, d) dimensions.
 */
typedef struct {
  unsigned int a, b, c, d;
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
Tensor create_tensor(unsigned int a, unsigned int b, unsigned int c,
                     unsigned int d);

/*!
 * \brief Creates a 4D tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \param c Third dimension.
 * \param d Fourth dimension.
 * \return The tensor.
 */
Tensor create_tensor4(unsigned int a, unsigned int b, unsigned int c,
                      unsigned int d);

/*!
 * \brief Creates a 3D tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \param c Third dimension.
 * \return The tensor.
 */
Tensor create_tensor3(unsigned int a, unsigned int b, unsigned int c);

/*!
 * \brief Creates a 2D tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \return The tensor.
 */
Tensor create_tensor2(unsigned int a, unsigned int b);

/*!
 * \brief Creates a 1D tensor.
 * \param a First dimension.
 * \return The tensor.
 */
Tensor create_tensor1(unsigned int a);

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
void reshape(Tensor *t, unsigned int a, unsigned int b, unsigned int c,
             unsigned int d);

/*!
 * \brief Reshapes a 4D tensor.
 * \param t The input tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \param c Third dimension.
 * \param d Fourth dimension.
 */
void reshape4(Tensor *t, unsigned int a, unsigned int b, unsigned int c,
              unsigned int d);

/*!
 * \brief Reshapes a 3D tensor.
 * \param t The input tensor.
 * \param a First dimension.
 * \param b Second dimension.
 * \param c Third dimension.
 */
void reshape3(Tensor *t, unsigned int a, unsigned int b, unsigned int c);

/*!
 * \brief Reshapes a 2D tensor.
 * \param t The input tensor.
 * \param a First dimension.
 * \param b Second dimension.
 */
void reshape2(Tensor *t, unsigned int a, unsigned int b);

/*!
 * \brief Reshapes a 1D tensor.
 * \param t The input tensor.
 * \param a First dimension.
 */
void reshape1(Tensor *t, unsigned int a);

/*!
 * \brief 4D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \param c The index in the third dimension.
 * \param d The index in the fourth dimension.
 * \return A pointer to the requested Element.
 */
float *at(Tensor *t, unsigned int a, unsigned int b, unsigned int c,
          unsigned int d);

/*!
 * \brief 4D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \param c The index in the third dimension.
 * \param d The index in the fourth dimension.
 * \return A pointer to the requested Element.
 */
float *at4(Tensor *t, unsigned int a, unsigned int b, unsigned int c,
           unsigned int d);

/*!
 * \brief 3D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \param c The index in the third dimension.
 * \return A pointer to the requested Element.
 */
float *at3(Tensor *t, unsigned int a, unsigned int b, unsigned int c);

/*!
 * \brief 2D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \return A pointer to the requested Element.
 */
float *at2(Tensor *t, unsigned int a, unsigned int b);

/*!
 * \brief 1D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \return A pointer to the requested Element.
 */
float *at1(Tensor *t, unsigned int a);

/*!
 * \brief Inline 4D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \param c The index in the third dimension.
 * \param d The index in the fourth dimension.
 * \return A pointer to the requested Element.
 */
#define AT(t, ai, bi, ci, di)                                                  \
  (t.data[(size_t)(ai)*t.b * t.c * t.d + (size_t)(bi)*t.c * t.d +              \
          (size_t)(ci)*t.d + (size_t)(di)])

/*!
 * \brief Inline 4D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \param c The index in the third dimension.
 * \param d The index in the fourth dimension.
 * \return A pointer to the requested Element.
 */
#define AT4(t, ai, bi, ci, di)                                                 \
  (t.data[(size_t)(ai)*t.b * t.c * t.d + (size_t)(bi)*t.c * t.d +              \
          (size_t)(ci)*t.d + (size_t)(di)])

/*!
 * \brief Inline 3D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \param c The index in the third dimension.
 * \return A pointer to the requested Element.
 */
#define AT3(t, ai, bi, ci)                                                     \
  (t.data[(size_t)(ai)*t.c * t.d + (size_t)(bi)*t.d + (size_t)(ci)])

/*!
 * \brief Inline 2D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \param b The index in the second dimension.
 * \return A pointer to the requested Element.
 */
#define AT2(t, ai, bi) (t.data[(size_t)(ai)*t.d + (size_t)(bi)])

/*!
 * \brief Inline 1D Tensor indexing function.
 * \param t The input tensor.
 * \param a The index in the first dimension.
 * \return A pointer to the requested Element.
 */
#define AT1(t, ai) (t.data[(size_t)(ai)])

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H

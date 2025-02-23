#include "cifar10.h"
#include "cifar10_weights.h"

/* CIFAR10 model */
Tensor cifar10(Tensor X) {
  /* first convolutional block */
  Tensor c1 = conv2d(X, C1W, C1B, 1, 1, 1, 1);
  Tensor a1 = relu(c1);
  Tensor p1 = maxpool2d(a1, 2, 2, 2, 2, 0);

  /* second convolutional block */
  Tensor c2 = conv2d(p1, C2W, C2B, 1, 1, 1, 1);
  Tensor a2 = relu(c2);
  Tensor p2 = maxpool2d(a2, 2, 2, 2, 2, 0);

  /* third convolutional block */
  Tensor c3 = conv2d(p2, C3W, C3B, 1, 1, 1, 1);
  Tensor a3 = relu(c3);
  Tensor p3 = maxpool2d(a3, 2, 2, 2, 2, 0);

  /* fourth convolutional block */
  Tensor c4 = conv2d(p3, C4W, C4B, 1, 1, 1, 1);
  Tensor a4 = relu(c4);
  Tensor p4 = maxpool2d(a4, 2, 2, 2, 2, 0);

  /* first fully connected block */
  reshape(&p4, 1, 1, p4.a, p4.b * p4.c * p4.d);
  Tensor d1 = dense(p4, D1W, D1B);
  Tensor a5 = relu(d1);

  /* second fully connected block */
  Tensor d2 = dense(a5, D2W, D2B);
  Tensor a6 = softmax(d2);

  /* create output */
  Tensor out = create_tensor1(a6.c);
  uint8_t i, j;
  for (i = 0; i < out.d; ++i)
    AT1(out, i) = 0.;

  /* find most likely class */
  for (i = 0; i < a6.c; ++i)
    for (j = 0; j < a6.d; ++j)
      if (AT2(a6, i, j) > AT2(a6, i, AT1(out, i)))
        AT1(out, i) = j;

  /* free unneeded tensors */
  free_tensor(c1);
  free_tensor(c2);
  free_tensor(c3);
  free_tensor(c4);

  free_tensor(a1);
  free_tensor(a2);
  free_tensor(a3);
  free_tensor(a4);
  free_tensor(a5);
  free_tensor(a6);

  free_tensor(p1);
  free_tensor(p2);
  free_tensor(p3);
  free_tensor(p4);

  free_tensor(d1);
  free_tensor(d2);

  return out;
}

/* get class name */
const char *get_class_name(uint8_t class_id) {
  switch (class_id) {
  case 0:
    return "airplane";
  case 1:
    return "automobile";
  case 2:
    return "bird";
  case 3:
    return "cat";
  case 4:
    return "deer";
  case 5:
    return "dog";
  case 6:
    return "frog";
  case 7:
    return "horse";
  case 8:
    return "ship";
  case 9:
    return "truck";
  default:
    return "Error: No such class!";
  }
}

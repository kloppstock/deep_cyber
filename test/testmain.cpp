#include "../include/deep_cyber.h"
#include "../include/tensor.h"
#include <gtest/gtest.h>

/*
 * Tensor tests
 */

// creation
TEST(TensorTest, create_tensorTest) {
  Tensor t = create_tensor(1, 2, 3, 4);

  EXPECT_EQ(t.a, 1);
  EXPECT_EQ(t.b, 2);
  EXPECT_EQ(t.c, 3);
  EXPECT_EQ(t.d, 4);

  for (unsigned int i = 0; i < 2 * 3 * 4; ++i)
    EXPECT_NO_FATAL_FAILURE(t.data[i] = 12.8f);

  free_tensor(t);
}

TEST(TensorTest, create_tensor4Test) {
  Tensor t = create_tensor(1, 2, 3, 4);

  EXPECT_EQ(t.a, 1);
  EXPECT_EQ(t.b, 2);
  EXPECT_EQ(t.c, 3);
  EXPECT_EQ(t.d, 4);

  for (unsigned int i = 0; i < 2 * 3 * 4; ++i)
    EXPECT_NO_FATAL_FAILURE(t.data[i] = 12.8f);

  free_tensor(t);
}

TEST(TensorTest, create_tensor3Test) {
  Tensor t = create_tensor3(1, 2, 3);

  EXPECT_EQ(t.a, 1);
  EXPECT_EQ(t.b, 1);
  EXPECT_EQ(t.c, 2);
  EXPECT_EQ(t.d, 3);

  for (unsigned int i = 0; i < 2 * 3; ++i)
    EXPECT_NO_FATAL_FAILURE(t.data[i] = 12.8f);

  free_tensor(t);
}

TEST(TensorTest, create_tensor2Test) {
  Tensor t = create_tensor2(1, 2);

  EXPECT_EQ(t.a, 1);
  EXPECT_EQ(t.b, 1);
  EXPECT_EQ(t.c, 1);
  EXPECT_EQ(t.d, 2);

  for (unsigned int i = 0; i < 2; ++i)
    EXPECT_NO_FATAL_FAILURE(t.data[i] = 12.8f);

  free_tensor(t);
}

TEST(TensorTest, create_tensor1Test) {
  Tensor t = create_tensor1(5);

  EXPECT_EQ(t.a, 1);
  EXPECT_EQ(t.b, 1);
  EXPECT_EQ(t.c, 1);
  EXPECT_EQ(t.d, 5);

  for (unsigned int i = 0; i < 5; ++i)
    EXPECT_NO_FATAL_FAILURE(t.data[i] = 12.8f);

  free_tensor(t);
}

// free empty tensor
TEST(TensorTest, free_tensorTest) {
  Tensor t;
  t.data = NULL;
  EXPECT_NO_FATAL_FAILURE(free_tensor(t));
}

// reshape
TEST(TensorTest, reshapeTest) {
  Tensor t = create_tensor(1, 2, 3, 4);

  reshape(&t, 4, 3, 2, 1);

  EXPECT_EQ(t.a, 4);
  EXPECT_EQ(t.b, 3);
  EXPECT_EQ(t.c, 2);
  EXPECT_EQ(t.d, 1);

  free_tensor(t);
}

TEST(TensorTest, reshape4Test) {
  Tensor t = create_tensor(1, 2, 3, 4);

  reshape4(&t, 4, 3, 2, 1);

  EXPECT_EQ(t.a, 4);
  EXPECT_EQ(t.b, 3);
  EXPECT_EQ(t.c, 2);
  EXPECT_EQ(t.d, 1);

  free_tensor(t);
}

TEST(TensorTest, reshape3Test) {
  Tensor t = create_tensor(1, 2, 3, 4);

  reshape3(&t, 4, 3, 2);

  EXPECT_EQ(t.a, 1);
  EXPECT_EQ(t.b, 4);
  EXPECT_EQ(t.c, 3);
  EXPECT_EQ(t.d, 2);

  free_tensor(t);
}

TEST(TensorTest, reshape2Test) {
  Tensor t = create_tensor(1, 2, 3, 4);

  reshape2(&t, 4, 3 * 2);

  EXPECT_EQ(t.a, 1);
  EXPECT_EQ(t.b, 1);
  EXPECT_EQ(t.c, 4);
  EXPECT_EQ(t.d, 3 * 2);

  free_tensor(t);
}

TEST(TensorTest, reshape1Test) {
  Tensor t = create_tensor(1, 2, 3, 4);

  reshape1(&t, 2 * 3 * 4);

  EXPECT_EQ(t.a, 1);
  EXPECT_EQ(t.b, 1);
  EXPECT_EQ(t.c, 1);
  EXPECT_EQ(t.d, 2 * 3 * 4);

  free_tensor(t);
}

// at
TEST(TensorTest, atTest) {
  unsigned int a = 1;
  unsigned int b = 2;
  unsigned int c = 3;
  unsigned int d = 4;

  Tensor t = create_tensor(a, b, c, d);

  float i = 0.f;

  for (unsigned int ai = 0; ai < a; ++ai)
    for (unsigned int bi = 0; bi < b; ++bi)
      for (unsigned int ci = 0; ci < c; ++ci)
        for (unsigned int di = 0; di < d; ++di)
          *at(&t, ai, bi, ci, di) = i++;

  i = 0;

  for (unsigned int ai = 0; ai < a; ++ai)
    for (unsigned int bi = 0; bi < b; ++bi)
      for (unsigned int ci = 0; ci < c; ++ci)
        for (unsigned int di = 0; di < d; ++di)
          EXPECT_EQ(*at(&t, ai, bi, ci, di), i++);

  free_tensor(t);
}

TEST(TensorTest, atContentsTest) {
  unsigned int a = 1;
  unsigned int b = 2;
  unsigned int c = 3;
  unsigned int d = 4;

  Tensor t = create_tensor(a, b, c, d);

  float i = 0.f;

  for (unsigned int ai = 0; ai < a; ++ai)
    for (unsigned int bi = 0; bi < b; ++bi)
      for (unsigned int ci = 0; ci < c; ++ci)
        for (unsigned int di = 0; di < d; ++di)
          *at(&t, ai, bi, ci, di) = i++;

  i = 0;

  for (unsigned int j = 0; j < a * b * c * d; ++j)
    EXPECT_EQ(t.data[j], i++);

  free_tensor(t);
}

TEST(TensorTest, at4Test) {
  unsigned int a = 1;
  unsigned int b = 2;
  unsigned int c = 3;
  unsigned int d = 4;

  Tensor t = create_tensor4(a, b, c, d);

  float i = 0.f;

  for (unsigned int ai = 0; ai < a; ++ai)
    for (unsigned int bi = 0; bi < b; ++bi)
      for (unsigned int ci = 0; ci < c; ++ci)
        for (unsigned int di = 0; di < d; ++di)
          *at4(&t, ai, bi, ci, di) = i++;

  i = 0;

  for (unsigned int ai = 0; ai < a; ++ai)
    for (unsigned int bi = 0; bi < b; ++bi)
      for (unsigned int ci = 0; ci < c; ++ci)
        for (unsigned int di = 0; di < d; ++di)
          EXPECT_EQ(*at4(&t, ai, bi, ci, di), i++);

  free_tensor(t);
}

TEST(TensorTest, at3Test) {
  unsigned int a = 1;
  unsigned int b = 2;
  unsigned int c = 3;

  Tensor t = create_tensor3(a, b, c);

  float i = 0.f;

  for (unsigned int ai = 0; ai < a; ++ai)
    for (unsigned int bi = 0; bi < b; ++bi)
      for (unsigned int ci = 0; ci < c; ++ci)
        *at3(&t, ai, bi, ci) = i++;

  i = 0;

  for (unsigned int ai = 0; ai < a; ++ai)
    for (unsigned int bi = 0; bi < b; ++bi)
      for (unsigned int ci = 0; ci < c; ++ci)
        EXPECT_EQ(*at3(&t, ai, bi, ci), i++);

  free_tensor(t);
}

TEST(TensorTest, at2Test) {
  unsigned int a = 2;
  unsigned int b = 3;

  Tensor t = create_tensor2(a, b);

  float i = 0.f;

  for (unsigned int ai = 0; ai < a; ++ai)
    for (unsigned int bi = 0; bi < b; ++bi)
      *at2(&t, ai, bi) = i++;

  i = 0;

  for (unsigned int ai = 0; ai < a; ++ai)
    for (unsigned int bi = 0; bi < b; ++bi)
      EXPECT_EQ(*at2(&t, ai, bi), i++);

  free_tensor(t);
}

TEST(TensorTest, at1Test) {
  unsigned int a = 2;

  Tensor t = create_tensor1(a);

  float i = 0.f;

  for (unsigned int ai = 0; ai < a; ++ai)
    *at1(&t, ai) = i++;

  i = 0;

  for (unsigned int ai = 0; ai < a; ++ai)
    EXPECT_EQ(*at1(&t, ai), i++);

  free_tensor(t);
}

/*
 * Conv2D tests
 */

TEST(Conv2DTest, conv2dEmptyTest) {
  // define input dimensions

  unsigned int a = 2;
  unsigned int b = 128;
  unsigned int c = 128;
  unsigned int d = 4;
  unsigned int filters = 5;
  unsigned int stride_cols = 2;
  unsigned int stride_rows = 2;
  unsigned int kernel_cols = 3;
  unsigned int kernel_rows = 3;
  unsigned int groups = 1;
  unsigned int padding = 0;

  // create input tensors
  Tensor X = create_tensor(a, b, c, d);
  Tensor weights = create_tensor(kernel_rows, kernel_cols, d, filters);
  Tensor bias = create_tensor1(filters);

  // conv2d
  Tensor out =
      conv2d(X, weights, bias, stride_rows, stride_cols, padding, groups);

  // free input and output tensors
  free_tensor(X);
  free_tensor(weights);
  free_tensor(bias);
  free_tensor(out);
}

TEST(Conv2DTest, bigConv2dEmptyTest) {
  // define input dimensions

  unsigned int a = 4;
  unsigned int b = 1;
  unsigned int c = 1;
  unsigned int d = 32768;
  unsigned int filters = 1;
  unsigned int stride_cols = 1;
  unsigned int stride_rows = 1;
  unsigned int kernel_cols = 1;
  unsigned int kernel_rows = 1;
  unsigned int groups = 1;
  unsigned int padding = 0;

  // create input tensors
  Tensor X = create_tensor(a, b, c, d);
  for (unsigned int i = 0; i < a * b * c * d; ++i)
    X.data[i] = 1.f;

  Tensor weights = create_tensor(kernel_rows, kernel_cols, d, filters);
  for (unsigned int i = 0; i < kernel_rows * kernel_cols * d * filters; ++i)
    weights.data[i] = 1.f;

  Tensor bias = create_tensor1(filters);
  for (unsigned int i = 0; i < filters; ++i)
    bias.data[i] = 1.f;

  // conv2d
  Tensor out =
      conv2d(X, weights, bias, stride_rows, stride_cols, padding, groups);

  EXPECT_EQ(AT(out, 0, 0, 0, 0), 32768.f + 1.f);
  EXPECT_EQ(AT(out, 1, 0, 0, 0), 32768.f + 1.f);

  // free input and output tensors
  free_tensor(X);
  free_tensor(weights);
  free_tensor(bias);
  free_tensor(out);
}

/*
 * Dense tests
 */

TEST(DenseTest, denseEmptyTest) {
  // define input dimensions

  unsigned int batches = 5;
  unsigned int size = 256;
  unsigned int units = 3;

  // create input tensors
  Tensor X = create_tensor2(batches, size);
  Tensor weights = create_tensor2(size, units);
  Tensor bias = create_tensor1(units);

  // dense
  Tensor out = dense(X, weights, bias);

  // free input and output tensors
  free_tensor(X);
  free_tensor(weights);
  free_tensor(bias);
  free_tensor(out);
}

/*
 * Activation tests
 */

TEST(ActivationTest, reluEmptyTest) {
  // define input dimensions

  unsigned int batches = 5;
  unsigned int size = 256;

  // create input tensors
  Tensor X = create_tensor2(batches, size);

  // relu
  Tensor out = relu(X);

  // free input and output tensors
  free_tensor(X);
  free_tensor(out);
}

TEST(ActivationTest, sigmoidEmptyTest) {
  // define input dimensions

  unsigned int batches = 5;
  unsigned int size = 256;

  // create input tensors
  Tensor X = create_tensor2(batches, size);

  // relu
  Tensor out = sigmoid(X);

  // free input and output tensors
  free_tensor(X);
  free_tensor(out);
}

TEST(ActivationTest, softmaxEmptyTest) {
  // define input dimensions

  unsigned int batches = 100;
  unsigned int size = 256;

  // create input tensors
  Tensor X = create_tensor2(batches, size);

  // relu
  Tensor out = softmax(X);

  // free input and output tensors
  free_tensor(X);
  free_tensor(out);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

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
  uint8_t a = 1;
  uint8_t b = 2;
  uint8_t c = 3;
  uint8_t d = 4;

  Tensor t = create_tensor(a, b, c, d);

  float i = 0.f;

  for (uint8_t ai = 0; ai < a; ++ai)
    for (uint8_t bi = 0; bi < b; ++bi)
      for (uint8_t ci = 0; ci < c; ++ci)
        for (uint8_t di = 0; di < d; ++di)
          *at(&t, ai, bi, ci, di) = i++;

  i = 0;

  for (uint8_t ai = 0; ai < a; ++ai)
    for (uint8_t bi = 0; bi < b; ++bi)
      for (uint8_t ci = 0; ci < c; ++ci)
        for (uint8_t di = 0; di < d; ++di)
          EXPECT_EQ(*at(&t, ai, bi, ci, di), i++);

  free_tensor(t);
}

TEST(TensorTest, at4Test) {
  uint8_t a = 1;
  uint8_t b = 2;
  uint8_t c = 3;
  uint8_t d = 4;

  Tensor t = create_tensor4(a, b, c, d);

  float i = 0.f;

  for (uint8_t ai = 0; ai < a; ++ai)
    for (uint8_t bi = 0; bi < b; ++bi)
      for (uint8_t ci = 0; ci < c; ++ci)
        for (uint8_t di = 0; di < d; ++di)
          *at4(&t, ai, bi, ci, di) = i++;

  i = 0;

  for (uint8_t ai = 0; ai < a; ++ai)
    for (uint8_t bi = 0; bi < b; ++bi)
      for (uint8_t ci = 0; ci < c; ++ci)
        for (uint8_t di = 0; di < d; ++di)
          EXPECT_EQ(*at4(&t, ai, bi, ci, di), i++);

  free_tensor(t);
}

TEST(TensorTest, at3Test) {
  uint8_t a = 1;
  uint8_t b = 2;
  uint8_t c = 3;

  Tensor t = create_tensor3(a, b, c);

  float i = 0.f;

  for (uint8_t ai = 0; ai < a; ++ai)
    for (uint8_t bi = 0; bi < b; ++bi)
      for (uint8_t ci = 0; ci < c; ++ci)
        *at3(&t, ai, bi, ci) = i++;

  i = 0;

  for (uint8_t ai = 0; ai < a; ++ai)
    for (uint8_t bi = 0; bi < b; ++bi)
      for (uint8_t ci = 0; ci < c; ++ci)
        EXPECT_EQ(*at3(&t, ai, bi, ci), i++);

  free_tensor(t);
}

TEST(TensorTest, at2Test) {
  uint8_t a = 2;
  uint8_t b = 3;

  Tensor t = create_tensor2(a, b);

  float i = 0.f;

  for (uint8_t ai = 0; ai < a; ++ai)
    for (uint8_t bi = 0; bi < b; ++bi)
      *at2(&t, ai, bi) = i++;

  i = 0;

  for (uint8_t ai = 0; ai < a; ++ai)
    for (uint8_t bi = 0; bi < b; ++bi)
      EXPECT_EQ(*at2(&t, ai, bi), i++);

  free_tensor(t);
}

TEST(TensorTest, at1Test) {
  uint8_t a = 2;

  Tensor t = create_tensor1(a);

  float i = 0.f;

  for (uint8_t ai = 0; ai < a; ++ai)
    *at1(&t, ai) = i++;

  i = 0;

  for (uint8_t ai = 0; ai < a; ++ai)
    EXPECT_EQ(*at1(&t, ai), i++);

  free_tensor(t);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

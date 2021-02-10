#include "../include/tensor.h"
#include <gtest/gtest.h>

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

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

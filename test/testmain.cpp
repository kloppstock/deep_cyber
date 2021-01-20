#include <gtest/gtest.h>

TEST(t, Test) {
    FAIL();
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

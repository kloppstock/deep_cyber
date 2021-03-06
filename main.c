#include "cifar10/cifar10.h"
#include "cifar10_data.h"
#include "include/deep_cyber.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  /* ruin the model on the CIFAR10 data */
  Tensor results = cifar10(CIFAR10);
  (void)argc;
  (void)argv;

  unsigned int i, correct = 0;

  /* print the results */
  for (i = 0; i < results.d; ++i) {
    /* check current image and print information */
    uint8_t ref = CIFAR10_REFERENCE[i];
    uint8_t res = AT1(results, i);
    const char *check = (ref == res) ? "correct" : "incorrect";
    printf("%d [%s]: %s vs. %s\n", i, check, get_class_name(ref),
           get_class_name(res));

    if (ref == res)
      ++correct;
  }

  /* calculate percentage of correctly predicted images */
  printf("Score: %f%%\n", (float)correct / (float)results.d * 100.f);

  /* not good enough */
  if ((float)correct / (float)results.d < 0.8f)
    return -1;

  return 0;
}

#ifndef INT_H
#define INT_H

#ifdef __cplusplus
extern "C" {
#endif

#if __STDC_VERSION__ >= 199901L || __cplusplus

#include <stdint.h>

#else

// define needed data types
typedef unsigned char uint8_t;
typedef char int8_t;
typedef unsigned short uint16_t;
typedef short int16_t;

#ifdef COMPAT

typedef unsigned long int uint32_t;
typedef long int int32_t;

#else

typedef unsigned int uint32_t;
typedef int int32_t;

#endif

#endif

#ifdef __cplusplus
}
#endif

#endif // INT_H

#ifndef DEEP_SHIT_H
#define DEEP_SHIT_H

#include <stdint.h>

typedef struct matrix {
  uint16_t width, height, channels;
  float *data;
} matrix;

matrix alloc_matrix(uint16_t width, uint16_t height, uint16_t channels);
void free_matrix(matrix);

typedef struct convolutional_layer{
  uint16_t width, height, channels;
  uint8_t size, stride, filters;
  bool padding;

  float *weigths;
  float *biases;

  void run(matrix, convolutional_layer, matrix);
  void (*activate)(matrix);
} convolutional_layer;

convolutional_layer alloc_convolutional_layer(uint16_t width, uint16_t height, uint16_t channels,
                  uint8_t size, uint8_t stride, uint8_t filters, bool padding);
void free_convolutional_layer(convolutional_layer);

typedef struct connected_layer{
  uint16_t width, height, channels;
  uint8_t size, stride, filters;
  bool padding;

  float *weigths;
  float *biases;

  void run(matrix, connected_layer, matrix);
  void (*activate)(matrix);
} connected_layer;

connected_layer alloc_connected_layer(uint16_t width, uint16_t height, uint16_t channels,
                  uint8_t size, uint8_t stride, uint8_t filters, bool padding);
void free_connected_layer(connected_layer);

typedef struct pooling_layer {
  uint8_t size, stride;

  void (*pool)(matrix, pooling_layer, matrix);
} pooling_layer;

typedef struct connection_layer {
    void concat(matrix, matrix, matrix);
    void add(matrix, matrix, matrix);
} connection_layer;

typedef enum LAYER_TYPE {
    CONVOLUTION,
    CONNECTED,
    POOLING,
    CONNECTION
} LAYER_TYPE;

typedef struct layer {
    matrix in, out;

    LAYER_TYPE type;
    union {
        convolutional_layer convolution;
        connected_layer connected;
        pooling_layer pool;
        connection_layer connection;
    };

    void run(layer);
} layer;

layer create_convolution(convolutional_layer);
layer create_connected(connected_layer);
layer create_pooling(pooling_layer);
layer create_connection(connection_layer);
void free_layer(layer);

typedef struct net {
    uint16_t n;
    layer* layers;
} net;

net allocate_net(uint16_t);
void free_net(net);

#endif // DEEP_SHIT_H

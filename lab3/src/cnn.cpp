#include <cmath>
#include <tapa.h>
#include "cnn.h"

/*
void read_input(tapa::mmap<float>...,
                tapa::ostream<float>...,
                const int...
                ...);

void read_weight(tapa::mmap<float>...,
                 tapa::ostream<float>...,
                 const int...
                 ...);

void read_bias(tapa::mmap<float>...,
               tapa::ostream<float>...,
               const int...
               ...);
            
void write_output(tapa::mmap<float>...,
                  tapa::istream<float>...,
                  const int...
                  ...);

void cnn_core(tapa::istream<float>...,
              ...,
              tapa::ostream<float>...
              const int...
              ...);
...
*/

void read_input(
  tapa::mmap<float> in_img,
  tapa::ostream<float> &in_img_stream,
  const int kNum,
  const int kKernel,
  const int kImSize,
  const int kInImSize
) {
  for (int j = 0; j < kNum; ++j) { // each kernel kNum channels
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int h = 0; h < kImSize; ++h) { 
    #pragma HLS loop_tripcount min=1 max=kImSize_0
      for (int w = 0; w < kImSize; ++w) { // each output pixel
      #pragma HLS loop_tripcount min=1 max=kImSize_0
        for (int p = 0; p < kKernel; ++p) {
        #pragma HLS loop_tripcount min=1 max=kKernel_0
          for (int q = 0; q < kKernel; ++q) { // perform single kernel channel
          #pragma HLS loop_tripcount min=1 max=kKernel_0
          #pragma HLS PIPELINE II=1
            in_img_stream.write(in_img(j, h + p, w + q));
          }
        }
      }
    }
  }
}

void read_weight(
  tapa::mmap<float> weight,
  tapa::ostream<float> &in_weight_stream,
  const int kNum,
  const int kKernel,
  const int kImSize
) {
  for (int i = 0; i < kNum; ++i) { // kNum kernels
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int j = 0; j < kNum; ++j) { // each kernel kNum channels
    #pragma HLS loop_tripcount min=1 max=kNum_0
      for (int h = 0; h < kImSize; ++h) { 
      #pragma HLS loop_tripcount min=1 max=kImSize_0
        for (int w = 0; w < kImSize; ++w) { // each output pixel
        #pragma HLS loop_tripcount min=1 max=kImSize_0
          for (int p = 0; p < kKernel; ++p) {
          #pragma HLS loop_tripcount min=1 max=kKernel_0
            for (int q = 0; q < kKernel; ++q) { // perform single kernel channel
            #pragma HLS loop_tripcount min=1 max=kKernel_0
              in_weight_stream.write(weight(i, j, p, q));
            }
          }
        }
      }
    }
  }
}

void read_bias(
  tapa::mmap<float> bias,
  tapa::ostream<float> &in_bias_stream,
  const int kNum,
  const int kKernel
) {
  for (int i = 0; i < kNum; ++i) {
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int h = 0; h < kImSize; ++h) {
    #pragma HLS loop_tripcount min=1 max=kImSize_0
      for (int w = 0; w < kImSize; ++w) {
      #pragma HLS loop_tripcount min=1 max=kImSize_0
        in_bias_stream.write(bias[i]);
      }
    }
  }
}

void cnncore(
  tapa::istream<float> &in_img_stream,
  tapa::istream<float> &in_weight_stream,
  tapa::ostream<float> &in_bias_stream,
  tapa::mmap<float> out_img,
  const int kNum,
  const int kKernel,
  const int kImSize,
  const int kInImSize,
  const int kOutImSize) {
  static float C[kNum_0][kImSize_0][kImSize_0];

  for (int i = 0; i < kNum; ++i) {
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int h = 0; h < kImSize; ++h) {
    #pragma HLS loop_tripcount min=1 max=kImSize_0
      for (int w = 0; w < kImSize; ++w) {
      #pragma HLS loop_tripcount min=1 max=kImSize_0
        C[i][h][w] = in_bias_stream.read();
      }
    }
  }

  // Convolution
  for (int i = 0; i < kNum; ++i) { // kNum kernels
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int j = 0; j < kNum; ++j) { // each kernel kNum channels
    #pragma HLS loop_tripcount min=1 max=kNum_0
      for (int h = 0; h < kImSize; ++h) { 
      #pragma HLS loop_tripcount min=1 max=kImSize_0
        for (int w = 0; w < kImSize; ++w) { // each output pixel
        #pragma HLS loop_tripcount min=1 max=kImSize_0
          for (int p = 0; p < kKernel; ++p) {
          #pragma HLS loop_tripcount min=1 max=kKernel_0
            for (int q = 0; q < kKernel; ++q) { // perform single kernel channel
            #pragma HLS loop_tripcount min=1 max=kKernel_0
              C[i][h][w] += in_weight_stream.read() * in_img_stream.read();
            }
          }
        }
      }
    }
  }
	
	// ReLU
	for (int i = 0; i < kNum; ++i) {
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int h = 0; h < kImSize; ++h) {
    #pragma HLS loop_tripcount min=1 max=kImSize_0
      for (int w = 0; w < kImSize; ++w) {
      #pragma HLS loop_tripcount min=1 max=kImSize_0
        C[i][h][w] = max(0.f, C[i][h][w]);
      }
    }
  }
	
	// Max pooling
  for (int i = 0; i < kNum; ++i) {
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int h = 0; h < kOutImSize; ++h) {
    #pragma HLS loop_tripcount min=1 max=kOutImSize_0
      for (int w = 0; w < kOutImSize; ++w) {
      #pragma HLS loop_tripcount min=1 max=kOutImSize_0
        out_img(i, h, w) = max(
          max(C[i][h * 2][w * 2    ], C[i][h * 2 + 1][w * 2    ]),
          max(C[i][h * 2][w * 2 + 1], C[i][h * 2 + 1][w * 2 + 1]));
      }
    }
  }
}

void CnnKernel(
  tapa::mmap<float> in_img,
  tapa::mmap<float> weight,
  tapa::mmap<float> bias,
  tapa::mmap<float> out_img,
  const int kNum,
  const int kKernel,
  const int kImSize,
  const int kInImSize,
  const int kOutImSize) {
  
  tapa::stream<float, 2> in_img_stream("q_in_image_0");
  tapa::stream<float, 2> in_weight_stream("w_in_image_0");
  tapa::stream<float, 2> in_bias_stream("b_in_image_0");

  tapa::task()
    .invoke(read_input, in_img, in_img_stream, kNum, kKernel, kImSize, kInImSize);
  tapa::task()
    .invoke(read_weight, weight, in_weight_stream, kNum, kKernel, kImSize);
  tapa::task()
    .invoke(read_bias, bias, in_bias_stream, kNum, kKernel);
  tapa::task()
    .invoke(cnncore, in_img_stream, in_weight_stream, in_bias_stream, out_img, kNum, kKernel, kImSize, kInImSize, kOutImSize);
}


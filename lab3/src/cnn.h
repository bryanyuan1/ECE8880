#ifndef CNN_H_
#define CNN_H_

#include <tapa.h>

#define weight(i, j, p, q) \
    weight[(i) * kNum * kKernel * kKernel + (j) * kKernel * kKernel + \
    (p) * kKernel + (q)]
#define input(j, h, w) \
    input[(j) * kInImSize * kInImSize + (h) * kInImSize + (w)]
#define output(i, h, w) \
    output[(i) * kOutImSize * kOutImSize + (h) * kOutImSize + (w)]

#define max(a, b) ((a) > (b) ? (a) : (b))

//_0 values are for the default/max setting
constexpr int kNum_0 = 256;       // chnannel number
constexpr int kKernel_0 = 5;      // knernel size
constexpr int kImSize_0 = 224;    //image size (after conv)
constexpr int kInImSize_0 = 228;  //input image size
constexpr int kOutImSize_0 = 112; //output image size (after maxpool)

void CnnKernel(
    tapa::mmap<float> input,
    tapa::mmap<float> weight,
    tapa::mmap<float> bias,
    tapa::mmap<float> output,
    const int kNum,
    const int kKernel,
    const int kImSize,
    const int kInImSize,
    const int kOutImSize);

#endif

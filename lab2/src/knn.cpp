#include <cmath>
#include <tapa.h>
#include "knn.h"

using std::cout;
using std::endl;

const int kbytes_img = 3072;
// each image has 3072 bytes (32*32*3)

void read_image(
  tapa::mmap<uint8_t> img_mem,
  const int img_num,
  tapa::ostream<uint8_t> &q_out,
  const int rp_time
) {
  for (int rp = 0; rp < rp_time; rp++) {
    for (int i = 0; i < img_num * kbytes_img; i++) {
      q_out.write(img_mem[i]);
    }
  }
}

void knn(
  tapa::istream<uint8_t> &q_in_0,
  tapa::istream<uint8_t> &q_in_1,
  tapa::istream<uint8_t> &q_in_2,
  tapa::istream<uint8_t> &q_in_3,
  tapa::istream<uint8_t> &q_in_4,
  tapa::istream<uint8_t> &q_in_5,
  tapa::istream<uint8_t> &q_in_6,
  tapa::istream<uint8_t> &q_in_7,
  tapa::istream<uint8_t> &q_in_8,
  tapa::istream<uint8_t> &q_in_9,
  tapa::istream<uint8_t> &q_test,
  tapa::ostream<uint8_t> &q_prediction,
  const int test_image_num,
  const int train_image_each_class_num
){
  for (int t = 0; t < test_image_num; t++) {
    cout << "t = " << t << endl;
    uint8_t best_label = 0;
    int best_dist = 0x7FFFFFFF;
    uint8_t test_img[kbytes_img];
    for (int i = 0; i < kbytes_img; i++) {
      test_img[i] = q_test.read();
    }
    for (int tr = 0; tr < train_image_each_class_num; tr++) {
      for (int c = 0; c < 10; ++c) {
        int dist = 0;
        for (int i = 0; i < kbytes_img; ++i) {
          int16_t d = (int16_t)(c == 0 ? q_in_0.read() :
                                c == 1 ? q_in_1.read() :
                                c == 2 ? q_in_2.read() :
                                c == 3 ? q_in_3.read() :
                                c == 4 ? q_in_4.read() :
                                c == 5 ? q_in_5.read() :
                                c == 6 ? q_in_6.read() :
                                c == 7 ? q_in_7.read() :
                                c == 8 ? q_in_8.read() :
                                         q_in_9.read())
                  - (int16_t)test_img[i];
          dist += int(d * d);
        }
        if (dist < best_dist) {
          best_dist = dist;
          best_label = c;
        }
      }
    }
    q_prediction.write(best_label);
  }
}

void write_label(
  tapa::istream<uint8_t> &q_in,
  tapa::mmap<uint8_t> label_mem,
  const int img_num
) {
  for (int i = 0; i < img_num; i++) {
    label_mem[i] = q_in.read();
  }
}

void KNNKernel(
  tapa::mmap<uint8_t> train_image_0,
  tapa::mmap<uint8_t> train_image_1,
  tapa::mmap<uint8_t> train_image_2,
  tapa::mmap<uint8_t> train_image_3,
  tapa::mmap<uint8_t> train_image_4,
  tapa::mmap<uint8_t> train_image_5,
  tapa::mmap<uint8_t> train_image_6,
  tapa::mmap<uint8_t> train_image_7,
  tapa::mmap<uint8_t> train_image_8,
  tapa::mmap<uint8_t> train_image_9,
  tapa::mmap<uint8_t> test_image,
  tapa::mmap<uint8_t> predict_label,
  const int test_image_num,
  const int train_image_each_class_num
) {
  tapa::streams<uint8_t, 10, 2> q_tr_img("q_trin_image");
  // for tapa::streams, the parameters to use is <data_type, num_streams, depth>

  tapa::stream<uint8_t, 2> q_t_img("q_test_image");
  tapa::stream<uint8_t, 2> q_label("predict_label");

  tapa::task()
    .invoke(read_image, train_image_0, train_image_each_class_num, q_tr_img[0], test_image_num)
    .invoke(read_image, train_image_1, train_image_each_class_num, q_tr_img[1], test_image_num)
    .invoke(read_image, train_image_2, train_image_each_class_num, q_tr_img[2], test_image_num)
    .invoke(read_image, train_image_3, train_image_each_class_num, q_tr_img[3], test_image_num)
    .invoke(read_image, train_image_4, train_image_each_class_num, q_tr_img[4], test_image_num)
    .invoke(read_image, train_image_5, train_image_each_class_num, q_tr_img[5], test_image_num)
    .invoke(read_image, train_image_6, train_image_each_class_num, q_tr_img[6], test_image_num)
    .invoke(read_image, train_image_7, train_image_each_class_num, q_tr_img[7], test_image_num)
    .invoke(read_image, train_image_8, train_image_each_class_num, q_tr_img[8], test_image_num)
    .invoke(read_image, train_image_9, train_image_each_class_num, q_tr_img[9], test_image_num)
    .invoke(read_image, test_image, test_image_num, q_t_img, 1)
    .invoke(knn, q_tr_img[0], q_tr_img[1], q_tr_img[2], q_tr_img[3], q_tr_img[4],
                 q_tr_img[5], q_tr_img[6], q_tr_img[7], q_tr_img[8], q_tr_img[9],
                 q_t_img, q_label,
                 test_image_num, train_image_each_class_num)
    .invoke(write_label, q_label, predict_label, test_image_num)
    ;
}

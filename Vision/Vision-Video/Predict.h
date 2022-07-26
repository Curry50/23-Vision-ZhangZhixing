//
// Created by z on 7/25/22.
//

#ifndef IMGTEST_PREDICT_H
#define IMGTEST_PREDICT_H
#include <opencv2/opencv.hpp>

using namespace cv;
class Prediction
{
private:
    double coordinate_x[3],coordinate_y[3];
public:
    void prediction(int,Mat,Point2f);
};
#endif //IMGTEST_PREDICT_H

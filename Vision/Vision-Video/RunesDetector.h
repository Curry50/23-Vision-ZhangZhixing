//
// Created by z on 22-7-10.
//

#ifndef VISION_RUNESDETECTOR_H
#define VISION_RUNESDETECTOR_H
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace  cv;
using namespace std;
class detector
{
private:
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Point2f points[4],pt[30],pt1[30],pt2[30],left_up[30],right_up[30],left_down[305],right_down[30];
    Mat smallImg, hsv_img, img,image,struct1,channels[3];
    double k1,k2,k3,k4,d1,d2,d3,x_distance_i,y_distance_i,area_i,x_distance_j,y_distance_j,area_j,areaRatio;
    VideoCapture video;
    int counter;
public:
    void cap(string);
    void pre_process();
    void find_contours();
    void matching(int);
};
#endif //VISION_RUNESDETECTOR_H

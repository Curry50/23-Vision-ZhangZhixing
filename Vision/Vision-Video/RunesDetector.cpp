//
// Created by z on 22-7-10.
//
#include "RunesDetector.h"
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;
void detector::cap(string dir)
{
    VideoCapture video1(dir);
    video = video1;
}

void detector::pre_process() {
    video >> image;
    struct1 = getStructuringElement(0, Size(2, 2));
    split(image, channels);
    Mat diff = channels[2] - channels[0] != channels[2] - channels[2];
    bool eq = countNonZero(diff) != 0;
    if (eq) {
        channels[2] = channels[2] - channels[0];
    }
    Mat diff_2 = channels[1] - channels[0] != channels[2] - channels[2];
    bool eq_2 = countNonZero(diff_2) != 0;
    if (eq_2) {
        channels[1] = channels[1] - channels[0];
    }
    merge(channels, 3, image);
    resize(image, smallImg, Size(), 0.25, 0.25, INTER_AREA);
    cvtColor(smallImg, hsv_img, COLOR_BGR2HSV);
    inRange(hsv_img, Scalar(100, 43, 46), Scalar(124, 255, 255), img);
    dilate(img, img, struct1);
    erode(img, img, struct1);
    imshow("binary", img);
}
void detector::find_contours() {
    counter = 0;
    findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
    for (int t = 0; t < contours.size(); t++) {
        Rect rect = boundingRect(contours[t]);
        double aspectRatio = double(rect.width) / double(rect.height);
        double area = contourArea(contours[t]);
        float AreaRatio = contourArea(contours[t]) / (rect.width * rect.height);
        if (aspectRatio <= 0.65 && aspectRatio > 0.2 && AreaRatio > 0.1) {
            rectangle(smallImg, rect, Scalar(0, 0, 255), 2, 8, 0);
            RotatedRect rrect = minAreaRect(contours[t]);
            rrect.points(points);
            pt[counter] = (points[0] + points[2]) / 2;
            pt1[counter] = (points[0] + points[1]) / 2;
            pt2[counter] = (points[2] + points[3]) / 2;
            left_up[counter] = points[0];
            right_up[counter] = points[1];
            right_down[counter] = points[2];
            left_up[counter] = points[3];
            counter += 1;
        }
    }

}
void detector::matching() {
    for (int i = 0; i < counter; i++) {
        for (int j = 1 + i; j < counter; j++) {
            k1 = double(pt1[i].y - pt2[j].y) / double(pt1[i].x - pt2[j].x);
            k2 = double(pt1[i].y - pt2[j].y) / double(pt1[i].x - pt2[j].x);
            k3 = double(pt[i].y - pt[j].y) / double(pt[i].x - pt[j].x);
            k4 = double(pt1[i].x - pt2[i].x) / double(pt1[i].y - pt2[i].y);
            d1 = sqrt(pow(pt1[i].x - pt2[i].x, 2) + pow(pt1[i].y - pt2[i].y, 2));
            d2 = sqrt(pow(pt1[j].x - pt2[j].x, 2) + pow(pt1[j].y - pt2[j].y, 2));
            d3 = sqrt(pow(pt[i].x - pt[j].x, 2) + pow(pt[i].y - pt[j].y, 2));
            x_distance_i = sqrt(pow(left_up[i].x - right_up[i].x, 2) + pow(right_up[i].y - left_up[i].y, 2));
            y_distance_i = sqrt(pow(left_up[i].x - left_down[i].x, 2) + pow(left_up[i].y - left_down[i].y, 2));
            area_i = x_distance_i * y_distance_i;
            x_distance_j = sqrt(pow(left_up[j].x - right_up[j].x, 2) + pow(right_up[j].y - left_up[j].y, 2));
            y_distance_j = sqrt(pow(left_up[i].x - left_down[i].x, 2) + pow(left_up[i].y - left_down[i].y, 2));
            area_j = x_distance_j * y_distance_j;
            areaRatio = (area_i) / (area_j);
            for (int k = 0; k < 2; k++) {
                if (k3 - k4 < 0.08 && k3 - k4 > -0.07 && d1 / d3 > 0.2 + 0.2 * k && d1 / d3 < 0.6 && areaRatio > 0.8 &&
                    areaRatio < 1.2) {
                    line(smallImg, pt[i], pt[j], Scalar(0, 255, 0), 2, 8, 0);
                } else if (k3 - k4 < 0.08 && k3 - k4 > -0.07 && k3 / k4 < 1 && k3 / k4 > -2 &&
                           d1 / d3 > 0.2 + 0.2 * k && d1 / d3 < 0.6 && areaRatio > 0.8 && areaRatio < 1.2) {
                    line(smallImg, pt[i], pt[j], Scalar(0, 255, 0), 2, 8, 0);
                } else if (k3 - k4 < 0.08 && k3 - k4 > -0.07 && (k3 / k4 > 0.9 * DBL_MAX || k3 / k4 < 0.9 * DBL_MIN) &&
                           k3 / k4 > -3 && d1 / d3 > 0.2 + 0.2 * k && d1 / d3 < 0.6 &&
                           areaRatio > 0.8 && areaRatio < 1.2) {
                    line(smallImg, pt[i], pt[j], Scalar(0, 255, 0), 2, 8, 0);
                }
            }
        }
    }
    imshow("result", smallImg);
}



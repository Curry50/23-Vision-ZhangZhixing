//
// Created by z on 7/25/22.
//
#include <opencv2//opencv.hpp>
#include "Predict.h"
#include "iostream"
#include "RunesDetector.h"
using namespace std;
using namespace cv;
void Prediction::prediction(int k,Mat binary,Point2f pt)
{
    detector detect;
    if(k<2)
    {
        coordinate_x[k]=pt.x;
        coordinate_y[k]=pt.y;
    }
    else{
        double delta_x = coordinate_x[1] - coordinate_x[0];//x轴方向单位时间的变化量
        double delta_y = coordinate_y[1] -coordinate_y[0];//y轴方向单位时间的变化量
        coordinate_x[2] = pt.x + delta_x*5;//预测点的x坐标
        coordinate_y[2] = pt.y + delta_y*5;//预测点的y坐标
        circle(binary,Point(coordinate_x[2],coordinate_y[2]),
               5,Scalar(0,0,255),-1);
        //更新坐标
        for(int m=0;m<1;m++)
        {
            coordinate_x[m] = coordinate_x[m+1];
            coordinate_y[m] = coordinate_y[m+1];
        }
        coordinate_x[1] = pt.x;
        coordinate_y[1] = pt.y;
    }
}

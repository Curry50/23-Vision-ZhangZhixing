#include "Kalm.h"
void Kalm::kalmanFilterFun(float x)
{
    float predictValue = info.A*info.filterValue + info.B*info.u;//计算预测值
    info.P = info.A*info.A*info.P + info.Q;//求协方差
    info.kalmanGain = info.P * info.H / (info.P * info.H * info.H + info.R);//计算卡尔曼增益
    info.filterValue = predictValue + (x - predictValue)*info.kalmanGain;//计算输出的值
    info.predictValue= x + 5*(x - predictValue)*info.kalmanGain;
    info.P = (1 - info.kalmanGain* info.H)*info.P;//更新协方差

}


void Kalm::initKalmanFilter(KalmanInfo *info)
{
    info->A = 1;
    info->H = 1;
    info->P = 0.1;
    info->Q = 1e-7;
    info->R = 1e-4;
    info->B = 0.1;
    info->u = 0;
    info->filterValue = 0;
    info->predictValue = 0;
}

Kalm::Kalm()
{

    info.A = 1;
    info.H = 1;
    info.P = 0.1;
    info.Q = 0.05;
    info.R = 0.1;
    info.B = 0.1;
    info.u = 0;
    info.filterValue = 0;
    info.predictValue = 0;
}

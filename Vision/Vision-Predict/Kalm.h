#pragma once
typedef struct {
    float filterValue;//滤波后的值
    float kalmanGain;//Kalamn增益
    float A;//状态矩阵
    float H;//观测矩阵
    float Q;//状态矩阵的方差
    float R;//观测矩阵的方差
    float P;//预测误差
    float B;
    float u;
    float predictValue;
}KalmanInfo;
class Kalm
{
public:
    void kalmanFilterFun(float new_value);
    void initKalmanFilter(KalmanInfo *info);
    Kalm();
    KalmanInfo info;
};
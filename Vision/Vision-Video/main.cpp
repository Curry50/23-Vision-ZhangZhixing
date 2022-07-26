#include "RunesDetector.h"

using namespace std;
VideoCapture video;
Mat img,smallImg;
int main() {
    detector dt;
    dt.cap("video1.avi");
    for(int k=0;;k++)
    {
        dt.pre_process();//图像预处理
        dt.find_contours();//寻找轮廓
        dt.matching(k);//匹配灯条
        waitKey(30);
    }
}

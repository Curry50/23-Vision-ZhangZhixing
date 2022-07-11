#include "RunesDetector.h"

using namespace std;
VideoCapture video;
Mat img,smallImg;
int main() {
    detector dt;
    dt.cap("video1.avi");
    while(1)
    {
        dt.pre_process();
        dt.find_contours();
        dt.matching();
        waitKey(100);
    }
}
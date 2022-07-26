#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
using namespace cv;
using namespace std;
// %EndTag(INCLUDES)%

// %Tag(INIT)%
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
Point2f points[4],pt[30],pt1[30],pt2[30],left_up[30],right_up[30],right_down[30],left_down[30];
Mat smallImg, hsv_img, img,image,struct1,channels[3];
double areaRatio;
double pose_x,pose_y,pose_z,theta_x,theta_y,theta_z;
VideoCapture video;
int counter;
float x,y;
cv::VideoCapture cap("/home/z/vision_ws/src/vision_markers/src/1.avi");
void pre_process() {
    cap >> image;
    struct1 = getStructuringElement(0, Size(2, 2));
    resize(image, smallImg, Size(), 1, 1, INTER_AREA);
    cvtColor(smallImg, hsv_img, COLOR_BGR2HSV);
    inRange(hsv_img, Scalar(26, 43, 46), Scalar(34, 255, 255), img);
    dilate(img, img, struct1);
    erode(img, img, struct1);
    imshow("binary", img);
}
void find_contours() {
    counter = 0;
    findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
    for (int t = 0; t < contours.size(); t++) {
        Rect rect = boundingRect(contours[t]);
        double aspectRatio = double(rect.width) / double(rect.height);
        double area = contourArea(contours[t]);
        float AreaRatio = contourArea(contours[t]) / (rect.width * rect.height);
        if (aspectRatio <= 0.9 && area >100) {
            rectangle(smallImg, rect, Scalar(0, 0, 255), 2, 8, 0);
            RotatedRect rrect = minAreaRect(contours[t]);
            rrect.points(points);

            if(rrect.size.height>rrect.size.width)
            {
                pt[counter] = (points[0] + points[2]) / 2;
                pt1[counter] = (points[2] + points[1]) / 2;
                pt2[counter] = (points[0] + points[3]) / 2;
                left_up[counter] = points[1];
                right_up[counter] = points[2];
                right_down[counter] = points[3];
                left_down[counter] = points[0];
                counter += 1;
            }
            else
            {
                pt[counter] = (points[0] + points[2]) / 2;
                pt1[counter] = (points[0] + points[1]) / 2;
                pt2[counter] = (points[2] + points[3]) / 2;
                left_up[counter] = points[2];
                right_up[counter] = points[3];
                right_down[counter] = points[0];
                left_down[counter] = points[1];
                counter += 1;
            }
        }
    }

}
void matching() {
    for (int i = 0; i < counter; i++) {
        for (int j = 1 + i; j < counter; j++) {
            vector<Point2d> image_points;
            vector<Point3d> model_points;
            //line(smallImg,pt1[i],pt2[j],Scalar(0, 255, 0), 2, 8, 0);
            //line(smallImg,pt2[i],pt1[j],Scalar(0, 255, 0), 2, 8, 0);
            line(smallImg,pt[i],pt[j],Scalar(0,255,0),2,8,0);
            if(left_up[i].x < left_up[j].x)
            {
                line(smallImg,left_up[i],right_down[i],Scalar(0, 255, 0), 2, 8, 0);
                line(smallImg,left_down[i],right_up[i],Scalar(0, 255, 0), 2, 8, 0);

                image_points.push_back(Point2d(left_up[i].x,left_up[i].y));
                image_points.push_back(Point2d(right_up[i].x,right_up[i].y));
                image_points.push_back(Point2d(right_down[i].x,right_down[i].y));
                image_points.push_back(Point2d(left_down[i].x,left_down[i].y));
                double l = 5;

                model_points.push_back(Point3d(-l/2,l,0));
                model_points.push_back(Point3d(l/2,l,0));
                model_points.push_back(Point3d(l/2,-l,0));
                model_points.push_back(Point3d(-l/2,-l,0));
            }
            else{
                line(smallImg,left_up[j],right_down[j],Scalar(0, 255, 0), 2, 8, 0);
                line(smallImg,left_down[j],right_up[j],Scalar(0, 255, 0), 2, 8, 0);

                image_points.push_back(Point2d(left_up[j].x,left_up[j].y));
                image_points.push_back(Point2d(right_up[j].x,right_up[j].y));
                image_points.push_back(Point2d(right_down[j].x,right_down[j].y));
                image_points.push_back(Point2d(left_down[j].x,left_down[j].y));
                double l = 5;

                model_points.push_back(Point3d(-l/2,l,0));
                model_points.push_back(Point3d(l/2,l,0));
                model_points.push_back(Point3d(l/2,-l,0));
                model_points.push_back(Point3d(-l/2,-l,0));
            }


            Mat camMatrix = (Mat_<double>(3, 3) << 598.29493, 0, 304.76898, 0, 597.56086, 233.34673, 0, 0, 1);
            Mat distCoeff = (Mat_<double>(5, 1) << -0.53572,1.35993,-0.00244,0.00620,0.00000);
            Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
            Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
            solvePnP(model_points, image_points, camMatrix, distCoeff, rvec, tvec, false, 2);
            pose_x = tvec.at<double>(0,0) ;
            pose_y = tvec.at<double>(0,1) ;
            pose_z = tvec.at<double>(0,2) ;

            double rm[9];

            cv::Mat rotMat(3, 3, CV_64FC1, rm);

            Rodrigues(rvec, rotMat);

            theta_z = atan2(rotMat.at<double>(1,0), rotMat.at<double>(0,0))*57.2958;
            cout << theta_z <<endl;
            theta_y = atan2(-rotMat.at<double>(2,0), sqrt(rotMat.at<double>(2,1) * rotMat.at<double>(2,1) + rotMat.at<double>(2,2) * rotMat.at<double>(2,2)))*57.2958;
            cout <<theta_y <<endl;
            theta_x = atan2(rotMat.at<double>(2,1), rotMat.at<double>(2,2))*57.2958;
            cout <<theta_x <<endl;


        }
    }

    
}


int main( int argc, char** argv )
{
  ros::init(argc, argv, "basic_shapes");
  ros::NodeHandle n;
  ros::Rate r(1);
  ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1);
  image_transport::ImageTransport it(n);
  image_transport::Publisher pub = it.advertise("camera/image", 1);
// %EndTag(INIT)%

  // Set our initial shape type to be a cube
// %Tag(SHAPE_INIT)%
  uint32_t shape = visualization_msgs::Marker::CUBE;
// %EndTag(SHAPE_INIT)%
// %Tag(MARKER_INIT)%
  while (ros::ok() && cap.isOpened())
  {
    pre_process();
    find_contours();
    matching();
    imshow("result", smallImg);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", smallImg).toImageMsg();
    pub.publish(msg);
    visualization_msgs::Marker marker;
    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = "my_frame";
    marker.header.stamp = ros::Time::now();
// %EndTag(MARKER_INIT)%

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
// %Tag(NS_ID)%
    marker.ns = "basic_shapes";
    marker.id = 0;
// %EndTag(NS_ID)%

    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
// %Tag(TYPE)%
    marker.type = shape;

// %EndTag(TYPE)%

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
// %Tag(ACTION)%
    marker.action = visualization_msgs::Marker::ADD;
// %EndTag(ACTION)%

    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
// %Tag(POSE)%
    marker.pose.position.x = -0.04*pose_z;
    marker.pose.position.y = 0.04*pose_x;
    marker.pose.position.z = -0.04*pose_y;
    marker.pose.orientation.x = theta_x;
    marker.pose.orientation.y = theta_y;
    marker.pose.orientation.z = theta_z;
    marker.pose.orientation.w = 1.0;

/*    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;*/
// %EndTag(POSE)%

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
// %Tag(SCALE)%
    marker.scale.x = 0.5;
    marker.scale.y = 0.5;
    marker.scale.z = 0.5;
// %EndTag(SCALE)%

    // Set the color -- be sure to set alpha to something non-zero!
// %Tag(COLOR)%
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;
// %EndTag(COLOR)%

// %Tag(LIFETIME)%
    marker.lifetime = ros::Duration();
// %EndTag(LIFETIME)%

    // Publish the marker
// %Tag(PUBLISH)%
    while (marker_pub.getNumSubscribers() < 1)
    {
      if (!ros::ok())
      {
        return 0;
      }
      ROS_WARN_ONCE("Please create a subscriber to the marker");
//      sleep(0.1);
    }
    marker_pub.publish(marker);
    ROS_WARN_ONCE("published");
// %EndTag(PUBLISH)%

    // Cycle between different shapes
// %Tag(CYCLE_SHAPES)%
	
    
// %EndTag(CYCLE_SHAPES)%

// %Tag(SLEEP_END)%
    cv::waitKey(50);
  }
// %EndTag(SLEEP_END)%
}
// %EndTag(FULLTEXT)%

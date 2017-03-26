/*
 * Copyright (C) Roland Meertens and Peng Lu
 *
 * This file is part of paparazzi
 *
 * paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 */
/**
 * @file "modules/computer_vision/opencv_contour.cpp"
 * @author Roland Meertens and Peng Lu
 *
 */

#include "opencv_contour.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv_image_functions.h"
#include <vector>
#include <string>

using namespace cv;
using namespace std;

struct contour_estimation cont_est;
struct contour_threshold cont_thres;

RNG rng(12345);

// YUV in opencv convert to YUV on Bebop
void yuv_opencv_to_yuv422(Mat image, char *img, int width, int height)
{
//Turn the opencv RGB colored image back in a YUV colored image for the drone
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      // Extract pixel color from image
      cv::Vec3b &c = image.at<cv::Vec3b>(row, col);

      // Set image buffer values
      int i = row * width + col;
      img[2 * i + 1] = c[0]; // y;
      img[2 * i] = col % 2 ? c[1] : c[2]; // u or v
    }
  }
}

void uyvy_opencv_to_yuv_opencv(Mat image, Mat image_in, int width, int height)
{
//Turn the opencv RGB colored image back in a YUV colored image for the drone
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      // Extract pixel color from image
      cv::Vec3b c = image_in.at<cv::Vec3b>(row, col);
      cv::Vec3b c_m1 = image_in.at<cv::Vec3b>(row, col);
      cv::Vec3b c_p1 = image_in.at<cv::Vec3b>(row, col);
      if (col > 0) {
        c_m1 = image_in.at<cv::Vec3b>(row, col - 1);
      }
      if (col < width) {
        c_p1 = image_in.at<cv::Vec3b>(row, col + 1);
      }
      image.at<cv::Vec3b>(row, col)[0] = c[1] ;
      image.at<cv::Vec3b>(row, col)[1] = col % 2 ? c[0] : c_m1[0];
      image.at<cv::Vec3b>(row, col)[2] = col % 2 ? c_p1[0] : c[0];

    }
  }
}

void find_contour(char *img, int width, int height)
{
  // Create a new image, using the original bebop image.
  Mat M(width, height, CV_8UC2, img); // original
  Mat image, edge_image, thresh_image;

  // convert UYVY in paparazzi to YUV in opencv
  cvtColor(M, M, CV_YUV2RGB_Y422);
  cvtColor(M, M, CV_RGB2YUV);

  // Threshold all values within the indicted YUV values.
  inRange(M, Scalar(cont_thres.lower_y, cont_thres.lower_u, cont_thres.lower_v), Scalar(cont_thres.upper_y,
          cont_thres.upper_u, cont_thres.upper_v), thresh_image);

  /// Find contours
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  edge_image = thresh_image;
  int edgeThresh = 35;
  Canny(edge_image, edge_image, edgeThresh, edgeThresh * 3);
  findContours(edge_image, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

  // Get the moments
  vector<Moments> mu(contours.size());
  for (unsigned int i = 0; i < contours.size(); i++) {
    mu[i] = moments(contours[i], false);
  }

  //  Get the mass centers:
  vector<Point2f> mc(contours.size());
  for (unsigned int i = 0; i < contours.size(); i++) {
    mc[i] = Point2f(mu[i].m10 / mu[i].m00 , mu[i].m01 / mu[i].m00);
  }

  /// Draw contours
  Mat drawing = Mat::zeros(edge_image.size(), CV_8UC3);
  for (unsigned int i = 0; i < contours.size(); i++) {
    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
    circle(drawing, mc[i], 4, color, -1, 8, 0);
  }

  // Find Largest Contour
  int largest_contour_index = 0;
  int largest_area = 0;
  Rect bounding_rect;

  // iterate through each contour.
  for (unsigned int i = 0; i < contours.size(); i++) {
    //  Find the area of contour
    double a = contourArea(contours[i], false);
    if (a > largest_area) {
      largest_area = a;
      // Store the index of largest contour
      largest_contour_index = i;
      // Find the bounding rectangle for biggest contour
      bounding_rect = boundingRect(contours[i]);
    }
  }
  Scalar color(255, 255, 255);
  // Draw the contour and rectangle
  drawContours(M, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy);

  rectangle(M, bounding_rect,  Scalar(0, 255, 0), 2, 8, 0);

  // some figure can cause there are no largest circles, in this case, do not draw circle
  circle(M, mc[largest_contour_index], 4, Scalar(0, 255, 0), -1, 8, 0);
  Point2f rect_center(bounding_rect.x + bounding_rect.width / 2 , bounding_rect.y + bounding_rect.height / 2);
  circle(image, rect_center, 4, Scalar(0, 0, 255), -1, 8, 0);

  // Convert back to YUV422, and put it in place of the original image
  grayscale_opencv_to_yuv422(M, img);  // , width, height
  float contour_distance_est;
  //estimate the distance in X, Y and Z direction
  float area = bounding_rect.width * bounding_rect.height;
  if (area > 28000.) {
    contour_distance_est = 0.1;
  }
  if ((area > 16000.) && (area < 28000.)) {
    contour_distance_est = 0.5;
  }
  if ((area > 11000.) && (area < 16000.)) {
    contour_distance_est = 1;
  }
  if ((area > 3000.) && (area < 11000.)) {
    contour_distance_est = 1.5;
  }
  if (area < 3000.) {
    contour_distance_est = 2.0;
  }
  cont_est.contour_d_x = contour_distance_est;
  float Im_center_w = width / 2.;
  float Im_center_h = height / 2.;
  float real_size = 1.; // real size of the object
  cont_est.contour_d_y = -(rect_center.x - Im_center_w) * real_size / float(bounding_rect.width); // right hand
  cont_est.contour_d_z = -(rect_center.y - Im_center_h) * real_size / float(bounding_rect.height); // point downwards
}



/// our function
//
//struct image_data_struct
//{
//	vector<vector<Point> > contours;
//	Mat thres;
//	Mat thres2;
//
//};
//
//struct final_struct
//{
//	bool safeToGoForward;
//	Mat img;
//};
//
//image_data_struct image_preprocess (Mat &img)
//{    printf("c1 \n");
//	image_data_struct imgdata;
//	Point jb;
//	jb.x = 200;
//	jb.y = 100;
//	circle (img, jb, 6, (255, 255, 255), -1 );
//	printf("c2 \n");
//	Mat gray;
//	printf("c3 \n");
//	vector<vector<Point> > contours;
//	printf("c4 \n");
//	cvtColor(img, gray, CV_BGR2GRAY);
//	printf("c5 \n");
//	threshold(img, imgdata.thres, 220, 255, THRESH_BINARY);
//	printf("c6 \n");
//	cvtColor(imgdata.thres, gray, CV_BGR2GRAY);
//	printf("c7 \n");
//	threshold(gray, imgdata.thres2, 0, 255, THRESH_BINARY);
//	printf("c8 \n");
//	findContours(imgdata.thres2, imgdata.contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
//	printf("c9 \n");
//	//drawContours(imgdata.thres2, imgdata.contours, -1, (128, 255, 255), 3);
//printf("qwertyuiop!!!!!!!!!!!!!!!!!!!!!! \n");
////	imshow("image",imgdata.thres2);
////	waitKey(0);
//	return imgdata;
//}
//
//float Variance( float samples[2])
//{
//     int size = 3;
//     float mean_value=(samples[0]+samples[1]+samples[2])/3;
//     float variance = 0;
//     // float t = samples[0];
//     // for (int i = 1; i < size; i++)
//     // {
//     //      t += samples[i];
//     //      float diff = ((i + 1) * samples[i]) - t;
//     //      variance += (diff * diff) / ((i + 1.0) *i);
//     // }
//     variance=((samples[0]-mean_value)*(samples[0]-mean_value)+(samples[1]-mean_value)*(samples[1]-mean_value)+(samples[2]-mean_value)*(samples[2]-mean_value));
//     return variance / (size - 1);
//}
//
//vector<float> get_point(int num_contours, std::vector<vector<Point> > contours)
//{
//	float y_value[3];
//	float pair_array[4];
//	vector<float> points_data(0);
//	vector<float> value_y;
////	    cout<< "sizeof" <<endl;
////    	cout<< sizeof(contours[num_contours]) <<endl;
//	for (int y = 0; y < contours[num_contours].size() - 2; y++)//!!!!!!
//	{
//
//		y_value[0] = contours[num_contours][y].y;
//		y_value[1] = contours[num_contours][y + 1].y;
//		y_value[2] = contours[num_contours][y + 2].y;
//
//
//		//cout << y_value[0] << " "<< y_value[1] << " " << y_value[2] << endl;
//		//std::vector<double> valuey (y_value, y_value + sizeof(y_value) / sizeof(double));
//
//		float var2 = Variance(y_value);
//
//
//
//		pair_array[0] = var2;
//		pair_array[1] = (float)y;
//		pair_array[2] = (float)contours[num_contours][y].x;
//		pair_array[3] = (float)contours[num_contours][y].y;
//
//		// pair_array[0] = var2;
//		// pair_array[1] = (float)y;
//		// pair_array[2] = (float)contours[num_contours][y].x;
//		// pair_array[3] = (float)contours[num_contours][y].y;
//
//		// for(int k = 0; k < 4; k++)
//		// {
//
//		// 	points_data.push_back (pair_array[k]);
//		// 	cout << points_data [4*y+k] << endl;
//		// }
//		points_data.push_back(pair_array[0]);
//		points_data.push_back(pair_array[1]);
//		points_data.push_back(pair_array[2]);
//		points_data.push_back(pair_array[3]);
//		//points_data.push_back (pair_array);
//
//	}
////	    cout<< "sizeofPPPPPP" <<endl;
////    	cout<< sizeof(points_data) <<endl;
//
//	return points_data;
//}
//
//vector<int> draw_obstacle(vector<float> points_data, Mat &img)
//{
//
//
//	vector<int> obstacle_position;
//	int n = points_data.size()/4;
//
////	cout<< "n" <<endl;
////    cout<< n <<endl;
//
//	//vector<vector<Point> > obstacle_data;
// 	Point obstacle_data;
//
//	for ( int i = 0; i < n  ; i = i + 1 )
//	{
////		cout<< "points_data[4*i]" <<endl;
////    	cout<< points_data[4*i] <<endl;
//    	int var_data = (int)points_data [4*i];
////    	cout<< "var_data" <<endl;
////    	cout<< var_data <<endl;
//    	if (var_data > 7)
//    	{
//    		//int j = 0;
//    		obstacle_data.x = (int)points_data[4*(i+2) + 2];
//    		obstacle_data.y = (int)points_data[4*(i+2) + 3];
//  			//j++;
//    		circle (img, obstacle_data, 2, (0, 255, 255), -1 );///????
//
//    		//obstacle_position
//    		obstacle_position.push_back(obstacle_data.x);
//    		obstacle_position.push_back(obstacle_data.y);
//    	}
//	}
////	imshow( "Display window", img);
////	waitKey(0);
//
//	return obstacle_position;
//}
//
//final_struct judge_go_forward(Mat &img, float epsilon)
//{
//	vector<int> obstacle_position_data;
//	printf("s1 \n");
//	final_struct safe_or_not;
//	printf("s2 \n");
//	vector<float> process_data;
//	printf("s3 \n");
//	vector<int> contour_to_use;
//	printf("s4 \n");
//	vector<Point> contour_in_process;
//	printf("s5 \n");
//	image_data_struct imgdata = image_preprocess (img);//stable
//	printf("s6 \n");
// 	safe_or_not.safeToGoForward = false;
//	printf("s7 \n");
//// 	cout << "1" << endl;
//
//	printf("imgdata.contours.size() %d\n",imgdata.contours.size());
//	for(int x = 0 ; x < imgdata.contours.size(); x++ )
//	{	printf("s8 \n");
////		cout << "2" << endl;
//		double ret = contourArea(imgdata.contours[x]);
//		printf("ret!!!!!!!!!!!!!!!!!!!!!!! %f\n",ret);
//		printf("address!!!!!!!!!!!!!!!!!!!!!!!  %d\n",&imgdata.contours[x]);
//		printf("s9 \n");
//		if(ret < 100)
//		{
//			printf("s2000 \n");
////			cout << "3" << endl;
////			// drawContours(imgdata.thres2, imgdata.contours[x], -1, (0, 0, 0), 25 );
////			cout << "6" << endl;
//		}
//		else
//		{
////			cout << "4" << endl;
//			contour_to_use.push_back(x);
//			printf("s10 \n");
//		}
//		printf("s4000 \n");
////		cout << "contour_to" << endl;
////			cout << contour_to_use.size() << endl;
//
//
//
//
//	}
//	printf("s3000 \n");
//	int  width_image = img.size().height;
//	int length_image = img.size().width;
//
////	for(int k = 0; k < contour_to_use.size(); k++)
////	{
////		cout << "contour_to_use" << endl;
////		cout << contour_to_use[k] << endl;
////	}
//
//
//	float threshold_x_min = epsilon * (float)length_image;
//	float threshold_x_max = (1-epsilon) * (float)length_image;
//	Point x_y_1;
//	x_y_1.x = (int)threshold_x_min;
//	x_y_1.y = 0;
//	Point x_y_2;
//	x_y_2.x = (int)threshold_x_max;
//	x_y_2.y = 0;
//	printf("s100 \n");
////
//
//	circle (img, x_y_1, 10, (255, 255, 255), -1 );
//	printf("s11 \n");
//	circle (img, x_y_2, 10, (255, 255, 255), -1 );
//	printf("s12 \n");
//	int p = 0;
//	int jj = 0;
//	int number_all_points_count = 0;
//	printf("contour_to_use.size() %d\n",contour_to_use.size());
//	for (int y=0 ; y < contour_to_use.size() ; y ++)
//	{
//		contour_in_process = imgdata.contours [contour_to_use[y]];
//		printf("s13 \n");
//		process_data = get_point(contour_to_use[y], imgdata.contours);
//		printf("s14 \n");
//		obstacle_position_data = draw_obstacle(process_data, img);
//		printf("s15 \n");
//		int number_obstacle = obstacle_position_data.size();
//		printf("obstacle_position_data.size() %d\n",obstacle_position_data.size());
//		int where_obstacle = number_obstacle/2;
//		printf("swhere_obstacle %d\n",where_obstacle);
//
//		for(int q = 0; q < where_obstacle; q++)
//		{
//			int x_point = obstacle_position_data[q*2];
//			printf("s18 \n");
//			if(((float)x_point > threshold_x_min) & ((float)x_point < threshold_x_max))
//			{
//				safe_or_not.safeToGoForward = false;
//				printf("s19 \n");
//				jj = jj + 1;
//			}
//			else
//			{
//				p = p + 1;
//				printf("s20 \n");
//			}
//
//		}
//		number_all_points_count = number_all_points_count + obstacle_position_data.size()/2;
//		printf("s21 \n");
//		printf("how many points in  threshold??????? %d\n",jj);
//		printf("how many points not in  threshold??????? %d\n",p);
////	cout << "len(obstacle_position_data)"<< obstacle_position_data.size() << endl;
////
////	cout << "p"<< p << endl;
////
////	cout << "number_all_points_count" << number_all_points_count << endl;
//
//	if((p - number_all_points_count) == 0 )
//	{
//		safe_or_not.safeToGoForward = true;
//		printf("s22 \n");
//	}
//
//	}
///////// puttext
//
//
//	Point jb;
//	jb.x = 200;
//	jb.y = 100;
//	if(safe_or_not.safeToGoForward == true)
//	{
//
//		putText(img, "true",jb,FONT_HERSHEY_SIMPLEX,5,(255,255,255) );
//	}
//	else
//	{
//
//		putText(img, "FALSE",jb,FONT_HERSHEY_SIMPLEX,5,(255,255,255) );
//	}
//
//
//
//////////
//
//	safe_or_not.img=img;
//	printf("s23 \n");
//	return safe_or_not;
//}
//
//
//void find_contour(char *img, int width, int height)
//{
//  // Create a new image, using the original bebop image.
//  Mat M(width, height, CV_8UC2, img); // original
//	printf("s24 \n");
//  Mat image, edge_image, thresh_image;
//	printf("s25 \n");
//
//  // convert UYVY in paparazzi to YUV in opencv
//  cvtColor(M, M, CV_YUV2RGB_Y422);
//	printf("s26 \n");
//  cvtColor(M, M, CV_RGB2YUV);
//	printf("s27 \n");
//	printf("width~~~~~~~~ %d\n",width);
//	printf("height......... %d\n",height);
//
//  // Threshold all values within the indicted YUV values.
//  inRange(M, Scalar(cont_thres.lower_y, cont_thres.lower_u, cont_thres.lower_v), Scalar(cont_thres.upper_y,
//          cont_thres.upper_u, cont_thres.upper_v), thresh_image);
//
//  /// Find contours
//  //float epsilon= 0.35;
//
//  final_struct safe = judge_go_forward(M,0.25);////////!!!!!!!!!!!
//	printf("s28 \n");
//  printf("safe.safeToGoForward %d\n",safe.safeToGoForward);
// return ;
//}
//
//
//
//










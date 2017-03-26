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
 * @file "modules/computer_vision/detect_contour.c"
 * @author Roland Meertens and Peng Lu
 *
 */

#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/detect_contour.h"
#include "modules/computer_vision/opencv_contour.h"
#include "modules/computer_vision/lib/vision/image.h"

//struct video_listener *listenerd = NULL;

// Filter Settings
uint8_t detect_lum_min = 55;
uint8_t detect_lum_max = 205;
uint8_t detect_cb_min  = 52;
uint8_t detect_cb_max  = 140;
uint8_t detect_cr_min  = 127;
uint8_t detect_cr_max  = 255;

// Function
int detect_count = 0;
struct image_t *contour_func(struct image_t *img);
struct image_t *contour_func(struct image_t *img)
{
	//struct image_t imgout;

  if (img->type == IMAGE_YUV422) {
    // Call OpenCV (C++ from paparazzi C function)
//	  detect_count = image_yuv422_colorfilt(img, img,
//			  detect_lum_min, detect_lum_max,
//			  detect_cb_min, detect_cb_max,
//			  detect_cr_min, detect_cr_max
//	                                      );
   // find_contour((char *) img->buf, img->w, img->h);
	  find_contour((char *) img->buf, img->w, img->h);
  }
  return img;
}

void detect_contour_init(void)
{
	cv_add_to_device(&DETECT_CONTOUR_CAMERA, contour_func);
  // in the mavlab, bright
  cont_thres.lower_y = 16;  cont_thres.lower_u = 135; cont_thres.lower_v = 80;
  cont_thres.upper_y = 100; cont_thres.upper_u = 175; cont_thres.upper_v = 165;
  //
  // for cyberzoo: Y:12-95, U:129-161, V:80-165, turn white.
  //int y1=16;  int u1=129; int v1=80; % cyberzoo, dark
  //int y2=100; int u2=161; int v2=165;
}

//void detect_contour_run(void)
//{
//	//cv_add_to_device(&DETECT_CONTOUR_CAMERA, contour_func);
//  // in the mavlab, bright
//  //cont_thres.lower_y = 16;  cont_thres.lower_u = 135; cont_thres.lower_v = 80;
//  //cont_thres.upper_y = 100; cont_thres.upper_u = 175; cont_thres.upper_v = 165;
//  printf("!!!!!!!!!! \n");
//  //
//  // for cyberzoo: Y:12-95, U:129-161, V:80-165, turn white.
//  //int y1=16;  int u1=129; int v1=80; % cyberzoo, dark
//  //int y2=100; int u2=161; int v2=165;
//}


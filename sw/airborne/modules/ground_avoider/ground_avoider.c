/*
 * Copyright (C) ych
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
 * @file "modules/ground_avoider/ground_avoider.c"
 * @author ych
 * avoid the ground
 */

#include "modules/ground_avoider/ground_avoider.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "firmwares/rotorcraft/navigation.h"

#include "generated/flight_plan.h"
#include "modules/computer_vision/colorfilter.h"
#include "modules/orange_avoider/orange_avoider.h"

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

uint8_t safeToGoForwards        = false;
int tresholdColorCount          = 0.05 * 124800; // 520 x 240 = 124.800 total pixels
float incrementForAvoidance;
uint16_t trajectoryConfidence   = 1;
float maxDistance               = 2.25;


/*
 * Initialisation function, setting the colour filter, random seed and incrementForAvoidance
 */
void ground_avoider_init()
{
  // Initialise the variables of the colorfilter to accept green***
  color_lum_min = 0;
  color_lum_max = 255;
  color_cb_min  = 0;
  color_cb_max  = 106;
  color_cr_min  = 0;
  color_cr_max  = 141;
  // Initialise random values
  srand(time(NULL));
  chooseRandomIncrementAvoidance();
}

/*
 * Function that checks it is safe to move forwards, and then moves a waypoint forward or changes the heading
 */
 void ground_avoider_periodic() 
{


}



/**
* This file is part of DSO.
*
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once


#include "util/NumType.h"

namespace dso
{

//====================================================================
// MIKEK: update his class to hold reference-depth related derivatives
//====================================================================

// NOTE: it would appear that only one frame's parameters are considered here
// otherise, say, d[x,y]/d[xi] would be 2x12 and not 2x6

struct RawResidualJacobian
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  // ================== new structure: save independently =============.
  VecNRf resF;

  // the two rows of d[x,y]/d[xi].
  Vec6f Jpdxi[2];			// 2x6 (TODO: change to 3x6)

  // the two rows of d[x,y]/d[C].
  VecCf Jpdc[2];			// 2x4 (TODO: change to 3x4)

  // the two rows of d[x,y]/d[idepth].
  Vec2f Jpdd;				// 2x1 (TODO: change to 3x1)

  // the two columns of d[r]/d[x,y].
  VecNRf JIdx[2];			// 8x2 (one residual per pixel in path)
  // TODO: change to 8x3 (d[r]/d[x, y, dj])

  // = the two columns of d[r] / d[ab]
  VecNRf JabF[2];			// 8x2 (one residual per pixel in path)

  // = JIdx^T * JIdx (inner product). Only as a shorthand.
  Mat22f JIdx2;				// 2x2 (TODO: change to 3x3)

  // = Jab^T * JIdx (inner product). Only as a shorthand.
  Mat22f JabJIdx;			// 2x2 (TODO: change to 2x3)

  // = Jab^T * Jab (inner product). Only as a shorthand.
  Mat22f Jab2;			// 2x2

};
}


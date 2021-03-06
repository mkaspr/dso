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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"

#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "FullSystem/ResidualProjections.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso
{
int PointFrameResidual::instanceCounter = 0;


long runningResID=0;


PointFrameResidual::PointFrameResidual(){assert(false); instanceCounter++;}

PointFrameResidual::~PointFrameResidual(){assert(efResidual==0); instanceCounter--; delete J;}

PointFrameResidual::PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_) :
  point(point_),
  host(host_),
  target(target_)
{
  efResidual=0;
  instanceCounter++;
  resetOOB();
  J = new RawResidualJacobian();
  assert(((long)J)%16==0);

  isNew=true;
}

//########################################################################
//########################################################################
// MIKEK: this is where all of my light-aware changes need to be made
//########################################################################
//########################################################################
double PointFrameResidual::linearize(CalibHessian* HCalib)
{
  // initialize new cost as invalid value
  state_NewEnergyWithOutlier=-1;

  // check if current residual state is "out-of-bounds"
  if (state_state == ResState::OOB)
  {
    // assign out-of-bounds to new state as well
    state_NewState = ResState::OOB;

    // return current energy
    return state_energy;
  }

  // get reference to precalculated values for frame-frame pair
  FrameFramePrecalc* precalc = &(host->targetPrecalc[target->idx]);

  // initialize total cost for keypoint patch
  float energyLeft=0;

  // fetch precalculated values for frame-frame observation
  // TODO: denote what each of these represents
  const Eigen::Vector3f* dIl = target->dI;
  const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
  const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
  const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
  const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
  const float* const color = point->color;
  const float* const weights = point->weights;
  Vec2f affLL = precalc->PRE_aff_mode;
  float b0 = precalc->PRE_b0_mode;

  // allocate partial derivatives
  Vec6f d_xi_x, d_xi_y;
  Vec4f d_C_x, d_C_y;
  float d_d_x, d_d_y;

  {
    // declare output parameters populated by projection function
    float drescale, u, v, new_idepth;
    float Ku, Kv;
    Vec3f KliP;

    // check if keypoint does notsuccessfully projections into reference frame
    // NOTE: this has the side of affect of computing Ku, Kv, and new_idepth
    // TODO: what is "zero-scaled" mean here?
    if(!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0,HCalib,
        PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
    {
      // set new state to out-of-bounds
      state_NewState = ResState::OOB;

      // return current energy
      return state_energy;
    }

    // get center-point and depth of keypoint in new reference frame
    centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

    //===========================================================
    // NOTE: these are all derivs for change in uv given params
    // however, I'll need to compute change in depth givie params
    // then actually use these values later during optimization
    //===========================================================

    // compute derivatives for change in uv as per keypoint depth
    // NOTE: light-aware has no change on camera geometry

    d_d_x = drescale * (PRE_tTll_0[0]-PRE_tTll_0[2]*u)*
        SCALE_IDEPTH*HCalib->fxl();

    d_d_y = drescale * (PRE_tTll_0[1]-PRE_tTll_0[2]*v)*
        SCALE_IDEPTH*HCalib->fyl();

    // derivatives for the change in uv as per camera calibration
    // NOTE: light-aware has no change on camera geometry
    // NOTE: I believe the camera parameters are global for window

    d_C_x[2] = drescale*(PRE_RTll_0(2,0)*u-PRE_RTll_0(0,0));

    d_C_x[3] = HCalib->fxl() * drescale*(PRE_RTll_0(2,1)*u-PRE_RTll_0(0,1)) *
        HCalib->fyli();

    d_C_x[0] = KliP[0]*d_C_x[2];
    d_C_x[1] = KliP[1]*d_C_x[3];

    d_C_y[2] = HCalib->fyl() * drescale*(PRE_RTll_0(2,0)*v-PRE_RTll_0(1,0)) *
        HCalib->fxli();

    d_C_y[3] = drescale*(PRE_RTll_0(2,1)*v-PRE_RTll_0(1,1));
    d_C_y[0] = KliP[0]*d_C_y[2];
    d_C_y[1] = KliP[1]*d_C_y[3];

    d_C_x[0] = (d_C_x[0]+u)*SCALE_F;
    d_C_x[1] *= SCALE_F;
    d_C_x[2] = (d_C_x[2]+1)*SCALE_C;
    d_C_x[3] *= SCALE_C;

    d_C_y[0] *= SCALE_F;
    d_C_y[1] = (d_C_y[1]+v)*SCALE_F;
    d_C_y[2] *= SCALE_C;
    d_C_y[3] = (d_C_y[3]+1)*SCALE_C;

    // compute derivatives for the change in uv as per camera pose
    // NOTE: light-aware has no change on camera geometry
    // NOTE: it would seem we are only considerinng one transform for this error
    // does that mean only the pose of the latest keyframes is optimized?

    d_xi_x[0] = new_idepth*HCalib->fxl();
    d_xi_x[1] = 0;
    d_xi_x[2] = -new_idepth*u*HCalib->fxl();
    d_xi_x[3] = -u*v*HCalib->fxl();
    d_xi_x[4] = (1+u*u)*HCalib->fxl();
    d_xi_x[5] = -v*HCalib->fxl();

    d_xi_y[0] = 0;
    d_xi_y[1] = new_idepth*HCalib->fyl();
    d_xi_y[2] = -new_idepth*v*HCalib->fyl();
    d_xi_y[3] = -(1+v*v)*HCalib->fyl();
    d_xi_y[4] = u*v*HCalib->fyl();
    d_xi_y[5] = u*HCalib->fyl();
  }

  // compose Jacobian matrix from partial derivatives
  // NOTE: two values as chain-rule ends with 2D image gradient
  {
    // update with pose derivatives
    J->Jpdxi[0] = d_xi_x;
    J->Jpdxi[1] = d_xi_y;

    // update with camera derivatives
    J->Jpdc[0] = d_C_x;
    J->Jpdc[1] = d_C_y;

    // update with depth derivatives
    J->Jpdd[0] = d_d_x;
    J->Jpdd[1] = d_d_y;
  }

  // allocate scalars for Jacobian-based calculations
  float JIdxJIdx_00=0, JIdxJIdx_11=0, JIdxJIdx_10=0;
  float JabJIdx_00=0, JabJIdx_01=0, JabJIdx_10=0, JabJIdx_11=0;
  float JabJab_00=0, JabJab_01=0, JabJab_11=0;
  float wJI2_sum = 0;

  // for each pixel in keypoint pixel-patch
  for(int idx=0;idx<patternNum;idx++)
  {
    // allocate projection coords for current pixel in patch
    float Ku, Kv;

    // check if pixel in patch does not project into current image
    // NOTE: this has the side affect of computing Ku and Kv coordinates
    // TODO: what is "scaled" mean here and why different from above?
    if (!projectPoint(point->u+patternP[idx][0], point->v+patternP[idx][1],
          point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
    {
      // set state to "out-of-bounds"
      state_NewState = ResState::OOB;

      // return current energy
      return state_energy;
    }

    // record screen coords of pixel in reference image
    projectedTo[idx][0] = Ku;
    projectedTo[idx][1] = Kv;

    // sample intensity and 2D derivatives for given coordinate
    Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

    //=======================================================================
    // finally, compute photometric error (TODO: update with light-aware cost)
    // NOTE: weight still needs to be applied (so update derivs accordingly)
    // NOTE: hitColor is currently only changed by uv-coords
    // however, in light-aware, it is also change by relative depth
    // ASSUME: color[idx] has already been converted to "global" brightness
    //=======================================================================
    float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);

    //========================================================================
    // compute partial-derivative of photometric error w.r.t. affine-scale (a)
    // TODO: should this be a negative value? what is b0?
    //========================================================================
    float drdA = (color[idx]-b0);

    // check if hitColor is invalid (ASSUME: over-/under-saturated)
    if (!std::isfinite((float)hitColor[0]))
    {
      // set state to "out-of-bounds"
      state_NewState = ResState::OOB;

      // return current energy
      return state_energy;
    }

    //=============================================================
    // compute residual weight based on magnitude of image gradient
    // w = sqrt(c / (c + norm(dI))
    // TODO: update with my own weighting function
    //=============================================================
    float w = sqrtf(setting_outlierTHSumComponent /
        (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));

    //================================================
    // average weight with main keyframe weight
    // TODO: where does these other weights come from?
    // TODO: perhaps I should change this as well
    //================================================
    w = 0.5f * (w + weights[idx]);

    // compute huber norm weighting
    float hw = fabsf(residual) < setting_huberTH ?
        1 : setting_huberTH / fabsf(residual);

    // compute final residual and add to current total
    // will be summed over all pixels in keypoint patch
    energyLeft += w*w*hw *residual*residual*(2-hw);

    {
      // check if huber-norm weight below 1.0
      if (hw < 1)
      {
        // increase weight (sqrt(a) > a : where 0 < a < 1)
        hw = sqrtf(hw);
      }

      // compute final residual weight (huber & image-based)
      hw = hw*w;

      // update image gradient with weight
      hitColor[1]*=hw;
      hitColor[2]*=hw;


      //===================================================
      // missing derivatives for J_cost_depth (light-aware)
      // I need to add new RawResdiaulJacobian members
      // and populate them accordingly here
      // then, when solving I'll need to use them too
      //===================================================

      // compute the weight-residual for current pixel in patch
      J->resF[idx] = residual*hw;

      // compute change is cost as per change in u-coord
      J->JIdx[0][idx] = hitColor[1];

      // compute change is cost as per change in v-coord
      J->JIdx[1][idx] = hitColor[2];

      // NOTE: it would seems that only one the affine brightness is considered
      // does that mean that only last keyframe's brightness is optimized?

      // compute change in cost as per change in brightness-scale
      J->JabF[0][idx] = drdA*hw;

      // compute change in cost as per change in brightness-bias
      J->JabF[1][idx] = hw;


      // update select cells of the hessian
      // ASSUME: nothing needs to be updated here

      //===============================================
      // missing hessian for J_cost_depth (light-aware)
      // I need to add new RawResdiaulJacobian members
      // and populate the hessian accordingly here
      // then, when solving I'll need to use them too
      //===============================================

      JIdxJIdx_00+=hitColor[1]*hitColor[1];
      JIdxJIdx_11+=hitColor[2]*hitColor[2];
      JIdxJIdx_10+=hitColor[1]*hitColor[2];

      JabJIdx_00+= drdA*hw * hitColor[1];
      JabJIdx_01+= drdA*hw * hitColor[2];
      JabJIdx_10+= hw * hitColor[1];
      JabJIdx_11+= hw * hitColor[2];

      JabJab_00+= drdA*drdA*hw*hw;
      JabJab_01+= drdA*hw*hw;
      JabJab_11+= hw*hw;

      //========================================================
      // update total weight-intensity value for threshold check
      // TODO: make sure my weight system doesn't muck this up
      //========================================================
      wJI2_sum += hw*hw*(hitColor[1]*hitColor[1]+hitColor[2]*hitColor[2]);

      // zero-out deriative if brightness-scale disbaled
      if(setting_affineOptModeA < 0) J->JabF[0][idx]=0;

      // zero-out deriative if brightness-bias disbaled
      if(setting_affineOptModeB < 0) J->JabF[1][idx]=0;
    }

  } // end of processing each pixel in patch

  // compose pre-computed Jacobian-based matrices
  J->JIdx2(0,0) = JIdxJIdx_00;
  J->JIdx2(0,1) = JIdxJIdx_10;
  J->JIdx2(1,0) = JIdxJIdx_10;
  J->JIdx2(1,1) = JIdxJIdx_11;
  J->JabJIdx(0,0) = JabJIdx_00;
  J->JabJIdx(0,1) = JabJIdx_01;
  J->JabJIdx(1,0) = JabJIdx_10;
  J->JabJIdx(1,1) = JabJIdx_11;
  J->Jab2(0,0) = JabJab_00;
  J->Jab2(0,1) = JabJab_01;
  J->Jab2(1,0) = JabJab_01;
  J->Jab2(1,1) = JabJab_11;

  // store residual value before outlier removal
  state_NewEnergyWithOutlier = energyLeft;

  // check if new cost is larger than either frame's threshold
  // or if the total weight applied to the residual is below a fixed threshold
  if(energyLeft > std::max<float>(host->frameEnergyTH, target->frameEnergyTH) ||
    wJI2_sum < 2)
  {
    // set enerage to max of host and target frames energy threshold
    // NOTE: these are computed as some scale of the previous tracking error
    energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);

    // set residual state to "outlier"
    state_NewState = ResState::OUTLIER;
  }
  else
  {
    // set residual state to "inlier"
    state_NewState = ResState::IN;
  }

  // record new total cost for keypoint patch
  state_NewEnergy = energyLeft;

  // return new total cost for keypoint patch
  return energyLeft;
}



void PointFrameResidual::debugPlot()
{
  if(state_state==ResState::OOB) return;
  Vec3b cT = Vec3b(0,0,0);

  if(freeDebugParam5==0)
  {
    float rT = 20*sqrt(state_energy/9);
    if(rT<0) rT=0; if(rT>255)rT=255;
    cT = Vec3b(0,255-rT,rT);
  }
  else
  {
    if(state_state == ResState::IN) cT = Vec3b(255,0,0);
    else if(state_state == ResState::OOB) cT = Vec3b(255,255,0);
    else if(state_state == ResState::OUTLIER) cT = Vec3b(0,0,255);
    else cT = Vec3b(255,255,255);
  }

  for(int i=0;i<patternNum;i++)
  {
    if((projectedTo[i][0] > 2 && projectedTo[i][1] > 2 && projectedTo[i][0] < wG[0]-3 && projectedTo[i][1] < hG[0]-3 ))
      target->debugImage->setPixel1((float)projectedTo[i][0], (float)projectedTo[i][1],cT);
  }
}

void PointFrameResidual::applyRes(bool copyJacobians)
{
  // check if we're copying the residual jacobians
  if (copyJacobians)
  {
    // check if keypoint "out-of-bounds"
    if (state_state == ResState::OOB)
    {
      assert(!efResidual->isActiveAndIsGoodNEW);
      return;	// can never go back from OOB
    }

    // check if keypoint is inlier
    if(state_NewState == ResState::IN)
    {
      // set status to good
      efResidual->isActiveAndIsGoodNEW=true;
      efResidual->takeDataF();
    }
    // otherwise outlier
    else
    {
      // set status to bad
      efResidual->isActiveAndIsGoodNEW=false;
    }
  }

  // copy new residual state to current state
  setState(state_NewState);

  // copy state energy to current energy
  state_energy = state_NewEnergy;
}

}

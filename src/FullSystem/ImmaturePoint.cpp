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



#include "FullSystem/ImmaturePoint.h"
#include "util/FrameShell.h"
#include "FullSystem/ResidualProjections.h"

namespace dso
{
ImmaturePoint::ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib)
: u(u_), v(v_), host(host_), my_type(type), idepth_min(0), idepth_max(NAN), lastTraceStatus(IPS_UNINITIALIZED)
{
  gradH.setZero();

  // process each patch pixel
  for(int idx=0;idx<patternNum;idx++)
  {
    // get patch pixel offset from center
    int dx = patternP[idx][0];
    int dy = patternP[idx][1];

    // get patch pixel intensity and gradients
    Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u+dx, v+dy,wG[0]);

    // assign intensity
    color[idx] = ptc[0];

    // if single patch pixel invalid all is invalid, return now
    if(!std::isfinite(color[idx])) {energyTH=NAN; return;}

    // add [ gx ; gy ] * [ gx ; gy ]' to 2x2 hessian
    gradH += ptc.tail<2>() * ptc.tail<2>().transpose();

    // compute weight as per gradient norm (lower gradient = higher weight)
    weights[idx] = sqrtf(setting_outlierTHSumComponent /
        (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
  }

  energyTH = patternNum*setting_outlierTH;
  energyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;

  idepth_GT=0;
  quality=10000;
}

ImmaturePoint::~ImmaturePoint()
{
}



/*
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */

// NOTE: this is where the epipolar line search is performed to update keypoints
// this should also be where the photometric error is computed
// so this where my light-aware updates could be made
// but I think tracking and optimization will be where it really takes place
ImmaturePointStatus ImmaturePoint::traceOn(FrameHessian* frame,
    const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt,
    const Vec2f& hostToFrame_affine, CalibHessian* HCalib, bool debugPrint)
{
  // skip updating keypointns that are mature / marginalized
  if(lastTraceStatus == ImmaturePointStatus::IPS_OOB) return lastTraceStatus;

  // disable debug
  debugPrint = false;

  // compuate max pixel search
  // NOTE: with current settings on a 640x480 image, this is ~30 pixels
  float maxPixSearch = (wG[0]+hG[0])*setting_maxPixSearch;

  // print debug message (never happens)
  if(debugPrint)
  {
    printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f %f!\n",
        u,v,
        host->shell->id, frame->shell->id,
        idepth_min, idepth_max,
        hostToFrame_Kt[0],hostToFrame_Kt[1],hostToFrame_Kt[2]);
  }

  //	const float stepsize = 1.0;				      // stepsize for initial discrete search.
  //	const int GNIterations = 3;				      // max # GN iterations
  //	const float GNThreshold = 0.1;          // GN stop after this stepsize.
  //	const float extraSlackOnTH = 1.2;       // for energy-based outlier check, be slightly more relaxed by this factor.
  //	const float slackInterval = 0.8;		 	  // if pixel-interval is smaller than this, leave it be.
  //	const float minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.

  // ============== project min and max. return if one of them is OOB ==========

  // project point at min depth into current frame
  Vec3f pr = hostToFrame_KRKi * Vec3f(u,v, 1);
  Vec3f ptpMin = pr + hostToFrame_Kt*idepth_min;
  float uMin = ptpMin[0] / ptpMin[2];
  float vMin = ptpMin[1] / ptpMin[2];

  // check if projected point is out-of-bounds (OOB)
  if(!(uMin > 4 && vMin > 4 && uMin < wG[0]-5 && vMin < hG[0]-5))
  {
    // print debug message (never happens)
    if(debugPrint)
    {
      printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n",
          u,v,uMin, vMin,  ptpMin[2], idepth_min, idepth_max);
    }

    // fail if minn depth not in image

    // reset trace variables
    lastTraceUV = Vec2f(-1,-1);
    lastTracePixelInterval=0;

    // return new trace status
    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
  }

  float dist;
  float uMax;
  float vMax;
  Vec3f ptpMax;

  // check that max depth is valid
  if (std::isfinite(idepth_max))
  {
    // project point at max depth into current frame
    ptpMax = pr + hostToFrame_Kt*idepth_max;
    uMax = ptpMax[0] / ptpMax[2];
    vMax = ptpMax[1] / ptpMax[2];

    // check if out-of-bounds
    if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
    {
      // fail if max depth not in image
      if(debugPrint) printf("OOB uMax  %f %f - %f %f!\n",u,v, uMax, vMax);
      lastTraceUV = Vec2f(-1,-1);
      lastTracePixelInterval=0;
      return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    }

    // ====== check their distance. everything below 2px is OK (-> skip). ======

    // compute distance (in pixels) between max and min depth points
    dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax);
    dist = sqrtf(dist);

    // pixel distance too small (current settings < 1.5 pixels) don't bother
    if(dist < setting_trace_slackInterval)
    {
      if(debugPrint)
        printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);

      // return skipped status (valid by nothing to search)
      lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
      lastTracePixelInterval=dist;
      return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
    }
    assert(dist>0);
  }
  // max depth is not valid
  // TODO: when would this be the case? when not initialized?
  else
  {
    // start with min position + max pixel search length
    dist = maxPixSearch;

    // project to arbitrary depth to get direction.
    ptpMax = pr + hostToFrame_Kt*0.01;
    uMax = ptpMax[0] / ptpMax[2];
    vMax = ptpMax[1] / ptpMax[2];

    // compute direction.
    float dx = uMax-uMin;
    float dy = vMax-vMin;
    float d = 1.0f / sqrtf(dx*dx+dy*dy);

    // compute image coordinages of max pixel search position
    uMax = uMin + dist*dx*d;
    vMax = vMin + dist*dy*d;

    // check if out of bounds
    if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
    {
      // skip search as we must search the full length (~30 pixels)
      if(debugPrint) printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax,  ptpMax[2]);
      lastTraceUV = Vec2f(-1,-1);
      lastTracePixelInterval=0;
      return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    }
    assert(dist>0);
  }

  // set OOB if scale change too big.
  // - if min depth is initialized
  // - and min-depth < 0.75 or min-depth > 1.5 (unitless)
  // TODO: exactly why is this check necessary
  // - also how is scale initialized? I assume it must be relative to this
  if(!(idepth_min<0 || (ptpMin[2]>0.75 && ptpMin[2]<1.5)))
  {
    // return out-of-bounds status
    if(debugPrint) printf("OOB SCALE %f %f %f!\n", uMax, vMax,  ptpMin[2]);
    lastTraceUV = Vec2f(-1,-1);
    lastTracePixelInterval=0;
    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
  }

  // ============== compute error-bounds on result in pixel ===============
  // ============== if the new interval is not at least 1/2 of the old, SKIP ===

  // compute total vertical and horizontal search distanceb
  // current settings stepsize = 1.0
  // NOTE: so norm(dx, dy) can be up 1.4 larger than norm(max - min)
  float dx = setting_trace_stepsize*(uMax-uMin);
  float dy = setting_trace_stepsize*(vMax-vMin);

  // TODO: not sure what this is doing exactly
  // ASSUME: trying to assess if patch gradient is parallel with search dir
  // - so strong vert gradient with horizontal search not ideal
  float a = (Vec2f(dx,dy).transpose() * gradH * Vec2f(dx,dy));
  float b = (Vec2f(dy,-dx).transpose() * gradH * Vec2f(dy,-dx));

  // TODO: where does the 0.2s come from?
  float errorInPixel = 0.2f + 0.2f * (a+b) / a;

  // check if good conditions for epipolar line search
  if(errorInPixel*setting_trace_minImprovementFactor > dist &&
      std::isfinite(idepth_max))
  {
    if(debugPrint)
      printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);

    // return bad search condition status
    lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
    lastTracePixelInterval=dist;
    return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
  }

  // saturate error to a max of 10 pixels
  // TODO: where else is this error used?
  if(errorInPixel > 10) errorInPixel = 10;

  // ============== do the discrete search ===================

  // normalize step direction
  dx /= dist;
  dy /= dist;

  // print debug message (never happens)
  if(debugPrint)
  {
    printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n",
        u,v,
        host->shell->id, frame->shell->id,
        idepth_min, uMin, vMin,
        idepth_max, uMax, vMax,
        errorInPixel
        );
  }

  // fix search length to max pixel search
  if(dist > maxPixSearch)
  {
    uMax = uMin + maxPixSearch*dx;
    vMax = vMin + maxPixSearch*dy;
    dist = maxPixSearch;
  }

  // compute number of discrete steps
  // NOTE: seems to force it to be 2 steps longer than needed
  int numSteps = 1.9999f + dist / setting_trace_stepsize;

  // TODO: what does the top left corner represent?
  Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2,2>();

  // computes a "random" shift but using the decimal values of the real numbers
  // essentially this will be a value between 0 and 1.0
  float randShift = uMin*1000-floorf(uMin*1000);

  // compute initialize pixel pos
  // NOTE: never shortens the search, only starts it earlier
  float ptx = uMin-randShift*dx;
  float pty = vMin-randShift*dy;

  // allocate buffer for rotated patch pixel offsetts
  Vec2f rotatetPattern[MAX_RES_PER_POINT];

  // for each patch pixel
  for(int idx=0;idx<patternNum;idx++)
  {
    // rotate offset as per current frame-frame transform
    rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);
  }

  // if either direction is invalid abort
  // NOTE: could have done this much earlier
  if(!std::isfinite(dx) || !std::isfinite(dy))
  {
    lastTracePixelInterval=0;
    lastTraceUV = Vec2f(-1,-1);
    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
  }

  // allocate error buffer
  // NOTE: odd that's allocated with a fixed size
  // - granted the number of steps shouldn't be much more than 30 in pratice
  // - but still a poor way to go about it
  float errors[100];

  // initialize best position and error
  float bestU=0, bestV=0, bestEnergy=1e10;

  // initialize best step index
  int bestIdx=-1;

  // force number of steps to be less than 100
  // NOTE: odd that we have allocated 1 more error value than we need
  if(numSteps >= 100) numSteps = 99;

  // perform each step along epipolar line
  for(int i=0;i<numSteps;i++)
  {
    // initialize current error to zero
    float energy=0;

    // for each pixel patch
    for(int idx=0;idx<patternNum;idx++)
    {
      // sample intensity in current image
      float hitColor = getInterpolatedElement31(frame->dI,
                    (float)(ptx+rotatetPattern[idx][0]),
                    (float)(pty+rotatetPattern[idx][1]),
                    wG[0]);

      // if color invalid, skip step altogether
      // TODO: what makes a pixel not finite? (over/under-saturated?)
      if(!std::isfinite(hitColor)) {energy+=1e5; continue;}

      // compute photometric error
      // #######################################################################
      // #######################################################################
      // MIKEK: this is where my new cost function would be added
      // #######################################################################
      // #######################################################################
      float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] +
          hostToFrame_affine[1]);

      // compute huber norm weighting for residual
      float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

      // add patch pixel weighted residual to total error
      energy += hw *residual*residual*(2-hw);
    }

    // print debug message (never happens)
    if(debugPrint)
    {
      printf("step %.1f %.1f (id %f): energy = %f!\n",
          ptx, pty, 0.0f, energy);
    }

    // record current error to history
    errors[i] = energy;

    // check if better match found
    if(energy < bestEnergy)
    {
      // update best match stats
      bestU = ptx;
      bestV = pty;
      bestEnergy = energy;
      bestIdx = i;
    }

    // take step along line
    ptx+=dx;
    pty+=dy;

  } // end of epipolar line search

  // find best score outside a +-2px radius.
  // NOTE: use for an ambiguity filter

  // initialize second best cost to be very high
  float secondBest=1e10;

  // process each recorded cost value
  for(int i=0;i<numSteps;i++)
  {
    // if index of cost far away from best match (+/- 2 pixels away)
    // and its error is better than the current second best
    if((i < bestIdx-setting_minTraceTestRadius ||
        i > bestIdx+setting_minTraceTestRadius) && errors[i] < secondBest)
    {
      // update second beset cost
      secondBest = errors[i];
    }
  }

  // get ration of second best to best score
  float newQuality = secondBest / bestEnergy;

  // if better quality found or a very large search length
  // update the historic quality of the search
  if(newQuality < quality || numSteps > 10) quality = newQuality;

  // ============== do GN optimization ===================

  // save backup of best match coordinates
  float uBak=bestU, vBak=bestV, gnstepsize=1, stepBack=0;

  // initialize new best cost (as very high)
  if(setting_trace_GNIterations>0) bestEnergy = 1e5;

  // initialize step stats
  int gnStepsGood=0, gnStepsBad=0;

  // for each gauss-newton iteration (3 iterations)
  for(int it=0;it<setting_trace_GNIterations;it++)
  {
    // initialize 1D "Ax = b" problem
    float H = 1, b=0, energy=0;

    // for each patch pixel
    for(int idx=0;idx<patternNum;idx++)
    {
      // sample pixel intensity and gradients
      Vec3f hitColor = getInterpolatedElement33(frame->dI,
          (float)(bestU+rotatetPattern[idx][0]),
          (float)(bestV+rotatetPattern[idx][1]),wG[0]);

      // if invalid color skip remaining evaluation
      if(!std::isfinite((float)hitColor[0])) {energy+=1e5; continue;}

      // compute photometric error
      // #######################################################################
      // #######################################################################
      // MIKEK: this is where my new cost function would be added
      // #######################################################################
      // #######################################################################
      float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] +
          hostToFrame_affine[1]);

      // compute jacobian (using search direction and pixel gradient)
      float dResdDist = dx*hitColor[1] + dy*hitColor[2];

      // compute huber norm weight
      float hw = fabs(residual) < setting_huberTH ?
          1 : setting_huberTH / fabs(residual);

      // added to weighted hessian
      H += hw*dResdDist*dResdDist;

      // added to weighted gradient
      b += hw*residual*dResdDist;

      // add pixel patch error to total
      energy += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
    }

    // check if step produced worse result
    if(energy > bestEnergy)
    {
      gnStepsBad++;

      // do a smaller step from old point.
      stepBack*=0.5;
      bestU = uBak + stepBack*dx;
      bestV = vBak + stepBack*dy;

      // print debug message (never happens)
      if(debugPrint)
      {
        printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
            it, energy, H, b, stepBack,
            uBak, vBak, bestU, bestV);
      }
    }
    // step produced better result
    else
    {
      // increment successful step count
      gnStepsGood++;

      // compute GN update step
      float step = -gnstepsize*b/H;

      // ensure step magnitude is less than half a pixel
      if(step < -0.5) step = -0.5;
      else if(step > 0.5) step=0.5;

      // if invalid step, zero out
      if(!std::isfinite(step)) step=0;

      // record previous best step
      uBak=bestU;
      vBak=bestV;
      stepBack=step;

      // apply new best step
      bestU += step*dx;
      bestV += step*dy;
      bestEnergy = energy;

      // print debug message (never happens)
      if(debugPrint)
      {
        printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
            it, energy, H, b, step,
            uBak, vBak, bestU, bestV);
      }
    }

    // exit if step update is too small
    if(fabsf(stepBack) < setting_trace_GNThreshold) break;

  } // end of gauss-netwon optimization


  // ============== detect energy-based outlier. ===================

  // check if best cost is above absolute threshold
  if(!(bestEnergy < energyTH*setting_trace_extraSlackOnTH))
  {
    // if it happens once, change status to "outlier" (temporary)
    // it it happens twice, change status to out-of=bounds (permanent)

    if(debugPrint)
      printf("OUTLIER!\n");

    lastTracePixelInterval=0;
    lastTraceUV = Vec2f(-1,-1);

    if(lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
      return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    else
      return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
  }

  // ============== set new interval ===================

  // if more horizontal search direction
  if(dx*dx>dy*dy)
  {
    // update min/max depth for future searches
    // TODO: understand this better
    idepth_min = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU-errorInPixel*dx));
    idepth_max = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU+errorInPixel*dx));
  }
  // if more vertical search direction
  else
  {
    // update min/max depth for future searches
    // TODO: understand this better
    idepth_min = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV-errorInPixel*dy));
    idepth_max = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV+errorInPixel*dy));
  }

  // ensure min depth is smaller than max depth
  if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);

  // check if invalid depth values
  // TODO: when would this happen?
  if(!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max<0))
  {
    // change status to outlier
    lastTracePixelInterval=0;
    lastTraceUV = Vec2f(-1,-1);
    return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
  }

  // finally if we've gotten here, return good status
  lastTracePixelInterval=2*errorInPixel;
  lastTraceUV = Vec2f(bestU, bestV);
  return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
}


float ImmaturePoint::getdPixdd(
    CalibHessian *  HCalib,
    ImmaturePointTemporaryResidual* tmpRes,
    float idepth)
{
  FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);
  const Vec3f &PRE_tTll = precalc->PRE_tTll;
  float drescale, u=0, v=0, new_idepth;
  float Ku, Kv;
  Vec3f KliP;

  projectPoint(this->u,this->v, idepth, 0, 0,HCalib,
      precalc->PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth);

  float dxdd = (PRE_tTll[0]-PRE_tTll[2]*u)*HCalib->fxl();
  float dydd = (PRE_tTll[1]-PRE_tTll[2]*v)*HCalib->fyl();
  return drescale*sqrtf(dxdd*dxdd + dydd*dydd);
}


float ImmaturePoint::calcResidual(
    CalibHessian *  HCalib, const float outlierTHSlack,
    ImmaturePointTemporaryResidual* tmpRes,
    float idepth)
{
  FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

  float energyLeft=0;
  const Eigen::Vector3f* dIl = tmpRes->target->dI;
  const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
  const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
  Vec2f affLL = precalc->PRE_aff_mode;

  for(int idx=0;idx<patternNum;idx++)
  {
    float Ku, Kv;
    if(!projectPoint(this->u+patternP[idx][0], this->v+patternP[idx][1], idepth, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
      {return 1e10;}

    Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
    if(!std::isfinite((float)hitColor[0])) {return 1e10;}
    //if(benchmarkSpecialOption==5) hitColor = (getInterpolatedElement13BiCub(tmpRes->target->I, Ku, Kv, wG[0]));

    float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

    float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
    energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
  }

  if(energyLeft > energyTH*outlierTHSlack)
  {
    energyLeft = energyTH*outlierTHSlack;
  }
  return energyLeft;
}




double ImmaturePoint::linearizeResidual(
    CalibHessian *  HCalib, const float outlierTHSlack,
    ImmaturePointTemporaryResidual* tmpRes,
    float &Hdd, float &bd,
    float idepth)
{
  if(tmpRes->state_state == ResState::OOB)
    { tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy; }

  FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

  // check OOB due to scale angle change.

  float energyLeft=0;
  const Eigen::Vector3f* dIl = tmpRes->target->dI;
  const Mat33f &PRE_RTll = precalc->PRE_RTll;
  const Vec3f &PRE_tTll = precalc->PRE_tTll;
  //const float * const Il = tmpRes->target->I;

  Vec2f affLL = precalc->PRE_aff_mode;

  for(int idx=0;idx<patternNum;idx++)
  {
    int dx = patternP[idx][0];
    int dy = patternP[idx][1];

    float drescale, u, v, new_idepth;
    float Ku, Kv;
    Vec3f KliP;

    if(!projectPoint(this->u,this->v, idepth, dx, dy,HCalib,
        PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth))
      {tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}


    Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

    if(!std::isfinite((float)hitColor[0])) {tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}
    float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

    float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
    energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);

    // depth derivatives.
    float dxInterp = hitColor[1]*HCalib->fxl();
    float dyInterp = hitColor[2]*HCalib->fyl();
    float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);

    hw *= weights[idx]*weights[idx];

    Hdd += (hw*d_idepth)*d_idepth;
    bd += (hw*residual)*d_idepth;
  }


  if(energyLeft > energyTH*outlierTHSlack)
  {
    energyLeft = energyTH*outlierTHSlack;
    tmpRes->state_NewState = ResState::OUTLIER;
  }
  else
  {
    tmpRes->state_NewState = ResState::IN;
  }

  tmpRes->state_NewEnergy = energyLeft;
  return energyLeft;
}



}

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
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

namespace dso
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;



FullSystem::FullSystem()
{
  // initialize system status number
  int retstat = 0;

  // check if we are logging
  if (setting_logStuff)
  {
    // recreate log folder
    retstat += system("rm -rf logs");
    retstat += system("mkdir logs");

    // recreate mat folder
    // NOTE: no where in this codebase is mats folder used
    // - so this code is pointless
    retstat += system("rm -rf mats");
    retstat += system("mkdir mats");

    calibLog = new std::ofstream();
    calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
    calibLog->precision(12);

    numsLog = new std::ofstream();
    numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
    numsLog->precision(10);

    coarseTrackingLog = new std::ofstream();
    coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
    coarseTrackingLog->precision(10);

    eigenAllLog = new std::ofstream();
    eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
    eigenAllLog->precision(10);

    eigenPLog = new std::ofstream();
    eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
    eigenPLog->precision(10);

    eigenALog = new std::ofstream();
    eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
    eigenALog->precision(10);

    DiagonalLog = new std::ofstream();
    DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
    DiagonalLog->precision(10);

    variancesLog = new std::ofstream();
    variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
    variancesLog->precision(10);


    nullspacesLog = new std::ofstream();
    nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
    nullspacesLog->precision(10);
  }
  else
  {
    // initialize null-pointers for file handles
    nullspacesLog=0;
    variancesLog=0;
    DiagonalLog=0;
    eigenALog=0;
    eigenPLog=0;
    eigenAllLog=0;
    numsLog=0;
    calibLog=0;
  }

  // check that no system errors occured
  assert(retstat!=293847);

  // TODO: understand these classes and data-structures
  // - I imagine I'll understand them as I read through addActiveFrame
  // TODO: what is the difference between the coarse tracker and initializer?
  // TODO: what is the difference between the tracker and new KF tracker?

  // related to keypoint selection
  // this is populated by pixelSelector->makeMaps
  // if a cell value != zero then it becomes a candidate
  // that cell value is also passed to the keypoint constructor
  selectionMap = new float[wG[0]*hG[0]];

  // ASSUME: this is used to activate new keypoints
  // - that is, determine closest adjacent points during density check
  coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);

  // used for tracking new frames
  // this is the frame tracker that's being used for tracking
  coarseTracker = new CoarseTracker(wG[0], hG[0]);

  // used for tracking new frames
  // this is the frame tracker that's being built from new KF
  coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);

  // another type of frame tracker
  // but is used during the bootstrapping stage
  coarseInitializer = new CoarseInitializer(wG[0], hG[0]);

  // used for keypoint detection
  // key function is makeMaps
  // which I assume to compute gradients and select keypoints
  pixelSelector = new PixelSelector(wG[0], hG[0]);

  // initialize SLAM system statistics
  // TODO: why backwards and forwards?
  statistics_lastNumOptIts=0;
  statistics_numDroppedPoints=0;
  statistics_numActivatedPoints=0;
  statistics_numCreatedPoints=0;
  statistics_numForceDroppedResBwd = 0;
  statistics_numForceDroppedResFwd = 0;
  statistics_numMargResFwd = 0;
  statistics_numMargResBwd = 0;

  // initalize last tracking photometric error
  // this will be used to determine tracking failure
  // TODO: why store in Vec5?
  lastCoarseRMSE.setConstant(100);

  // I assume this to be the minimum distance (in pixels)
  // in which a candidate keypoint can be from an active keypoint
  // in order to active that candidate
  currentMinActDist=2;

  // initialize SLAM state
  initialized=false;
  isLost=false;
  initFailed=false;

  // TODO: understand this
  // assuming this is the objective function to be minimized
  // so error terms and parameters will be added/remove during runtime
  ef = new EnergyFunctional();

  // TODO: follow this, may not be in addActiveFrame
  // get thread for minimized objective
  // not sure if this is for the window optimization
  // or the tracking optimization
  ef->red = &this->treadReduce;


  // TODO: understand this
  // I assume it denotes something about dropping new keyframes
  // but the "after" confuses me, I'll have to see what this means
  // could it be "after" processing it fully as a simple frame?
  // or is it "after" x frames have been processed?
  needNewKFAfter = -1;

  // TODO: understand this
  // not sure what this is, but I know it to have something to do with
  // timestamps and real-time / playback speeds;
  linearizeOperation=true;

  // TODO: understand this
  // does this indicate DSO can be run as a localization-only system?
  // or is mapping refer to something else here
  runMapping=true;

  // TODO: follow this, may not be in addActiveFrame
  // I assume this creates a separate thread for mapping
  // why is it in a separate thread
  // does that mean VO is happening in another thread
  // and the window optimization is happening in another?
  mappingThread = boost::thread(&FullSystem::mappingLoop, this);

  // TODO: understand this
  lastRefStopID=0;

  // I assume this is for keypoint visualization using JET colormap
  minIdJetVisDebug = -1;
  maxIdJetVisDebug = -1;
  minIdJetVisTracker = -1;
  maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem()
{
  blockUntilMappingIsFinished();

  if (setting_logStuff)
  {
    calibLog->close(); delete calibLog;
    numsLog->close(); delete numsLog;
    coarseTrackingLog->close(); delete coarseTrackingLog;
    //errorsLog->close(); delete errorsLog;
    eigenAllLog->close(); delete eigenAllLog;
    eigenPLog->close(); delete eigenPLog;
    eigenALog->close(); delete eigenALog;
    DiagonalLog->close(); delete DiagonalLog;
    variancesLog->close(); delete variancesLog;
    nullspacesLog->close(); delete nullspacesLog;
  }

  delete[] selectionMap;

  for (FrameShell* s : allFrameHistory)
    delete s;
  for (FrameHessian* fh : unmappedTrackedFrames)
    delete fh;

  delete coarseDistanceMap;
  delete coarseTracker;
  delete coarseTracker_forNewKF;
  delete coarseInitializer;
  delete pixelSelector;
  delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib,
    int originalW, int originalH)
{

}

void FullSystem::setGammaFunction(float* BInv)
{
  if (BInv==0) return;

  // copy BInv.
  memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


  // invert.
  for (int i=1;i<255;i++)
  {
    // find val, such that Binv[val] = i.
    // I dont care about speed for this, so do it the stupid way.

    for (int s=1;s<255;s++)
    {
      if (BInv[s] <= i && BInv[s+1] >= i)
      {
        Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
        break;
      }
    }
  }

  Hcalib.B[0] = 0;
  Hcalib.B[255] = 255;
}

void FullSystem::printResult(std::string file)
{
  boost::unique_lock<boost::mutex> lock(trackMutex);
  boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

  std::ofstream myfile;
  myfile.open (file.c_str());
  myfile << std::setprecision(15);

  for (FrameShell* s : allFrameHistory)
  {
    if (!s->poseValid) continue;

    if (setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

    myfile << s->timestamp <<
      " " << s->camToWorld.translation().transpose()<<
      " " << s->camToWorld.so3().unit_quaternion().x()<<
      " " << s->camToWorld.so3().unit_quaternion().y()<<
      " " << s->camToWorld.so3().unit_quaternion().z()<<
      " " << s->camToWorld.so3().unit_quaternion().w() << "\n";
  }
  myfile.close();
}


// this function is responsible for generating a list initialize poses
// it will then attempt to track with each of these initialize poses
// it will do this using CoarseTracker::trackNewestCoarse
// this function will keep track of the best tracking results
// perhaps even exiting early if a sufficient track was found
Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
{
  // check that we have frames to track against
  assert(allFrameHistory.size() > 0);

  // notify listeners of system state
  for (IOWrap::Output3DWrapper* ow : outputWrapper)
      ow->pushLiveFrame(fh);

  // get reference frame
  FrameHessian* lastF = coarseTracker->lastRef;

  // initialize brightness
  AffLight aff_last_2_l = AffLight(0,0);

  // create list of initial poses
  // NOTE: this list initially will have size == 0
  std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;

  // check if we have processed only one frame before
  // NOTE: the second frame will be the current
  // - it has already been pushed onto the list at this point
  if (allFrameHistory.size() == 2)
  {
    // NOTE: this does nothing!
    // list is empty, so iterating over list does nothing
    for (unsigned int i=0;i<lastF_2_fh_tries.size();i++)
    {
      lastF_2_fh_tries.push_back(SE3());
    }
  }
  // we have processed 2+ frames
  else
  {
    // NOTE: this function doesn't get called with less than 2 frames
    // so if we've reached this point, we can assume there are 3+ frames

    // get previous frame
    // NOTE: the last frame in allFrameHistory is the current frame
    FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];

    // get frame before previous frame
    FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];

    // allocate k-1 to k-2 transform
    // (assuming we are now processing frame k)
    SE3 slast_2_sprelast;

    // allocate k-1 to reference transform
    SE3 lastF_2_slast;

    {
      // locate relative poses for frames
      // NOTE: don't want optimization window messing with it
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

      // assign k-1 to k-2 transform
      slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;

      // assign k-1 to reference transform
      lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;

      // get brightness transform
      // assume same global transfer as previous frame
      aff_last_2_l = slast->aff_g2l;
    }

    // using a constant mostion model
    // we assume T_k1_k0 is the same as T_k2_k1
    SE3 fh_2_slast = slast_2_sprelast;

    // add different transform initializations

    // add constant-motion
    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);

    // add double-motion
    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);

    // add half-motion
    lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast);

    // add no-motion (from last frame)
    lastF_2_fh_tries.push_back(lastF_2_slast);

    // ad no-motion (from reference keyframe)
    lastF_2_fh_tries.push_back(SE3());

    // just try a TON of different initializations (all rotations). In the end,
    // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
    // also, if tracking fails here we lose, so we really, really want to avoid that.

    // NOTE: this for-loop will only performing one iteration
    // - so in total there will only be 26 attempts
    for (float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
    {
      // using constant motion model, modulate T_k0_ref by small rotations
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
    }

    // check if we're working with valid poses
    // NOTE: could this check have happend first?
    if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
    {
      // remove previously added initialize poses
      lastF_2_fh_tries.clear();

      // just add no motion from last keyframe
      lastF_2_fh_tries.push_back(SE3());
    }

  } // done adding initial poses to list


  // initialize optical flow values
  Vec3 flowVecs = Vec3(100,100,100);

  // initialize T_new_ref
  SE3 lastF_2_fh = SE3();

  // initilize brightness transfer
  AffLight aff_g2l = AffLight(0,0);

  // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
  // I'll keep track of the so-far best achieved residual for each level in achievedRes.
  // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

  // initialize best residual seen with NaN
  Vec5 achievedRes = Vec5::Constant(NAN);

  // initialize flag for if we have ever made a valid track
  bool haveOneGood = false;

  // initialize number of tracking attempts to zero
  int tryIterations=0;

  // attempt tracking will all initialize poses in list
  for (unsigned int i=0;i<lastF_2_fh_tries.size();i++)
  {
    // initialize global brightness transfer to be same as prevoius frame
    AffLight aff_g2l_this = aff_last_2_l;

    // seed initial pose from current list index
    SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

    // attempt to track with current pose
    // NOTE: we pass in "achievedRes" as an argument here
    // - this allows quick exit if any level does worse
    // - assuming coarser levels will always have lower residuals
    // - as they are dealing with fewer pixels
    // - this must be total error (and not MSE)
    bool trackingIsGood = coarseTracker->trackNewestCoarse(
        fh, lastF_2_fh_this, aff_g2l_this,
        pyrLevelsUsed-1,
        achievedRes);

    // increment tracking attempt count
    tryIterations++;

    // print debug message if first attempt wasn't successful
    if (i != 0)
    {
      printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d "
             "(ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
          i,
          i, pyrLevelsUsed-1,
          aff_g2l_this.a,aff_g2l_this.b,
          achievedRes[0],
          achievedRes[1],
          achievedRes[2],
          achievedRes[3],
          achievedRes[4],
          coarseTracker->lastResiduals[0],
          coarseTracker->lastResiduals[1],
          coarseTracker->lastResiduals[2],
          coarseTracker->lastResiduals[3],
          coarseTracker->lastResiduals[4]);
    }

    // do we have a new winner?
    // check if tracking was successful
    // and we have a valid residual value
    // and the current residual is NOT worse than the previous
    // NOTE: note the "NOT worse" is needed as we might be comparing with NaN
    if (trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0])
        && !(coarseTracker->lastResiduals[0] >= achievedRes[0]))
    {
      // record best optical flow vectors
      flowVecs = coarseTracker->lastFlowIndicators;

      // record best brightness transform
      aff_g2l = aff_g2l_this;

      // record best SE3 transform
      lastF_2_fh = lastF_2_fh_this;

      // flag that we have found something usable
      haveOneGood = true;
    }

    // take over achieved res (always).
    if (haveOneGood)
    {
      // for each of the 5 elements of the tracking residual vector
      for (int i=0;i<5;i++)
      {
        // take over if achievedRes is either bigger or NAN.
        // record all time best residual PER element
        if (!std::isfinite((float)achievedRes[i]) ||
            achievedRes[i] > coarseTracker->lastResiduals[i])
        {
          // assign new best value
          achievedRes[i] = coarseTracker->lastResiduals[i];
        }
      }
    }

    // check if we have a valid track
    // and the current residual is than 1.5x the previous frame RMSE
    if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] *
        setting_reTrackThreshold)
    {
      // we've found a suitable track and can exit early
      break;
    }

  } // end of all tracking attempts

  // check if tracking has failed completely
  if (!haveOneGood)
  {
    printf("BIG ERROR! tracking failed entirely. "
           "Taking predicted pose and hope we may somehow recover.\n");

    // zero out optical flow vectors
    flowVecs = Vec3(0,0,0);

    // assign same global affine brightness as previous pose
    aff_g2l = aff_last_2_l;

    // assign same SE3 global transform as previous pose
    lastF_2_fh = lastF_2_fh_tries[0];
  }

  // record previous tracking residual
  lastCoarseRMSE = achievedRes;

  // assign tracking results to current frame
  // NOTE: no lock required, as fh is not used anywhere yet.
  fh->shell->camToTrackingRef = lastF_2_fh.inverse();
  fh->shell->trackingRef = lastF->shell;
  fh->shell->aff_g2l = aff_g2l;
  fh->shell->camToWorld = fh->shell->trackingRef->camToWorld *
      fh->shell->camToTrackingRef;

  // check if first time tracking
  if (coarseTracker->firstCoarseRMSE < 0)
  {
    // assign achieve residual to first error ever recorded
    coarseTracker->firstCoarseRMSE = achievedRes[0];
  }

  // debug as required
  if (!setting_debugout_runquiet)
  {
    printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n",
        aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);
  }

  // log as required
  if (setting_logStuff)
  {
    (*coarseTrackingLog) << std::setprecision(16)
            << fh->shell->id << " "
            << fh->shell->timestamp << " "
            << fh->ab_exposure << " "
            << fh->shell->camToWorld.log().transpose() << " "
            << aff_g2l.a << " "
            << aff_g2l.b << " "
            << achievedRes[0] << " "
            << tryIterations << "\n";
  }

  // return tracking results
  // - total tracking error
  // - 3x optical flow values
  // ASSUME: flow values are used for keyframe detection
  return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FullSystem::traceNewCoarse(FrameHessian* fh)
{
  // acquire mapping lock (as we will be updating keypoint candidates)
  // - this could conflict with mappingLoop when makeKeyframe is called
  boost::unique_lock<boost::mutex> lock(mapMutex);

  // initialize state
  int trace_total=0;
  int trace_good=0;
  int trace_oob=0;
  int trace_out=0;
  int trace_skip=0;
  int trace_badcondition=0;
  int trace_uninitialized=0;

  // build camera projection matrix
  Mat33f K = Mat33f::Identity();
  K(0,0) = Hcalib.fxl();
  K(1,1) = Hcalib.fyl();
  K(0,2) = Hcalib.cxl();
  K(1,2) = Hcalib.cyl();

  // process all active frames in window
  for (FrameHessian* host : frameHessians)
  {
    // compute reference to new frame transform
    // actually doing: T_c1_w * T_w_c2 => T_c1_c2
    SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;

    // compute projection geometry for epipolar line searches
    Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
    Vec3f Kt = K * hostToNew.translation().cast<float>();

    // get affine brightness transfrom for two frames
    Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure,
        fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

    // for each inactive keypoint
    for (ImmaturePoint* ph : host->immaturePoints)
    {
      // ASSUME: perform epipolar line search to update keypoint
      // TODO: look into this function more
      ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );

      // increment keypoint trace statistics
      if (ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
      if (ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
      if (ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
      if (ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
      if (ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
      if (ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;

      // increase trace count
      trace_total++;
    }
  }
}

void FullSystem::activatePointsMT_Reductor(
    std::vector<PointHessian*>* optimized,
    std::vector<ImmaturePoint*>* toOptimize,
    int min, int max, Vec10* stats, int tid)
{
  ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
  for (int k=min;k<max;k++)
  {
    (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
  }
  delete[] tr;
}



void FullSystem::activatePointsMT()
{

  if (ef->nPoints < setting_desiredPointDensity*0.66)
    currentMinActDist -= 0.8;
  if (ef->nPoints < setting_desiredPointDensity*0.8)
    currentMinActDist -= 0.5;
  else if (ef->nPoints < setting_desiredPointDensity*0.9)
    currentMinActDist -= 0.2;
  else if (ef->nPoints < setting_desiredPointDensity)
    currentMinActDist -= 0.1;

  if (ef->nPoints > setting_desiredPointDensity*1.5)
    currentMinActDist += 0.8;
  if (ef->nPoints > setting_desiredPointDensity*1.3)
    currentMinActDist += 0.5;
  if (ef->nPoints > setting_desiredPointDensity*1.15)
    currentMinActDist += 0.2;
  if (ef->nPoints > setting_desiredPointDensity)
    currentMinActDist += 0.1;

  if (currentMinActDist < 0) currentMinActDist = 0;
  if (currentMinActDist > 4) currentMinActDist = 4;

    if (!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



  FrameHessian* newestHs = frameHessians.back();

  // make dist map.
  coarseDistanceMap->makeK(&Hcalib);
  coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

  //coarseTracker->debugPlotDistMap("distMap");

  std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000);


  for (FrameHessian* host : frameHessians)		// go through all active frames
  {
    if (host == newestHs) continue;

    SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
    Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


    for (unsigned int i=0;i<host->immaturePoints.size();i+=1)
    {
      ImmaturePoint* ph = host->immaturePoints[i];
      ph->idxInImmaturePoints = i;

      // delete points that have never been traced successfully, or that are outlier on the last trace.
      if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
      {
//				immature_invalid_deleted++;
        // remove point.
        delete ph;
        host->immaturePoints[i]=0;
        continue;
      }

      // can activate only if this is true.
      bool canActivate = (ph->lastTraceStatus == IPS_GOOD
          || ph->lastTraceStatus == IPS_SKIPPED
          || ph->lastTraceStatus == IPS_BADCONDITION
          || ph->lastTraceStatus == IPS_OOB )
              && ph->lastTracePixelInterval < 8
              && ph->quality > setting_minTraceQuality
              && (ph->idepth_max+ph->idepth_min) > 0;


      // if I cannot activate the point, skip it. Maybe also delete it.
      if (!canActivate)
      {
        // if point will be out afterwards, delete it instead.
        if (ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
        {
//					immature_notReady_deleted++;
          delete ph;
          host->immaturePoints[i]=0;
        }
//				immature_notReady_skipped++;
        continue;
      }


      // see if we need to activate point due to distance map.
      Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
      int u = ptp[0] / ptp[2] + 0.5f;
      int v = ptp[1] / ptp[2] + 0.5f;

      if ((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
      {

        float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

        if (dist>=currentMinActDist* ph->my_type)
        {
          coarseDistanceMap->addIntoDistFinal(u,v);
          toOptimize.push_back(ph);
        }
      }
      else
      {
        delete ph;
        host->immaturePoints[i]=0;
      }
    }
  }


//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

  std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

  if (multiThreading)
    treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

  else
    activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);


  for (unsigned k=0;k<toOptimize.size();k++)
  {
    PointHessian* newpoint = optimized[k];
    ImmaturePoint* ph = toOptimize[k];

    if (newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
    {
      newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
      newpoint->host->pointHessians.push_back(newpoint);
      ef->insertPoint(newpoint);
      for (PointFrameResidual* r : newpoint->residuals)
        ef->insertResidual(r);
      assert(newpoint->efPoint != 0);
      delete ph;
    }
    else if (newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
    {
      delete ph;
      ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
    }
    else
    {
      assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
    }
  }


  for (FrameHessian* host : frameHessians)
  {
    for (int i=0;i<(int)host->immaturePoints.size();i++)
    {
      if (host->immaturePoints[i]==0)
      {
        host->immaturePoints[i] = host->immaturePoints.back();
        host->immaturePoints.pop_back();
        i--;
      }
    }
  }


}






void FullSystem::activatePointsOldFirst()
{
  assert(false);
}

void FullSystem::flagPointsForRemoval()
{
  assert(EFIndicesValid);

  std::vector<FrameHessian*> fhsToKeepPoints;
  std::vector<FrameHessian*> fhsToMargPoints;

  {
    // I suppose this was intented to populate list of keyframes to keep
    // NOTE: however, the conditional expression in the for-loop is never true
    // - init i = size - 1, cond: i >= size, i only gets smaller
    // - not sure why they felt desending order was needed for this operation
    for (int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
      if (!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

    // add each keyframe flag for marginalization to list
    for (int i=0; i< (int)frameHessians.size();i++)
      if (frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
  }

  // initialize removal stats
  int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

  // process each active keyframe in window
  for (FrameHessian* host : frameHessians)
  {
    // process each keypoint in current keyframe
    for (unsigned int i=0;i<host->pointHessians.size();i++)
    {
      // get pointer to current keypoint
      PointHessian* ph = host->pointHessians[i];

      // skip null keypoint pointers
      if (ph==0) continue;

      // keypoint has invalid depth or zero residuals
      // this keypoint has no active effect on the total optimization
      // - essentially can this be dropped without marginalization
      if (ph->idepth_scaled < 0 || ph->residuals.size()==0)
      {
        // put on list of keypoints to remove
        host->pointHessiansOut.push_back(ph);

        // mark keypoint for straight up removal (not marginalization)
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;

        // set host pointer to null for this keypoint
        // NOTE: this appears to happen anyway outside this if-statement
        // - probably could just be removed
        host->pointHessians[i]=0;

        // increment dropped keypoint count
        flag_nores++;
      }
      // check if point can be marginalized or the whole keyframe should be
      // TODO: what does it mean if a keypoint should be but not the keyframe?
      // NOTE: the order of this boolean expressiions should be switched
      else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) ||
          host->flaggedForMarginalization)
      {
        // increment marginalized keypoint count
        flag_oob++;

        // check if keypoint is an inlier
        // TODO: how is this determined
        if (ph->isInlierNew())
        {
          // increment inlier keypoint count
          flag_in++;

          // initialize count of good residuals
          int ngoodRes=0;

          // process each residual with keypoint
          for (PointFrameResidual* r : ph->residuals)
          {
            // reset all residuals
            r->resetOOB();
            r->linearize(&Hcalib);
            r->efResidual->isLinearized = false;
            r->applyRes(true);

            // if residual is still active
            if (r->efResidual->isActive())
            {
              // fix linearization point
              // TODO: what does this do exactly?
              r->efResidual->fixLinearizationF(ef);
              ngoodRes++;
            }
          }

          // check if depth hessian is valid for marginalization
          // TODO: what does this mean in theory?
          if (ph->idepth_hessian > setting_minIdepthH_marg)
          {
            // increment marginalized inliear count
            flag_inin++;

            // flag for marginalization
            ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;

            // push keypoint onto list of keypoints to marginalize
            host->pointHessiansMarginalized.push_back(ph);
          }
          // keypoint depth hessian is not valid for marginalization
          else
          {
            // mark keypoint for straight-up removal
            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
            host->pointHessiansOut.push_back(ph);
          }
        }
        // keypoint is an outlier
        // NOTE: can just remove as we don't want it in our optimization
        else
        {
          // add keypoint to be dropped
          host->pointHessiansOut.push_back(ph);

          // mark keypoint for straight-up removal
          ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        }

        // set host pointer to null for this keypoint
        host->pointHessians[i]=0;
      }
    }

    // process each keypoint in current keyframe (again)
    for (int i=0;i<(int)host->pointHessians.size();i++)
    {
      // check if keypoint pointer is null
      if (host->pointHessians[i]==0)
      {
        // compact keypoint list to be sequential list of non-null keypoints
        host->pointHessians[i] = host->pointHessians.back();
        host->pointHessians.pop_back();
        i--;
      }
    }
  }
}

// This function takes raw image data and creates a frame
// This new frame includes other image data (pyramid, gradient, corrections)
// it then initializes and trackers the current frame
// it then determines if the frame should be a keyframe
// finally it sends to new frame one for futher processing
// it is also where the Keyframe decision is finally made
// it will pass its results on to FullSystem::deliverTrackedFrame
void FullSystem::addActiveFrame( ImageAndExposure* image, int id )
{
  // do nothing if lost
  // TODO: how do I relocalize if I don't process anything?
  if (isLost) return;

  // lock tracking mutex
  // TODO: what data (and what threads) are we protecting (against)?
  // NOTE: it would seem that a scoped lock would be better
  // - as they are explicitly unlocking before leaving the function
  boost::unique_lock<boost::mutex> lock(trackMutex);

  // ===========================================================================
  // add into allFrameHistory
  // ===========================================================================

  // create new Hessian for frame
  // holds optimization details (ex: points, marginalization, state, null-space)
  // NOTE: this is a pretty large structure
  // - I think this should be created once and retrieved from a pool
  // - frame shells are added to history, but not this
  // - this struct may be deleted before this function ends
  FrameHessian* fh = new FrameHessian();

  // create new shell for frame
  // holds id, timestamps, pose, status
  // NOTE: this is also a fairly large structure
  // - it too could benefit from a struct pool
  // - however, this gets added to the "allFrameHistory" list
  // - so perhaps for logging purposes, all data must be retained
  FrameShell* shell = new FrameShell();

  // initialize pose as identity
  shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.

  // initialize its affine brightness transfer parameters
  shell->aff_g2l = AffLight(0,0);

  // TODO: why marginalized at frame count index?
  // - not sure what that member variable is used for
  shell->marginalizedAt = shell->id = allFrameHistory.size();

  // assign frame timestamp
  shell->timestamp = image->timestamp;

  // assign frame id
  shell->incoming_id = id;

  // assign frame-shell to frame-hessian
  fh->shell = shell;

  // add new frame SHELL to history
  // NOTE: not the hessian
  allFrameHistory.push_back(shell);

  // ===========================================================================
  // make image pyramids, derivatives, etc.
  // ===========================================================================

  // assign frame exposure time
  fh->ab_exposure = image->exposure_time;

  // make image pyramids, corrected images, image gradients from source image
  // NOTE: are image gradients needed if not used as a KF?
  // - we could save some potential work / memory there
  // - if not, are they using the IC form of the LK?
  fh->makeImages(image->image, &Hcalib);

  // check if SLAM system not initialized
  if (!initialized)
  {
    // this is the first frame ever
    if (coarseInitializer->frameID < 0)
    {
      // assignn current frame to the initializer
      // TODO: what does this do? read further
      coarseInitializer->setFirst(&Hcalib, fh);
    }
    // attemp to track current frame
    // returns true if "snapped"
    // TODO: how is tracking performed?
    // TODO: what does "snapped" mean
    else if (coarseInitializer->trackFrame(fh, outputWrapper))
    {
      // TODO: what what does this do?
      // TODO: what is this done here?
      initializeFromInitializer(fh);

      // unlock current tracking thread
      lock.unlock();

      // send current frame to be processed
      // send frame allong with whether is should become a keyframe
      // TODO: follow this function further
      // what does "snapped" have to do with keyframe creation?
      deliverTrackedFrame(fh, true);
    }
    // current frame didn't "snap"
    // TODO: what does that mean?
    else
    {
      // mark the hessian
      fh->shell->poseValid = false;

      // we do delete the hessian
      delete fh;
    }

    // TODO: is lock released when scope dies?
    // - if not, we can exist without releasing it
    return;
  }
  // process current frame with initialized SLAM system
  else
  {
    // check if tracker reference should be swapped
    // GUESS: tracker must be built from updated window's lastest KF
    // - but this process may be slow
    // - so we manage two trackers, one that can be updated
    // - while the other is actively used
    // - we check the frame ID to determine if the "other" is more recent
    if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
    {
      // create lock for trackers
      // TODO: who else is using trackers
      boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);

      // swap tracker references
      // NOTE: could just use std::swap
      CoarseTracker* tmp = coarseTracker;
      coarseTracker=coarseTracker_forNewKF;
      coarseTracker_forNewKF=tmp;
    }

    // track frame with current tracker
    // TODO: what else does this function do, I need to follow
    // TODO: how does this different than coarse initializer?
    // TODO: what does the tres 4 values represent?
    Vec4 tres = trackNewCoarse(fh);

    // check if tracking results are bogus
    if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) ||
        !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
    {
      // change SLAM state to lost
      printf("Initial Tracking failed: LOST!\n");
      isLost=true;
      return;
    }

    // default to not making a keyframe
    bool needToMakeKF = false;

    // check if limit on keyframes per second
    if (setting_keyframesPerSecond > 0)
    {
      // new keyframe if first frame
      // or elapsed time for new keyframe exceeded
      needToMakeKF = allFrameHistory.size() == 1 ||
          (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) >
          0.95f / setting_keyframesPerSecond;
    }
    // no limit on keyframes per second
    // perhaps meaning elapsed time does not affect keyframe creation
    else
    {
      // compute relative brightness parameters from two frames
      Vec2 refToFh=AffLight::fromToVecExposure(
          coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
          coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

      // new keyframe if first frame
      // or max optical flow really high???
      // - TODO: read where this was discussed in their paper
      // or current error is really high???
      needToMakeKF = allFrameHistory.size()== 1 ||
          setting_kfGlobalWeight*setting_maxShiftWeightT *
              sqrtf((double)tres[1]) / (wG[0]+hG[0]) +
          setting_kfGlobalWeight*setting_maxShiftWeightR *
              sqrtf((double)tres[2]) / (wG[0]+hG[0]) +
          setting_kfGlobalWeight*setting_maxShiftWeightRT *
              sqrtf((double)tres[3]) / (wG[0]+hG[0]) +
          setting_kfGlobalWeight*setting_maxAffineWeight *
              fabs(logf((float)refToFh[0])) > 1 ||
          2 * coarseTracker->firstCoarseRMSE < tres[0];
    }

    // report camera pose to all output handlers
    for (IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->publishCamPose(fh->shell, &Hcalib);

    // unlock current tracking thread
    lock.unlock();

    // send current frame to be processed
    // send frame allong with whether is should become a keyframe
    // TODO: follow this function further
    deliverTrackedFrame(fh, needToMakeKF);
  }
}

// this function gets called by FullSystem::addActiveFrame
// it is mainly responsible for add frames to the unmappedTrackedFrames list
// this list will eventually be processed by the FullSystem::mappingLoop
// it will also update needNewKFAfter if a keyframe was desired
void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{
  // if using timestamps and speed up
  if (linearizeOperation)
  {
    // check if walk through debugging
    if (goStepByStep && lastRefStopID != coarseTracker->refFrameID)
    {
      MinimalImageF3 img(wG[0], hG[0], fh->dI);
      IOWrap::displayImage("frameToTrack", &img);
      while (true)
      {
        char k=IOWrap::waitKey(0);
        if (k==' ') break;
        handleKey( k );
      }
      lastRefStopID = coarseTracker->refFrameID;
    }
    else handleKey( IOWrap::waitKey(1) );

    if (needKF) makeKeyFrame(fh);
    else makeNonKeyFrame(fh);
  }
  // dataset has not timestamps
  else
  {
    // acquire tracking-mapping lock
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

    // add frame to queue
    // TODO: is this for updating keypoints?
    // - at this stage the frame is already tracked
    // - and needKF could be false
    // - so this won't be added to the window
    // - but it could be used for updating points
    // - I need to look at how unmappedTrackedFrames is used
    unmappedTrackedFrames.push_back(fh);

    // TODO: where else is needNewKFAfter used?
    // - does this ever overwrite a frame queued but not processed?
    // - it would appear that is sets not this frame's id, but it's ref frame's
    if (needKF) needNewKFAfter = fh->shell->trackingRef->id;

    // notify mapping system that new frame is availble
    // - mappingLoop() is waiting for this
    trackedFrameSignal.notify_all();

    // simply block until a tracker has a reference frame
    while (coarseTracker_forNewKF->refFrameID == -1 &&
        coarseTracker->refFrameID == -1 )
    {
      // mappingLoop() will notify us when this is ready
      // we have nothing to use for tracking until this is done
      // - release lock while waiting
      // TODO: does this notification mean trackers are ready?
      // - or that there will be soon? If the latter, then have two wait for two
      mappedFrameSignal.wait(lock);
    }

    // release lock
    lock.unlock();
  }
}

// this function starts an infinite loop that runs in a separate thread
// it is responsible for processing all frames in unmappedTrackedFrames
// it will either call makeKeyframe or makeNonKeyframe with them
// if making a non-keyframe then the "immature" keypoints will be updated
// if making a keyframe it will be added to the window and optimized
void FullSystem::mappingLoop()
{
  // acquire tracking-mapping lock
  // this is released and re-acquired throughout the loop
  // so that the deliverTrackedFrame can eventually acquire it
  boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

  // run until system shutown
  while (runMapping)
  {
    // wait for a new tracked frame available for mapping
    while (unmappedTrackedFrames.size()==0)
    {
      // wait for tracked frame notification
      // - release lock while waiting
      trackedFrameSignal.wait(lock);

      // or stop waiting if shutting down
      if (!runMapping) return;
    }

    // get reference to next frame to be mapped
    FrameHessian* fh = unmappedTrackedFrames.front();

    // remove frame from queue
    unmappedTrackedFrames.pop_front();

    // guaranteed to make a KF for the very first two tracked frames.
    if (allKeyFramesHistory.size() <= 2)
    {
      // release lock
      lock.unlock();

      // make keyframe from current frame
      // TODO: follow this function
      makeKeyFrame(fh);

      // acquire lock again
      lock.lock();

      // notify other threads of new keyframe
      // - deliverTrackedFrame is waiting for this
      // - as it relates to potentially new tracking states
      mappedFrameSignal.notify_all();

      // jump to next loop iteration
      // - no other processing to be done
      // does the second frame not help the first one initialize?
      continue;
    }

    // check if we have a lot of frames still left to be mapped
    if (unmappedTrackedFrames.size() > 3)
    {
      // once this is set to true, it can only be unset when keyframe is added
      // even if in subsequent iterations this tracked frames list is empty
      needToKetchupMapping=true;
    }

    // if there are other frames left to track, do that first.
    if (unmappedTrackedFrames.size() > 0)
    {
      // release lock
      lock.unlock();

      // process frame as non-keyframe
      // TODO: what is this doing, need to follow
      makeNonKeyFrame(fh);

      // acquire lock
      lock.lock();

      // check if we're falling behind on processing frames
      if (needToKetchupMapping && unmappedTrackedFrames.size() > 0)
      {
        // get the next frame from the queue
        FrameHessian* fh = unmappedTrackedFrames.front();

        // remove new frame from the queue
        unmappedTrackedFrames.pop_front();

        {
          // acquire frame pose lock
          // TODO: how else is acquiring this lock?
          boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

          // check that frame has reference frame used for tracking
          assert(fh->shell->trackingRef != 0);

          // compute this frames pose in world-frame
          // TODO: why is this updated if it will soon be deleted?
          fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

          // TODO: why is this updated if it will soon be deleted?
          fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);

          // this block seems similar to that of makeNonKeyframe
          // except it doesn't make the traceNewCoarse call
          // which would explain why these updates would be needed
          // but without it, I think this is wasted computation
        }

        // it would seem we just delete frames when we're following behind
        delete fh;
      }
    }
    // unmapped frames list is empty
    else
    {
      // check if frame should be made into new keyframe
      // - a frame is tracked w.r.t. a reference frame
      // - when that frame is deemed to be a candidate keyframe
      // - it's reference frame's id is assigned to needNewKFAfter
      // - so when the next frame reaches this point
      // - that arrived after the reference frame, it will be the next keyframe
      // - this seems wrong, if not done in lock-stop
      // - a previous frame not deemed to be a keyframe could also pass this
      if (setting_realTimeMaxKF ||
          needNewKFAfter >= frameHessians.back()->shell->id)
      {
        // release lock
        lock.unlock();

        // process frame as keyframe
        // TODO: what is this doing, need to follow
        makeKeyFrame(fh);

        // we have now caught up
        needToKetchupMapping=false;

        // acquire lock again
        lock.lock();
      }
      else
      {
        // release lock
        lock.unlock();

        // process frame as non-keyframe
        // TODO: what is this doing, need to follow
        // - why doesn't this need the lock?
        makeNonKeyFrame(fh);

        // acquire lock again
        lock.lock();
      }
    }

    // notify other threads that frame has been mapped
    mappedFrameSignal.notify_all();
  }

  printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
  boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
  runMapping = false;
  trackedFrameSignal.notify_all();
  lock.unlock();

  mappingThread.join();

}

// this function is mainly responsible for calling FullSystem::trackNewCoarse
// which updates current immature keypoints by performing epipolar search
void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
  // needs to be set by mapping thread.
  // no lock required since we are in mapping thread.
  {
    // acquire pose lock
    // ASSUME: this simply reports the current pose of the camera
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

    // assert this frame has a valid reference frame
    assert(fh->shell->trackingRef != 0);

    // compute frames pose in world-frame
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld *
        fh->shell->camToTrackingRef;

    // assign evaluation point (pose and affine brightness)
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
  }

  // trace new frame
  // ASSUME: this is updating inactive keypoint candidates
  // TODO: follow this function to find out more
  traceNewCoarse(fh);

  delete fh;
}

// like makeNonKeyframe this function is calls FullSystem::trackNewCoarse
// which updates current immature keypoints by performing epipolar search
// however, it also updates the keyframe window:
// - add / removes keypoint residuals
// - optimizes the new window
// - performs marginalization
// it also updates the CoarseTracker depth map for tracking frames
void FullSystem::makeKeyFrame( FrameHessian* fh)
{
  // needs to be set by mapping thread.
  // no lock required since we are in mapping thread.
  {
    // same code as makeNonKeyframe
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    assert(fh->shell->trackingRef != 0);

    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld *
        fh->shell->camToTrackingRef;

    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
  }

  // still update keypoints as with non-keyframes
  traceNewCoarse(fh);

  // acquire mapping lock
  // - prevent keypoint updates from changing during activation process
  boost::unique_lock<boost::mutex> lock(mapMutex);

  // =========================== Flag Frames to be Marginalized. ===============

  // simply flags keyframes for marginalization
  // no other work is done
  // TODO: when are they actually removed?
  // - are they optimized first before removal?
  flagFramesForMarginalization(fh);

  // =========================== add New Frame to Hessian Struct. ==============

  // assign new id to hessian
  fh->idx = frameHessians.size();

  // add hessian to active hessian list
  frameHessians.push_back(fh);

  // assign new id to keyframe
  fh->frameID = allKeyFramesHistory.size();

  // add keyframe to keyframe history list
  allKeyFramesHistory.push_back(fh->shell);

  // add keyframe to objective function
  // initializes new keyframe objective function structure
  // updates connectivity map
  ef->insertFrame(fh, &Hcalib);

  // perform object function frame-frame  precalcutions
  // TODO: understand this better
  setPrecalcValues();

  // =========================== add new residuals for old points ==============

  // NOTE: this counter doesn't seem to be used
  int numFwdResAdde=0;

  // process all active keyframes
  for (FrameHessian* fh1 : frameHessians)
  {
    // skip newest keyframe
    // (has no active keypoints)
    if (fh1 == fh) continue;

    // for each keypoint in keyframe
    for (PointHessian* ph : fh1->pointHessians)
    {
      // create residual for keypoint in latest keyframe
      // NOTE: what if point doesn't project into keyframe tracked pose?
      // - could we save computation by just not adding it?
      PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);

      // initialize keypoint to be inlier
      // TODO: when does this change and how?
      r->setState(ResState::IN);

      // record residual to keypoint
      ph->residuals.push_back(r);

      // add residual to objective function
      ef->insertResidual(r);

      // shift back previous "last residual" for this keypoint
      ph->lastResiduals[1] = ph->lastResiduals[0];

      // create new "last residual" for this keypoint
      // TODO: how is this used?
      // - couldn't ph->residuals.back() accomplish the same thing?
      ph->lastResiduals[0] =
          std::pair<PointFrameResidual*, ResState>(r, ResState::IN);

      // increment residual count
      // NOTE: this seems pointless
      numFwdResAdde+=1;
    }
  }

  // ================= Activate Points (& flag for marginalization). ===========

  // activate new keypoints for candidate pool
  // TODO: investigate this function further
  activatePointsMT();

  // recreate frame list and assign updated ids
  // recreate point list and assign updated ids
  // recreate residual with new host and target ids
  ef->makeIDX();

  // =========================== OPTIMIZE ALL =========================

  // NOTE: this seems redundant
  // - frameHessians.back() should be the same as fh
  fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;

  // perform keyframe window optimization
  // TODO: look into this function more
  // NOTE: it would seem keyframes flagged for marg are still in optimization
  float rmse = optimize(setting_maxOptIterations);

  // =========================== Figure Out if INITIALIZATION FAILED ===========

  // check if we've only had 4 or fewer keyframes
  if (allKeyFramesHistory.size() <= 4)
  {
    // check if initialization failed
    // - threshold changes as per keyframe count
    // NOTE: I think "else if" should be used here

    if (allKeyFramesHistory.size()==2 &&
        rmse > 20*benchmark_initializerSlackFactor)
    {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed=true;
    }
    if (allKeyFramesHistory.size()==3 &&
        rmse > 13*benchmark_initializerSlackFactor)
    {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed=true;
    }
    if (allKeyFramesHistory.size()==4 &&
        rmse > 9*benchmark_initializerSlackFactor)
    {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed=true;
    }
  }

  // exit early if we're lost
  if (isLost) return;

  // =========================== REMOVE OUTLIER =========================

  // drops keypoints from window if residual count is empty
  // TODO: how does their residual count change?
  // - could this happen during optimization?
  // - could this be determined in advance?
  removeOutliers();

  // =========================== UPDATE TRACKER DEPTHMAP =======================

  {
    // acquire tracker swap lock, so cannot be swapped while updating
    boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);

    // assign new camera projection as per optimization
    coarseTracker_forNewKF->makeK(&Hcalib);

    // assign reference frame and create new tracking depth map
    coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);

    // plot new depth map for debugging
    coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker,
        &maxIdJetVisTracker, outputWrapper);

    // plot new depth map for debugging
    coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
  }

  // plot optimization results
  debugPlot("post Optimize");

  // =========================== (Activate-)Marginalize Points =================

  // determine which points should be removed and which marginalized
  // flag them as such and add them to the appropriate lists
  flagPointsForRemoval();

  // drop all points that will not be marginalized
  ef->dropPointsF();

  //  get null-spaces of entire system
  // TODO: correct use of the term "null-space"?
  // TODO: what is this doing exactly and why?
  // ASSUME: determine which terms don't need marginalization update
  // - as they were not affected by the parameter being marginalized out
  getNullspaces(
      ef->lastNullspaces_pose,
      ef->lastNullspaces_scale,
      ef->lastNullspaces_affA,
      ef->lastNullspaces_affB);

  // perform keypoint marginalization
  // TODO: see how this is performed
  ef->marginalizePointsF();

  // =========================== add new Immature points & new residuals =======

  // create new keypoint candidates
  // no residuals yet, they still need to be initialized
  makeNewTraces(fh, 0);

  // for each output handler
  for (IOWrap::Output3DWrapper* ow : outputWrapper)
  {
    // notify of new pose graph and keyframes
    ow->publishGraph(ef->connectivityMap);
    ow->publishKeyframes(frameHessians, false, &Hcalib);
  }

  // =========================== Marginalize Frames =========================

  // check each keyframe in window
  for (unsigned int i=0;i<frameHessians.size();i++)
  {
    // check if flagged for marginalization
    if (frameHessians[i]->flaggedForMarginalization)
    {
      // marginalize keyframe
      // TODO: look into this function
      marginalizeFrame(frameHessians[i]);

      // resize for-loop index
      // TODO: what's going on in marginalize that we have too redo everything
      i=0;
    }
  }

  // =========================== Write log output =========================

  // write to logs as needed
  printLogLine();
}


void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
  boost::unique_lock<boost::mutex> lock(mapMutex);

  // add firstframe.
  FrameHessian* firstFrame = coarseInitializer->firstFrame;
  firstFrame->idx = frameHessians.size();
  frameHessians.push_back(firstFrame);
  firstFrame->frameID = allKeyFramesHistory.size();
  allKeyFramesHistory.push_back(firstFrame->shell);
  ef->insertFrame(firstFrame, &Hcalib);
  setPrecalcValues();

  //int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
  //int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

  firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
  firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
  firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);


  float sumID=1e-5, numID=1e-5;
  for (int i=0;i<coarseInitializer->numPoints[0];i++)
  {
    sumID += coarseInitializer->points[0][i].iR;
    numID++;
  }
  float rescaleFactor = 1 / (sumID / numID);

  // randomly sub-select the points I need.
  float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if (!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
                (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );

  for (int i=0;i<coarseInitializer->numPoints[0];i++)
  {
    if (rand()/(float)RAND_MAX > keepPercentage) continue;

    Pnt* point = coarseInitializer->points[0]+i;
    ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);

    if (!std::isfinite(pt->energyTH)) { delete pt; continue; }


    pt->idepth_max=pt->idepth_min=1;
    PointHessian* ph = new PointHessian(pt, &Hcalib);
    delete pt;
    if (!std::isfinite(ph->energyTH)) {delete ph; continue;}

    ph->setIdepthScaled(point->iR*rescaleFactor);
    ph->setIdepthZero(ph->idepth);
    ph->hasDepthPrior=true;
    ph->setPointStatus(PointHessian::ACTIVE);

    firstFrame->pointHessians.push_back(ph);
    ef->insertPoint(ph);
  }



  SE3 firstToNew = coarseInitializer->thisToNext;
  firstToNew.translation() /= rescaleFactor;


  // really no lock required, as we are initializing.
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    firstFrame->shell->camToWorld = SE3();
    firstFrame->shell->aff_g2l = AffLight(0,0);
    firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
    firstFrame->shell->trackingRef=0;
    firstFrame->shell->camToTrackingRef = SE3();

    newFrame->shell->camToWorld = firstToNew.inverse();
    newFrame->shell->aff_g2l = AffLight(0,0);
    newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
    newFrame->shell->trackingRef = firstFrame->shell;
    newFrame->shell->camToTrackingRef = firstToNew.inverse();

  }

  initialized=true;
  printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
{
  // TODO: what does this setting do?
  // - why is it always set at this stage?
  pixelSelector->allowFast = true;

  // create pixel selection map from new keyframe
  // ASSUME: results in gradient median for threshold computation
  int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,
        setting_desiredImmatureDensity);

  // reserve memory for all selected keypoints
  newFrame->pointHessians.reserve(numPointsTotal*1.2f);
  newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
  newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);

  // for each row in image (except for border)
  for (int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
  {
    // for each column in image (except for border)
    for (int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
    {
      // get index of pixel
      int i = x+y*wG[0];

      // skip points that were not selected
      // NOTE: couldn't this be created during the assingment stage?
      // - seems like were touching all the pixels again for now reason
      if (selectionMap[i]==0) continue;

      // NOTE: does this have to be a pointer?
      // - could we at least check "energyTH" first?
      // - seems like were allocated dynamic memory that could be deleted
      ImmaturePoint* impt =
          new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);

      // check if valid point, delete if not
      if (!std::isfinite(impt->energyTH)) delete impt;
      // if valid add to this frames list of keypoints
      else newFrame->immaturePoints.push_back(impt);

    }
  }
}

void FullSystem::setPrecalcValues()
{
  for (FrameHessian* fh : frameHessians)
  {
    fh->targetPrecalc.resize(frameHessians.size());
    for (unsigned int i=0;i<frameHessians.size();i++)
      fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
  }

  ef->setDeltaF(&Hcalib);
}


void FullSystem::printLogLine()
{
  if (frameHessians.size()==0) return;

    if (!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                ef->resInA,
                ef->resInL,
                ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                allKeyFramesHistory.back()->aff_g2l.a,
                allKeyFramesHistory.back()->aff_g2l.b,
                frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                (int)frameHessians.size());


  if (!setting_logStuff) return;

  if (numsLog != 0)
  {
    (*numsLog) << allKeyFramesHistory.back()->id << " "  <<
        statistics_lastFineTrackRMSE << " "  <<
        (int)statistics_numCreatedPoints << " "  <<
        (int)statistics_numActivatedPoints << " "  <<
        (int)statistics_numDroppedPoints << " "  <<
        (int)statistics_lastNumOptIts << " "  <<
        ef->resInA << " "  <<
        ef->resInL << " "  <<
        ef->resInM << " "  <<
        statistics_numMargResFwd << " "  <<
        statistics_numMargResBwd << " "  <<
        statistics_numForceDroppedResFwd << " "  <<
        statistics_numForceDroppedResBwd << " "  <<
        frameHessians.back()->aff_g2l().a << " "  <<
        frameHessians.back()->aff_g2l().b << " "  <<
        frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
        (int)frameHessians.size() << " "  << "\n";
    numsLog->flush();
  }


}



void FullSystem::printEigenValLine()
{
  if (!setting_logStuff) return;
  if (ef->lastHS.rows() < 12) return;


  MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
  MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
  int n = Hp.cols()/8;
  assert(Hp.cols()%8==0);

  // sub-select
  for (int i=0;i<n;i++)
  {
    MatXX tmp6 = Hp.block(i*8,0,6,n*8);
    Hp.block(i*6,0,6,n*8) = tmp6;

    MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
    Ha.block(i*2,0,2,n*8) = tmp2;
  }
  for (int i=0;i<n;i++)
  {
    MatXX tmp6 = Hp.block(0,i*8,n*8,6);
    Hp.block(0,i*6,n*8,6) = tmp6;

    MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
    Ha.block(0,i*2,n*8,2) = tmp2;
  }

  VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
  VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
  VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
  VecX diagonal = ef->lastHS.diagonal();

  std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
  std::sort(eigenP.data(), eigenP.data()+eigenP.size());
  std::sort(eigenA.data(), eigenA.data()+eigenA.size());

  int nz = std::max(100,setting_maxFrames*10);

  if (eigenAllLog != 0)
  {
    VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
    (*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
    eigenAllLog->flush();
  }
  if (eigenALog != 0)
  {
    VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
    (*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
    eigenALog->flush();
  }
  if (eigenPLog != 0)
  {
    VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
    (*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
    eigenPLog->flush();
  }

  if (DiagonalLog != 0)
  {
    VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
    (*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
    DiagonalLog->flush();
  }

  if (variancesLog != 0)
  {
    VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
    (*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
    variancesLog->flush();
  }

  std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
  (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
  for (unsigned int i=0;i<nsp.size();i++)
    (*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
  (*nullspacesLog) << "\n";
  nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
  if (!setting_logStuff) return;


  boost::unique_lock<boost::mutex> lock(trackMutex);

  std::ofstream* lg = new std::ofstream();
  lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
  lg->precision(15);

  for (FrameShell* s : allFrameHistory)
  {
    (*lg) << s->id
      << " " << s->marginalizedAt
      << " " << s->statistics_goodResOnThis
      << " " << s->statistics_outlierResOnThis
      << " " << s->movedByOpt;



    (*lg) << "\n";
  }





  lg->close();
  delete lg;

}


void FullSystem::printEvalLine()
{
  return;
}





}

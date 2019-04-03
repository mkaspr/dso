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

#include <cmath>

#include <algorithm>

namespace dso
{





void FullSystem::linearizeAll_Reductor(bool fixLinearization,
    std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats,
    int tid)
{
  for(int k=min;k<max;k++)
  {
    PointFrameResidual* r = activeResiduals[k];
    (*stats)[0] += r->linearize(&Hcalib);

    if(fixLinearization)
    {
      r->applyRes(true);

      if(r->efResidual->isActive())
      {
        if(r->isNew)
        {
          PointHessian* p = r->point;
          Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll * Vec3f(p->u,p->v, 1);	// projected point assuming infinite depth.
          Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll*p->idepth_scaled;	// projected point with real depth.
          float relBS = 0.01*((ptp_inf.head<2>() / ptp_inf[2])-(ptp.head<2>() / ptp[2])).norm();	// 0.01 = one pixel.


          if(relBS > p->maxRelBaseline)
            p->maxRelBaseline = relBS;

          p->numGoodResiduals++;
        }
      }
      else
      {
        toRemove[tid].push_back(activeResiduals[k]);
      }
    }
  }
}


void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max,
    Vec10* stats, int tid)
{
  for(int k=min;k<max;k++)
  {
    activeResiduals[k]->applyRes(true);
  }
}

void FullSystem::setNewFrameEnergyTH()
{

  // collect all residuals and make decision on TH.
  allResVec.clear();
  allResVec.reserve(activeResiduals.size()*2);
  FrameHessian* newFrame = frameHessians.back();

  for(PointFrameResidual* r : activeResiduals)
    if(r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame)
    {
      allResVec.push_back(r->state_NewEnergyWithOutlier);

    }

  if(allResVec.size()==0)
  {
    newFrame->frameEnergyTH = 12*12*patternNum;
    return;		// should never happen, but lets make sure.
  }


  int nthIdx = setting_frameEnergyTHN*allResVec.size();

  assert(nthIdx < (int)allResVec.size());
  assert(setting_frameEnergyTHN < 1);

  std::nth_element(allResVec.begin(), allResVec.begin()+nthIdx, allResVec.end());
  float nthElement = sqrtf(allResVec[nthIdx]);






    newFrame->frameEnergyTH = nthElement*setting_frameEnergyTHFacMedian;
  newFrame->frameEnergyTH = 26.0f*setting_frameEnergyTHConstWeight + newFrame->frameEnergyTH*(1-setting_frameEnergyTHConstWeight);
  newFrame->frameEnergyTH = newFrame->frameEnergyTH*newFrame->frameEnergyTH;
  newFrame->frameEnergyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;
}

Vec3 FullSystem::linearizeAll(bool fixLinearization)
{
  double lastEnergyP = 0;
  double lastEnergyR = 0;
  double num = 0;


  std::vector<PointFrameResidual*> toRemove[NUM_THREADS];
  for(int i=0;i<NUM_THREADS;i++) toRemove[i].clear();

  if(multiThreading)
  {
    treadReduce.reduce(boost::bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4), 0, activeResiduals.size(), 0);
    lastEnergyP = treadReduce.stats[0];
  }
  else
  {
    Vec10 stats;
    linearizeAll_Reductor(fixLinearization, toRemove, 0,activeResiduals.size(),&stats,0);
    lastEnergyP = stats[0];
  }


  setNewFrameEnergyTH();


  if(fixLinearization)
  {

    for(PointFrameResidual* r : activeResiduals)
    {
      PointHessian* ph = r->point;
      if(ph->lastResiduals[0].first == r)
        ph->lastResiduals[0].second = r->state_state;
      else if(ph->lastResiduals[1].first == r)
        ph->lastResiduals[1].second = r->state_state;



    }

    int nResRemoved=0;
    for(int i=0;i<NUM_THREADS;i++)
    {
      for(PointFrameResidual* r : toRemove[i])
      {
        PointHessian* ph = r->point;

        if(ph->lastResiduals[0].first == r)
          ph->lastResiduals[0].first=0;
        else if(ph->lastResiduals[1].first == r)
          ph->lastResiduals[1].first=0;

        for(unsigned int k=0; k<ph->residuals.size();k++)
        {
          if(ph->residuals[k] == r)
          {
            ef->dropResidual(r->efResidual);
            deleteOut<PointFrameResidual>(ph->residuals,k);
            nResRemoved++;
            break;
          }
        }
      }
    }
  }

  return Vec3(lastEnergyP, lastEnergyR, num);
}

// applies step to linearization point.
bool FullSystem::doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD)
{
//	float meanStepC=0,meanStepP=0,meanStepD=0;
//	meanStepC += Hcalib.step.norm();

  Vec10 pstepfac;
  pstepfac.segment<3>(0).setConstant(stepfacT);
  pstepfac.segment<3>(3).setConstant(stepfacR);
  pstepfac.segment<4>(6).setConstant(stepfacA);


  float sumA=0, sumB=0, sumT=0, sumR=0, sumID=0, numID=0;

  float sumNID=0;

  if(setting_solverMode & SOLVER_MOMENTUM)
  {
    Hcalib.setValue(Hcalib.value_backup + Hcalib.step);
    for(FrameHessian* fh : frameHessians)
    {
      Vec10 step = fh->step;
      step.head<6>() += 0.5f*(fh->step_backup.head<6>());

      fh->setState(fh->state_backup + step);
      sumA += step[6]*step[6];
      sumB += step[7]*step[7];
      sumT += step.segment<3>(0).squaredNorm();
      sumR += step.segment<3>(3).squaredNorm();

      for(PointHessian* ph : fh->pointHessians)
      {
        float step = ph->step+0.5f*(ph->step_backup);
        ph->setIdepth(ph->idepth_backup + step);
        sumID += step*step;
        sumNID += fabsf(ph->idepth_backup);
        numID++;

                ph->setIdepthZero(ph->idepth_backup + step);
      }
    }
  }
  else
  {
    Hcalib.setValue(Hcalib.value_backup + stepfacC*Hcalib.step);
    for(FrameHessian* fh : frameHessians)
    {
      fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
      sumA += fh->step[6]*fh->step[6];
      sumB += fh->step[7]*fh->step[7];
      sumT += fh->step.segment<3>(0).squaredNorm();
      sumR += fh->step.segment<3>(3).squaredNorm();

      for(PointHessian* ph : fh->pointHessians)
      {
        ph->setIdepth(ph->idepth_backup + stepfacD*ph->step);
        sumID += ph->step*ph->step;
        sumNID += fabsf(ph->idepth_backup);
        numID++;

                ph->setIdepthZero(ph->idepth_backup + stepfacD*ph->step);
      }
    }
  }

  sumA /= frameHessians.size();
  sumB /= frameHessians.size();
  sumR /= frameHessians.size();
  sumT /= frameHessians.size();
  sumID /= numID;
  sumNID /= numID;



    if(!setting_debugout_runquiet)
        printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
                sqrtf(sumA) / (0.0005*setting_thOptIterations),
                sqrtf(sumB) / (0.00005*setting_thOptIterations),
                sqrtf(sumR) / (0.00005*setting_thOptIterations),
                sqrtf(sumT)*sumNID / (0.00005*setting_thOptIterations));


  EFDeltaValid=false;
  setPrecalcValues();



  return sqrtf(sumA) < 0.0005*setting_thOptIterations &&
      sqrtf(sumB) < 0.00005*setting_thOptIterations &&
      sqrtf(sumR) < 0.00005*setting_thOptIterations &&
      sqrtf(sumT)*sumNID < 0.00005*setting_thOptIterations;
//
//	printf("mean steps: %f %f %f!\n",
//			meanStepC, meanStepP, meanStepD);
}



// sets linearization point.
void FullSystem::backupState(bool backupLastStep)
{
  if(setting_solverMode & SOLVER_MOMENTUM)
  {
    if(backupLastStep)
    {
      Hcalib.step_backup = Hcalib.step;
      Hcalib.value_backup = Hcalib.value;
      for(FrameHessian* fh : frameHessians)
      {
        fh->step_backup = fh->step;
        fh->state_backup = fh->get_state();
        for(PointHessian* ph : fh->pointHessians)
        {
          ph->idepth_backup = ph->idepth;
          ph->step_backup = ph->step;
        }
      }
    }
    else
    {
      Hcalib.step_backup.setZero();
      Hcalib.value_backup = Hcalib.value;
      for(FrameHessian* fh : frameHessians)
      {
        fh->step_backup.setZero();
        fh->state_backup = fh->get_state();
        for(PointHessian* ph : fh->pointHessians)
        {
          ph->idepth_backup = ph->idepth;
          ph->step_backup=0;
        }
      }
    }
  }
  else
  {
    Hcalib.value_backup = Hcalib.value;
    for(FrameHessian* fh : frameHessians)
    {
      fh->state_backup = fh->get_state();
      for(PointHessian* ph : fh->pointHessians)
        ph->idepth_backup = ph->idepth;
    }
  }
}

// sets linearization point.
void FullSystem::loadSateBackup()
{
  Hcalib.setValue(Hcalib.value_backup);
  for(FrameHessian* fh : frameHessians)
  {
    fh->setState(fh->state_backup);
    for(PointHessian* ph : fh->pointHessians)
    {
      ph->setIdepth(ph->idepth_backup);

            ph->setIdepthZero(ph->idepth_backup);
    }

  }


  EFDeltaValid=false;
  setPrecalcValues();
}


double FullSystem::calcMEnergy()
{
  // if forced to accept step, return zero error
  // this way it cannot be any worse
  if(setting_forceAceptStep) return 0;

  // otherwise, return the actual error
  return ef->calcMEnergyF();

}


void FullSystem::printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b)
{
  printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
      res[0],
      sqrtf((float)(res[0] / (patternNum*ef->resInA))),
      ef->resInA,
      ef->resInM,
      a,
      b
  );

}


// perform keyframe window optimization
// here is where the residuals will be computed
// and where my light-aware updates will need to be made
float FullSystem::optimize(int mnumOptIts)
{
  // if number of frames too small, skip
  if(frameHessians.size() < 2) return 0;

  // if number of frames small, do more iterations
  if(frameHessians.size() < 3) mnumOptIts = 20;
  if(frameHessians.size() < 4) mnumOptIts = 15;

  // clear list of residuals objects
  activeResiduals.clear();

  int numPoints = 0;
  int numLRes = 0;

  // process each keyframe in window
  for(FrameHessian* fh : frameHessians)
  {
    // process each keypoint in current keyframe
    for(PointHessian* ph : fh->pointHessians)
    {
      // process each residual (observation) for the current keypoint
      for(PointFrameResidual* r : ph->residuals)
      {
        // check if keypoint not marginalized
        if(!r->efResidual->isLinearized)
        {
          // add residual as normal
          activeResiduals.push_back(r);
          // reset residual status
          r->resetOOB();
        }
        else
        {
          // increment linearized residual count
          numLRes++;
        }
      }

      // increment points used
      // NOTE: probably better as numPoints += pointHessians.size()
      numPoints++;
    }
  }

  // print debug message if needed
  if(!setting_debugout_runquiet)
  {
    printf("OPTIMIZE %d pts, %d active res, %d lin res!\n",
        ef->nPoints,(int)activeResiduals.size(), numLRes);
  }

  // ASSUME: this is recomputing new derivatives
  // that way the following two functions can just do Ax - b
  // to get the current error, without approximating anything
  // these Jacobians/Hessian can also be used later
  Vec3 lastEnergy = linearizeAll(false);

  // ASSUME: this is the photometric error data term
  double lastEnergyL = calcLEnergy();

  // ASSUME: this is the regularization cost on brightness change
  double lastEnergyM = calcMEnergy();

  // check if multi-threading
  if(multiThreading)
  {
    // copy residual new state to current state (multi-threaded)
    // copy jacobian=true, and apply to all residuals
    treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor,
        this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
  }
  else
  {
    // copy residual new state to current state (single-threaded)
    // copy jacobian=true, and apply to all residuals
    applyRes_Reductor(true,0,activeResiduals.size(),0,0);
  }

  // print debug message as needed
  if(!setting_debugout_runquiet)
  {
    printf("Initial Error       \t");

    printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0,
        frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
  }

  // debug plot as needed
  debugPlotTracking();

  // initialize LM lambda
  double lambda = 1e-1;

  // initialize trust region size
  float stepsize=1;

  // initialize previous parameter update vector
  VecX previousX = VecX::Constant(CPARS+ 8*frameHessians.size(), NAN);

  // for each optimization iteration
  for(int iteration=0;iteration<mnumOptIts;iteration++)
  {
    // backup optimization state
    // ASSUME: this if for if a bad step is taken and we want to revert
    backupState(iteration!=0);

    // perform LM step
    // TODO: follow this function
    solveSystem(iteration, lambda);

    // get relative update direction change
    // ASSUME: this is for exiting if little change?
    double incDirChange = (1e-20 + previousX.dot(ef->lastX)) /
        (1e-20 + previousX.norm() * ef->lastX.norm());

    // assign previous update vector
    previousX = ef->lastX;

    // check if valid direction change value and using step momentum
    if(std::isfinite(incDirChange) &&
        (setting_solverMode & SOLVER_STEPMOMENTUM))
    {
      // compute new update vector using previous iterations momentum
      float newStepsize = exp(incDirChange*1.4);
      if(incDirChange<0 && stepsize>1) stepsize=1;
      stepsize = sqrtf(sqrtf(newStepsize*stepsize*stepsize*stepsize));

      // clamp step size
      if(stepsize > 2) stepsize=2;
      if(stepsize <0.25) stepsize=0.25;
    }

    // TODO: review ========================

    bool canbreak =
        doStepFromBackup(stepsize,stepsize,stepsize,stepsize,stepsize);

    // eval new energy!
    // TODO: follow this function
    Vec3 newEnergy = linearizeAll(false);
    // TODO: follow this function
    double newEnergyL = calcLEnergy();
    // TODO: follow this function
    double newEnergyM = calcMEnergy();

    // print debug message as needed
    if(!setting_debugout_runquiet)
    {
      printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",

      (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
          lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ?
          "ACCEPT" : "REJECT",

      iteration,
      log10(lambda),
      incDirChange,
      stepsize);

      printOptRes(newEnergy, newEnergyL, newEnergyM , 0, 0,
          frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
    }

    // if forced to always take step
    // or new step error is better than previous
    if(setting_forceAceptStep ||
        (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
        lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
    {
      // check if multi-threading
      if(multiThreading)
      {
        // copy residual new state to current state (multi-threaded)
        // copy jacobian=true, and apply to all residuals
        treadReduce.reduce(
              boost::bind(&FullSystem::applyRes_Reductor, this, true,
              _1, _2, _3, _4), 0, activeResiduals.size(), 50);
      }
      else
      {
        // copy residual new state to current state (single-threaded)
        // copy jacobian=true, and apply to all residuals
        applyRes_Reductor(true,0,activeResiduals.size(),0,0);
      }

      // update previous step errors
      lastEnergy = newEnergy;
      lastEnergyL = newEnergyL;
      lastEnergyM = newEnergyM;

      // be more like GN
      lambda *= 0.25;
    }
    // step rejected
    else
    {
      // restore previous step state
      loadSateBackup();

      // restore costs
      // NOTE: why isn't this appart of backup state?
      lastEnergy = linearizeAll(false);
      lastEnergyL = calcLEnergy();
      lastEnergyM = calcMEnergy();

      // be more like gradient descent
      lambda *= 1e2;
    }

    // if can exit optimization early, do so
    if(canbreak && iteration >= setting_minOptIterations) break;
  }

  Vec10 newStateZero = Vec10::Zero();

  newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);

  frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam,
      newStateZero);

  EFDeltaValid=false;
  EFAdjointsValid=false;
  ef->setAdjointsF(&Hcalib);
  setPrecalcValues();

  lastEnergy = linearizeAll(true);

  if(!std::isfinite((double)lastEnergy[0]) ||
      !std::isfinite((double)lastEnergy[1]) ||
      !std::isfinite((double)lastEnergy[2]))
  {
    printf("KF Tracking failed: LOST!\n");
    isLost=true;
  }

  statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] /
      (patternNum*ef->resInA)));

  // write to log as needed
  if(calibLog != 0)
  {
    (*calibLog) << Hcalib.value_scaled.transpose() <<
        " " << frameHessians.back()->get_state_scaled().transpose() <<
        " " << sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA))) <<
        " " << ef->resInM << "\n";
    calibLog->flush();
  }

  {
    // acquire frame lock as we'll be updating keyframe poses
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

    // for each active keyframe in window
    for(FrameHessian* fh : frameHessians)
    {
      // apply SE3 and brightness updates
      fh->shell->camToWorld = fh->PRE_camToWorld;
      fh->shell->aff_g2l = fh->aff_g2l();
    }
  }

  // debug plot as needed
  debugPlotTracking();

  // return RMSE for window
  return sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));
}

void FullSystem::solveSystem(int iteration, double lambda)
{
  // get null-spaces for logging purposes
  // TODO: determine how these are computed
  ef->lastNullspaces_forLogging = getNullspaces(
      ef->lastNullspaces_pose,
      ef->lastNullspaces_scale,
      ef->lastNullspaces_affA,
      ef->lastNullspaces_affB);

  // compute the LM state update
  ef->solveSystemF(iteration, lambda, &Hcalib);
}



double FullSystem::calcLEnergy()
{
  // if we are forced to always access an optimization step
  // then teturn zero, soi no other work is really needed
  if(setting_forceAceptStep) return 0;

  double Ef = ef->calcLEnergyF_MT();
  return Ef;

}


void FullSystem::removeOutliers()
{
  int numPointsDropped=0;
  for(FrameHessian* fh : frameHessians)
  {
    for(unsigned int i=0;i<fh->pointHessians.size();i++)
    {
      PointHessian* ph = fh->pointHessians[i];
      if(ph==0) continue;

      if(ph->residuals.size() == 0)
      {
        fh->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        fh->pointHessians[i] = fh->pointHessians.back();
        fh->pointHessians.pop_back();
        i--;
        numPointsDropped++;
      }
    }
  }
  ef->dropPointsF();
}




std::vector<VecX> FullSystem::getNullspaces(
    std::vector<VecX> &nullspaces_pose,
    std::vector<VecX> &nullspaces_scale,
    std::vector<VecX> &nullspaces_affA,
    std::vector<VecX> &nullspaces_affB)
{
  // clear all input vectors
  nullspaces_pose.clear();
  nullspaces_scale.clear();
  nullspaces_affA.clear();
  nullspaces_affB.clear();


  // compute total number of "dense" parameters in system
  // dense in the sense that they affect almost everything
  // - camera parameters (CPARS): 4 (fx, fy, cx, cy)
  // - keyframe parameters (per frame): 8 (6 DoF pose, 2 brightness)
  int n = CPARS + frameHessians.size() * 8;

  std::vector<VecX> nullspaces_x0_pre;

  // for each 6 DoF of pose
  for(int i=0;i<6;i++)
  {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();

    // process each keyframe in window
    for(FrameHessian* fh : frameHessians)
    {
      // get null spaces for pose
      // ASSUME: these are cells of the hessian with zero for parameter?
      nullspace_x0.segment<6>(CPARS+fh->idx*8) = fh->nullspaces_pose.col(i);

      // transform translation hessian (multiply by 2)
      // TODO: why?
      nullspace_x0.segment<3>(CPARS+fh->idx*8) *= SCALE_XI_TRANS_INVERSE;

      // transform rotation hessian (multiply by 1)
      // TODO: why?
      nullspace_x0.segment<3>(CPARS+fh->idx*8+3) *= SCALE_XI_ROT_INVERSE;
    }

    nullspaces_x0_pre.push_back(nullspace_x0);
    nullspaces_pose.push_back(nullspace_x0);
  }

  for(int i=0;i<2;i++)
  {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for(FrameHessian* fh : frameHessians)
    {
      nullspace_x0.segment<2>(CPARS+fh->idx*8+6) = fh->nullspaces_affine.col(i).head<2>();
      nullspace_x0[CPARS+fh->idx*8+6] *= SCALE_A_INVERSE;
      nullspace_x0[CPARS+fh->idx*8+7] *= SCALE_B_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    if(i==0) nullspaces_affA.push_back(nullspace_x0);
    if(i==1) nullspaces_affB.push_back(nullspace_x0);
  }

  VecX nullspace_x0(n);
  nullspace_x0.setZero();
  for(FrameHessian* fh : frameHessians)
  {
    nullspace_x0.segment<6>(CPARS+fh->idx*8) = fh->nullspaces_scale;
    nullspace_x0.segment<3>(CPARS+fh->idx*8) *= SCALE_XI_TRANS_INVERSE;
    nullspace_x0.segment<3>(CPARS+fh->idx*8+3) *= SCALE_XI_ROT_INVERSE;
  }
  nullspaces_x0_pre.push_back(nullspace_x0);
  nullspaces_scale.push_back(nullspace_x0);

  return nullspaces_x0_pre;
}

}

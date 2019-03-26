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



#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"



#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"


std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start=0;
int end=100000;
bool prefetch = false;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload=false;
bool useSampleOutput=false;


int mode=0;

bool firstRosSpin=false;

using namespace dso;


void my_exit_handler(int s)
{
  printf("Caught signal %d\n",s);
  exit(1);
}

void exitThread()
{
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = my_exit_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  firstRosSpin=true;
  while(true) pause();
}



void settingsDefault(int preset)
{
  printf("\n=============== PRESET Settings: ===============\n");
  if(preset == 0 || preset == 1)
  {
    printf("DEFAULT settings:\n"
        "- %s real-time enforcing\n"
        "- 2000 active points\n"
        "- 5-7 active frames\n"
        "- 1-6 LM iteration each KF\n"
        "- original image resolution\n", preset==0 ? "no " : "1x");

    playbackSpeed = (preset==0 ? 0 : 1);
    preload = preset==1;
    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations=6;
    setting_minOptIterations=1;

    setting_logStuff = false;
  }

  if(preset == 2 || preset == 3)
  {
    printf("FAST settings:\n"
        "- %s real-time enforcing\n"
        "- 800 active points\n"
        "- 4-6 active frames\n"
        "- 1-4 LM iteration each KF\n"
        "- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

    playbackSpeed = (preset==2 ? 0 : 5);
    preload = preset==3;
    setting_desiredImmatureDensity = 600;
    setting_desiredPointDensity = 800;
    setting_minFrames = 4;
    setting_maxFrames = 6;
    setting_maxOptIterations=4;
    setting_minOptIterations=1;

    benchmarkSetting_width = 424;
    benchmarkSetting_height = 320;

    setting_logStuff = false;
  }

  printf("==============================================\n");
}






void parseArgument(char* arg)
{
  int option;
  float foption;
  char buf[1000];


    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

  if(1==sscanf(arg,"preset=%d",&option))
  {
    settingsDefault(option);
    return;
  }


  if(1==sscanf(arg,"rec=%d",&option))
  {
    if(option==0)
    {
      disableReconfigure = true;
      printf("DISABLE RECONFIGURE!\n");
    }
    return;
  }



  if(1==sscanf(arg,"noros=%d",&option))
  {
    if(option==1)
    {
      disableROS = true;
      disableReconfigure = true;
      printf("DISABLE ROS (AND RECONFIGURE)!\n");
    }
    return;
  }

  if(1==sscanf(arg,"nolog=%d",&option))
  {
    if(option==1)
    {
      setting_logStuff = false;
      printf("DISABLE LOGGING!\n");
    }
    return;
  }
  if(1==sscanf(arg,"reverse=%d",&option))
  {
    if(option==1)
    {
      reverse = true;
      printf("REVERSE!\n");
    }
    return;
  }
  if(1==sscanf(arg,"nogui=%d",&option))
  {
    if(option==1)
    {
      disableAllDisplay = true;
      printf("NO GUI!\n");
    }
    return;
  }
  if(1==sscanf(arg,"nomt=%d",&option))
  {
    if(option==1)
    {
      multiThreading = false;
      printf("NO MultiThreading!\n");
    }
    return;
  }
  if(1==sscanf(arg,"prefetch=%d",&option))
  {
    if(option==1)
    {
      prefetch = true;
      printf("PREFETCH!\n");
    }
    return;
  }
  if(1==sscanf(arg,"start=%d",&option))
  {
    start = option;
    printf("START AT %d!\n",start);
    return;
  }
  if(1==sscanf(arg,"end=%d",&option))
  {
    end = option;
    printf("END AT %d!\n",start);
    return;
  }

  if(1==sscanf(arg,"files=%s",buf))
  {
    source = buf;
    printf("loading data from %s!\n", source.c_str());
    return;
  }

  if(1==sscanf(arg,"calib=%s",buf))
  {
    calib = buf;
    printf("loading calibration from %s!\n", calib.c_str());
    return;
  }

  if(1==sscanf(arg,"vignette=%s",buf))
  {
    vignette = buf;
    printf("loading vignette from %s!\n", vignette.c_str());
    return;
  }

  if(1==sscanf(arg,"gamma=%s",buf))
  {
    gammaCalib = buf;
    printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
    return;
  }

  if(1==sscanf(arg,"rescale=%f",&foption))
  {
    rescale = foption;
    printf("RESCALE %f!\n", rescale);
    return;
  }

  if(1==sscanf(arg,"speed=%f",&foption))
  {
    playbackSpeed = foption;
    printf("PLAYBACK SPEED %f!\n", playbackSpeed);
    return;
  }

  if(1==sscanf(arg,"save=%d",&option))
  {
    if(option==1)
    {
      debugSaveImages = true;
      if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
      if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
      if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
      if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
      printf("SAVE IMAGES!\n");
    }
    return;
  }

  if(1==sscanf(arg,"mode=%d",&option))
  {

    mode = option;
    if(option==0)
    {
      printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
    }
    if(option==1)
    {
      printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
    }
    if(option==2)
    {
      printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
    }
    return;
  }

  printf("could not parse argument \"%s\"!!!!\n", arg);
}



int main(int argc, char** argv)
{
  // parse command line argumentes
  for (int i = 1; i < argc; ++i)
  {
    parseArgument(argv[i]);
  }

  // create thread for capturing control-c exit command
  boost::thread exThread = boost::thread(exitThread);

  // create image reader for loading camera images files
  ImageFolderReader* reader;
  reader = new ImageFolderReader(source,calib, gammaCalib, vignette);
  reader->setGlobalCalibration();

  // check if photometric calibration provided if required
  if (setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
  {
    printf("ERROR: dont't have photometric calibation. "
           "Need to use commandline options mode=1 or mode=2 ");

    exit(1);
  }

  // initialize frame start and end indices
  int lstart=start;
  int lend = end;
  int linc = 1;

  // check if reverse order requested
  if (reverse)
  {
    printf("REVERSE!!!!");

    // set start to end
    lstart=end-1;

    // check if fewer images read than indices indicate
    if (lstart >= reader->getNumImages())
    {
      // ensure that start is actually last image file
      lstart = reader->getNumImages()-1;
    }

    // update end and increment accordingly
    lend = start;
    linc = -1;
  }

  // create the SLAM system
  // NOTE: some MAGIC should be happening here
  FullSystem* fullSystem = new FullSystem();

  // set photometric calibration of system
  fullSystem->setGammaFunction(reader->getPhotometricGamma());

  // TODO: understand this member variable's role
  fullSystem->linearizeOperation = (playbackSpeed==0);

  // declare pointer to pangolin view
  IOWrap::PangolinDSOViewer* viewer = 0;

  // check if GUI required
  if (!disableAllDisplay)
  {
    // create pangolin view
    viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);

    // TODO: understand what happens to output wrappers
    fullSystem->outputWrapper.push_back(viewer);
  }

  // check if sample output required
  if (useSampleOutput)
  {
    // create example output handler object and add to system
    // NOTE: I could use this to evaluate runtime status on synthetic data
    fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());
  }

  // create thread for running SLAM system
  std::thread runthread([&]()
  {
    // declare list of frame indices
    // NOTE: this will be populated in the next for-loop
    std::vector<int> idsToPlay;

    // declare list of frame timestamps
    // NOTE: this will be populated in the next for-loop
    std::vector<double> timesToPlayAt;

    // process each frame, start with initial frame
    // this for-loop just initializes idsToPlay and timesToPlayAt
    for (int i = lstart;
        // check if valid frame index as per available frames and per request
        i >= 0 && i < reader->getNumImages() && linc * i < linc * lend;
        // increment frame index according to direction order
        i+=linc)
    {
      // add current frame index to list
      idsToPlay.push_back(i);

      // essentially: if first frame processed
      if (timesToPlayAt.empty())
      {
        // initialize with zero timestamp
        timesToPlayAt.push_back((double)0);
      }
      // essentially: for all subsequent frames
      else
      {
        // get timestamp of current frame index (may not be sequential)
        double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size()-1]);

        // get timestamp of previous frame index (may not be sequential)
        double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size()-2]);

        // increment previous timestamp with frame timestamp difference
        // must be absolute value, in case it is reversed order
        // NOTE: could playbackSpeed be zero? if so this will be NaN
        // - do timestamps just not matter when preset=0?
        timesToPlayAt.push_back(timesToPlayAt.back() +
            fabs(tsThis-tsPrev) / playbackSpeed);
      }
    }

    // delcare list of captured images
    std::vector<ImageAndExposure*> preloadedImages;

    // check if preloading images was requested
    if(preload)
    {
      printf("LOADING ALL IMAGES!\n");

      // for each frame index to be processed
      for(int ii=0;ii<(int)idsToPlay.size(); ii++)
      {
        // get index of frame
        int i = idsToPlay[ii];

        // load image and append to list
        preloadedImages.push_back(reader->getImage(i));
      }
    }

    // struct for holding current system time when first frame processed
    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);

    // seemily duplicated work as above
    clock_t started = clock();

    // initial log-time of first frame
    double sInitializerOffset=0;

    // for each frame to be processed
    for(int ii=0;ii<(int)idsToPlay.size(); ii++)
    {
      // check if first frame procssed
      if(!fullSystem->initialized)
      {
        // get current system time
        gettimeofday(&tv_start, NULL);
        started = clock();

        // get first log time for first frame as offset
        sInitializerOffset = timesToPlayAt[ii];
      }

      // get index of current frame
      int i = idsToPlay[ii];

      // pointer for current frame image
      ImageAndExposure* img;

      // check if images were preloaded
      if (preload)
      {
        // get preloaded pointer
        img = preloadedImages[ii];
      }
      else
      {
        // read image from file
        img = reader->getImage(i);
      }

      // bool indicating if the current frame should be skipped
      bool skipFrame=false;

      // check if runtime/variable playback speed requested
      // NOTE: not the case when preset=0
      if (playbackSpeed != 0)
      {
        // get current system time
        struct timeval tv_now;
        gettimeofday(&tv_now, NULL);

        // get elapsed time since the start of process
        // NOTE: offset by the log-time of first frame
        double sSinceStart = sInitializerOffset +
            // add elapsed system time (in seconds)
            ((tv_now.tv_sec-tv_start.tv_sec) +
            // add elapsed system time (in microseconds converted to secs)
            (tv_now.tv_usec-tv_start.tv_usec)/(1000.0f*1000.0f));

        // if elapsed time is still before next frame time
        if(sSinceStart < timesToPlayAt[ii])
        {
          // this means SLAM system running faster than framerate
          // so we will process this frame, but first need to wait
          // sleep for time remaining until next frame should be read
          usleep((int)((timesToPlayAt[ii]-sSinceStart)*1000*1000));
        }
        // check if we've taken too long on the previos frame
        // that means we've missed processing this new frame
        // NOTE: not sure what the added 0.6 is for.
        // - these look like absolute numbers, ignoring actual framerate
        // - I believe 0.5 here is have a second
        // - sho you'd have be be lagging behind by that before skipping
        // - maybe this is for initialization purposes
        // - as skipping one frame won't catch up with 0.5s
        // - so it's like a buffer that will eventually get full
        // - and only then will it start dropping frames
        else if(sSinceStart > timesToPlayAt[ii]+0.5+0.1*(ii%2))
        {
          printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii,
              timesToPlayAt[ii], sSinceStart);

          skipFrame=true;
        }
      }

      // if not skipping, add frame to SLAM system
      // NOTE: this is where the real MAGIC happens!
      if(!skipFrame) fullSystem->addActiveFrame(img, i);

      // delete image once processed
      delete img;

      // check if current SLAM state is initializating
      if (fullSystem->initFailed || setting_fullResetRequested)
      {
        // check if actually a reset
        // process actually deletes SLAM objects and creates it again
        // NOTE: I assume that initialization takes around 250 frames
        // - so we assume that if were are within the first 250 it's nothing new
        if (ii < 250 || setting_fullResetRequested)
        {
          printf("RESETTING!\n");

          // get all old output handlers
          std::vector<IOWrap::Output3DWrapper*> wraps;
          wraps = fullSystem->outputWrapper;

          // delete SLAM system
          delete fullSystem;

          // delete old output handlers
          for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

          // create brand new SLAM system
          fullSystem = new FullSystem();

          // assign same photometric calibration function
          fullSystem->setGammaFunction(reader->getPhotometricGamma());

          // TODO: understand this member variable's role
          fullSystem->linearizeOperation = (playbackSpeed==0);

          // assign new output handlers to SLAM system
          fullSystem->outputWrapper = wraps;

          // reset SLAM reset request
          setting_fullResetRequested=false;
        }
      }

      // check if SLAM state is lost
      if(fullSystem->isLost)
      {
        // print notification
        printf("LOST!!\n");

        // don't continue processing current frame
        break;
      }

    } // end of the actually SLAM processing loop

    // wait for SLAM mapping to finish
    fullSystem->blockUntilMappingIsFinished();

    // get end system time
    clock_t ended = clock();
    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);

    // write pose-graph to file
    fullSystem->printResult("result.txt");

    // get run statistics for printing results
    int numFramesProcessed = abs(idsToPlay[0]-idsToPlay.back());
    double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0])-reader->getTimestamp(idsToPlay.back()));
    double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
    double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);

    // print out run summary
    printf("\n======================"
            "\n%d Frames (%.1f fps)"
            "\n%.2fms per frame (single core); "
            "\n%.2fms per frame (multi core); "
            "\n%.3fx (single core); "
            "\n%.3fx (multi core); "
            "\n======================\n\n",
            numFramesProcessed, numFramesProcessed/numSecondsProcessed,
            MilliSecondsTakenSingle/numFramesProcessed,
            MilliSecondsTakenMT / (float)numFramesProcessed,
            1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
            1000 / (MilliSecondsTakenMT / numSecondsProcessed));

    // check if logging data
    if (setting_logStuff)
    {
      // open output log file
      std::ofstream tmlog;
      tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);

      // print out performance framerates
      tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*
          reader->getNumImages()) << " " << ((tv_end.tv_sec-tv_start.tv_sec)*
          1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) /
          (float)reader->getNumImages() << "\n";

      // close log file
      tmlog.flush();
      tmlog.close();
    }

  }); // end of SLAM thread

  // check if GUI requested
  if(viewer != 0)
  {
    // run pangoline view in main thread
    // NOTE: I assume this is a blocking function
    // - meaning the window has to be closed to proceed
    viewer->run();
  }

  // wait for SLAM thread to complete
  runthread.join();

  // wait for each output handler to complete
  for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
  {
    ow->join();
    delete ow;
  }

  // perform the final cleanup before exiting
  printf("DELETE FULLSYSTEM!\n");
  delete fullSystem;

  printf("DELETE READER!\n");
  delete reader;

  printf("EXIT NOW!\n");
  return 0;
}

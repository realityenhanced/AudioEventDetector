#install.packages('caTools');
#install.packages('audio');
require(caTools);
require(audio);

# Current Runtime: 2hrs

# Logistic Regression for binary classification of Audio data
# Positive Inputs are under the positive folder and Negative Inputs under the negative folder
# Audio files need to be uncompressed wav files.
# Sample rate of the training wav files should be 16Khz.
# Only the first channel will be used for training, if a multi channel wave file is used as input.
# Duration of the training wav files is not mandated, only the first 10ms wndow will be used for training.
# So make sure the first 10ms contain the event to be detected.

# Configuration Variables
PLOT_ONLY <<- FALSE; # Only Plot the data in training files
LOG_FILE <<- "log.log";
OPTIMAL_THETA_LOGFILE <<- "OptimalTheta.log";
POSITIVE_FOLDER <<- "positive";
NEGATIVE_FOLDER <<- "negative";
MIN_THRESHOLD <<- 0.01; # Treat samples below this as minimum
NUM_SAMPLES <<- 0.010 * 16000; # 0.01s worth of samples at 16KHz
SECTION_SIZE <<- 5;
AUDIO_THRESHOLD <<- 0.05; # % increase before event detection starts
NUM_ITERATIONS <<- 1000000;
LAST_KNOWN_THETA <<- c(48.98584,-34.08085,36.07657,10.15354,-14.58598,
                       -29.82637,-19.51422,-37.75456,3.337353,10.25387,
                       -7.539513,-9.702672,-39.51088,-21.29289,-4.482931,
                       -7.846956,26.01872,-55.03574,-13.75364,-0.1781504,
                       -11.93301,-19.44979,38.56394,-8.32193,-4.273079,
                       -13.5322,15.44775,5.26963,16.70343,-16.45484,
                       3.417254,20.80411,35.17422,-10.05808,1.950941,
                       23.34977,-28.93218,-45.59006,23.93392,14.12858,
                       6.503352,23.75147,-12.87637,87.42623,-22.60923,
                       24.31866,7.952283,4.946256,39.12044,19.29568,
                       -6.252109,14.99636,16.37063,-6.825156,-3.202251,
                       0.3693936,21.90368,-5.936639,27.43503,-48.8979,
                       3.640914,4.96002,22.30016,27.45199,18.70437,
                       27.94296,-25.34845,73.47367,-15.49899,-9.371605,
                       13.43658,32.62597,31.00726,23.01913,9.945299,
                       7.889511,33.32596,19.29016,1.027091,7.777448,
                       -8.647991,-54.20201,-31.03197,0.3036094,17.43253,
                       37.62831,5.52369,-0.4008886,12.59131,-16.99769,
                       -19.47173,-42.97385,64.01443,16.04932,-16.24305,
                       -29.89545,28.77299,15.62211,12.83778,12.04214,
                       14.03664,-50.42192,-77.81623,5.650083,-20.33065,
                       -21.54364,-24.90889,-9.685399,-26.72614,-20.62394,
                       -42.38044,1.975372,-5.262332,-2.27349,-3.503568,
                       4.900912,10.90172,44.93784,6.003533,-19.51973,
                       -12.24102,13.06057,-24.23417,4.694212,6.863126,
                       10.31773,-28.03063,-2.606782,4.436853,18.54782,
                       -16.15499,-15.15439,-59.6563,16.25569,-31.08876,
                       5.451538,-13.70069,14.73496,-35.64097,-11.32442,
                       -31.22165,-18.26392,3.263229,-11.33552,-1.152434,
                       16.64332,-15.73408,-11.98606,-6.222856,-23.24712,
                       75.63466,-2.908966,57.26583,10.46732,-1.701779,
                       11.72756,-8.14354,1.713875,-13.20112,-2.378029,
                       -6.683554);

# GLOBALS
X <<- 0; # features
Y <<- 0; # results
opttheta <<- 0; # final theta calculated

# HELPER FUNCTIONS
# Initialize logs
InitLog <- function()
{
  write(paste(proc.time()[3], "Init Logs"), LOG_FILE, append=FALSE);
}

# Logging helper func
Log <- function (...)
{
    arguments <- list(...); 
    numArgs <- length(arguments);
    str <- proc.time()[3]; # Elapsed time 
    for(i in (1:numArgs))
    {
      str <- paste(str, arguments[i], sep=" ");
    }
    write(str, LOG_FILE, append=TRUE);
    print(str);
}

# Sigmoid function
Sigmoid <- function(z)
{
  g <- 1/(1+exp(-z));
  return(g);
}

# Cost Function
CostFunction <- function(theta)
{
  m <- nrow(X);
  g <- Sigmoid(X %*% theta);
  J <- (1/m)*sum((-Y*log(g)) - ((1-Y)*log(1-g)));
  return(J)
}

# Feature Extract
GetFeatures <- function(data)
{
  if (length(data) != NUM_SAMPLES)
  {
    Log("Length of data != NUM_SAMPLES " , length(data));
    stop();  
  }
  
  ffts <- fft(data);
  
  # Use the real part (contribution of the cos waves)
  features <- Re(ffts);

  # TEST: Do not Filter out extreme values
  #lapply(features, function(x) { if (x <= 0.05) { return (0);} else if (x >= 0.95){ return (0.95); } })
  
  return(features);
}
#

# Get the Sectional Energy
GetSectionalEnergy <- function(data)
{
  numSections <- length(data)%/%SECTION_SIZE;
  energies <- matrix(0, nrow = numSections, ncol = 1);
  for (i in 1:numSections)
  {
    energies[i] <- sum(abs(data[(1+((i-1)*SECTION_SIZE)) : (i*SECTION_SIZE)]));
  }
  
  return (energies);
}

# Get deltas between consecutive elements
GetDeltas <- function(elements)
{
  delta <- elements[1:length(elements)-1];
  delta <- (abs(delta - elements[2:length(elements)])/delta) ;
  
  return (delta);
}
#

# Plot waveforms
PlotAudioData <- function(filePath)
{
  data <- LoadDataFromWav(filePath)[1:NUM_SAMPLES];
  
  # Create a new plot for audio waveform
  dev.new();
  
  # Create a 2x2 grid plot and store the old par val for restoring later
  old.par <- par(mfrow=c(2,3));
  
  plot(data, main=filePath, xlab="Time", ylab="Amplitude");
  
  ffts <- fft(data);
  
  # Create a new plot for Re(ffts)
  reffts = Re(ffts);
  plot(reffts);
  
  # Create a new plot for Im(ffts)
  imffts = Im(ffts);
  plot(imffts);
  
  # Create a new plot for Mod(ffts)
  modffts = Mod(ffts)
  plot(modffts);
  
  # DISABLED: New plot for moving window average
  #WINDOW_SIZE <- 5;
  #slidingMean <- runmean(abs(data), WINDOW_SIZE, endrule="mean", align="left");
  #plot(slidingMean[0:20]);
  
  # Plot deltas between consecutive samples
  #delta <- slidingMean[1:length(slidingMean)-1];
  #delta <- (abs(delta - slidingMean[2:length(slidingMean)])/delta) ;
  #plot(delta[0:20]);
  
  # Plot the sectional energy
  energies <- GetSectionalEnergy(data);
  barplot(energies);

  # Plot deltas between consecutive samples
  deltas <- GetDeltas(energies);
  barplot(deltas);
  
  par(old.par);
}
#

# Load a wav file and extract features
LoadDataFromWav <- function(filePath)
{
  Log("Loading data from: ", filePath);
  
  data <- load.wave(filePath);
  if (data$rate != 16000)
  {
    Log("ERROR! Sample rate != 16Khz, instead", data$rate);
    stop();
  }
  
  if (length(data) < NUM_SAMPLES)
  {  
    Log("ERROR! Not enoough samples. Expected : ", NUM_SAMPLES, "but got ", length(data));
    stop();
  }
  
  # Remove values less than 0.001
  data <- sapply(data, function(x) { if (x <= MIN_THRESHOLD) { return (MIN_THRESHOLD);} else { return (x); } });
  
  return (data);
}
#

# Check for the first event in the data
CheckForEvent <- function(data, optimalTheta)
{
  energies <- GetSectionalEnergy(data);
  deltas <- GetDeltas(energies);
  start <- 0;
  
  for (i in (1:length(deltas)))
  {
    # Choose the section in which the sudden jump in energy was seen
    if (deltas[i] >= AUDIO_THRESHOLD)
    {
      start <- i*SECTION_SIZE;
      
      # Check if we have enough samples to use for training 
      if (start + NUM_SAMPLES - 1 >= length(data))
      {
        # No more samples
        break;
      }
      
      features <- GetFeatures(data[start:(start+NUM_SAMPLES-1)]);
      features <- c(1, features);
      probability <- Sigmoid(features%*%optimalTheta);
      if (probability > 0.5)
      {
        Log("EVENT FOUND AT ", start);
        return (TRUE);
      }
    }
  }
  
  if (start == 0)
  {
    Log("NO JUMP FOUND");
  }
  
  return (FALSE);
}
#

# Test non-training samples
TestNonTrainingSamples <- function (optimalTheta)
{
  POSITIVE_TEST_FOLDER <- "TestFiles/positive";
  NEGATIVE_TEST_FOLDER <- "TestFiles/negative";
  
  positiveTestFiles <- list.files(POSITIVE_TEST_FOLDER, pattern = "*.wav");
  negativeTestFiles <- list.files(NEGATIVE_TEST_FOLDER, pattern = "*.wav");
  numTestInputs <- length(positiveTestFiles) + length(negativeTestFiles);

  numIncorrect <- 0;
  
  # Test positive data
  for (file in positiveTestFiles)
  {
    filePath <- paste(POSITIVE_TEST_FOLDER, file, sep = "/");
    Log(filePath);
    
    data <- LoadDataFromWav(filePath);
    
    wasEventFound <- CheckForEvent(data, optimalTheta)
    if (!wasEventFound)
    {
      Log("FAILED: Event not found in positive sample");
      numIncorrect <- numIncorrect + 1;
    }
  }
  
  # Test negative data
  for (file in negativeTestFiles)
  {
    filePath <- paste(NEGATIVE_TEST_FOLDER, file, sep = "/");
    
    data <- LoadDataFromWav(filePath);

    wasEventFound <- CheckForEvent(data, optimalTheta)
    if (wasEventFound)
    {
      Log("FAILED: Event was found in negative sample");
      numIncorrect <- numIncorrect + 1;
    }
  }
  
  Log("NUM INCORRECT = ", numIncorrect);
}

# Main entry point
Main <- function()
{
  # Files
  positiveFiles <- list.files(POSITIVE_FOLDER, pattern = "*.wav");
  negativeFiles <- list.files(NEGATIVE_FOLDER, pattern = "*.wav");
  numInputs <- length(positiveFiles) + length(negativeFiles);
  
  # Vectors to accumulate features and results before converting to matrix form
  Xvec <<- NULL;
  Yvec <<- NULL;
  
  # Load positive data
  for (file in positiveFiles)
  {
    filePath <- paste(POSITIVE_FOLDER, file, sep = "/");
    if (!PLOT_ONLY)
    { 
      data <- LoadDataFromWav(filePath);
      
      # Find the sectional which has a jump in energy to mark the start
      start <- 0; # Invalid index
      energies <- GetSectionalEnergy(data);
      deltas <- GetDeltas(energies);
      for (i in (1:length(deltas)))
      {
        # Choose the section in which the sudden jump in energy was seen
        if (deltas[i] >= AUDIO_THRESHOLD)
        {
          start <- i*SECTION_SIZE;
          break;
        }
      }
      Log("DELTAS = ", deltas);
      
      # Check if we have enough samples to use for training 
      if (start + NUM_SAMPLES - 1 >= length(data))
      {
        Log("ERROR: No jump found in wav");
        stop();
      }

      Xvec <<- c(Xvec, GetFeatures(data[start:(start + NUM_SAMPLES - 1)]));
      Yvec <<- c(Yvec, 1);
    }
    else
    {
      # Plot out discrete audio waveform, real and imaginary parts of the fft
      PlotAudioData(filePath);
    }
  }
  
  # Load negative data
  for (file in negativeFiles)
  {
    filePath <- paste(NEGATIVE_FOLDER, file, sep = "/");
    if (!PLOT_ONLY)
    {  
      data <- LoadDataFromWav(filePath);
      
      # Find the sectional which has a jump in energy to mark the start
      start <- 0; # Invalid index
      energies <- GetSectionalEnergy(data);
      deltas <- GetDeltas(energies);
      for (i in (1:length(deltas)))
      {
        # Choose the section in which the sudden jump in energy was seen
        if (deltas[i] >= AUDIO_THRESHOLD)
        {
          start <- i*SECTION_SIZE;
          
          # Check if we have enough samples to use for training 
          if (start + NUM_SAMPLES - 1 >= length(data))
          {
            break;
          }

          Xvec <<- c(Xvec, GetFeatures(data[start: (start + NUM_SAMPLES - 1)]));
          Yvec <<- c(Yvec, 0);
        }
      }
    }
    else
    {
      # Plot out discrete audio waveform, real and imaginary parts of the fft
      PlotAudioData(filePath);
    }
  }
  
  if (!PLOT_ONLY)
  {
    # Convert the features and results to matrix form
    X <<- matrix(Xvec, ncol=NUM_SAMPLES, byrow=T);
    Y <<- matrix(Yvec, ncol=1);
    
    # Add ones to X
    X <<- cbind(rep(1, nrow(X)), X);
    
    # Intial theta
    initialTheta <- rep(0, ncol(X));
    
    # Cost at inital theta
    cost <- CostFunction(initialTheta);
    
    # Get optimal theta using gradient descent
    optimalTheta <- optim(par=initialTheta, fn=CostFunction, control = list(maxit = NUM_ITERATIONS));
    opttheta <<- optimalTheta$par;
    plot(opttheta);
    
    # Cost at optimal value of the theta
    Log("OPTIMAL THETA VAL = ", optimalTheta$value);
    Log("OPTIMAL THETA = " , opttheta);
    
    # Write theta out
    fileConn <- file(OPTIMAL_THETA_LOGFILE);
    write(opttheta, fileConn);
    close(fileConn);
  }
}

# Start Timing the training
ptm <- proc.time()

# Init log
InitLog();
Log("Starting");

if (!TEST_LKG_THETA_ONLY)
{
  # Run the main entry point
  Main();
  
  if (!PLOT_ONLY)
  {
    TestNonTrainingSamples(optimalTheta=opttheta);
  }
}
else
{
    TestNonTrainingSamples(optimalTheta=LAST_KNOWN_THETA);
}

# Print time elapsed
Log("TIME ELAPSED: ", (proc.time() - ptm)[3]);


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
POSITIVE_FOLDER <<- "positive";
NEGATIVE_FOLDER <<- "negative";
NUM_SAMPLES <<- 0.010 * 16000; # 0.01s worth of samples at 16KHz
SECTION_SIZE <<- 5;
AUDIO_THRESHOLD <<- 1.0; # 100% increase
NUM_ITERATIONS <<- 1000000;
LAST_KNOWN_THETA <<- c(-27.8164919, -3.0226850, 19.0595619, 39.2543972,  2.1650547,  -10.5485699, 11.8562808,  -10.0397605, -4.2093663,
                     18.0918260,  4.4826782,  -11.4337075,  1.7611361, 23.6021712, 13.2373426, 10.8262022,  6.3772729, 51.1833244,
                     -22.3437540,  -19.0647905, -7.8559344,  -21.8503614, -1.2691935,  -72.4302657, -8.2654490,  -22.6887958, -0.8781434,
                     33.6132753, -5.7572174,  9.8232980,  -64.7179481, 15.2351885, 15.6986117,  -12.0769334,  -23.9166253,  -31.0287779,
                     -11.6682373,   31.1061442,  -15.6104925,  8.5981416, -105.3439151, 17.2884851, 11.4259107, 20.5875689, 18.4730356,
                     36.5807801,  -19.7352117,    1.1097305,    7.2424743,  -32.8935622,  -14.4214863,   16.8791742,  -24.7333529,   83.5332348,
                     -6.5529256,    5.6602476,    3.2629208,   -7.3327598,  -37.7872961,  -30.4834054,  -36.1845442,   30.2478353,    5.0843796,
                     7.5903735,    1.7129605,   16.1006062,   55.1108557,  -16.2201632,   19.8286771,   16.4237267,    6.4736836,  -27.5675588,
                     -7.5071095,  -10.2859179,   -9.1975850,   25.0146315,   18.7301720,    3.2477635,   -2.7771972,   -7.6688111,   47.7914605,
                     8.5982414,   36.9307862,  126.7117815,  -12.8621962,   75.1519638,  -12.7980810,   14.7223907,  -10.5578728,  -1.7769516,
                     -7.9579436,   44.5045707,  -38.6368642,  -24.2154141,   12.2256529,  -36.6042055,   18.4965069,   -0.6358399,  -23.0002608,
                     -19.9429190,  -19.4849338,  -23.5531932,    0.6245739,  -83.9497376,    3.7820109,   -1.4902101,  -10.7905249,   -2.1321146,
                     13.1548949,  -50.2088694,   26.4564482,  -10.2281130,   19.8019665,  -30.3071886,    3.7619951,  51.5682140,  -25.7555855,
                     -17.9835730,   34.0622505,  -46.7124568,  -73.3928253,   24.5228007,    4.8563233,   21.5373128,    2.8660628,  -40.7459433,
                     -28.1900834,  -19.0662304,  -13.9713577,   16.2681342,  -18.0371543,   -0.6531429,   32.8164512,  -24.8722705,  -22.5157578,
                     2.2266209,   34.5146448,   18.0950284,   10.3973320,   75.4644192,    7.4392098,    8.3710820,   -2.1770807,  -23.2666710,
                     4.8791209,  -19.7972173,   -9.5381385,   29.4380440,  -3.6968011,  -10.7317907,    3.4487535,   32.9200031,   15.3617103,
                     -5.1097885,  -25.9643015,   20.9482219,    6.3808591,   27.5028573,   15.4133280,  -43.7522664,   15.0088310);

# GLOBALS
X <<- 0; # features
Y <<- 0; # results
opttheta <<- 0; # final theta calculated

# HELPER FUNCTIONS

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
  data <- load.wave(filePath)[1:NUM_SAMPLES];
  
  # Create a new plot for audio waveform
  dev.new();
  
  # Create a 2x2 grid plot and store the old par val for restoring later
  old.par <- par(mfrow=c(2,3));
  
  plot(data, main=filePath, xlab="Amplitude", ylab="Time");
  
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
  plot(energies);

  # Plot deltas between consecutive samples
  deltas <- GetDeltas(energies);
  plot(deltas);
  
  par(old.par);
}
#

# Load a wav file and extract features
LoadDataFromWav <- function(filePath)
{
  print(filePath);
  
  data <- load.wave(filePath);
  if (data$rate != 16000)
  {
    print(data$rate);
    print("ERROR! Sample rate != 16Khz");
    stop();
  }
  
  if (length(data) < NUM_SAMPLES)
  {
    print(length(data));
    print(NUM_SAMPLES);      
    print("ERROR! Not enoough samples");
    stop();
  }

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
      if (start + NUM_SAMPLES >= length(data))
      {
        # No more samples
        break;
      }
      
      features <- GetFeatures(data[start:(start+NUM_SAMPLES-1)]);
      features <- c(1, features);
      probability <- Sigmoid(features%*%optimalTheta);
      if (probability > 0.5)
      {
        print(paste("EVENT FOUND AT ", start));
        return (TRUE);
      }
    }
  }
  
  if (start == 0)
  {
    print("NO JUMP FOUND");
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
    print(filePath);
    
    data <- load.wave(filePath);
    
    wasEventFound <- CheckForEvent(data, optimalTheta)
    if (!wasEventFound)
    {
      print("FAILED: Event not found in positive sample");
      numIncorrect <- numIncorrect + 1;
    }
  }
  
  # Test negative data
  for (file in negativeTestFiles)
  {
    filePath <- paste(NEGATIVE_TEST_FOLDER, file, sep = "/");
    print(filePath);
    
    data <- load.wave(filePath);
    
    wasEventFound <- CheckForEvent(data, optimalTheta)
    if (wasEventFound)
    {
      print("FAILED: Event was found in negative sample");
      numIncorrect <- numIncorrect + 1;
    }
  }
  
  print(paste("NUM INCORRECT = ", numIncorrect));
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
    print(max(deltas));
    
    # Check if we have enough samples to use for training 
    if (start + NUM_SAMPLES >= length(data))
    {
      print("ERROR: No jump found in wav");
      stop();
    }
    
    Xvec <<- c(Xvec, GetFeatures(data[start:(start + NUM_SAMPLES - 1)]));
    Yvec <<- c(Yvec, 1);
    
    # Plot out discrete audio waveform, real and imaginary parts of the fft
    PlotAudioData(filePath);
  }
  
  # Load negative data
  for (file in negativeFiles)
  {
    filePath <- paste(NEGATIVE_FOLDER, file, sep = "/");
    
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
        if (start + NUM_SAMPLES >= length(data))
        {
          break;
        }
        
        Xvec <<- c(Xvec, GetFeatures(data[start: (start + NUM_SAMPLES - 1)]));
        Yvec <<- c(Yvec, 0);
      }
    }

    # Plot out discrete audio waveform, real and imaginary parts of the fft
    PlotAudioData(filePath);
  }
  
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
  print(optimalTheta$value);
}

# Start Timing the training
ptm <- proc.time()

# Run the main entry point
Main();

# TODO: Compare against non-training samples
TestNonTrainingSamples(optimalTheta=opttheta);
#TestNonTrainingSamples(optimalTheta=LAST_KNOWN_THETA);

# Print time elapsed
print(paste("TIME ELAPSED: ", (proc.time() - ptm)[3]));


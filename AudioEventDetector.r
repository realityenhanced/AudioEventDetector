#install.packages('caTools');
#install.packages('audio');
require(caTools);
require(audio);

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
  SECTION_SIZE <- 5;
  numSections <- length(data)%/%SECTION_SIZE;
  energies <- matrix(0, nrow = numSections, ncol = 1);
  for (i in 1:numSections)
  {
    energies[i] <- sum(abs(data[(1+((i-1)*SECTION_SIZE)) : (i*SECTION_SIZE)]));
  }
  plot(energies);

  # Plot deltas between consecutive samples
  delta <- energies[1:length(energies)-1];
  delta <- (abs(delta - energies[2:length(energies)])/delta) ;
  plot(delta);
  
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
    
    wasEventFound <- FALSE;
    for (start in (1: (length(data) - NUM_SAMPLES)))
    {
      features <- GetFeatures(data[start:(start+NUM_SAMPLES-1)]);
      features <- c(1, features);
      probability <- Sigmoid(features%*%optimalTheta);
      if (probability > 0.5)
      {
        wasEventFound <- TRUE;
        print(paste("EVENT FOUND AT ", start));
        break;
      }
    }
    
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
    
    wasEventFound <- FALSE;
    for (start in (1: (length(data) - NUM_SAMPLES)))
    {
      features <- GetFeatures(data[start:(start+NUM_SAMPLES-1)]);
      features <- c(1, features);
      probability <- Sigmoid(features%*%optimalTheta);
      if (probability > 0.5)
      {
        wasEventFound <- TRUE;
        print(paste("EVENT FOUND AT ", start));
        break;
      }
    }
    
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
    Xvec <<- c(Xvec, GetFeatures(data[1:NUM_SAMPLES]));
    Yvec <<- c(Yvec, 1);
    
    # Plot out discrete audio waveform, real and imaginary parts of the fft
    PlotAudioData(filePath);
  }
  
  # Load negative data
  for (file in negativeFiles)
  {
    filePath <- paste(NEGATIVE_FOLDER, file, sep = "/");
    
    data <- LoadDataFromWav(filePath);
    
    # Treat all NUM_SAMPLES sections as negative cases
    for (start in 1:(length(data) - NUM_SAMPLES))
    {
      Xvec <<- c(Xvec, GetFeatures(data[start: (start + NUM_SAMPLES - 1)]));
      Yvec <<- c(Yvec, 0);
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
  optimalTheta <- optim(par=initialTheta, fn=CostFunction, control = list(maxit = 200000));
  opttheta <<- optimalTheta$par;
  plot(opttheta);
  
  # Cost at optimal value of the theta
  print(optimalTheta$value);
  
  # TODO: Compare against non-training samples
  TestNonTrainingSamples(optimalTheta=opttheta);
}

# Run the main entry point
Main();

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
PlotAudioData <- function(data, title)
{
  # Create a new plot for audio waveform
  dev.new();
  
  # Create a 2x2 grid plot and store the old par val for restoring later
  old.par <- par(mfrow=c(2,3));
  
  plot(data, main=title, xlab="Amplitude", ylab="Time");
  
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
  
  # Plot the sectional means
  SECTION_SIZE <- 25;
  numMeans <- length(data)%/%SECTION_SIZE;
  means <- matrix(0, nrow = numMeans, ncol = 1);
  for (i in 1:numMeans)
  {
    means[i] <- mean(abs(data[(1+((i-1)*SECTION_SIZE)) : (i*SECTION_SIZE)]));
  }
  plot(means);

  # Plot deltas between consecutive samples
  delta <- means[1:length(means)-1];
  delta <- (abs(delta - means[2:length(means)])/delta) ;
  plot(delta);
  
  par(old.par);
}
#

# Load a wav file and extract features
LoadFeaturesFromWav <- function(filePath)
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
  
  # TODO: Up/Downsample the data to 16KHz & experiment with features
  # TMP: Use real parts of the FFT
  return (GetFeatures(data[1:NUM_SAMPLES]));
}
#

# Main entry point
Main <- function()
{
  # Files
  positiveFiles <- list.files(POSITIVE_FOLDER, pattern = "*.wav");
  negativeFiles <- list.files(NEGATIVE_FOLDER, pattern = "*.wav");
  numInputs <- length(positiveFiles) + length(negativeFiles);
  
  # Regression Variables
  X <<- matrix(NA, nrow = numInputs, ncol = NUM_SAMPLES);
  Y <<- matrix(NA, nrow = numInputs, ncol = 1);
  
  # TODO: Move this to a helper function
  # Load positive data
  currentRow <- 1;
  for (file in positiveFiles)
  {
    filePath <- paste(POSITIVE_FOLDER, file, sep = "/");
    print(filePath);
    
    X[currentRow,] <<- LoadFeaturesFromWav(filePath);
    Y[currentRow] <<- 1;
    
    # Plot out discrete audio waveform, real and imaginary parts of the fft
    PlotAudioData(X[currentRow,], filePath);
    
    currentRow <- currentRow + 1;
  }
  
  # Load negative data
  for (file in negativeFiles)
  {
    filePath <- paste(NEGATIVE_FOLDER, file, sep = "/");
    
    X[currentRow,] <<- LoadFeaturesFromWav(filePath);
    Y[currentRow] <<- 0;
    
    # Plot out discrete audio waveform, real and imaginary parts of the fft
    PlotAudioData(X[currentRow,], filePath);
    
    currentRow <- currentRow + 1;
  }
  
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
  
  # Print Prob values for all Training samples
  numIncorrect <- 0;
  for (i in 1:nrow(X))
  {
      prob <- Sigmoid(X[i,]%*%opttheta);
      if (prob > 0.5 && i <= length(positiveFiles))
      {
        print("AUDIO EVENT CORRECTLY DETECTED");
      }
      else if (prob <= 0.5 && i > length(positiveFiles))
      {
        print("NON-AUDIO CORRECTLY DETECTED");
      }
      else
      {
        print("WRONG JUDGEMENT");
        numIncorrect <- numIncorrect + 1;
      }
      print(prob);
  }
  print("NUM INCORRECT = ");
  print(numIncorrect);
  
  # TODO: Compare against non-training samples
  # ...
}

# Run the main entry point
Main();

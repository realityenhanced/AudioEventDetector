require(audio);

# Logistic Regression for binary classification of Audio data
# Positive Inputs are under the positive folder and Negative Inputs under the negative folder
# Audio files need to be uncompressed wav files.
# Sample rate of the training wav files could be anything and will be re-sampled to 16Khz, here.
# Only the first channel will be used for training, if a multi channel wave file is used as input.
# Duration of the training wav files is not mandated, only the first 60ms wndow will be used for training.
# So make sure the first 60ms contain the event to be detected.

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
  features <- Mod(ffts);
  
  lapply(features, function(x) { if (x <= 0.05) { return (0);} else if (x >= 0.95){ return (0.95); } })
  
  return(features);
}

# Plot waveforms
PlotAudioData <- function(data, title)
{
  # Create a new plot for audio waveform
  dev.new();
  
  # Create a 2x2 grid plot and store the old par val for restoring later
  old.par <- par(mfrow=c(2,2));
  
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
  
  par(old.par);
}
#

# Configuration Variables
POSITIVE_FOLDER <- "positive";
NEGATIVE_FOLDER <- "negative";
NUM_SAMPLES <- 0.06 * 16000; # 0.06s worth of samples at 16KHz

# Files
positiveFiles <- list.files(POSITIVE_FOLDER, pattern = "*.wav");
negativeFiles <- list.files(NEGATIVE_FOLDER, pattern = "*.wav");
numInputs <- length(positiveFiles) + length(negativeFiles);

# Regression Variables
X <- matrix(NA, nrow = numInputs, ncol = NUM_SAMPLES);
Y <- matrix(NA, nrow = numInputs, ncol = 1);

# TODO: Move this to a helper function
# Load positive data
currentRow <- 1;
for (file in positiveFiles)
{
  filePath <- paste(POSITIVE_FOLDER, file, sep = "/");
  
  data <- load.wave(filePath);
  print(length(data));
  
  # TODO: Up/Downsample the data to 16KHz & experiment with features
  # TMP: Use real parts of the FFT
  X[currentRow,] <- GetFeatures(data[1:NUM_SAMPLES]);
  Y[currentRow] <- 1;
  
  # Plot out discrete audio waveform, real and imaginary parts of the fft
  PlotAudioData(data[1:NUM_SAMPLES], filePath);
  
  currentRow <- currentRow + 1;
}

# Load negative data
for (file in negativeFiles)
{
  filePath <- paste(NEGATIVE_FOLDER, file, sep = "/");
  data <- load.wave(filePath);
  print(length(data));
  
  # TODO: Up/Downsample the data to 16KHz & experiment with features
  # TMP: Use mod of the FFT
  X[currentRow,] <- GetFeatures(data[1:NUM_SAMPLES]);
  Y[currentRow] <- 0;
  
  # Plot out discrete audio waveform, real and imaginary parts of the fft
  PlotAudioData(data[1:NUM_SAMPLES], filePath);
  
  currentRow <- currentRow + 1;
}

# Add ones to X
X <- cbind(rep(1, nrow(X)), X);

# Intial theta
initialTheta <- rep(0, ncol(X));

# Cost at inital theta
cost <- CostFunction(initialTheta);

# Get optimal theta using gradient descent
optimalTheta <- optim(par=initialTheta, fn=CostFunction, control = list(maxit = 200000));
theta <- optimalTheta$par;
plot(theta);

# Cost at optimal value of the theta
print(optimalTheta$value);

# Print Prob values for all Training samples
for (i in 1:nrow(X))
{
    prob <- Sigmoid(X[i,]%*%theta);
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
    }
    print(prob);
}

# TODO: Compare against non-training samples
# ...

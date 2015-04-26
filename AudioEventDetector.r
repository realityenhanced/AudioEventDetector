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
LOG_FILE <<- "log.log";
OPTIMAL_THETA_LOGFILE <<- "OptimalTheta.log";
POSITIVE_FOLDER <<- "positive";
NEGATIVE_FOLDER <<- "negative";
NUM_SAMPLES <<- 0.010 * 16000; # 0.01s worth of samples at 16KHz
SECTION_SIZE <<- 5;
AUDIO_THRESHOLD <<- 1.58; # % increase before event detection starts
NUM_ITERATIONS <<- 1000000;
LAST_KNOWN_THETA <<- c(-9.460925,-0.1676232,-11.45989,6.617613,-8.509732,-2.756163,26.00656,5.433123,-5.1109,-10.09702,-16.23733,-0.05093115,8.050109,-0.6314149,-0.8663847,
                       -11.16448,-7.606944,19.91732,5.978457,21.54498,-7.59011,-7.560708,-0.2843999,13.21496,-14.91531,5.159829,0.1982577,-8.455772,37.56543,-28.39935,
                       -7.83255,8.483995,-19.52272,8.583576,-25.24718,-8.742351,0.210617,-3.650255,-20.15341,-15.08544,7.416293,-10.89001,9.520285,14.39171,-23.16119,
                       -0.7648932,-4.760438,4.115804,-6.216832,-1.29017,6.860531,5.519303,24.78752,-14.65086,0.007336539,-2.137842,28.49022,-1.092005,2.727617,5.945737,
                       12.27002,1.223457,-21.90908,-4.764118,-31.2146,10.57923,-3.284467,-0.7315763,13.54201,10.29515,5.882691,-16.1281,9.250581,8.74087,-1.47653,
                       -0.6516207,6.362457,23.36108,12.61138,-13.99769,35.59,-9.914563,-25.32388,8.214159,14.93207,7.854837,-14.25614,4.137037,37.18404,10.64073,
                       -0.2137118,0.3182497,-11.87251,13.47162,-3.425885,-3.406238,-3.476826,2.104124,-8.46065,-14.90963,-16.7238,16.35584,1.74954,2.840431,20.9783,
                       -6.729115,4.470805,8.331585,-4.528757,-23.18686,13.94955,-3.478773,-8.3861,-9.642631,-7.446763,8.401106,-2.853355,2.04898,2.589609,-16.32461,
                       -8.572835,5.922126,-1.77466,2.70211,-7.660334,12.89781,-20.01668,-6.817063,-20.35047,-9.878235,-4.646381,-8.043022,-7.356339,24.86493,-4.047431,
                       2.897343,-9.198013,1.65108,-2.51988,-1.372048,22.12706,11.4689,4.221151,-8.020132,-9.583633,12.22753,11.01183,18.78962,-4.656781,2.190093,
                       -20.85469,9.341185,17.59717,8.878193,-2.542694,-2.510963,-19.03513,-1.501389,5.559725,-2.622976, -1.269152);

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
  
  # TODO: Vectorize this. For now treat zeroes as a very small value to prevent div by zero.
  for (i in (1:length(delta)))
  {
    if (delta[i] == 0)
    {
      delta[i] <- 0.00001;
    }
  }
  
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
    
    data <- load.wave(filePath);
    if (data$rate != 16000)
    {
      Log("ERROR! Sample rate != 16Khz, instead", data$rate);
      stop();
    }
    
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
    print(filePath);
    
    data <- load.wave(filePath);
    if (data$rate != 16000)
    {
      Log("ERROR! Sample rate != 16Khz, instead", data$rate);
      stop();
    }
    
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
    if (start + NUM_SAMPLES >= length(data))
    {
      Log("ERROR: No jump found in wav");
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
  Log("OPTIMAL THETA VAL = ", optimalTheta$value);
  Log("OPTIMAL THETA = " , opttheta);
  
  # Write theta out
  fileConn <- file(OPTIMAL_THETA_LOGFILE);
  write(opttheta, fileConn);
  writeLine("Cost", fileConn);
  close(fileConn);
}

# Start Timing the training
ptm <- proc.time()

# Init log
InitLog();
Log("Starting");

# Run the main entry point
Main();

# TODO: Compare against non-training samples
TestNonTrainingSamples(optimalTheta=opttheta);
TestNonTrainingSamples(optimalTheta=LAST_KNOWN_THETA);

# Print time elapsed
Log("TIME ELAPSED: ", (proc.time() - ptm)[3]);


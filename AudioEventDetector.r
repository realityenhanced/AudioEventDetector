require(audio);

# Logistic Regression for binary classification of Audio data
# Positive Inputs are under the positive folder and Negative Inputs under the negative folder
# Audio files need to be uncompressed wav files.
# Sample rate of the training wav files could be anything and will be re-sampled to 16Khz, here.
# Only the first channel will be used for training, if a multi channel wave file is used as input.
# Duration of the training wav files is not mandated, only the first 60ms wndow will be used for training.
# So make sure the first 60ms contain the event to be detected.

# Configuration Variables
POSITIVE_FOLDER <- "positive";
NEGATIVE_FOLDER <- "negative";

# Load positive data
positiveFiles <- list.files(POSITIVE_FOLDER, pattern = "*.wav");
for (file in positiveFiles)
{
  data <- load.wave(paste(POSITIVE_FOLDER, file, sep = "/"));
  
  # TODO: Create the X matrix after extracting feature sets from the data
  #       TBD.
}

# Load negative data
negativeFiles <- list.files(NEGATIVE_FOLDER, pattern = "*.wav");
for (file in negativeFiles)
{
  data <- load.wave(paste(NEGATIVE_FOLDER, file, sep = "/"));
}

require(audio);
wav <- load.wave('files/test.wav');
ffts <- fft(wav);
plot(ffts)

# Using LSTM to detect spoofing attacks in an Air-Ground network

## Specifications
- IDE: Spider
- Packages: 
  - Tensorflow 2.1.0
  - Keras
  - NumPy
  - Scikit-learn
  - Matplotlib

## Datasets:
- Training dataset is `trainX_H0__LSTM_IN.npy` that contains normal time-series data samples.
- Testing datasets include `testX_H0__LSTM_IN.npy` and `testX_H1__LSTM_IN.npy`.
  - `testX_H0__LSTM_IN.npy` contains **normal** time-series data samples
  - `testX_H1__LSTM_IN.npy` contains **abnormal** time-series data samples.
  - Note that `H0` means the hypothesis that there is no spoofing attack, and `H1` means the hypothesis that there is a spoofing attack from some spoofers/impersonators.
- The shape of each dataset is (82, 15, 15), 
where the first number (82) is the length of a time-series data sample, 
the second number (15) is the number of previous time slots that we want to look back for learning-from-the-past purposes,
and the last number (15) is the number of receive antennas (or the number of features).
- The following figure illustrates 2 time-series data samples, corresponding to `H0` and `H1`, respectively.
  
## Goals
- Train an LSTM autoencoder in order for it to learn the `H0` data samples.
- Once the LSTM autoencoder has been trained, it can capture the most significant characteristics of the `H0`-normal data.
- Test whether the testing datasets contain ``H1`-abnormal data samples that are associated with spoofing attacks.
- A detection rule relies on contrasting the output and the input of the LSTM autoencoder. 
Imagine that if the input is a `H0`-normal data sample, then the output should look similar to the input. In this case, the difference between the input and the output is insignificant.
On the other hand, if the input is a `H1`-abnormal data sample, then there is a big difference between the input and the output, because the trained LSTM autoencoder is meant to learn normal data samples.

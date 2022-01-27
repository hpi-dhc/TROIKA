# TROIKA heart rate monitoring

# TROIKA algorithm

This projects implements the [TROIKA](https://arxiv.org/pdf/1409.5181.pdf) framework for heart monitoring using Wrist-Type
Photoplethysmographic Signals.
TROIKA is a multi-stage algorithm for computing heart rate from wristband PPG sensors.
It operates on overlapping time windows of the time-domain PPG signal and produces an HR estimation for each window.
The computation for each window consists of the stages *signal decomposition*, *SSR (Sparse Signal Reconstruction)*, and *spectral peak tracking*.
Signal decomposition builds upon Singular Value decomposition (SVD) and divides the PPG signal into several signals of similar frequencies used for filtering noise.
SSR is an optimization algorithm to obtain a sparse frequency-domain representation of the signal with only few peaks representing heart rate.
Spectral peak tracking performs corrections of the projected HR frequency by making sure that the projected HR is similar across adjacent time windows.
Because each time window is based on the estimated HR of the previous time window, the algorithm is strictly linear and multiple time windows cannot be parallelized.

# State of this framework

This framework contains a Python reference implementation of TROIKA.
The signal decomposition stage and Spectral Peak Tracking stage work well.
The implementation is unfinished however, as the SSR algorithm imposed a major difficulty.
SSR is an optimization algorithm on large matrices with the main difficulty being reliable results in tolerable amounts of time.

# Contributing

This section offers an overview over the repository for everybody who wishes to contribute:

- **py_ppg_package** is a Python package containing the TROIKA framework and can be installed using pip
- **troika_main.py** can be used as an entry point for running and debugging the package
- **datasets** contains PPG signal datasets from the 2015 Signal Processing Cup. The same datasets are used in the [TROIKA paper](https://arxiv.org/pdf/1409.5181.pdf) for verification.

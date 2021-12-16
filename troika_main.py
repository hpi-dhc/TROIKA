import numpy as np
import pandas as pd
import plotly.express as px
import pypg
import scipy.io

import ppg_package

ppg_path = 'datasets/IEEE_SPC_2015/Training_data/DATA_01_TYPE01.mat'
hr_path = 'datasets/IEEE_SPC_2015/Training_data/DATA_01_TYPE01_BPMtrace.mat'
ppg_dataset = scipy.io.loadmat(ppg_path)
hr_dataset = scipy.io.loadmat(hr_path)
ppg = ppg_dataset['sig'][1]
acc = ppg_dataset['sig'][3:,:]
ground_truth_hr = hr_dataset['BPM0']

troika = ppg_package.Troika()

troika_predictions = []
i = 0
for pred in troika.transform(ppg, acc):
    print(f"pred: {pred}, real: {ground_truth_hr[i]}")
    i += 1

a = 9

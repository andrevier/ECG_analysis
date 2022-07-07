# ECG_analysis
This repository is an example of analysis of 3 datasets of electrocardiogram (ECG) signals using wavelet-based feature extraction and a support vector machine (SVM) classifier. The purpose of this project is to exercise the analysis of data with python and the APIs. The ECG dataset analysis is inspired by the MatWorks [tutorial](https://www.mathworks.com/help/wavelet/ug/ecg-classification-using-wavelet-features.html) and the original dataset is in the Matlab's [Git repository](https://github.com/mathworks/physionet_ECG_data/). 

The example uses 162 ECG recordings from three PhysioNet databases with 96 recordings from persons with arrhythmia, 30 recordings from persons with congestive heart failure, and 36 recordings from persons with normal sinus rhythms. The goal is to train a classifier to distinguish between arrhythmia (ARR), congestive heart failure (CHF), and normal sinus rhythm (NSR). 

## File tree

```
root
│   main.py
│   README.md    
│
└───physionet_ECG_data
│   │   ECGData.mat
│   │   License.txt
│   │   Modified_physionet_data.txt
|   |   README.md
```

- main.py: concentrates the manipulation of the dataset.
- README.md: Purpose of the project and description of the contents.
- ECGData.mat: dataset.
- License.txt: License file from the dataset owners.
- Modified_physionet_data.txt: A descripition of the preparation of the data.




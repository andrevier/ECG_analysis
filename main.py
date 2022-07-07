"""ECG dataset analysis inspired by the tutorial at
https://www.mathworks.com/help/wavelet/ug/ecg-classification-using-wavelet-features.html

The example uses 162 ECG recordings from three PhysioNet databases
with 96 recordings from persons with arrhythmia, 30 recordings from
persons with congestive heart failure, and 36 recordings from persons
with normal sinus rhythms. The goal is to train a classifier to 
distinguish between arrhythmia (ARR), congestive heart failure (CHF),
and normal sinus rhythm (NSR).
"""
from scipy import io
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

print("\nECG dataset analysis\n")

# Change the directory to the path of the respective file.
dir = "C:\\Users\\range\\Documents\\code\\ECG\\physionet_ECG_data"


os.chdir(dir)

# Transform the .mat into a dictionary with loadmat function.
matfile = io.loadmat("ECGData.mat")

# At the matlab browser, ECGData.mat appears as a struct 1x1
# with 2 fields: Data and Labels. Each field is a matrix with 
# the same number of rows. 'Labels' are the classification and 
# 'Data' are data signals.  The convertions to a dictionnary
# gives the followin keys:
print("\nConverting matfile to dict. The keys are:\n",matfile.keys())

# The data is inside ECGData key and everything else are metadata included during
# the conversion. The data is allocated in an numpy ndarray format.
ECGData_type = type(matfile["ECGData"])
print(f"data type of the key 'ECGData': {ECGData_type} \n ")

# The index [0] has a numpy.ndarray with the Data table content,
# whereas the [1] has the Labels. 
print("\ntype of ['ECGData'][0][0][0]\n",type(matfile["ECGData"][0][0][0])) 

# Conversion to pandas Dataframe.
data = pd.DataFrame(matfile["ECGData"][0][0][0])
labels = pd.DataFrame(matfile["ECGData"][0][0][1], columns=["Labels"])

print("\nConversion to Pandas Dataframe:\n Dataset:\n", data)
print("\nLabels in the Pandas DataFrame:\n Labels:\n", labels)

print("\nShape of the Data dataframe: ", data.shape)
print("\nShape of the Labels dataframe: ", labels.shape)

# Ramdomly split the dataset into training and test according to a 
# split ratio. The major part will be the training set and the indexes
# have to be integers.
split_ratio = 0.7
training_size = round(data.shape[0]*split_ratio)
test_size = data.shape[0] - training_size
print("\nChoosing Test and Training data sets.\n")
print("training set size: ",training_size)
print("test set size", test_size)

indexes = list(range(data.shape[0]))

training_id = random.sample(indexes, training_size)

# After the random sample for the training set, the test has the 
# remaining indexes.
test_id = indexes.copy()

for it in training_id:
    test_id.remove(it)

# Selection by list of indexes.
training_df = data.iloc[training_id]
test_df = data.iloc[test_id]

print("\ntraining data\n",training_df.head())
print("\ntest data\n",test_df.head())

# Check if the sets have different elements.
count = 0
for it in test_id:
    if it in training_id:
        count += 1

if count == 0:
    print("\nTest and training sets have different elements.")
else:
    print(f"\nError: Test and training sets have {count} elements in\
              common.\n")

# Examine the percentages for each label. First, the labels are stored
# as numpy.ndarray and have to be converted to string format.
training_label = labels.iloc[training_id]
training_label = training_label["Labels"].apply(lambda x: x[0])

test_label = labels.iloc[test_id]
test_label = test_label["Labels"].apply(lambda x: x[0])

print("\ntraining label (head)\n ", training_label.head())
print("\ntest label (head)\n ", test_label.head())

print("\n Partition of the training set:")
print("AAR ", training_label.loc[lambda x: x == "ARR"].count())
print("NSR ", training_label.loc[lambda x: x == "NSR"].count())
print("CHF ", training_label.loc[lambda x: x == "CHF"].count())
print("total: ", training_label.loc[lambda x: x == "ARR"].count()
               + training_label.loc[lambda x: x == "NSR"].count()
               + training_label.loc[lambda x: x == "CHF"].count())

print("\n Partition of the test set:")
print("AAR ", test_label.loc[lambda x: x == "ARR"].count())
print("NSR ", test_label.loc[lambda x: x == "NSR"].count())
print("CHF ", test_label.loc[lambda x: x == "CHF"].count())
print("total: ", test_label.loc[lambda x: x == "ARR"].count()
               + test_label.loc[lambda x: x == "NSR"].count()
               + test_label.loc[lambda x: x == "CHF"].count())

# Plot random data from all the labels.
plt.figure(figsize=(8,6))
plt.title("Plots from the training set.")
plt.xlabel("Samples")
plt.ylabel("Volts")
training_df.iloc[random.randint(0, training_size),:3000].plot()
plt.show()








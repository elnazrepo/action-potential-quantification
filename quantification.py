import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import find_peaks
# the class is used for detection of first index greater or smaller than a specific threshold
class threshold_detector:
    def __init__(self,signal,threshold):
        self.result(signal,threshold)
    def result(self,signal,threshold):
      self.thresholded_data = signal > threshold
      self.threshold_edges = np.convolve([1, -1], self.thresholded_data, mode='same')
      self.thresholded_edge_indices_high = np.where(self.threshold_edges == 1)[0]
      self.thresholded_edge_indices_low=np.where(self.threshold_edges==-1)[0]
# number of step is required for calculating threshold. how many steps are required to acquire time window of 1ms
frequency=50
step_number=int(1/(1/frequency))
# loading the data using pandas
result = pd.read_csv('sample_AP_train.csv')
data_array = np.array(result['Membrane Potential (mV)'])
time_array=np.array(result['Time (ms)'])
detector=threshold_detector(data_array,0)
h=detector.thresholded_edge_indices_high
# length of array of indexes where zero-crossing has occurred in leading edge, returns event numbers (action potentials)
number_of_events = len(h)
# time of events has been defined as the time array in which zero crossing has occurred
events=np.array([result._get_value(i,'Time (ms)') for i in h])
# further quantification can be applied. value of peaks is obtained
indexes, _ = scipy.signal.find_peaks(data_array, height=7, distance=2.1)
number_of_peaks=len(indexes)
peak=np.array([result._get_value(i,'Membrane Potential (mV)') for i in indexes ])
# threshold values are calculated
# since dv/dt is 20mv/ms for threshold voltage and each 50 rows (50*20=1ms) gives us 1ms, slope has been calculated
# first element which slope passes 20 mv , gives us the threshold
th=np.array([data_array[i+step_number]-data_array[i]  for i in range(len(data_array)-step_number)])
spike_threshold=threshold_detector(th,20)
spike_threshold1=spike_threshold.thresholded_edge_indices_high
thresholds=np.array([result._get_value(i,'Membrane Potential (mV)') for i in spike_threshold1])
thresholded_time=np.array([result._get_value(i,'Time (ms)') for i in spike_threshold1])
# rise time can be also calculated as a quantification parameter
# number of peaks and number of events are equal (both approaches can be used for event detection)
print('numbers of events is {}'.format(number_of_events))
print('number of peaks is {}'.format(number_of_peaks))
refractory=threshold_detector(data_array,-65)
refractory1=refractory.thresholded_edge_indices_low
output=[]
for index, item in enumerate(refractory1):
    if index == 0:
        output.append(item)
        output.append(refractory1[index+1])
    else:
        if index + 1 == len(refractory1):
            break
        else:
            next_item = refractory1[index + 1]
            if next_item - refractory1[index] > 40:
                output.append(next_item)
output.pop(len(output)-1)
refractory_periods=np.array([result._get_value(i,'Time (ms)') for i in output])
# absolute refractory period has been considered as time difference between threshold and resting voltage time(-65)
refractory_duration=(refractory_periods)-(thresholded_time)
# quantified data can be returned as pandas dataframe and be stored in a csv file
numpy_data = {'event time': events, 'peak values': peak,'threshold values':thresholds,'absolute refractory period':refractory_duration}
df = pd.DataFrame(numpy_data)
print(df)
df.to_csv('quantified data.csv')
# data can be plotted to visualize action potentials
result.plot(x='Time (ms)', y='Membrane Potential (mV)', style='-')
plt.plot(thresholded_time,thresholds,'r o')
plt.show()




### Import required packages
import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Helps to obtain the FFT
import scipy.fftpack    
# Various operations on signals (waveforms)
import scipy.signal as signal
                                                    
### Obtain ECG sample from CSV file using pandas
dataset = pd.read_csv("noise.csv")
y = [e for e in dataset.hart]

# Number of sample points
N = len(y)
# Sample spacing
Fs = 1000
T = 1.0 / Fs
# Compute x-axis
x = np.linspace(0.0, N*T, N)

# Compute FFT
yf = scipy.fftpack.fft(y)
# Compute frequency x-axis
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

## Declare plots for time-domain and frequency-domain plots
fig_td = plt.figure()
fig_td.canvas.manager.set_window_title('Time domain signals')
fig_fd = plt.figure()
fig_fd.canvas.manager.set_window_title('Frequency domain signals')

ax1 = fig_td.add_subplot(211)
ax1.set_title('Before filtering')
ax2 = fig_td.add_subplot(212)
ax2.set_title('After filtering')
ax3 = fig_fd.add_subplot(211)
ax3.set_title('Before filtering')
ax4 = fig_fd.add_subplot(212)
ax4.set_title('After filtering')     

# Plot non-filtered inputs
ax1.plot(x, y, color='r', linewidth=0.7)
ax3.plot(xf, 2.0/N * np.abs(yf[:N//2]), color='r', linewidth=0.7, label='raw')
ax3.set_ylim([0, 0.2])

### Compute filtering coefficients to eliminate 50Hz noise ###
# band_filt = np.array([45, 55])
# b, a = signal.butter(2, band_filt/(Fs/2), 'bandstop', analog=False)
b, a = signal.butter(4, 50/(Fs/2), 'low')

# Compute filtered signal
tempf = signal.filtfilt(b, a, y)
tempf = signal.filtfilt(b, a, y)
yff = scipy.fftpack.fft(tempf)

### Compute Kaiser window co-effs to eliminate baseline drift noise ###
nyq_rate = Fs / 2.0
# The desired width of the transition from pass to stop.
width = 5.0 / nyq_rate
# The desired attenuation in the stop band, in dB.
ripple_db = 60.0
# Compute the order and Kaiser parameter for the FIR filter.
O, beta = signal.kaiserord(ripple_db, width)
# The cutoff frequency of the filter.
cutoff_hz = 4.0

### Use firwin with a Kaiser window to create a lowpass FIR filter ###
taps = signal.firwin(O, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)
# Use lfilter to filter x with the FIR filter.
y_filt = signal.lfilter(taps, 1.0, tempf)
yff = scipy.fftpack.fft(y_filt)

# Plot filtered outputs
ax4.plot(xf, 2.0/N * np.abs(yff[:N//2]), color='g', linewidth=0.7)
ax4.set_ylim([0, 0.2])
ax2.plot(x, y_filt, color='g', linewidth=0.7)

### Compute beats ###
dataset['filt'] = y_filt

# Calculate moving average with 0.75s in both directions, then append to dataset
hrw = 1  # One-sided window size, as proportion of the sampling frequency
fs = 333  # The example dataset was recorded at 300Hz

mov_avg = dataset.filt.rolling(int(hrw * fs)).mean()

# Impute where moving average returns NaN (beginning of signal)
avg_hr = np.mean(dataset.filt)
mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
mov_avg = [(0.5 + x) for x in mov_avg]
mov_avg = [x * 1.2 for x in mov_avg]  # raise average by 20%
dataset['filt_rollingmean'] = mov_avg

# Mark regions of interest
window = []
peaklist = []
listpos = 0

for datapoint in dataset.filt:
    rollingmean = dataset.filt_rollingmean[listpos]
    
    if (datapoint < rollingmean) and (len(window) < 1):
        listpos += 1

    elif datapoint > rollingmean:
        window.append(datapoint)
        listpos += 1
        
    else:
        if len(window) > 0:
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window)))
            peaklist.append(beatposition)
            window = []
        listpos += 1

ybeat = [dataset.filt[x] for x in peaklist]

fig_hr = plt.figure()
fig_hr.canvas.manager.set_window_title('Peak detector')
ax5 = fig_hr.add_subplot(111)
ax5.set_title("Detected peaks in signal")
ax5.plot(dataset.filt, alpha=0.5, color='blue')
ax5.plot(mov_avg, color='green')
ax5.scatter(peaklist, ybeat, color='red')

# Compute heart rate
RR_list = []
cnt = 0
while cnt < (len(peaklist) - 1):
    RR_interval = (peaklist[cnt+1] - peaklist[cnt])
    ms_dist = ((RR_interval / fs) * 1000.0)
    RR_list.append(ms_dist)
    cnt += 1

bpm = 60000 / np.mean(RR_list)
print("\n\nAverage Heart Beat is: %.01f bpm" % bpm)
print("Number of peaks detected:", len(peaklist))

plt.show()
